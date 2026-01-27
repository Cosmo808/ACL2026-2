import itertools
import time
import math
from tqdm import tqdm
from collections import defaultdict

from spikingjelly.activation_based import functional

import torch.cuda
from torch.nn.parallel import DistributedDataParallel

from utils.distributed import *
from utils.exp_utils import *
from boundary_predictor import SNN


class Trainer:
    def __init__(self, data_iters, model, model_config, optimizer, scheduler, vocab, args, scaler):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_iter = data_iters['train']
        self.valid_iter = data_iters['valid']
        self.test_iter = data_iters['test']

        self.model = model
        self.model_config = model_config
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.vocab = vocab
        self.args = args
        self.scaler = scaler

        self.epoch = 1
        self.train_step = 0
        self.last_iter = 0
        self.init_time = time.time()
        self.best_val_loss = np.inf

        self.load()

    def train(self):
        for epoch in itertools.count(start=self.epoch):
            self.epoch = epoch
            if self.args.roll:
                self.train_iter.roll(seed=self.args.seed + epoch)

            train_iter = self.train_iter.get_fixlen_iter(start=self.last_iter, shuffle=self.args.shuffle,
                                                         seed=self.args.seed + epoch, nw=self.args.nw)
            self.run_epoch(train_iter, training=True)

            if self.train_step == self.args.max_step:
                print('End of training')
                break

    def run_epoch(self, data_iter, training):
        train_loss = 0
        target_tokens = 0
        log_step = 0
        log_start_time = time.time()
        stats_agg = defaultdict(list)

        eval_loss = 0
        eval_len = 0

        if training:
            self.model.train()
        else:
            self.model.eval()
            try:
                self.model.boundary_predictor.node.training = True
            except AttributeError:
                pass
            data_iter = data_iter.get_fixlen_iter()

        for batch, (data, target, seq_len, boundaries_gt) in enumerate(data_iter, start=1):
            # Load data
            data = data.to(self.device, non_blocking=True)
            data_chunks = torch.chunk(data, self.args.batch_chunk, 1)
            target = target.to(self.device, non_blocking=True)
            target_chunks = torch.chunk(target, self.args.batch_chunk, 1)
            boundaries_gt = boundaries_gt.to(self.device, non_blocking=True)
            boundaries_gt_chunks = torch.chunk(boundaries_gt, self.args.batch_chunk, 1)

            # Train on batch
            self.optimizer.zero_grad()
            for i in range(self.args.batch_chunk):
                if i < self.args.batch_chunk - 1 and isinstance(self.model, DistributedDataParallel):
                    with self.model.no_sync():
                        train_loss_chunk, stats = self.run_iter(i, data_chunks, target_chunks, boundaries_gt_chunks, training)
                else:
                    train_loss_chunk, stats = self.run_iter(i, data_chunks, target_chunks, boundaries_gt_chunks, training)

                if training:
                    train_loss += train_loss_chunk
                else:
                    eval_loss += train_loss_chunk * seq_len
                    eval_len += seq_len

            for k, v in stats.items():
                stats_agg[k].append(v)

            if isinstance(self.model.boundary_predictor, SNN):
                functional.reset_net(self.model.boundary_predictor)
                self.model.boundary_predictor.I = []

            if training:
                if self.args.fp16:
                    self.scaler.unscale_(self.optimizer)
                try:
                    grad_l2 = (sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5)
                    weights_l2 = (sum(p.detach().norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5)
                except AttributeError:
                    grad_l2, weights_l2 = 0., 0.
                stats_agg['grad_l2'].append(grad_l2)
                stats_agg['weights_l2'].append(weights_l2)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                self.train_step += 1

                # Step-wise learning rate annealing
                if self.args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.train_step < self.args.warmup_step:
                    curr_lr = self.args.lr * self.train_step / self.args.warmup_step
                    self.optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    self.scheduler.step(self.train_step - self.args.warmup_step)

                # Logging
                log_step += 1
                target_tokens += target.numel()
                if self.train_step % self.args.log_interval == 0 or self.train_step == 1:
                    self.logging(batch, train_loss, target_tokens, log_step, log_start_time, training)
                    train_loss, log_step, target_tokens, log_start_time = 0, 0, 0, time.time()

                # Valid
                do_periodic_eval = self.train_step % self.args.eval_interval == 0
                is_final_step = self.train_step == self.args.max_step
                if do_periodic_eval or is_final_step:
                    eval_start_time = time.time()

                    val_loss, stats_val = self.run_epoch(self.valid_iter, training=False)
                    self.logging(batch, val_loss, None, None, eval_start_time, False)

                    log_start_time += time.time() - eval_start_time
                    last_iter = self.train_iter.last_iter

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(self.args, self.model, self.model_config, self.optimizer, self.scheduler, self.vocab,
                                        self.epoch, batch, last_iter, self.train_step, self.args.work_dir, self.scaler)

                if is_final_step:
                    break

        if not training:
            self.model.train()
            eval_loss = all_reduce_item(eval_loss / eval_len, op='mean')
            return eval_loss, stats_agg

    def run_iter(self, i, data, target, boundaries_gt, training):
        data = data[i].contiguous()
        target = target[i].contiguous()
        boundaries_gt = boundaries_gt[i].contiguous()

        with torch.cuda.amp.autocast(self.args.fp16):
            seq_loss, stats, aux_loss, _ = self.model(data, target, boundaries_gt)
            seq_loss = seq_loss.float().mean().type_as(seq_loss)
            total_loss = (seq_loss + aux_loss) / self.args.batch_chunk

        if training:
            if self.args.fp16:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

        return seq_loss.item() / self.args.batch_chunk, stats

    def logging(self, batch, loss, target_tokens, log_step, log_start_time, training):
        if training:
            cur_loss = loss / log_step
            cur_loss = all_reduce_item(cur_loss, op='mean')
            lr = self.optimizer.param_groups[0]['lr']

            throughput = target_tokens / (time.time() - log_start_time)
            throughput = all_reduce_item(throughput, op='sum')

            log_str = '| epoch {:3d} step {:>7d} | batches {:>6d} / {:d} | lr {:.3e} | tok/s {:6.0f} | loss {:5.3f} | elapse {:5.1f} min'.format(
                self.epoch, self.train_step, batch, self.train_iter.n_batch, lr, throughput, cur_loss, (time.time() - self.init_time) / 60)
            print_once(log_str, self.args)
        else:
            print_once('-' * 100, self.args)
            log_str = '| Eval {:3d} at step {:>7d} | time: {:3.1f} min | valid loss {:5.5f}'.format(
                self.train_step // self.args.eval_interval, self.train_step, (time.time() - log_start_time) / 60, loss)
            print_once(log_str, self.args)
            print_once('-' * 100, self.args)

    def evaluate(self):
        test_loss, stats_test = self.run_epoch(self.test_iter, False)
        print_once('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(test_loss, test_loss / math.log(2)), self.args)
        self.cal_SF()

    def cal_SF(self):
        # Calculate shorten factor
        self.model.eval()
        try:
            self.model.boundary_predictor.node.training = True
        except AttributeError:
            pass
        data_iter = self.test_iter.get_fixlen_iter()

        boundary_num = 0
        for data, _, _, _ in tqdm(data_iter):
            data = data.to(self.device, non_blocking=True)
            boundary_num = boundary_num + self.model.get_boundary_num(data)
        print('Boundary number:', boundary_num)

    def load(self):
        if self.args.ckpt:
            print(f"Loading checkpoint from {self.args.ckpt}")
            ckpt = torch.load(self.args.ckpt, map_location='cuda', weights_only=False)
            # self.args = ckpt['args']
            # self.model_config = ckpt['model_config']
            self.model.load_state_dict(ckpt['model_state'], strict=False)
            # self.optimizer.load_state_dict(ckpt['optimizer_state'])
            # self.scheduler.load_state_dict(ckpt['scheduler_state'])
            # self.scaler.load_state_dict(ckpt['scaler'])
            # self.vocab = ckpt['vocab']
            # self.epoch = ckpt['epoch']
            # self.train_step = ckpt['train_step']