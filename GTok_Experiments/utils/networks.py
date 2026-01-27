import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
import copy


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class IFNode(neuron.IFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class LIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class ConditionalLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []
        self.reset_mask = None  # [T, B, 1]
        self.current_step = 0  # track current time step

    def set_reset_mask(self, mask):
        """mask: [T, B, 1] or [T, B]"""
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [T, B] -> [T, B, 1]
            self.reset_mask = mask
        else:
            self.reset_mask = None
        self.current_step = 0  # reset step counter

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.past_v.append(self.v)

        if self.v_reset is None:
            # soft reset (optional)
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            if self.reset_mask is not None:
                # Get reset condition for CURRENT time step
                assert self.current_step < self.reset_mask.shape[0], \
                    f"Step {self.current_step} >= T={self.reset_mask.shape[0]}"

                # reset_mask: [T, B, 1] -> current: [B, 1]
                current_reset = self.reset_mask[self.current_step]  # [B, 1]
                reset_condition = current_reset & (spike_d > 0)
                self.v = torch.where(reset_condition, torch.full_like(self.v, self.v_reset), self.v)
                self.current_step += 1  # move to next step
            else:
                # No reset at all
                pass

    def reset(self):
        self.past_v = []
        self.reset_mask = None
        self.current_step = 0
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class MembraneLoss(torch.nn.Module):
    def __init__(self, mse=torch.nn.MSELoss(), v_decay=1, i_decay=1, alpha=0., *args, **kwargs):
        """
        :param mse: loss function
        :param v_decay: coefficient of v
        :param i_decay: coefficient of I
        :param alpha: weight of upper bound
        """
        super().__init__(*args, **kwargs)
        self.mse = mse
        self.v_decay = v_decay
        self.i_decay = i_decay
        self.alpha_value = torch.nn.Parameter(torch.tensor(alpha))

    def __call__(self, mem_seq, I, gt_idx, Vth=1.):
        mem_loss = 0.
        mem_seq = torch.stack(mem_seq)
        B = mem_seq.shape[1]
        for b in range(B):
            gt_i = gt_idx[b]
            mem_v = mem_seq[gt_i, b].squeeze(-1)

            up_bound_target = (torch.tensor(Vth) * self.v_decay + self.i_decay * I[b, gt_i].detach().clamp(0)).clamp(min=Vth)
            low_bound_target = torch.tensor(Vth)
            target = self.alpha * up_bound_target + (1 - self.alpha) * low_bound_target
            mem_loss = mem_loss + self.mse(mem_v, target)

        return mem_loss / B

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_value)


def calc_stats(preds, gt):
    # B x T
    preds, gt = preds.bool(), gt.bool()
    TP = ((preds == gt) & preds).sum().item()
    FP = ((preds != gt) & preds).sum().item()
    FN = ((preds != gt) & (~preds)).sum().item()

    acc = (preds == gt).sum().item() / gt.numel()

    if TP == 0:
        precision, recall = 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    stats = {'acc': acc, 'precision': precision, 'recall': recall}
    return stats


