import torch

from utils.networks import *
from shortening import downsample, upsample
from boundary_predictor import *


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm, model_config,
                 activation_function, boundaries_type, spikes_left, temp, prior, entropy):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.entropy = entropy

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])
            return layers

        pre_layers, (shortened_layers,), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == 'none'
            self.layers = nn.ModuleList([create_decoder_layers(pre_layers)])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList([create_decoder_layers(pre_layers), create_decoder_layers(shortened_layers), create_decoder_layers(post_layers), ])
            self.down_ln = nn.LayerNorm(d_model)

            # Boundary predictor
            self.boundary_predictor = SNN(d_model, d_inner, activation_function, prior, boundaries_type)
            # self.boundary_predictor = BoundaryPredictor(d_model, d_inner, activation_function, temp, prior, boundaries_type)
            self.spikes_left = spikes_left

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()
        dec_attn_mask = torch.triu(core_input.new_ones(qlen, qlen), diagonal=1).bool()
        pos_seq = torch.arange(qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)
        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask)
        return core_out

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()
        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask
        return total

    def get_boundary_num(self, data):
        with torch.no_grad():
            word_emb = self.word_emb(data)
            hidden = self.drop(word_emb)
            hidden = self._forward(core_input=hidden, layers=self.layers[0])
            soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)
            return hard_boundaries.sum().item()

    def forward(self, data, target, boundaries_gt):
        """
            data: T x B
            target: T x B
            boundaries_gt: T x B or None
        """
        stats = {}

        # All batches should be of the same length, but last can be shorter
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Token_ids to vector embeddings -> T x B x C
        word_emb = self.word_emb(data)
        hidden = self.drop(word_emb)

        # Process input with Transformer blocks
        hidden = self._forward(core_input=hidden, layers=self.layers[0])

        # Tokenization
        residual = hidden
        if self.boundaries_type in ['fixed', 'whitespaces']:
            hard_boundaries = boundaries_gt.float().transpose(0, 1)  # B x T
        else:
            soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)

        # Downsampling
        hidden = downsample(boundaries=hard_boundaries, hidden=hidden, null_group=self.null_group, )
        hidden = self.down_ln(hidden)
        stats['p_ones'] = (hard_boundaries.sum() / hard_boundaries.numel()).item()
        stats['shortened_length'] = hidden.size(0)
        hidden = self._forward(core_input=hidden, layers=self.layers[1])

        # Upsampling
        back_hidden = upsample(boundaries=hard_boundaries, shortened_hidden=hidden, )
        hidden = back_hidden + residual
        hidden = self._forward(core_input=hidden, layers=self.layers[2])

        # Calculate loss
        hidden = hidden[-tgt_len:]
        logit = self.final_cast(hidden)

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)

            # Entropy
            if self.entropy:
                entropy = -torch.nn.functional.log_softmax(logit, dim=-1) * torch.nn.functional.softmax(logit, dim=-1)
                entropy = torch.sum(entropy, dim=-1)  # T x B
            else:
                entropy = None

            # Boundary predictor loss
            if self.boundaries_type == 'entropy':
                target_boundaries = self.get_spikes(entropy).transpose(0, 1)  # B x T
                # target_boundaries: B x T
            elif self.boundaries_type == 'unigram':
                # T x B
                target_boundaries = boundaries_gt[-tgt_len:].transpose(0, 1)
                # B x T
            elif self.boundaries_type == 'gumbel':
                target_boundaries = None

            soft_boundaries = soft_boundaries[:, -tgt_len:]
            hard_boundaries = hard_boundaries[:, -tgt_len:]

            if self.boundaries_type in ['unigram', 'entropy']:
                assert target_boundaries.sum().item() > 0
                # Pass entropy to calc_loss for reset supervision
                loss_boundaries = self.boundary_predictor.calc_loss(soft_boundaries, target_boundaries, entropy=entropy)
                bp_stats = calc_stats(hard_boundaries, target_boundaries)
            elif self.boundaries_type == 'gumbel':
                loss_boundaries = self.boundary_predictor.calc_loss(preds=hard_boundaries, gt=None)
                bp_stats = calc_stats(hard_boundaries, (data == 0)[-tgt_len:].transpose(0, 1))

            for k, v in bp_stats.items():
                stats[f'{k}'] = v
            stats['loss_boundaries'] = loss_boundaries.item()

            # LM loss
            logit = logit.view(-1, logit.size(-1))
            target = target.view(-1)

            loss = self.crit(logit, target)
            loss = loss.view(tgt_len, -1)

            return loss, stats, loss_boundaries, logit
        else:
            # Generation mode, we return raw logits
            return logit
