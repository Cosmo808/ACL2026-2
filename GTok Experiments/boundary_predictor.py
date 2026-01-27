import torch
import torch.nn as nn
from utils.networks import IFNode, LIFNode, ConditionalLIFNode, MembraneLoss
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer


class SNN(nn.Module):
    def __init__(self, d_model, d_inner, activation_function, prior, bp_type, v_reset=0.):
        super(SNN, self).__init__()
        self.prior = prior
        self.bp_type = bp_type
        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.I = []
        self.node = ConditionalLIFNode(step_mode='m', v_reset=v_reset)
        self.loss = nn.BCEWithLogitsLoss()
        self.snn_loss = MembraneLoss()
        self.entropy_top_mask = None
        self.reset_logits = None

        # Boundary predictor
        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )
        # Reset predictor (predicts where to reset)
        self.reset_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

    def forward(self, hidden):
        """
        Args:
            hidden: [seq_len, batch_size, d_model]
        Returns:
            soft_boundaries: [batch_size, seq_len]
            hard_boundaries: [batch_size, seq_len]
        """
        # Predict boundaries
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)  # [B, T]
        # soft_boundaries = torch.sigmoid(boundary_logits)
        soft_boundaries = boundary_logits

        # Predict reset mask (will be supervised by entropy in loss)
        reset_logits = self.reset_predictor(hidden).squeeze(-1).transpose(0, 1)  # [B, T]

        # Use predicted reset for neuron reset (inference & training)
        reset_mask = (torch.sigmoid(reset_logits) > 0.5).transpose(0, 1).unsqueeze(-1)  # [T, B, 1]
        self.node.set_reset_mask(reset_mask)

        snn_input = boundary_logits.transpose(0, 1).unsqueeze(-1)  # [T, B, 1]
        hard_boundaries = self.node(snn_input).squeeze(-1).transpose(0, 1)  # [B, T]

        self.I = soft_boundaries
        self.reset_logits = reset_logits  # save for loss
        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds, gt, entropy=None):
        # preds, gt: [B, T]
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None

            # Default enhanced_gt
            enhanced_gt = gt.clone()
            reset_target = None

            # If entropy is provided (training), use it to supervise
            if entropy is not None:
                # entropy: [T, B] → convert to [B, T]
                entropy_bt = entropy.transpose(0, 1)  # [B, T]
                B, T = entropy_bt.shape
                k = max(1, int(0.2 * T))
                entropy_top_mask = torch.zeros_like(entropy_bt, dtype=torch.bool)
                for b in range(B):
                    _, top_indices = torch.topk(entropy_bt[b], k, largest=True)
                    entropy_top_mask[b, top_indices] = True

                # Enhance boundary GT with high-entropy positions
                enhanced_gt = gt | entropy_top_mask
                reset_target = entropy_top_mask.float()  # [B, T]

            # bce_loss = self.loss(preds, enhanced_gt.float())

            # Reset predictor loss
            reset_loss = 0.0
            if reset_target is not None and self.reset_logits.shape == reset_target.shape:
                reset_loss = self.loss(self.reset_logits, reset_target)

            # SNN membrane loss
            gt_idx = [torch.where(enhanced_gt[b])[0] for b in range(enhanced_gt.shape[0])]
            snn_loss = self.snn_loss(self.node.past_v, self.I, gt_idx)

            functional.reset_net(self)
            self.I = []
            return (snn_loss + reset_loss) / 2.0

        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(preds.size(-1), probs=torch.Tensor([self.prior]).to(preds.device))
            loss_boundaries = -binomial.log_prob(preds.sum(dim=-1)).mean() / preds.size(-1)
            return loss_boundaries


class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function, temp, prior, bp_type, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temp, probs=boundary_probs)
            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (hard_boundaries - soft_boundaries.detach() + soft_boundaries)
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds, gt, entropy=None):
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(preds.size(-1), probs=torch.Tensor([self.prior]).to(preds.device))
            loss_boundaries = -binomial.log_prob(preds.sum(dim=-1)).mean() / preds.size(-1)
            return loss_boundaries

