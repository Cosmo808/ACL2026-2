import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
import copy


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


class ConditionalLIFNode(neuron.IFNode):
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

        self.v = self.jit_soft_reset(self.v, spike_d, 0.8*self.v_threshold)

        if self.reset_mask is not None:
            # Get reset condition for CURRENT time step
            assert self.current_step < self.reset_mask.shape[0], \
                f"Step {self.current_step} >= T={self.reset_mask.shape[0]}"

            # reset_mask: [T, B, 1] -> current: [B, 1]
            current_reset = self.reset_mask[self.current_step]  # [B, 1]
            reset_condition = current_reset & (spike_d > 0)
            self.v = self.jit_hard_reset(self.v, reset_condition, self.v_reset)
            self.current_step += 1  # move to next step

    def reset(self):
        self.past_v = []
        self.reset_mask = None
        self.current_step = 0
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class MembraneLoss(torch.nn.Module):
    def __init__(self, v_decay=1., i_decay=1., alpha=0., beta=0., *args, **kwargs):
        """
        :param mse: loss function
        :param v_decay: coefficient of v
        :param i_decay: coefficient of I
        :param alpha: weight of upper bound
        """
        super().__init__(*args, **kwargs)
        self.mse = torch.nn.MSELoss()
        self.v_decay = v_decay
        self.i_decay = i_decay
        self.alpha_value = torch.nn.Parameter(torch.tensor(alpha))
        self.beta_value = torch.nn.Parameter(torch.tensor(beta))

    def __call__(self, mem_seq, I, gt_idx, Vth=1.):
        mem_losses = 0.
        mem_seq = torch.stack(mem_seq)
        B, T = mem_seq.shape[1], mem_seq.shape[0]

        spike_num, not_spike_num = 0, 0
        spike_total, not_spike_total = 0, 0

        for b in range(B):
            # --- Encourage to Spike ---
            gt_i = gt_idx[b]
            mem_v = mem_seq[gt_i, b].squeeze(-1)
            up_bound_target = (torch.tensor(Vth) * self.v_decay + self.i_decay * I[b, gt_i].detach().clamp(0)).clamp(min=1.2*Vth, max=2*Vth)
            low_bound_target = torch.tensor(Vth)
            target = self.alpha * up_bound_target + (1 - self.alpha) * low_bound_target
            mse_loss = self.mse(mem_v, target)
            mem_losses = mem_losses + mse_loss

            # --- Discourage to Spike ---
            mask = torch.ones(T, dtype=torch.bool, device=mem_seq.device)
            mask[gt_i] = False
            mem_v_others = mem_seq[mask, b].squeeze(-1)
            up_bound_target = (self.i_decay * I[b, mask].detach().clamp(min=0, max=0.8*Vth))
            low_bound_target = torch.tensor(0.)
            target = self.beta * up_bound_target + (1 - self.beta) * low_bound_target
            mse_loss = self.mse(mem_v_others, target)
            mem_losses = mem_losses + mse_loss

            # --- Record Accuracy ---
            spike_num += (mem_v >= 1).sum().item()
            spike_total += len(mem_v)
            not_spike_num += (mem_v_others < 1).sum().item()
            not_spike_total += len(mem_v_others)

        return mem_losses / B, {'spike': spike_num / spike_total, 'not_spike': not_spike_num / not_spike_total}

    @property
    def alpha(self):
        # return torch.sigmoid(self.alpha_value)
        return 1.2

    @property
    def beta(self):
        # return torch.sigmoid(self.beta_value)
        return 0.6