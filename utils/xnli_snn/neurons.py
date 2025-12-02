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


