import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(
            self,
            n_head=8,
            d_model=512,
            d_head=64,
            d_inner=1024,
            dropout=0.1,
            dropatt=0.1,
            activation_function='gelu',
    ):
        super(ContextEncoder, self).__init__()

        self.attn = MultiHeadAttn(n_head, d_model, d_head, dropout, dropatt)
        self.pos_ff = PositionFF(d_model, d_inner, dropout, activation_function)

        self.drop = nn.Dropout(0.1)
        self.pos_emb = PositionalEmbedding(d_model)

        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head).zero_())
        
    def _forward(self, dec_inp, r, dec_attn_mask=None):
        # output = self.attn(dec_inp, r, self.r_w_bias, self.r_r_bias, attn_mask=dec_attn_mask)
        output = self.pos_ff(dec_inp)
        return output

    def forward(self, input):
        # input is of size (B x T x D)
        input = input.transpose(0, 1)
        qlen, _, _ = input.size()
        attn_mask = torch.triu(input.new_ones(qlen, qlen), diagonal=1).bool()
        pos_seq = torch.arange(qlen - 1, -1, -1.0, device=input.device, dtype=input.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)
        output = input
        output = self._forward(output, pos_emb, attn_mask).transpose(0, 1)  # (B x T x D)
        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt):
        super(MultiHeadAttn, self).__init__()

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

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
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
        attn_score = self.scale * (AC + BD)

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
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        output = self.layer_norm(w + attn_out)

        return output


class PositionFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, activation_function):
        super(PositionFF, self).__init__()

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

    def forward(self, inp):
        core_out = self.CoreNet(inp)
        output = self.layer_norm(inp + core_out)
        return output


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