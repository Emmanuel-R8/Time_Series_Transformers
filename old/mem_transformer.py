import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

from utils.proj_adaptive_sigmoid import ProjectedAdaptiveSigmoid

import data_utils


class PositionalEmbedding(LightningModule):
    def __init__(self, d_emb):
        super(PositionalEmbedding, self).__init__()

        assert d_emb % 2 == 0, "The embedding size n_model must be an even number"
        self.d_emb = d_emb
        self.n_emb = math.ceil(d_emb / 2)

        # Instead of writing sin(x / f) will use sin(x * inv_freq)
        # Frequencies range from 1 to 10000**2, sort of exponential progression
        # with exactly d_pos_emb frequencies
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_emb, 2.0) / d_emb))
        # inv_freq = inv_freq.rename('InvFreq') unsupported by torch.ger

        # Register this variable as a constant
        # DIMS: ceiling(d_emb / 2)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):

        # torch.ger = outer product
        # DIMS: pos_seq x d_emb
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)

        # DIMS: pos_seq x (2 x d_emb)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            # DIMS: pos_seq x n_batch x (2 x d_emb)
            pos_emb = pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            # DIMS: pos_seq x n_batch x (2 x d_emb)
            pos_emb = pos_emb[:, None, :]

        # DIMS: pos_seq x n_batch x (2 x n_emb)
        assert pos_emb.size()[0] == pos_seq.size()[
            0], "pos_emb.size()[0] != pos_seq"
        if bsz is not None:
            assert pos_emb.size()[1] == bsz, "pos_emb.size()[1] != n_batch"
        assert (
                pos_emb.size()[2] == 2 * self.n_emb
        ), "pos_emb.size()[2] != 2 * self.n_emb"

        return pos_emb


class PositionwiseFF(LightningModule):
    def __init__(self, n_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.n_model = n_model
        self.d_inner = d_inner
        self.dropout = dropout

        # DIMS: n_model x n_model
        self.CoreNet = nn.Sequential(
            # DIMS: n_model x d_inner
            nn.Linear(n_model, d_inner),
            # DIMS: d_inner
            nn.ReLU(inplace=True),
            # DIMS: d_inner
            nn.Dropout(dropout),
            # DIMS: d_inner x n_model
            nn.Linear(d_inner, n_model),
            # DIMS: n_model
            nn.Dropout(dropout),
        )

        # DIMS: n_model
        self.layer_norm = nn.LayerNorm(n_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, input):
        # DIMS: input -> tgt_len x n_batch x n_model
        assert (
                input.size()[2] == self.n_model
        ), "PositionWideFF/forward: input.size()[0] != self.n_model"

        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            # DIMS: core_out -> tgt_len x n_batch x n_model
            core_out = self.CoreNet(self.layer_norm(input))

            # residual connection
            # DIMS: output -> tgt_len x n_batch x n_model
            output = core_out + input
        else:
            # positionwise feed-forward
            # DIMS: core_out -> tgt_len x n_batch x n_model
            core_out = self.CoreNet(input)

            # residual connection + layer normalization
            # DIMS: output -> tgt_len x n_batch x n_model
            output = self.layer_norm(input + core_out)

        return output


class MultiHeadAttn(LightningModule):
    def __init__(self, n_head, n_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.n_model = n_model
        self.d_head = d_head
        self.dropout = dropout

        # DIMS: n_model x n_head*d_head
        self.q_net = nn.Linear(n_model, n_head * d_head, bias=False)

        # DIMS: n_model x 2*n_head*d_head
        self.kv_net = nn.Linear(n_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # DIMS: n_head*d_head x n_model
        self.o_net = nn.Linear(n_head * d_head, n_model, bias=False)

        # DIMS: n_model
        self.layer_norm = nn.LayerNorm(n_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # [hlen x n_batch x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        # DIMS: q_net -> n_model x n_head * d_head
        head_q = self.q_net(h)

        # DIMS: kv_net -> n_model x 2*n_head*d_head
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x n_batch x n_head]
        attn_score = torch.einsum("ibnd,jbnd->ijbn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None],
                                        -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x n_batch x n_head]
        attn_prob = torch.sigmoid(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x n_batch x n_head] x [klen x n_batch x n_head x d_head] ->
        # [qlen x n_batch x n_head x d_head]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        # DIMS: n_head*d_head x n_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(LightningModule):
    def __init__(
            self,
            n_head,
            n_model,
            d_head,
            dropout,
            dropatt=0,
            tgt_len=None,
            ext_len=None,
            mem_len=None,
            pre_lnorm=False,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.n_model = n_model
        self.d_head = d_head
        self.dropout = dropout

        # DIMS: n_model x 3*n_head*d_head
        self.qkv_net = nn.Linear(n_model, 3 * n_head * d_head, bias=False)

        # DIMS:
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # DIMS: n_head*d_head x n_model
        self.o_net = nn.Linear(n_head * d_head, n_model, bias=False)

        # DIMS: n_model
        self.layer_norm = nn.LayerNorm(n_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):

        # Create a block of zeros that will be added along the 4th dimension
        # DIMS: x0 x x1 x x2 x 1
        zero_pad = torch.zeros(
            (x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype
        )

        # Add along the 4th dimension
        # DIMS: x0 x x1 x x2 x (x3 + 1)
        x_padded = torch.cat([zero_pad, x], dim=3)

        # CHECK: Those 2 lines makes little sense
        # x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        # x = x_padded[:, :, 1:].view_as(x)

        # This version retains the original shape of x
        # DIMS: x0 x x1 x x2 x x3
        x = x_padded[:, :, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))

            # Return a lower triangular matrix
            x = x * torch.tril(ones, diagonal=x.size(3) - x.size(2))[None, None,
                    :, :]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        # DIMS: n_model x n_head*d_head
        self.r_net = nn.Linear(self.n_model, self.n_head * self.d_head,
                               bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        # DIMS: w -> tgt_len x n_batch x n_model
        # qlen = tgt_len
        # DIMS: r -> tgt_len x 1 x n_model
        # rlen = tgt_len
        # DIMS: r -> n_head x d_head
        # DIMS: r -> n_head x d_head

        if mems is not None:
            # DIMS: 0 at the beginning

            # Concatenate memories + current segment along 1st dimension
            # DIMS: mems -> n_mem x n_model
            # DIMS: w ->    tgt_len x n_batch x n_model
            # DIMS: cat ->  (n_mem+n_model) x n_model

            # CHECK: mems has to be copied to have one per training value in
            # the batch
            # DIMS: w -> tgt_len x n_batch x n_model
            # DIMS: mems -> ???
            # DIMS: cat -> (tgt_len + ???) x n_batch x n_model
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                # DIMS: cat ->     tgt_len x n_batch x n_model
                # DIMS: qkc_net -> n_model x 3*n_head*d_head
                # DIMS: w_heads -> (tgt_len + ???) x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                # DIMS: cat ->     tgt_len x n_batch x n_model
                # DIMS: qkc_net -> n_model x 3*n_head*d_head
                # DIMS: w_heads -> (tgt_len + ???) x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(cat)

            # DIMS: r -> tgt_len x 1 x n_model
            # DIMS: r_net -> n_model x n_head*d_head
            # DIMS: r_head_k -> tgt_len x n_head x d_head
            r_head_k = self.r_net(r)

            # DIMS: w_heads -> tgt_len x n_batch x 3*n_head*d_head
            # DIMS: w_head_q -> tgt_len x n_batch x n_head x d_head
            # DIMS: w_head_k -> tgt_len x n_batch x n_head x d_head
            # DIMS: w_head_v -> tgt_len x n_batch x n_head x d_head
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            # DIMS: w_head_q -> qlen x n_batch x n_head x d_head
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                # DIMS: w -> tgt_len x n_batch x n_model
                # DIMS: qkv_net -> n_model x 3*n_head*d_head
                # DIMS: w_heads -> tgt_len x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # DIMS: w -> tgt_len x n_batch x n_model
                # DIMS: qkv_net -> n_model x 3*n_head*d_head
                # DIMS: w_heads -> tgt_len x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(w)

            # DIMS: r -> tgt_len x 1 x n_model
            # DIMS: r_net -> n_model x n_head*d_head
            # DIMS: r_head_k -> tgt_len x n_head x d_head
            r_head_k = self.r_net(r)

            # DIMS: w_heads -> tgt_len x n_batch x 3*n_head*d_head
            # DIMS: w_head_q -> tgt_len x n_batch x n_head x d_head
            # DIMS: w_head_k -> tgt_len x n_batch x n_head x d_head
            # DIMS: w_head_v -> tgt_len x n_batch x n_head x d_head
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        # klen = tgt_len
        klen = w_head_k.size(0)

        # DIMS: qlen x n_batch x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)

        # DIMS: klen x n_batch x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)

        # DIMS: klen x n_batch x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        # DIMS: rlen x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        # compute attention score
        # DIMS: w_head_q ->  qlen x n_batch x n_head x d_head
        # DIMS: r_w_bias ->               n_head x n_head
        # DIMS: rw_head_q -> qlen x n_batch x n_head x d_head
        rw_head_q = w_head_q + r_w_bias

        # DIMS: rw_head_q -> qlen x n_batch x n_head x d_head
        # DIMS: w_head_k ->  klen x n_batch x n_head x d_head
        # DIMS: AC ->        n_batch x n_head x qlen x klen
        AC = torch.einsum("ibnd,jbnd->bnij", (rw_head_q, w_head_k))

        # DIMS: w_head_q ->  qlen x n_batch x n_head x d_head
        # DIMS: r_r_bias ->               n_head x n_head
        # DIMS: rr_head_q -> qlen x n_batch x n_head x d_head
        rr_head_q = w_head_q + r_r_bias

        # DIMS: rr_head_q -> qlen x n_batch x n_head x d_head
        # DIMS: r_head_k ->        rlen x n_head x d_head
        # DIMS: BD ->        n_batch x n_head x qlen x rlen
        BD = torch.einsum("ibnd,jnd->bnij", (rr_head_q, r_head_k))

        # DIMS: BD -> n_batch x n_head x qlen x rlen
        BD = self._rel_shift(BD)

        # DIMS: AC -> n_batch x n_head x qlen x klen
        # DIMS: BD -> n_batch x n_head x qlen x rlen
        # DIMS: attn_score -> n_batch x n_head x qlen x rlen
        # klen = rlen = tgt_len
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        # DIMS: attn_mask ->                 tgt_len x tgt_len x 1
        # DIMS: attn_score -> n_batch x n_head x tgt_len x tgt_len
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :],
                                        -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float("inf"))

        # [n_batch x n_head x qlen x klen]
        attn_prob = torch.sigmoid(attn_score)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("bnij,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x n_batch x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        # DIMS: n_head*d_head x n_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                # DIMS: n_model x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                # DIMS: n_model x 3*n_head*d_head
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                # DIMS: n_model x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # DIMS: n_model x 3*n_head*d_head
                w_heads = self.qkv_net(w)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # qlen x n_batch x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)

        # qlen x n_batch x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)

        # qlen x n_batch x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        # qlen x n_batch x n_head x d_head
        rw_head_q = w_head_q + r_w_bias[None]

        # qlen x klen x n_batch x n_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))

        # qlen x klen x n_batch x n_head
        B_ = torch.einsum("ibnd,jnd->ijbn", (w_head_q, r_emb))

        # 1    x klen x 1   x n_head
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x n_batch x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None],
                                        -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x n_batch x n_head]
        attn_prob = torch.sigmoid(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x n_batch x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        # DIMS: n_head*d_head x n_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(LightningModule):
    def __init__(self, n_head, n_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, n_model, d_head, dropout,
                                      **kwargs)
        self.pos_ff = PositionwiseFF(
            n_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(LightningModule):
    def __init__(self, n_head, n_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, n_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            n_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None,
                mems=None):
        output = self.dec_attn(
            dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(LightningModule):
    def __init__(self, n_head, n_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        # DIMS: n_model x n_head*d_head
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, n_model, d_head, dropout, **kwargs
        )

        # DIMS: n_model x n_model
        self.pos_ff = PositionwiseFF(
            n_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None,
                mems=None):
        # DIMS: r -> tgt_len x 1 x (n_model + 1)
        # DIMS: r_w_bias -> n_head x d_head
        # DIMS: r_r_bias -> n_head x d_head
        # DIMS: output ->
        output = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(LightningModule):
    def __init__(
            self, n_token, d_embed, d_proj, cutoffs, div_val=1,
            sample_softmax=False
    ):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0))
            )
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.d_proj], dtype=param.dtype,
                device=param.device,
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(LightningModule):
    def __init__(
            self,
            n_token: int,
            n_layer: int,
            n_head: int,
            n_model: int,
            d_head: int,
            d_inner: int,
            dropout: object,
            dropatt: object,
            tie_weight: object = False,
            d_embed: object = None,
            div_val: object = 1,
            tie_projs: object = [False],
            pre_lnorm: bool = False,
            tgt_len: object = None,
            ext_len: object = None,
            mem_len: object = None,
            cutoffs: object = [],
            adapt_inp: object = False,
            same_length: object = False,
            clamp_len: object = -1,
    ) -> object:
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = n_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.n_model = n_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, n_model, cutoffs, div_val=div_val
        )

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_n_keys = tgt_len + ext_len + mem_len

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head,
                    n_model,
                    d_head,
                    d_inner,
                    dropout,
                    tgt_len=tgt_len,
                    ext_len=ext_len,
                    mem_len=mem_len,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                )
            )

        self.crit = ProjectedAdaptiveSigmoid(
            n_token, d_embed, n_model, cutoffs, div_val=div_val
        )

        if tie_weight:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[
                    i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and n_model != d_embed:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len
        self.training_steps = 0
        self.compute = 0

        self._create_params()

    def _create_params(self):
        # default attention
        # DIMS: ceiling(d_emb / 2)
        self.pos_emb = PositionalEmbedding(self.n_model)

        # DIMS: n_head x n_head
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        # DIMS: n_head x n_head
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(
            mems
        ), "len(hids) != len(mems) ({len(hids)} != {len(mems)})"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = torch.triu(all_ones, 1 + mlen) + torch.tril(
                all_ones, -mask_shift_len
            )
            # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
            dec_attn_mask = dec_attn_mask.byte()
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen),
                                       diagonal=1 + mlen)
            # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
            dec_attn_mask = dec_attn_mask.byte()

        dec_attn_mask = dec_attn_mask.bool()  # Convert to bool

        hids = []

        # Default
        # DIMS: pos_seq -> tgt_len
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
        )
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(
                # DIMS: tgt_len x 1 x n_model
                core_out,
                # DIMS: tgt_len x 1 x (n_model+1)
                pos_emb,
                # DIMS: n_head x n_head
                self.r_w_bias,
                # DIMS: n_head x n_head
                self.r_r_bias,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
            )
            hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        if not mems:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)),
                         target.reshape(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")

    parser.add_argument("--n_layer", type=int, default=4, help="")
    parser.add_argument("--n_rel_layer", type=int, default=4, help="")
    parser.add_argument("--n_head", type=int, default=2, help="")
    parser.add_argument("--d_head", type=int, default=2, help="")
    parser.add_argument("--n_model", type=int, default=200, help="")
    parser.add_argument("--d_embed", type=int, default=200, help="")
    parser.add_argument("--d_inner", type=int, default=200, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--cuda", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=1111, help="")
    parser.add_argument("--multi_gpu", action="store_true", help="")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)
    diter = data_utils.OrderedIterator(
        data, B, tgt_len, device=device, ext_len=ext_len
    )

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(
                args.n_token,
                args.n_layer,
                args.n_head,
                args.n_model,
                args.d_head,
                args.d_inner,
                args.dropout,
                dropatt=args.dropout,
                tie_weight=True,
                d_embed=d_embed,
                div_val=div_val,
                tie_projs=tie_projs,
                pre_lnorm=True,
                tgt_len=tgt_len,
                ext_len=ext_len,
                mem_len=mem_len,
                cutoffs=cutoffs,
            ).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print("batch {}".format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
