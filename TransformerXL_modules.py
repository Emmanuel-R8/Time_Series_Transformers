from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

from utils.utils import ProjectedSigmoid

from utils import utils


class PositionalEmbedding(LightningModule):
    def __init__(self, d_pos_embed, debug=False):
        super(PositionalEmbedding, self).__init__()

        self.debug = debug

        assert (
                d_pos_embed % 2 == 0
        ), "The size of the positional  d_pos_embed must be an even number"

        self.d_pos_embed = d_pos_embed
        # TODO: check if useful: self.n_emb = math.ceil(d_pos_embed / 2)

        # Instead of writing sin(x / f), we use sin(input * inv_freq)
        # Frequencies range from 1 to 10000**2, sort of exponential progression
        # with exactly d_pos_emb frequencies
        inv_freq = 1 / (10000 ** (
                torch.arange(0.0, d_pos_embed, 2.0) / d_pos_embed))
        # inv_freq = inv_freq.rename('InvFreq') unsupported by torch.ger

        # Register this variable as a constant
        # DIMS: ceiling(d_pos_embed / 2)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, n_model):
        # torch.ger = outer product
        # DIMS: sinusoid_inp -> d_model d_pos_embed
        sinusoid_inp = torch.ger(n_model, self.inv_freq)

        # DIMS: pos_emb -> d_model 2*d_pos_embed
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if self.debug:
            print(f"PositionalEmbedding forward:"
                  f"    pos_emb size: {pos_emb.shape}")

        return pos_emb


class PositionwiseFF(LightningModule):
    def __init__(self, n_layer, layer_idx, d_model, d_hidden, d_inner, dropout,
                 pre_lnorm=False, debug=False):
        super(PositionwiseFF, self).__init__()

        self.debug = debug

        self.layer_idx = layer_idx

        if layer_idx == 0:
            self.d_input = d_model
            self.d_output = d_hidden
        elif layer_idx == n_layer - 1:
            self.d_input = d_hidden
            self.d_output = d_model
        else:
            self.d_input = d_hidden
            self.d_output = d_hidden

        self.d_inner = d_inner
        self.dropout = dropout

        # DIMS: d_model x d_model
        self.CoreNet = nn.Sequential(

            # DIMS: d_model x d_output
            nn.Linear(self.d_input, self.d_inner),

            # DIMS: d_output
            nn.ReLU(inplace=True),

            # DIMS: d_output
            nn.Dropout(dropout),

            # DIMS: d_output x d_model
            nn.Linear(d_inner, self.d_output),

            # DIMS: d_model
            nn.Dropout(dropout),
        )

        # DIMS: d_model
        self.pre_lnorm = pre_lnorm
        if pre_lnorm:
            self.layer_norm = nn.LayerNorm(self.d_input)
        else:
            self.layer_norm = nn.LayerNorm(self.d_output)

    def forward(self, input):

        if self.debug:
            print(f"PositionalFF forward:"
                  f"    input size: {input.shape}")

        # DIMS: input -> n_predict x d_model
        # assert (
        #        input.size()[1] == self.d_input
        # ), "PositionWideFF/forward: input.size()[1] != self.d_model"

        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            # DIMS: core_out -> n_predict x n_batch x d_model
            core_out = self.CoreNet(self.layer_norm(input))

            # residual connection
            # DIMS: output -> n_predict x n_batch x d_model
            output = core_out + input
        else:
            # positionwise feed-forward
            # DIMS: core_out -> n_predict x n_batch x d_model
            core_out = self.CoreNet(input)

            # residual connection + layer normalization
            # DIMS: output -> n_predict x n_batch x d_model
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

        # DIMS: d_model x n_head*d_head
        self.q_net = nn.Linear(n_model, n_head * d_head, bias=False)

        # DIMS: d_model x 2*n_head*d_head
        self.kv_net = nn.Linear(n_model, 2 * n_head * d_head, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # DIMS: n_head*d_head x d_model
        self.o_net = nn.Linear(n_head * d_head, n_model, bias=False)

        # DIMS: d_model
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

        # DIMS: q_net -> d_model x n_head * d_head
        head_q = self.q_net(h)

        # DIMS: kv_net -> d_model x 2*n_head*d_head
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
        # DIMS: n_head*d_head x d_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

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
            dropout=0.0,
            dropatt=0.0,
            n_predict=1,
            n_ext_ctx=None,
            n_mems=None,
            pre_lnorm=False,
            debug=False
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.debug = debug

        self.n_head = n_head
        self.n_model = n_model
        self.d_head = d_head
        self.dropout = dropout

        # DIMS: d_model x 3*n_head*d_head
        self.qkv_net = nn.Linear(n_model, 3 * n_head * d_head, bias=False)

        # DIMS:
        self.dropout = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # DIMS: n_head*d_head x d_model
        self.o_net = nn.Linear(n_head * d_head, n_model, bias=False)

        # DIMS: d_model
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
        # x_padded = x_padded.view(input.size(0), input.size(1), input.size(3) + 1, input.size(2))
        # input = x_padded[:, :, 1:].view_as(input)

        # This version retains the original shape of input
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
    def __init__(self, n_head, n_model, d_head, dropout, n_predict, n_ext_ctx,
                 n_mems, dropatt, pre_lnorm, debug=False):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(n_head, n_model,
                                                               d_head, dropout,
                                                               n_predict,
                                                               n_ext_ctx,
                                                               n_mems, dropatt,
                                                               pre_lnorm,
                                                               debug=False)

        self.debug = debug

        # DIMS: d_model x n_head*d_head
        self.r_net = nn.Linear(self.n_model, self.n_head * self.d_head,
                               bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # if self.debug:
        print(f"RelPartialLearnableMultiHeadAttn:"
              f"    w size: {w.shape}"
              f" -- r size: {r.shape}"
              f" -- r_w_bias size: {r_w_bias.shape}"
              f" -- r_r_bias size: {r_r_bias.shape}")

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        # TODO: DIMS obtained through debugging are clearly wrong.
        # DIMS: w -> n_batch x d_model x d_model
        # DIMS: r -> n_batch x 1 x d_pos_emb
        # DIMS: r_w_bias -> n_head x d_head
        # DIMS: r_r_bias -> n_head x d_head
        # qlen = n_batch
        # rlen = n_batch
        # bsz = d_model

        if mems is not None:
            # DIMS: 0 at the beginning

            # Concatenate memories + current segment along 1st dimension
            # TODO: CHECK mems DIMS. should reflect hidden state size
            # DIMS: mems -> n_mem x d_model
            # DIMS: w ->    n_batch x d_model x d_model
            # DIMS: cat ->  (n_mem+d_model) x d_model

            # CHECK: mems has to be copied to have one per training value in
            # the batch
            # DIMS: w -> n_batch x d_model x d_model
            # DIMS: mems -> ???
            # DIMS: cat -> (n_batch + ???) x d_model x d_model
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                # DIMS: cat ->     n_predict x n_batch x d_model
                # DIMS: qkc_net -> d_model x 3*n_head*d_head
                # DIMS: w_heads -> (n_predict + ???) x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                # DIMS: cat ->     n_predict x n_batch x d_model
                # DIMS: qkc_net -> d_model x 3*n_head*d_head
                # DIMS: w_heads -> (n_predict + ???) x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(cat)

            # DIMS: r -> n_predict x 1 x d_model
            # DIMS: r_net -> d_model x n_head*d_head
            # DIMS: r_head_k -> n_predict x n_head x d_head
            r_head_k = self.r_net(r)

            # DIMS: w_heads -> n_predict x n_batch x 3*n_head*d_head
            # DIMS: w_head_q -> n_predict x n_batch x n_head x d_head
            # DIMS: w_head_k -> n_predict x n_batch x n_head x d_head
            # DIMS: w_head_v -> n_predict x n_batch x n_head x d_head
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            # DIMS: w_head_q -> qlen x n_batch x n_head x d_head
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                # DIMS: w -> n_predict x n_batch x d_model
                # DIMS: qkv_net -> d_model x 3*n_head*d_head
                # DIMS: w_heads -> n_predict x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # DIMS: w -> n_batch x d_model x d_model
                # DIMS: qkv_net -> d_model x 3*n_head*d_head
                # DIMS: w_heads -> n_predict x n_batch x 3*n_head*d_head
                w_heads = self.qkv_net(w)

            # DIMS: r -> n_predict x 1 x d_model
            # DIMS: r_net -> d_model x n_head*d_head
            # DIMS: r_head_k -> n_predict x n_head x d_head
            r_head_k = self.r_net(r)

            # DIMS: w_heads -> n_predict x n_batch x 3*n_head*d_head
            # DIMS: w_head_q -> n_predict x n_batch x n_head x d_head
            # DIMS: w_head_k -> n_predict x n_batch x n_head x d_head
            # DIMS: w_head_v -> n_predict x n_batch x n_head x d_head
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        # klen = n_predict
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
        # klen = rlen = n_predict
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        # DIMS: attn_mask ->                 n_predict x n_predict x 1
        # DIMS: attn_score -> n_batch x n_head x n_predict x n_predict
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
        # DIMS: n_head*d_head x d_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

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

        self.debug = kwargs['debug']

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                # DIMS: d_model x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                # DIMS: d_model x 3*n_head*d_head
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                # DIMS: d_model x 3*n_head*d_head
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # DIMS: d_model x 3*n_head*d_head
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
        # DIMS: n_head*d_head x d_model
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(LightningModule):
    def __init__(self, d_input, d_output, n_head, d_head,
                 d_inner, dropout,
                 **kwargs):
        super(DecoderLayer, self).__init__()

        self.debug = kwargs.get("debug")
        self.pre_lnorm = kwargs.get("pre_lnorm")

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout,
                                      **kwargs)
        self.pos_ff = PositionwiseFF(n_layer, layer_idx, d_model, d_hidden,
                                     d_inner, dropout,
                                     pre_lnorm=self.pre_lnorm,
                                     debug=self.debug
                                     )

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(LightningModule):
    def __init__(self, n_head, n_model, d_head, d_inner, dropout, debug=False,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.debug = debug

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, n_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            n_model, d_inner, dropout, debug=debug,
            pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None,
                mems=None):
        output = self.dec_attn(
            dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(LightningModule):
    def __init__(self, n_head, n_model, d_head, d_inner, dropout, n_predict,
                 n_ext_ctx, n_mems, dropatt, pre_lnorm, debug=False):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.debug = debug

        # DIMS: d_model x n_head*d_head
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, n_model,
                                                         d_head, dropout,
                                                         n_predict, n_ext_ctx,
                                                         n_mems, dropatt,
                                                         pre_lnorm, debug=False)

        # DIMS: d_model x d_model
        self.pos_ff = PositionwiseFF(
            n_model, d_inner, dropout, debug=kwargs['debug'],
            pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None,
                mems=None):
        # DIMS: r -> n_predict x 1 x (d_model + 1)
        # DIMS: r_w_bias -> n_head x d_head
        # DIMS: r_r_bias -> n_head x d_head
        # DIMS: output ->
        output = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.pos_ff(output)

        return output


class Embedding(LightningModule):
    def __init__(
            self, d_model, d_pos_embed, d_proj, sample_softmax=False,
            debug=False
    ):
        super(Embedding, self).__init__()

        self.debug = debug

        self.d_model = d_model
        self.d_pos_embed = d_pos_embed

        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        self.emb_layers.append(
            nn.Embedding(d_model, d_pos_embed, sparse=(sample_softmax > 0))
        )
        if d_proj != d_pos_embed:
            self.emb_projs.append(
                nn.Parameter(torch.Tensor(d_proj, d_pos_embed)))

    def forward(self, input):
        if self.debug:
            print(f"Embedding forward: "
                  f"    input shape: {input.shape}"
                  f" -- size of emb_layers: {len(self.emb_layers)}")

        embed = self.emb_layers[0](input)
        if self.d_proj != self.d_pos_embed:
            embed = F.linear(embed, self.emb_projs[0])
        embed.mul_(self.emb_scale)

        return embed


class TransformerXL(LightningModule):
    def __init__(self, d_model: int, n_model: int, n_head: int, d_head: int,
                 d_FF_inner: int, n_layer: int, dropout: object, dropatt: object,
                 d_pos_enc: object = None, pre_lnorm: bool = False,
                 n_predict: object = None, n_ext_ctx: object = None,
                 n_mems: Optional[int] = None, adapt_inp: object = False,
                 same_length: object = False,
                 n_clamp_after: object = -1, debug=False) -> None:
        """

        :param d_model: dimensionality of the transformer_model's hidden states'
        :param n_model: the number of dates that the transformer_model takes (length of the transformer_model)
        :param n_head: number of attention heads for each attention layer in the Transformer encoder
        :param d_head: dimensionality of the transformer_model's heads
        :param d_FF_inner:
        :param n_layer: number of layers
        :param dropout:
        :param dropatt:
        :param d_pos_enc: dimensionality of the positional embeddings
        :param pre_lnorm:
        :param n_predict: number of tokens to predict
        :param n_ext_ctx:
        :param n_mems:
        :param adapt_inp:
        :param same_length:
        :param n_clamp_after:
        """
        super(TransformerXL, self).__init__()

        self.debug = debug

        self.d_model = d_model

        self.d_pos_embed = n_model if d_pos_enc is None else d_pos_enc
        self.n_model = n_model
        self.n_head = n_head
        self.d_head = d_head

        # TODO: Remove word embeddings
        # self.embedding = Embedding(
        #     d_model, d_pos_embed, d_model, debug=debug
        # )

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.n_predict = n_predict

        self.n_mems = n_mems
        self.mems = None  # Memories are all empty before training

        self.n_ext_ctx = n_ext_ctx
        self.max_n_keys = n_predict + n_ext_ctx + n_mems

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head,
                    n_model,
                    d_head,
                    d_FF_inner,
                    dropout,
                    n_predict=n_predict,
                    n_ext_ctx=n_ext_ctx,
                    n_mems=n_mems,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    debug=False
                )
            )

        self.crit = ProjectedSigmoid(d_model, d_pos_enc, n_model)

        self.same_length = same_length
        self.n_clamp_after = n_clamp_after
        self.training_steps = 0
        self.compute = 0

        # default attention
        # DIMS: ceiling(d_pos_embed / 2)
        self.pos_emb = PositionalEmbedding(self.d_pos_embed, debug=self.debug)

        # DIMS: n_head x d_head
        self.r_w_bias = nn.Parameter(torch.zeros(self.n_head, self.d_head, dtype=torch.float))

        # DIMS: n_head x d_head
        self.r_r_bias = nn.Parameter(torch.zeros(self.n_head, self.d_head, dtype=torch.float))


    def reset_length(self, n_predict, n_ext_ctx, n_mems):
        self.n_predict = n_predict
        self.n_mems = n_mems
        self.n_ext_ctx = n_ext_ctx

    def init_mems(self):
        if self.n_mems > 0:
            self.mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                self.mems.append(empty)

        return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if self.mems is None:
            return None

        # mems is not None
        assert len(hids) == len(
            self.mems
        ), "len(hids) != len(self.mems) ({len(hids)} != {len(mems)})"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `n_ext_ctx` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.n_ext_ctx - self.n_mems`
        # to `mlen + qlen - self.n_ext_ctx`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.n_ext_ctx)
            beg_idx = max(0, end_idx - self.n_mems)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

            self.mems = new_mems

        return new_mems

    def _forward(self, dec_inp, mems=None):
        if self.debug:
            print(f"TransformerXL _forward:"
                  f"    dec_inp: {dec_inp.shape}")

        qlen = dec_inp.shape[0]

        # current_embedding = self.embedding(dec_inp)

        mlen = self.mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # TODO: delete when debugged
        # if self.same_length:
        #     all_ones = current_embedding.new_ones(qlen, klen)
        #     mask_len = klen - self.n_mems
        #     if mask_len > 0:
        #         mask_shift_len = qlen - mask_len
        #     else:
        #         mask_shift_len = qlen
        #     dec_attn_mask = torch.triu(all_ones, 1 + mlen) + torch.tril(
        #         all_ones, -mask_shift_len
        #     )
        #     # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
        #     dec_attn_mask = dec_attn_mask.byte()
        # else:
        #     dec_attn_mask = torch.triu(current_embedding.new_ones(qlen, klen),
        #                                diagonal=1 + mlen)
        #     # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
        #     dec_attn_mask = dec_attn_mask.byte()

        if self.same_length:
            mask_len = klen - self.n_mems
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen

            dec_attn_mask = torch.triu(dec_inp, 1 + mlen) + \
                            torch.tril(dec_inp, -mask_shift_len
                                       )
            # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
            dec_attn_mask = dec_attn_mask.byte()

        else:
            dec_attn_mask = torch.triu(dec_inp, diagonal=1 + mlen)
            # REVERT? dec_attn_mask = dec_attn_mask.byte()[:, :, None]  # -1
            dec_attn_mask = dec_attn_mask.byte()

        dec_attn_mask = dec_attn_mask.bool()  # Convert to bool

        hids = []

        # Default
        # DIMS: d_model -> n_predict
        pos_seq = torch.arange(klen - 1, -1, -1.0)
        if self.n_clamp_after > 0:
            pos_seq.clamp_(max=self.n_clamp_after)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(dec_inp)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(
                # DIMS: n_predict x 1 x d_model
                core_out,
                # DIMS: n_predict x 1 x (d_model+1)
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

    def forward(self, input, output, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the transformer_model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        if self.debug:
            print(f"TransformerXL forward:"
                  f"    input shape: {input.shape}"
                  f" -- output shape: {output.shape}"
                  f" -- memory length: {len(mems)}")

        if (mems is None) or (not mems):
            mems = self.init_mems()

        n_predict = output.size(0)
        hidden, new_mems = self._forward(input, mems=mems)

        pred_hid = hidden[-n_predict:]
        loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)),
                         output.reshape(-1))
        loss = loss.view(n_predict, -1)

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
    parser.add_argument("--d_model", type=int, default=200, help="")
    parser.add_argument("--d_pos_embed", type=int, default=200, help="")
    parser.add_argument("--d_output", type=int, default=200, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--cuda", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=1111, help="")
    parser.add_argument("--multi_gpu", action="store_true", help="")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    n_predict, n_mems, n_ext_ctx = 36, 36, 0
    data_len = n_predict * 20
    args.d_model = 10000

    data = torch.LongTensor(data_len * B).random_(0, args.d_model).to(device)
    diter = utils.OrderedIterator(data, B, n_predict, device=device,
                                  n_ext_ctx=n_ext_ctx)

    for d_pos_embed in [200, 100]:
        model = TransformerXL(args.d_model, args.n_model, args.n_head,
                              args.d_head, args.d_inner, args.n_layer,
                              args.dropout, dropatt=args.dropout,
                              d_pos_enc=d_pos_embed, pre_lnorm=True,
                              n_predict=n_predict,
                              n_ext_ctx=n_ext_ctx, n_mems=n_mems).to(device)

        print(sum(p.numel() for p in model.parameters()))

        mems = tuple()
        for idx, (inp, tgt, seqlen) in enumerate(diter):
            print("batch {}".format(idx))
            out = model(inp, tgt, *mems)
            mems = out[1:]


