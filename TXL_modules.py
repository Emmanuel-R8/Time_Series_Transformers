from typing import *

import torch
import torch.nn as nn

from pytorch_lightning.core.lightning import LightningModule
from utils.proj_sigmoid import ProjectedSigmoid


class PositionEncoding(LightningModule):
    def __init__(self, d_pos_enc):
        super().__init__()

        power_range = -torch.arange(0.0, d_pos_enc, 2.0) / d_pos_enc
        inv_freq = 10000 ** power_range

        # register buffer tells pytorch that this tensor is part of the modle
        # this means that it will be saved in the state_dict and moved to the GPU
        # along with the model
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, n_model: torch.LongTensor) -> torch.FloatTensor:
        # outer product
        angle_values = torch.einsum("i,j->ij", n_model.float(), self.inv_freq)
        pos_emb = torch.cat([angle_values.sin(), angle_values.cos()], dim=-1)

        # DIMS: pos_enc -> (n_model, d_pos_enc)
        return pos_emb


class MultiHeadAttention(LightningModule):
    def __init__(self, d_input: int, d_output: int, d_pos_enc: int, n_head: int,
                 d_head: int, dropout: float, dropout_attn: float):
        super().__init__()

        # TODO: Relax this constraint
        assert d_input == d_output, \
            "MultiHeadAttention: d_input must equal d_output"
        self.d_input, self.d_output = d_input, d_output

        self.d_head = d_head
        self.n_head = n_head

        # Queries are only applied to vectors of the current sequence (not to
        # the memorized states)
        # No bias since simple matrix multiplication
        self.linear_q = nn.Linear(d_input, d_head * n_head, bias=False)

        # this layer applies the linear transformation required
        # for the keys and values for all heads at once for efficiency
        # 2 is for keys and values
        self.linear_kv = nn.Linear(d_input, d_head * n_head * 2, bias=False)

        # for position encodings
        self.linear_p = nn.Linear(d_pos_enc, d_head * n_head, bias=False)

        # for scaled dot product attention
        self.scale = 1 / (d_head ** 0.5)

        self.dropout_attn = nn.Dropout(dropout_attn)

        # we will use this to project to the output dimension
        self.layer_out = nn.Linear(self.d_head * self.n_head, self.d_output,
                                   bias=False)
        self.norm_out = nn.LayerNorm(self.d_output)
        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x):
        # DIMS: x -> (d1, d2, d3, ....)
        # DIMS: zero_pad -> (d1, 1, d3, ....)
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)

        # DIMS: joined -> (d1, 1+d2, d3,...)
        joined = torch.cat([zero_pad, x], dim=1)

        # DIMS: swapped -> (1+d2, d1, d3, ...)
        swapped = joined.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        # DIMS: clipped -> (d2, d1, d3, ...)
        clipped = swapped[1:]

        # DIMS: _rel_shift -> (d1, d2, d3, ...)
        return clipped.view_as(x)

    def forward(self, segment: torch.FloatTensor, pos_encs: torch.FloatTensor,
                memories: torch.FloatTensor, u: torch.FloatTensor,
                v: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None,
                ):
        """
        pos_encs: position encodings is separate to handle relative positions
        DIMS: segment -> (n_model, self.d_input)
        DIMS: pos_embs -> (n_model + n_mems, self.d_input)
        DIMS: output ->  (n_model, self.d_input)
        DIMS: u ->  (n_head, d_FF_inner)
        """

        # length of current segment
        n_model = segment.shape[0]

        # length of memory available
        n_current_mems = memories.shape[0]

        n_head, d_head = self.n_head, self.d_head

        # DIMS: memory_cat_input -> (n_model + n_current_mems, d_input)
        memory_cat_input = torch.cat([memories, segment], dim=0)

        # DIMS: input_ -> (n_model, d_input)
        # DIMS: self.linear_q -> (d_input, n_head * d_head)
        # DIMS: queries -> (n_model, b, n_head * d_head)
        queries = self.linear_q(segment)

        # DIMS: memory_cat_input -> (n_model + n_current_mems, d_input)
        # DIMS: self.linear_kv -> (d_input, d_head * n_head * 2)
        # DIMS: keys -> (n_model + n_current_mems, d_head * n_head)
        # DIMS: values -> (n_model + n_current_mems, d_head * n_head)
        keys, values = torch.chunk(self.linear_kv(memory_cat_input), 2, dim=-1)

        # DIMS: content_attn -> (n_model, n_model + n_current_mems, n_head)
        content_attn = torch.einsum(
            "ihd,jhd->ijh",
            ((queries.view(n_model, n_head, d_head) + u),
             keys.view(n_model + n_current_mems, n_head, d_head)))

        # position-based attention term ((b) + (d) in the paper)
        # this attention is solely based on the position of the key/values
        # (i.e. it does not take the content of the key/values into account)
        # (n_model, n_model + n_previous, b, n_head)
        # (n_model + n_previous, b, n_head* d_hidden)

        # DIMS: pos_enc -> (n_model, d_pos_enc)
        # DIMS: self.linear_p -> (d_pos_enc, d_head * n_head)
        # DIMS: positions -> (n_model, d_head * n_head)
        positions = self.linear_p(pos_encs)

        # DIMS: position_attn -> (n_model, n_model + n_current_mems, n_head)
        position_attn = torch.einsum(
            "ihd,jhd->ijh",
            ((queries.view(n_model, n_head, d_head) + v),
             positions.view(n_model + n_current_mems, n_head, d_head)))

        # Compute positional attention efficiently
        # DIMS: position_attn -> (n_model, n_model + n_current_mems, n_head)
        position_attn = self._rel_shift(position_attn)

        # the attention is the sum of content-based and position-based attention
        # DIMS: attn -> (n_model, n_model + n_current_mems, n_head)
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            attn = attn.masked_fill(
                mask[..., None], -float('inf'))

        # rescale to prevent values from exploding
        # normalize across the value sequence dimension
        attn = torch.softmax(attn * self.scale, dim=1)
        attn = self.dropout_attn(attn)

        # DIMS: attn -> (n_model, n_model + n_current_mems, n_head)
        # DIMS: values -> (n_model + n_current_mems, d_head * n_head)
        # DIMS: values.view -> (n_model + n_current_mems, n_head, d_head)
        # i: n_model
        # j: n_model + n_current_mems
        # h: n_head
        # d: d_head
        # DIMS: einsum -> (n_model, n_head, d_head)
        # DIMS: attn_weighted_values -> (n_model, n_head* d_head)
        attn_weighted_values = torch.einsum(
            "ijh,jhd->ihd",
            (attn, values.view(n_model + n_current_mems, n_head, d_head),)) \
            .contiguous() \
            .view(n_model, n_head * d_head)

        # Project back to input dimension and add residual connection
        # DIMS: self.layer_out() -> (d_head * n_head, d_output)
        # DIMS: attn_weighted_values -> (n_model, n_head* d_head)
        # DIMS: self_dropout(...) -> (n_model, d_output)
        # DIMS: segment -> (n_model, self.d_input)
        output = segment + self.dropout(self.layer_out(attn_weighted_values))
        output = self.norm_out(output)

        return output


class Positionwise_FeedFwd(LightningModule):
    def __init__(self, d_input, d_FF_inner, dropout):
        super().__init__()

        self.d_input = d_input
        self.d_FF_inner = d_FF_inner
        self.dropout = dropout

        self.ff = nn.Sequential(nn.Linear(d_input, d_FF_inner),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(d_FF_inner, d_input),
                                nn.Dropout(dropout))

        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, sequence: torch.FloatTensor) -> torch.FloatTensor:
        # DIMS: sequence -> (n_model, d_input)
        fed_fwd = self.ff(sequence)
        output = self.layer_norm(sequence + fed_fwd)

        # DIMS: output -> (cur_seq, bs, d_input)
        return output


class DecoderBlock(LightningModule):
    def __init__(self, n_head, d_input, d_output, d_head_inner, d_FF_inner,
                 dropout=0.0, dropout_attn=0.0):
        super().__init__()

        self.multi_heat_attn = MultiHeadAttention(d_input, d_output, d_pos_enc,
                                                  n_head=n_head,
                                                  d_head=d_head_inner,
                                                  dropout=dropout,
                                                  dropout_attn=dropout_attn)
        self.feed_fwd = Positionwise_FeedFwd(d_input, d_FF_inner, dropout)

    def forward(self, input_: torch.FloatTensor,  # (cur_seq, bs, d_input)
                pos_embs: torch.FloatTensor,  # (cur_seq + prev_seq, d_input),
                u: torch.FloatTensor,  # (H, d_input),
                v: torch.FloatTensor,  # (H, d_input),
                mask=None,
                mems=None,
                ):
        return self.feed_fwd(
            self.multi_heat_attn(input_, pos_embs, mems, u, v, mask=mask))


class Transformer_XL(LightningModule):
    def __init__(self, n_layer: int, d_hidden, d_pos_enc, n_head: int,
                 d_head: int, d_FF_inner: int, d_model: int, dropout: float,
                 dropout_attn: float, n_model: int, n_mems: int):
        super().__init__()

        self.n_layer = n_layer
        self.n_head, self.d_head, self.d_ff_inner = n_head, d_head, d_FF_inner
        self.d_model = d_model

        # Position encoding
        self.pos_enc = PositionEncoding(d_pos_enc)

        # Core transformer
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for l in range(n_layer):
            # All layers have d_hidden input size and d_hidden output size
            # apart from:
            #     layer 0: input size = d_model
            #     layer n_layer - 1: output size = d_model
            input_size = d_model if l == 0 else d_hidden
            output_size = d_model if l == n_layer - 1 else d_hidden

            self.layers.append(DecoderBlock(n_head, input_size, output_size,
                                            d_head_inner=d_head,
                                            d_FF_inner=d_FF_inner,
                                            dropout=dropout,
                                            dropout_attn=dropout_attn))

        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len, self.mem_len = n_model, n_mems

        # u and v are global parameters since position encodings are too.
        self.u = nn.Parameter(
            torch.zeros(self.n_head, self.d_head, dtype=torch.float))
        self.v = nn.Parameter(
            torch.zeros(self.n_head, self.d_head, dtype=torch.float))

    def init_memory(self, device=torch.device("cpu")) -> List[
        torch.FloatTensor]:
        return [torch.empty(0).to(device=device, dtype=torch.float)
                for _ in range(self.n_layer + 1)]

    def update_memory(self,
                      previous_memory: List[torch.FloatTensor],
                      hidden_states: List[torch.FloatTensor], ):
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)

        # For the updated memory, we use the most recent `self.mem_len`
        # states, including the previous memory
        # In other words, if `seq_len` < `self.mem_len` some of the previous memory
        # will carry over to the next memory
        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(previous_memory, hidden_states):
                # (mem_len + seq_len, bs, d)
                cat = torch.cat([m, h], dim=0)

                # (self.mem_len, bs, d)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def reset_length(self, seq_len, ext_len, mem_len):
        self.seq_len = seq_len
        self.mem_len = mem_len

    def forward(self, data: torch.FloatTensor,
                target: torch.FloatTensor,  # (n_model, bs)
                memory: Optional[List[torch.FloatTensor]] = None,
                ) -> Dict[str, torch.Tensor]:
        # DIMS: data -> (n_model, d_model)
        # DIMS: target -> (n_model, d_model)

        if memory is None:
            memory: List[torch.FloatTensor] = self.init_memory(data.device)

        assert len(memory) == len(self.layers) + 1

        n_sequence, n_batch = data.size()
        prev_seq = memory[0].size(0)

        # Construct attention mask
        dec_attn_mask = torch.triu(
            torch.ones((n_sequence, n_sequence + prev_seq)),
            diagonal=1 + prev_seq,
        ).byte()[..., None].to(data.device)

        current_segment = self.dropout(data)

        pos_idxs = torch.arange(n_sequence + prev_seq - 1, -1, -1.0,
                                dtype=torch.float).to(current_segment.device)
        pos_embs = self.dropout(self.pos_enc(pos_idxs))

        # Main part of forward pass
        hidden_states = [current_segment]
        layer_out = current_segment
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(layer_out, pos_embs, self.u, self.v,
                              mask=dec_attn_mask, mems=mem)
            hidden_states.append(layer_out)

        layer_out = self.dropout(layer_out)
        loss = self.loss_fn(layer_out.view(-1, layer_out.size(-1)),
                            target.view(-1))

        # Update memory
        # Ensure the memory is treated as a constant
        # and we do not back propagate through them
        new_memory = self.update_memory(memory, hidden_states)

        return {"loss": loss, "layer_out": layer_out, "memory": new_memory}
