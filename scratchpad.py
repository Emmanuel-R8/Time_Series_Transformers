#%%
import torch

#%%
d_pos_emb = 5

# Instead of writing sin(x / f) will use sin(x * inv_freq)
# Frequencies range from 1 to 10000
inv_freq = 1 / (10000 ** (torch.arange(0.0, d_pos_emb, 2.0 / d_pos_emb)))

inv_freq.size()[1]
# DIMS:


#%%

# torch.ger = outer product
# DIMS:
pos_seq = torch.rand(3)
sinusoid_inp = torch.ger(pos_seq, inv_freq)

#%%

pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
pos_emb.size()

#%%

bsz = 4

if bsz is not None:
    pos_emb = pos_emb[:, None, :].expand(-1, bsz, -1)
    pos_emb = pos_emb.rename("PosSin", "Batch", "PosCos")
else:
    pos_emb = pos_emb[:, None, :]

