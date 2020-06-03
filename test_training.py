#%%
import train_ts
from utils.argparsing import parser

#%%

# Test with very unusual value to track errors in tensor sizes
arguments = parser.parse_args(
    "--datadir ./data/enwik8 \
     --dataset enwik8 \
     --n_layer 3 \
     --d_model 253 \
     --n_head 4 \
     --d_head 61 \
     --d_inner 17 \
     --dropout 0.1 \
     --dropatt 0.0 \
     --optim adam \
     --lr 0.00025 \
     --warmup_step 0 \
     --max_step 10000 \
     --tgt_len 57 \
     --mem_len 70 \
     --eval_tgt_len 64 \
     --batch_size 22 ".split()
)

#%%
train_ts.train_ts(arguments)


# %%
