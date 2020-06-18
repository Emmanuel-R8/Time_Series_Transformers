#%%
import train_ts

from utils.argparsing import parser

#%%
# Test with very unusual value to track errors in tensor sizes
arguments = parser.parse_args(
    "--datadir ./data/etf \
     --dataset allData.csv \
     --n_layer 2 \
     --d_model 128 \
     --n_head 8 \
     --d_head 16 \
     --d_inner 17 \
     --dropout 0.1 \
     --dropatt 0.0 \
     --optim adam \
     --lr 0.00025 \
     --warmup_step 0 \
     --max_step 10000 \
     --tgt_len 64 \
     --mem_len 16 \
     --eval_tgt_len 32 \
     --batch_size 8 ".split()
)

#%%
train_ts.train_ts(arguments)
