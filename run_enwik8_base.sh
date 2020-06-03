#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_ts.py \
#        --cuda \
        --datadir ./data/enwik8 \
        --dataset enwik8 \
        --n_layer 3 \
        --d_model 64 \
        --n_head 4 \
        --d_head 64 \
        --d_inner 128 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 10000 \
        --tgt_len 64 \
        --mem_len 64 \
        --eval_tgt_len 64 \
        --batch_size 22 \
#        --multi_gpu \
#        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ./data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
