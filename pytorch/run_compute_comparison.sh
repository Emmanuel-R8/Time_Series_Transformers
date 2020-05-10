echo 'Run training...'
python train.py \
    --cuda \
    --data ../data/wikitext-103/ \
    --dataset wt103 \
    --adaptive \
    --n_layer 2 \
    --d_model 64 \
    --n_head 4 \
    --d_head 16 \
    --d_inner 256 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 200000 \
    --tgt_len 150 \
    --mem_len 150 \
    --eval_tgt_len 150 \
    --batch_size 60 \
    --batch_chunk 1 \
    --wandb transfoxl-compute \
    ${@:2}
