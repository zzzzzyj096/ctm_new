#!/bin/bash
RUN=1
ITERATIONS=100
MODEL_TYPE="lstm"
LOG_DIR="logs/parity/run${RUN}/${MODEL_TYPE}_${ITERATIONS}"
SEED=$((RUN - 1))

python -m tasks.parity.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --iterations $ITERATIONS \
    --model_type $MODEL_TYPE \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 857 \
    --d_input 512 \
    --heads 8 \
    --dropout 0.0 \
    --positional_embedding_type="custom-rotational-1d" \
    --backbone_type="parity_backbone" \
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping -1 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --dataset "parity" \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr=0.0001 \
    --training_iterations 200001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 10000 \
    --no-reload \
    --no-reload_model_only \
    --device 0 \
    --no-use_amp \
