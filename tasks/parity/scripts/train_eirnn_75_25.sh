#!/bin/bash
RUN=1
ITERATIONS=75
MEMORY_LENGTH=25
dmodel=64
LOG_DIR="logs/parity/run${RUN}/eirnn_${ITERATIONS}_${DMODEL}"
SEED=$((RUN - 1))

python -m tasks.parity.train \
    --model_type 'eirnn'\
    --log_dir $LOG_DIR \
    --seed $SEED \
    --iterations $ITERATIONS \
    --memory_length $MEMORY_LENGTH \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 64 \
    --d_input 256 \
    --n_synch_out 16 \
    --n_synch_action 16 \
    --heads 4 \
    --memory_hidden_dims 8 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping 0.9 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --dataset "parity" \
    --batch_size 32 \
    --batch_size_test 128 \
    --lr=0.00010 \
    --training_iterations 100001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 5000 \
    --reload \
    --no-reload_model_only \
    --device 0 \
    --no-use_amp \
