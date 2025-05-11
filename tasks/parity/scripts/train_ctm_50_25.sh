#!/bin/bash
RUN=1
ITERATIONS=50
MEMORY_LENGTH=25
LOG_DIR="logs/parity/run${RUN}/ctm_${ITERATIONS}_${MEMORY_LENGTH}"
SEED=$((RUN - 1))

python -m tasks.parity.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --iterations $ITERATIONS \
    --memory_length $MEMORY_LENGTH \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 1024 \
    --d_input 512 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 8 \
    --memory_hidden_dims 16 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --positional_embedding_type="custom-rotational-1d" \
    --backbone_type="parity_backbone" \
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping 0.9 \
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
    --neuron_select_type "random"