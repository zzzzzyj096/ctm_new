#!/bin/bash
RUN=3
MEMORY_LENGTH=30
MODEL_TYPE="lstm"
Q_NUM_REPEATS_PER_INPUT=10
LOG_DIR="logs/qamnist/run${RUN}/${MODEL_TYPE}_${Q_NUM_REPEATS_PER_INPUT}"
SEED=$((RUN - 1))

python -m tasks.qamnist.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --memory_length $MEMORY_LENGTH \
    --model_type $MODEL_TYPE \
    --q_num_images 3 \
    --q_num_images_delta 2 \
    --q_num_repeats_per_input 8 \
    --q_num_operations 3 \
    --q_num_operations_delta 2 \
    --q_num_answer_steps 10 \
    --n_test_batches 20 \
    --d_model 512 \
    --d_input 64 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 4 \
    --memory_hidden_dims 5 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --weight_decay 0.0 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --batch_size 16 \
    --batch_size_test 64 \
    --lr=0.0001 \
    --training_iterations 100001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 10000 \
    --no-reload \
    --no-reload_model_only \
    --device 0 \
    --no-use_amp \
    --neuron_select_type "random-pairing"
