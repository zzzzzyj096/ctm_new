#!/bin/bash
for RUN in 1 2 3; do
    ITERATIONS=5
    MODEL_TYPE="ctm"
    ENV_ID="CartPole-v1"
    LOG_DIR="logs/rl/${ENV_ID}/run${RUN}/${MODEL_TYPE}_${ITERATIONS}"
    RUN_NAME="run${RUN}_${ENV_ID}_${MODEL_TYPE}_${ITERATIONS}"
    TB_LOG_DIR="logs/runs/"
    SEED=$RUN

    python -m tasks.rl.train \
        --model_type $MODEL_TYPE \
        --env_id $ENV_ID \
        --log_dir $LOG_DIR \
        --tb_log_dir $TB_LOG_DIR \
        --seed $SEED \
        --iterations $ITERATIONS \
        --run_name $RUN_NAME \
        --d_model 128 \
        --d_input 128 \
        --memory_hidden_dims 4 \
        --n_synch_out 16 \
        --discount_gamma 0.99 \
        --gae_lambda 0.95 \
        --ent_coef 0.1 \
        --vf_coef 0.25 \
        --memory_length 50 \
        --max_environment_steps 200 \
        --total_timesteps 10000000 \
        --num_steps 100 \
        --anneal_lr \
        --num_envs 256 \
        --update_epochs 4 \
        --mask_velocity \
        --continuous_state_trace \
        --dropout 0.0 \
        --lr=0.001 \
        --track_every 1000 \
        --save_every 100 \
        --no-reload \
        --device 0 \
        --neuron_select_type "first-last"
done