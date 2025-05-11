#!/bin/bash
RUN=1
ITERATIONS=1
MODEL_TYPE="ctm"
ENV_ID="MiniGrid-FourRooms-v0"
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
    --d_model 512 \
    --d_input 128 \
    --memory_hidden_dims 16 \
    --n_synch_out 32 \
    --discount_gamma 0.99 \
    --gae_lambda 0.95 \
    --ent_coef 0.1 \
    --vf_coef 0.25 \
    --memory_length 10 \
    --max_environment_steps 300 \
    --total_timesteps 300000000 \
    --num_steps 50 \
    --anneal_lr \
    --num_envs 256 \
    --update_epochs 1 \
    --mask_velocity \
    --continuous_state_trace \
    --dropout 0.0 \
    --lr=0.0001 \
    --track_every 100 \
    --save_every 100 \
    --no-reload \
    --device 0 \
    --neuron_select_type "first-last"