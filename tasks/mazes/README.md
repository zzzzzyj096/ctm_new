# Mazes

This folder contains code for training and analysing 2D maze solving experiments


## Training
To run the maze training that we used for the paper, run the following command from the parent directory:
```
python -m tasks.mazes.train --d_model 2048 --d_input 512 --synapse_depth 4 --heads 8 --n_synch_out 64 --n_synch_action 32 --neuron_select_type first-last --iterations 75 --memory_length 25 --deep_memory --memory_hidden_dims 32 --dropout 0.1 --no-do_normalisation --positional_embedding_type none --backbone_type resnet34-2 --batch_size 64 --batch_size_test 64 --lr 1e-4 --training_iterations 1000001 --warmup_steps 10000 --use_scheduler --scheduler_type cosine --weight_decay 0.0 --log_dir logs/mazes/d=2048--i=512--h=8--ns=64-32--iters=75x25--h=32--drop=0.1--pos=none--back=34-2--seed=42 --dataset mazes-medium --save_every 2000 --track_every 5000 --seed 42 --n_test_batches 50
```

## Small training run
We also provide a 'mazes-small' dataset (see [here](https://drive.google.com/file/d/1cBgqhaUUtsrll8-o2VY42hPpyBcfFv86/view?usp=drivesdk)) for fast iteration and testing ideas. The following command can train a CTM locally without a GPU in 12-24 hours:
```
python -m tasks.mazes.train --dataset mazes-small --maze_route_length 50 --cirriculum_lookahead 5  --model ctm --d_model 1024 --d_input 256 --backbone_type resnet18-1 --synapse_depth 8 --heads 4 --n_synch_out 128 --n_synch_action 128 --neuron_select_type random-pairing --memory_length 25 --iterations 50 --training_iterations 100001 --lr 1e-4 --batch_size 64 --batch_size_test 32 --n_test_batches 50 --log_dir logs/mazes-small-tester --track_every 2000
```