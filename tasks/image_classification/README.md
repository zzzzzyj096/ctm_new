# Image classification

This folder contains code for training and analysing imagenet and cifar related experiments. 

## Accessing and loading imagenet

We use the [ILSRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset in our paper.

To get this to work for you, you will need to do the following:
1. Login to huggingface (make an account) to agree to TCs of this dataset, 
2. Make a new access token.
3. Install huggingface_hub on the target machine with ```pip install huggingface_hub``` 
4. Run ```huggingface-cli login``` and use your token. This will authenticate you on the backend and allow the code to run.
5. Simply run an imagenet experiment. It will auto download and do all that magic. 


## Training
There are two training files: `train.py` and `train_distributed.py`. The training code uses mixed precision. For the settings in the paper, the following command was used for distributed training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m tasks.image_classification.train_distributed --d_model 4096 --d_input 1024 --synapse_depth 12 --heads 16 --n_synch_out 150 --n_synch_action 150 --neuron_select_type random --iterations 75 --memory_length 25 --deep_memory --memory_hidden_dims 64 --dropout 0.05 --no-do_normalisation --positional_embedding_type none --backbone_type resnet152-4 --batch_size 60 --batch_size_test 64 --lr 5e-4 --training_iterations 500001 --warmup_steps 10000 --use_scheduler --scheduler_type cosine --weight_decay 0.0 --log_dir logs-lambda/imagenet-distributed-4april/d=4096--i=1024--h=16--ns=150-random--iters=75x25--h=64--drop=0.05--pos=none--back=152x4--seed=42 --dataset imagenet --save_every 2000 --track_every 5000 --seed 42 --n_test_batches 50 --use_amp
```

You can run the same setup on a single GPU with:
```
python -m tasks.image_classification.train --d_model 4096 --d_input 1024 --synapse_depth 12 --heads 16 --n_synch_out 150 --n_synch_action 150 --neuron_select_type random --iterations 75 --memory_length 25 --deep_memory --memory_hidden_dims 64 --dropout 0.05 --no-do_normalisation --positional_embedding_type none --backbone_type resnet152-4 --batch_size 60 --batch_size_test 64 --lr 5e-4 --training_iterations 500001 --warmup_steps 10000 --use_scheduler --scheduler_type cosine --weight_decay 0.0 --log_dir logs-lambda/imagenet-distributed-4april/d=4096--i=1024--h=16--ns=150-random--iters=75x25--h=64--drop=0.05--pos=none--back=152x4--seed=42 --dataset imagenet --save_every 2000 --track_every 5000 --seed 42 --n_test_batches 50 --use_amp --device 0
```

## Checkpoint

The checkpoint for the model used in the paper can be found [here](https://drive.google.com/file/d/1Lr_3RZU9X9SS8lBhAhECBiSZDKfKhDkJ/view?usp=drive_link).
