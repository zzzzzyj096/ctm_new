import argparse
import os
import random
import gc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.samplers import FastRandomDistributedSampler 
from tqdm.auto import tqdm

# Data/Task Specific Imports
from data.custom_datasets import MazeImageFolder

# Model Imports
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline

# Plotting/Utils Imports
from tasks.mazes.plotting import make_maze_gif
from tasks.image_classification.plotting import plot_neural_dynamics
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import maze_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, required=True, choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet34-2', help='Type of backbone featureiser.')
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads (CTM, LSTM).')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding (CTM, LSTM).', choices=['none',
                                                                       'learnable-fourier',
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])
    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=8, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch (CTM only).')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch (CTM only).')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=32, help='Hidden dimensions of the memory if using deep memory (CTM only).')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).')

    # Task Specific Args
    parser.add_argument('--maze_route_length', type=int, default=100, help='Length to truncate targets.')
    parser.add_argument('--cirriculum_lookahead', type=int, default=5, help='How far to look ahead for cirriculum.')

    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (per GPU).')
    parser.add_argument('--batch_size_test', type=int, default=64, help='Batch size for testing (per GPU).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--num_workers_train', type=int, default=0, help='Num workers training.')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--use_custom_sampler', action=argparse.BooleanOptionalAction, default=False, help='Use custom fast sampler to avoid reshuffling.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components.')

    # Logging and Saving
    parser.add_argument('--log_dir', type=str, default='logs/scratch', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='mazes-medium', help='Dataset to use.', choices=['mazes-medium', 'mazes-large'])
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?') # Default False based on user edit
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=False, help='Should use strict reload for model weights.')
    parser.add_argument('--ignore_metrics_when_reloading', action=argparse.BooleanOptionalAction, default=False, help='Ignore metrics when reloading (for debugging)?')

    # Tracking
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=2, help='How many minibatches to approx metrics. Set to -1 for full eval')

    # Precision
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    args = parser.parse_args()
    return args

# --- DDP Setup Functions ---
def setup_ddp():
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356' # Different port from image classification
        os.environ['LOCAL_RANK'] = '0'
        print("Running in non-distributed mode (simulated DDP setup).")
        if not torch.cuda.is_available() or int(os.environ['WORLD_SIZE']) == 1:
            dist.init_process_group(backend='gloo')
            print("Initialized process group with Gloo backend for single/CPU process.")
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            return rank, world_size, local_rank

    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank} setup on GPU {local_rank}")
    else:
         print(f"Rank {rank} setup on CPU")
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP cleanup complete.")

def is_main_process(rank):
    return rank == 0
# --- End DDP Setup ---


if __name__=='__main__':

    args = parse_args()

    rank, world_size, local_rank = setup_ddp()

    set_seed(args.seed + rank, False)

    # Rank 0 handles directory creation and initial logging
    if is_main_process(rank):
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        zip_python_code(f'{args.log_dir}/repo_state.zip')
        with open(f'{args.log_dir}/args.txt', 'w') as f:
            print(args, file=f)
    if world_size > 1: dist.barrier()


    assert args.dataset in ['mazes-medium', 'mazes-large']

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        if world_size > 1: warnings.warn("Running DDP on CPU is not recommended.")

    if is_main_process(rank):
        print(f'Main process (Rank {rank}): Using device {device}. World size: {world_size}. Model: {args.model}')


    prediction_reshaper = [args.maze_route_length, 5]
    args.out_dims = args.maze_route_length * 5

    # --- Model Definition (Conditional) ---
    model_base = None # Base model before DDP wrapping
    if args.model == 'ctm':
        model_base = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    elif args.model == 'lstm':
        model_base = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model_base = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Use pseudo-input *before* DDP wrapping
    try:
        # Determine pseudo input shape based on dataset
        h_w = 39 if args.dataset in ['mazes-small', 'mazes-medium'] else 99 # Example dimensions
        pseudo_inputs = torch.zeros((1, 3, h_w, h_w), device=device).float()
        model_base(pseudo_inputs)
    except Exception as e:
         print(f"Warning: Pseudo forward pass failed: {e}")

    if is_main_process(rank):
        print(f'Total params: {sum(p.numel() for p in model_base.parameters() if p.requires_grad)}')

    # Wrap model with DDP
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model_base, device_ids=[local_rank], output_device=local_rank)
    elif device.type == 'cpu' and world_size > 1:
        model = DDP(model_base)
    else:
        model = model_base
    # --- End Model Definition ---


    # Data Loading (After model setup to allow pseudo pass first)
    dataset_mean = [0,0,0]
    dataset_std = [1,1,1]
    which_maze = args.dataset.split('-')[-1]
    data_root = f'data/mazes/{which_maze}'

    train_data = MazeImageFolder(root=f'{data_root}/train/', which_set='train', maze_route_length=args.maze_route_length)
    test_data = MazeImageFolder(root=f'{data_root}/test/', which_set='test', maze_route_length=args.maze_route_length)

    train_sampler = (FastRandomDistributedSampler(train_data, num_replicas=world_size, rank=rank, seed=args.seed, epoch_steps=int(10e10))
                     if args.use_custom_sampler else
                     DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)

    num_workers_test = 1
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                                              num_workers=args.num_workers_train, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, sampler=test_sampler,
                                             num_workers=num_workers_test, pin_memory=True, drop_last=False)


    # Optimizer and scheduler
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip parameters that don't require gradients
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names) and is_main_process(rank):
        print(f'WARNING, excluding: {no_decay_names}')

    # Optimizer and scheduler (Common setup)
    if len(no_decay_names) and args.weight_decay!=0:
        optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay':args.weight_decay},
                                       {'params': no_decay_params, 'weight_decay':0}],
                                  lr=args.lr,
                                  eps=1e-8 if not args.use_amp else 1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    eps=1e-8 if not args.use_amp else 1e-6,
                                    weight_decay=args.weight_decay)

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError


    # Metrics tracking (Rank 0 stores history)
    start_iter = 0
    iters = []
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], [] # Avg Step Acc (scalar list)
    train_accuracies_most_certain, test_accuracies_most_certain = [], [] # Avg Step Acc @ Certain tick (scalar list)
    train_accuracies_most_certain_permaze, test_accuracies_most_certain_permaze = [], [] # Full Maze Acc @ Certain tick (scalar list)


    scaler = torch.amp.GradScaler("cuda" if device.type == 'cuda' else "cpu", enabled=args.use_amp)

    # Reloading Logic
    if args.reload:
        map_location = device
        chkpt_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(chkpt_path):
            print(f'Rank {rank}: Reloading from: {chkpt_path}')
            if not args.strict_reload: print('WARNING: not using strict reload for model weights!')

            checkpoint = torch.load(chkpt_path, map_location=map_location, weights_only=False)

            model_to_load = model.module if isinstance(model, DDP) else model
            state_dict = checkpoint['model_state_dict']
            has_module_prefix = all(k.startswith('module.') for k in state_dict)
            is_wrapped = isinstance(model, DDP)

            if has_module_prefix and not is_wrapped:
                state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
            elif not has_module_prefix and is_wrapped:
                load_result = model_to_load.load_state_dict(state_dict, strict=args.strict_reload)
                print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
                state_dict = None # Prevent loading again

            if state_dict is not None:
                load_result = model_to_load.load_state_dict(state_dict, strict=args.strict_reload)
                print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
                


            if not args.reload_model_only:
                print(f'Rank {rank}: Reloading optimizer, scheduler, scaler, iteration.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']

                if is_main_process(rank) and not args.ignore_metrics_when_reloading:
                    print(f'Rank {rank}: Reloading metrics history.')
                    iters = checkpoint['iters']
                    train_losses = checkpoint['train_losses']
                    test_losses = checkpoint['test_losses']
                    train_accuracies = checkpoint['train_accuracies'] # Reloading simplified avg step acc list
                    test_accuracies = checkpoint['test_accuracies']
                    train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                    test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                    train_accuracies_most_certain_permaze = checkpoint['train_accuracies_most_certain_permaze']
                    test_accuracies_most_certain_permaze = checkpoint['test_accuracies_most_certain_permaze']
                elif is_main_process(rank) and args.ignore_metrics_when_reloading:
                     print(f'Rank {rank}: Ignoring metrics history upon reload.')
            else:
                 print(f'Rank {rank}: Only reloading model weights!')

            if is_main_process(rank) and 'torch_rng_state' in checkpoint and not args.reload_model_only:
                print(f'Rank {rank}: Loading RNG states.')
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Rank {rank}: Reload finished, starting from iteration {start_iter}")
        else:
            print(f"Rank {rank}: Checkpoint not found at {chkpt_path}, starting from scratch.")

        
        if world_size > 1: dist.barrier()


    # Conditional Compilation
    if args.do_compile:
        if is_main_process(rank): print('Compiling model components...')
        model_to_compile = model.module if isinstance(model, DDP) else model
        if hasattr(model_to_compile, 'backbone'):
            model_to_compile.backbone = torch.compile(model_to_compile.backbone, mode='reduce-overhead', fullgraph=True)
        if args.model == 'ctm':
             model_to_compile.synapses = torch.compile(model_to_compile.synapses, mode='reduce-overhead', fullgraph=True)
        if world_size > 1: dist.barrier()
        if is_main_process(rank): print('Compilation finished.')


    # --- Training Loop ---
    model.train()
    pbar = tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True, disable=not is_main_process(rank))

    iterator = iter(trainloader)

    for bi in range(start_iter, args.training_iterations):

        # --- Evaluation and Plotting (Rank 0 + Aggregation) ---
        if bi % args.track_every == 0 and (bi != 0 or args.reload_model_only):
            model.eval()
            with torch.inference_mode():

                # --- Distributed Evaluation ---
                if is_main_process(rank): iters.append(bi) # Track iterations on rank 0

                # Initialize accumulators on device
                total_train_loss = torch.tensor(0.0, device=device)
                total_train_correct_certain = torch.tensor(0.0, device=device) # Sum correct steps @ certain tick
                total_train_mazes_solved = torch.tensor(0.0, device=device)    # Sum solved mazes @ certain tick
                total_train_steps = torch.tensor(0.0, device=device)           # Total steps evaluated (B * S)
                total_train_mazes = torch.tensor(0.0, device=device)           # Total mazes evaluated (B)

                # TRAIN METRICS
                train_eval_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False)
                train_eval_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, sampler=train_eval_sampler, num_workers=num_workers_test, pin_memory=True)

                pbar_inner_desc = 'Eval Train (Rank 0)' if is_main_process(rank) else None
                with tqdm(total=len(train_eval_loader), desc=pbar_inner_desc, leave=False, position=1, dynamic_ncols=True, disable=not is_main_process(rank)) as pbar_inner:
                    for inferi, (inputs, targets) in enumerate(train_eval_loader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True) # B, S
                        batch_size = inputs.size(0)
                        seq_len = targets.size(1)

                        loss_eval = None
                        pred_at_certain = None # Shape B, S
                        if args.model == 'ctm':
                            predictions_raw, certainties, _ = model(inputs)
                            predictions = predictions_raw.reshape(batch_size, -1, 5, predictions_raw.size(-1)) # B,S,C,T
                            loss_eval, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                            pred_at_certain = predictions.argmax(2)[torch.arange(batch_size, device=device), :, where_most_certain]
                        elif args.model == 'lstm':
                            predictions_raw, certainties, _ = model(inputs)
                            predictions = predictions_raw.reshape(batch_size, -1, 5, predictions_raw.size(-1)) # B,S,C,T
                            loss_eval, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False) # where = -1
                            pred_at_certain = predictions.argmax(2)[torch.arange(batch_size, device=device), :, where_most_certain]
                        elif args.model == 'ff':
                            predictions_raw = model(inputs) # B, S*C
                            predictions = predictions_raw.reshape(batch_size, -1, 5) # B,S,C
                            loss_eval, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False) # where = -1
                            pred_at_certain = predictions.argmax(2)

                        # Accumulate metrics
                        total_train_loss += loss_eval * batch_size # Sum losses
                        correct_steps = (pred_at_certain == targets) # B, S boolean
                        total_train_correct_certain += correct_steps.sum() # Sum correct steps across batch
                        total_train_mazes_solved += correct_steps.all(dim=-1).sum() # Sum mazes where all steps are correct
                        total_train_steps += batch_size * seq_len
                        total_train_mazes += batch_size

                        if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                        pbar_inner.update(1)

                # Aggregate Train Metrics
                if world_size > 1:
                    dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_correct_certain, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_mazes_solved, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_steps, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_mazes, op=dist.ReduceOp.SUM)

                # Calculate final Train metrics on Rank 0
                if is_main_process(rank) and total_train_mazes > 0:
                    avg_train_loss = total_train_loss.item() / total_train_mazes.item() # Avg loss per maze/sample
                    avg_train_acc_step = total_train_correct_certain.item() / total_train_steps.item() # Avg correct step %
                    avg_train_acc_maze = total_train_mazes_solved.item() / total_train_mazes.item() # Avg full maze solved %
                    train_losses.append(avg_train_loss)
                    train_accuracies_most_certain.append(avg_train_acc_step)
                    train_accuracies_most_certain_permaze.append(avg_train_acc_maze)
                    # train_accuracies list remains unused/placeholder for this simplified metric structure
                    print(f"Iter {bi} Train Metrics (Agg): Loss={avg_train_loss:.4f}, StepAcc={avg_train_acc_step:.4f}, MazeAcc={avg_train_acc_maze:.4f}")

                # TEST METRICS
                total_test_loss = torch.tensor(0.0, device=device)
                total_test_correct_certain = torch.tensor(0.0, device=device)
                total_test_mazes_solved = torch.tensor(0.0, device=device)
                total_test_steps = torch.tensor(0.0, device=device)
                total_test_mazes = torch.tensor(0.0, device=device)

                pbar_inner_desc = 'Eval Test (Rank 0)' if is_main_process(rank) else None
                with tqdm(total=len(testloader), desc=pbar_inner_desc, leave=False, position=1, dynamic_ncols=True, disable=not is_main_process(rank)) as pbar_inner:
                    for inferi, (inputs, targets) in enumerate(testloader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        batch_size = inputs.size(0)
                        seq_len = targets.size(1)

                        loss_eval = None
                        pred_at_certain = None
                        if args.model == 'ctm':
                            predictions_raw, certainties, _ = model(inputs)
                            predictions = predictions_raw.reshape(batch_size, -1, 5, predictions_raw.size(-1))
                            loss_eval, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                            pred_at_certain = predictions.argmax(2)[torch.arange(batch_size, device=device), :, where_most_certain]
                        elif args.model == 'lstm':
                            predictions_raw, certainties, _ = model(inputs)
                            predictions = predictions_raw.reshape(batch_size, -1, 5, predictions_raw.size(-1))
                            loss_eval, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False)
                            pred_at_certain = predictions.argmax(2)[torch.arange(batch_size, device=device), :, where_most_certain]
                        elif args.model == 'ff':
                            predictions_raw = model(inputs)
                            predictions = predictions_raw.reshape(batch_size, -1, 5)
                            loss_eval, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False)
                            pred_at_certain = predictions.argmax(2)

                        total_test_loss += loss_eval * batch_size
                        correct_steps = (pred_at_certain == targets)
                        total_test_correct_certain += correct_steps.sum()
                        total_test_mazes_solved += correct_steps.all(dim=-1).sum()
                        total_test_steps += batch_size * seq_len
                        total_test_mazes += batch_size

                        if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                        pbar_inner.update(1)

                # Aggregate Test Metrics
                if world_size > 1:
                    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_correct_certain, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_mazes_solved, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_steps, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_mazes, op=dist.ReduceOp.SUM)

                # Calculate and Plot final Test metrics on Rank 0
                if is_main_process(rank) and total_test_mazes > 0:
                    avg_test_loss = total_test_loss.item() / total_test_mazes.item()
                    avg_test_acc_step = total_test_correct_certain.item() / total_test_steps.item()
                    avg_test_acc_maze = total_test_mazes_solved.item() / total_test_mazes.item()
                    test_losses.append(avg_test_loss)
                    test_accuracies_most_certain.append(avg_test_acc_step)
                    test_accuracies_most_certain_permaze.append(avg_test_acc_maze)
                    print(f"Iter {bi} Test Metrics (Agg): Loss={avg_test_loss:.4f}, StepAcc={avg_test_acc_step:.4f}, MazeAcc={avg_test_acc_maze:.4f}\n")

                    # --- Plotting ---
                    figacc = plt.figure(figsize=(10, 10))
                    axacc_train = figacc.add_subplot(211)
                    axacc_test = figacc.add_subplot(212)

                    # Plot Avg Step Accuracy
                    axacc_train.plot(iters, train_accuracies_most_certain, 'k-', alpha=0.7, label=f'Avg Step Acc ({train_accuracies_most_certain[-1]:.3f})')
                    axacc_test.plot(iters, test_accuracies_most_certain, 'k-', alpha=0.7, label=f'Avg Step Acc ({test_accuracies_most_certain[-1]:.3f})')
                    # Plot Full Maze Accuracy
                    axacc_train.plot(iters, train_accuracies_most_certain_permaze, 'r-', alpha=0.6, label=f'Full Maze Acc ({train_accuracies_most_certain_permaze[-1]:.3f})')
                    axacc_test.plot(iters, test_accuracies_most_certain_permaze, 'r-', alpha=0.6, label=f'Full Maze Acc ({test_accuracies_most_certain_permaze[-1]:.3f})')

                    axacc_train.set_title('Train Accuracy (Aggregated)')
                    axacc_test.set_title('Test Accuracy (Aggregated)')
                    axacc_train.legend(loc='lower right')
                    axacc_test.legend(loc='lower right')
                    axacc_train.set_xlim([0, args.training_iterations])
                    axacc_test.set_xlim([0, args.training_iterations])
                    axacc_train.set_ylim([0, 1])
                    axacc_test.set_ylim([0, 1])

                    figacc.tight_layout()
                    figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                    plt.close(figacc)

                    # Loss Plot
                    figloss = plt.figure(figsize=(10, 5))
                    axloss = figloss.add_subplot(111)
                    axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train (Agg): {train_losses[-1]:.4f}')
                    axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test (Agg): {test_losses[-1]:.4f}')
                    axloss.legend(loc='upper right')
                    axloss.set_xlabel("Iteration")
                    axloss.set_ylabel("Loss")
                    axloss.set_xlim([0, args.training_iterations])
                    axloss.set_ylim(bottom=0)
                    figloss.tight_layout()
                    figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                    plt.close(figloss)
                    # --- End Plotting ---


                # --- Visualization (Rank 0, Conditional) ---
                if is_main_process(rank) and args.model in ['ctm', 'lstm']:
                    # try:
                    model_module = model.module if isinstance(model, DDP) else model
                    # Use a consistent batch for viz if possible, or just next batch
                    inputs_viz, targets_viz = next(iter(testloader))
                    inputs_viz = inputs_viz.to(device)
                    targets_viz = targets_viz.to(device)
                    longest_index = (targets_viz!=4).sum(-1).argmax() # 4 assumed padding

                    pbar.set_description('Tracking (Rank 0): Viz Fwd Pass')
                    predictions_viz_raw, _, _, _, post_activations_viz, attention_tracking_viz = model_module(inputs_viz, track=True)
                    predictions_viz = predictions_viz_raw.reshape(predictions_viz_raw.size(0), -1, 5, predictions_viz_raw.size(-1))

                    att_shape = (model.module.kv_features.shape[2], model.module.kv_features.shape[3])
                    attention_tracking_viz = attention_tracking_viz.reshape(
                        attention_tracking_viz.shape[0], 
                        attention_tracking_viz.shape[1], -1, att_shape[0], att_shape[1])

                    pbar.set_description('Tracking (Rank 0): Dynamics Plot')
                    plot_neural_dynamics(post_activations_viz, 100, args.log_dir, axis_snap=True)

                    pbar.set_description('Tracking (Rank 0): Maze GIF')
                    if attention_tracking_viz is not None:
                            make_maze_gif((inputs_viz[longest_index].detach().cpu().numpy()+1)/2,
                                        predictions_viz[longest_index].detach().cpu().numpy(),
                                        targets_viz[longest_index].detach().cpu().numpy(),
                                        attention_tracking_viz[:, longest_index],
                                        args.log_dir)
                        # else:
                        #      print("Skipping maze GIF due to attention shape issue.")

                    # except Exception as e_viz:
                    #     print(f"Rank 0 visualization failed: {e_viz}")
                # --- End Visualization ---

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if world_size > 1: dist.barrier()
        model.train()
        # --- End Evaluation Block ---




        if hasattr(train_sampler, 'set_epoch'): # Check if sampler has set_epoch
            train_sampler.set_epoch(bi)

        current_lr = optimizer.param_groups[-1]['lr']

        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Defaults for logging
        loss = torch.tensor(0.0, device=device) # Need loss defined for logging scope
        accuracy_finegrained = 0.0
        where_most_certain_val = -1.0
        where_most_certain_std = 0.0
        where_most_certain_min = -1
        where_most_certain_max = -1
        upto_where_mean = -1.0
        upto_where_std = 0.0
        upto_where_min = -1
        upto_where_max = -1

        with torch.autocast(device_type="cuda" if device.type == 'cuda' else "cpu", dtype=torch.float16, enabled=args.use_amp):
            if args.do_compile: torch.compiler.cudagraph_mark_step_begin()

            if args.model == 'ctm':
                predictions_raw, certainties, _ = model(inputs)
                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=True)
                with torch.no_grad(): # Calculate local accuracy for logging
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=device), :, where_most_certain] == targets).float().mean().item()
            elif args.model == 'lstm':
                predictions_raw, certainties, _ = model(inputs)
                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False) # where = -1
                with torch.no_grad():
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=device), :, where_most_certain] == targets).float().mean().item()
            elif args.model == 'ff':
                predictions_raw = model(inputs) # B, S*C
                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5) # B,S,C
                loss, where_most_certain, upto_where = maze_loss(predictions.unsqueeze(-1), None, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False) # where = -1
                with torch.no_grad():
                    accuracy_finegrained = (predictions.argmax(2) == targets).float().mean().item()

            # Extract stats from loss outputs
            if torch.is_tensor(where_most_certain):
                where_most_certain_val = where_most_certain.float().mean().item()
                where_most_certain_std = where_most_certain.float().std().item()
                where_most_certain_min = where_most_certain.min().item()
                where_most_certain_max = where_most_certain.max().item()
            elif isinstance(where_most_certain, int):
                 where_most_certain_val = float(where_most_certain); where_most_certain_min = where_most_certain; where_most_certain_max = where_most_certain
            if isinstance(upto_where, (np.ndarray, list)) and len(upto_where) > 0:
                 upto_where_mean = np.mean(upto_where); upto_where_std = np.std(upto_where); upto_where_min = np.min(upto_where); upto_where_max = np.max(upto_where)

        # Backprop / Step
        scaler.scale(loss).backward()
        if args.gradient_clipping!=-1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # --- Aggregation and Logging (Rank 0) ---
        loss_log = loss.detach()
        if world_size > 1: dist.all_reduce(loss_log, op=dist.ReduceOp.AVG)

        if is_main_process(rank):
             pbar_desc = f'Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_finegrained:.3f} LR={current_lr:.6f}'
             if args.model in ['ctm', 'lstm'] or torch.is_tensor(where_most_certain):
                  pbar_desc += f' Cert={where_most_certain_val:.2f}'#+-{where_most_certain_std:.2f}' # Removed std for brevity
             if isinstance(upto_where, (np.ndarray, list)) and len(upto_where) > 0:
                  pbar_desc += f' Path={upto_where_mean:.1f}'#+-{upto_where_std:.1f}'
             pbar.set_description(f'{args.model.upper()} {pbar_desc}')
        # --- End Aggregation and Logging ---


        


        # --- Checkpointing (Rank 0) ---
        if (bi % args.save_every == 0 or bi == args.training_iterations - 1) and bi != start_iter and is_main_process(rank):
            pbar.set_description('Rank 0: Saving checkpoint...') 
            save_path = f'{args.log_dir}/checkpoint.pt'
            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

            checkpoint_data = {
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'iteration': bi,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies, # Saving simplified scalar list
                'test_accuracies': test_accuracies,   # Saving simplified scalar list
                'train_accuracies_most_certain': train_accuracies_most_certain,
                'test_accuracies_most_certain': test_accuracies_most_certain,
                'train_accuracies_most_certain_permaze': train_accuracies_most_certain_permaze,
                'test_accuracies_most_certain_permaze': test_accuracies_most_certain_permaze,
                'iters': iters,
                'args': args,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
            }
            torch.save(checkpoint_data, save_path)
        # --- End Checkpointing ---


        if world_size > 1: dist.barrier()

        if is_main_process(rank):
            pbar.update(1)
    # --- End Training Loop ---

    if is_main_process(rank):
        pbar.close()

    cleanup_ddp()