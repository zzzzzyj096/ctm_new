import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')
import torch.nn as nn 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.samplers import FastRandomDistributedSampler 
from tqdm.auto import tqdm

from tasks.image_classification.train import get_dataset # Use shared get_dataset

# Model Imports
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline

# Plotting/Utils Imports
from tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss # For CTM, LSTM
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="UserWarning: Metadata Warning, tag 274 had too many entries: 4, expected 1")
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, required=True, choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    # Common
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet18-4', help='Type of backbone featureiser.')
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (CTM, LSTM).') 
    parser.add_argument('--iterations', type=int, default=50, help='Number of internal ticks (CTM, LSTM).') 
    parser.add_argument('--positional_embedding_type', type=str, default='none', help='Type of positional embedding (CTM, LSTM).',
                        choices=['none',
                                 'learnable-fourier',
                                 'multi-learnable-fourier',
                                 'custom-rotational'])
    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch (CTM only).')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch (CTM only).')
    parser.add_argument('--neuron_select_type', type=str, default='first-last', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=256, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory (CTM only).')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).')

    # Training 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (per GPU).')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing (per GPU).')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--num_workers_train', type=int, default=1, help='Num workers training.')
    parser.add_argument('--use_custom_sampler', action=argparse.BooleanOptionalAction, default=False, help='Use custom fast sampler to avoid reshuffling.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components.')

    # Housekeeping 
    parser.add_argument('--log_dir', type=str, default='logs/scratch', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use.')
    parser.add_argument('--data_root', type=str, default='data/', help='Where to save dataset.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True, help='Should use strict reload for model weights.') 
    parser.add_argument('--ignore_metrics_when_reloading', action=argparse.BooleanOptionalAction, default=False, help='Ignore metrics when reloading?') 

    # Tracking 
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--plot_indices', type=int, default=[0], nargs='+', help='Which indices in test data to plot?') # Defaulted to 0

    # Precision
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    args = parser.parse_args()
    return args

# --- DDP Setup Functions ---
def setup_ddp():
    if 'RANK' not in os.environ:
        # Basic setup for non-distributed run
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # Ensure this port is free
        os.environ['LOCAL_RANK'] = '0'
        print("Running in non-distributed mode (simulated DDP setup).")
        # Need to manually init if only 1 process desired for non-GPU testing
        if not torch.cuda.is_available() or int(os.environ['WORLD_SIZE']) == 1:
            dist.init_process_group(backend='gloo') # Gloo backend for CPU
            print("Initialized process group with Gloo backend for single/CPU process.")
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            return rank, world_size, local_rank


    # Standard DDP setup
    dist.init_process_group(backend='nccl') # 'nccl' for NVIDIA GPUs
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank} setup on GPU {local_rank}")
    else:
         print(f"Rank {rank} setup on CPU (GPU not available or requested)")
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

    set_seed(args.seed + rank, False) # Add rank for different seeds per process

    # Rank 0 handles directory creation and initial logging
    if is_main_process(rank):
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        zip_python_code(f'{args.log_dir}/repo_state.zip')
        with open(f'{args.log_dir}/args.txt', 'w') as f:
            print(args, file=f)
    if world_size > 1: dist.barrier() # Sync after rank 0 setup


    assert args.dataset in ['cifar10', 'cifar100', 'imagenet']

    # Data Loading
    train_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(args.dataset, args.data_root)

    # Setup Samplers
    # This custom sampler is useful when using large batch sizes for Cifar. Otherwise the reshuffle happens tediously often
    train_sampler = (FastRandomDistributedSampler(train_data, num_replicas=world_size, rank=rank, seed=args.seed, epoch_steps=int(10e10))
                     if args.use_custom_sampler else
                     DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed) # No shuffle needed for test; consistent

    # Setup DataLoaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                                              num_workers=args.num_workers_train, pin_memory=True, drop_last=True) # drop_last=True often used in DDP
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, sampler=test_sampler,
                                             num_workers=1, pin_memory=True, drop_last=False)


    prediction_reshaper = [-1]  # Task specific
    args.out_dims = len(class_labels)

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        if world_size > 1:
             warnings.warn("Running DDP on CPU is not recommended.")
    if is_main_process(rank):
        print(f'Main process (Rank {rank}): Using device {device}. World size: {world_size}. Model: {args.model}')

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
            start_type=args.start_type,
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

    # Initialize lazy modules if any
    try:
        pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
        model_base(pseudo_inputs)
    except Exception as e:
         print(f"Warning: Pseudo forward pass failed: {e}")

    # Wrap model with DDP
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model_base, device_ids=[local_rank], output_device=local_rank)
    elif device.type == 'cpu' and world_size > 1:
        model = DDP(model_base) # No device_ids for CPU
    else: # Single process run
        model = model_base # No DDP wrapping needed

    if is_main_process(rank):
        # Access underlying model for param count
        param_count = sum(p.numel() for p in model.module.parameters() if p.requires_grad) if world_size > 1 else sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total trainable params: {param_count}')
    # --- End Model Definition ---


    # Optimizer and scheduler
    # Use model.parameters() directly, DDP handles it
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


    # Metrics tracking (on Rank 0)
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = [] # Placeholder for potential detailed accuracy
    test_accuracies = []  # Placeholder for potential detailed accuracy
    # Conditional metrics
    train_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None # Scalar accuracy list
    test_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None  # Scalar accuracy list
    train_accuracies_standard = [] if args.model == 'ff' else None # Standard accuracy list for FF
    test_accuracies_standard = [] if args.model == 'ff' else None  # Standard accuracy list for FF
    iters = []

    scaler = torch.amp.GradScaler("cuda" if device.type == 'cuda' else "cpu", enabled=args.use_amp) 
    # Reloading Logic
    if args.reload:
        map_location = device # Load directly onto the process's device
        chkpt_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(chkpt_path):
            print(f'Rank {rank}: Reloading from: {chkpt_path}')
            checkpoint = torch.load(chkpt_path, map_location=map_location, weights_only=False)

            # Determine underlying model based on whether DDP wrapping occurred
            model_to_load = model.module if isinstance(model, DDP) else model

            # Handle potential 'module.' prefix in saved state_dict
            state_dict = checkpoint['model_state_dict']
            has_module_prefix = all(k.startswith('module.') for k in state_dict)
            is_wrapped = isinstance(model, DDP)

            if has_module_prefix and not is_wrapped:
                # Saved with DDP, loading into non-DDP model -> remove prefix
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
                scaler_state_dict = checkpoint['scaler_state_dict']
                if scaler.is_enabled():
                    print("Loading non-empty GradScaler state dict.")
                    try:
                        scaler.load_state_dict(scaler_state_dict)
                    except Exception as e:
                        print(f"Error loading GradScaler state dict: {e}")
                        print("Continuing with a fresh GradScaler state.")

                start_iter = checkpoint['iteration']
                # Only rank 0 loads metric history
                if is_main_process(rank) and not args.ignore_metrics_when_reloading:
                    print(f'Rank {rank}: Reloading metrics history.')
                    iters = checkpoint['iters']
                    train_losses = checkpoint['train_losses']
                    test_losses = checkpoint['test_losses']
                    train_accuracies = checkpoint['train_accuracies']
                    test_accuracies = checkpoint['test_accuracies']
                    if args.model in ['ctm', 'lstm']:
                        train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                        test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                    elif args.model == 'ff':
                        train_accuracies_standard = checkpoint['train_accuracies_standard']
                        test_accuracies_standard = checkpoint['test_accuracies_standard']
                elif is_main_process(rank) and args.ignore_metrics_when_reloading:
                     print(f'Rank {rank}: Ignoring metrics history upon reload.')

            else:
                 print(f'Rank {rank}: Only reloading model weights!')

            # Load RNG states
            if is_main_process(rank) and 'torch_rng_state' in checkpoint and not args.reload_model_only:
                print(f'Rank {rank}: Loading RNG states (may need DDP adaptation for full reproducibility).')
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu()) # Load CPU state
                # Add CUDA state loading if needed, ensuring correct device handling
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"Rank {rank}: Reload finished, starting from iteration {start_iter}")
        else:
            print(f"Rank {rank}: Checkpoint not found at {chkpt_path}, starting from scratch.")
        if world_size > 1: dist.barrier() # Sync after loading


    # Conditional Compilation
    if args.do_compile:
        if is_main_process(rank): print('Compiling model components...')
        # Compile on the underlying model if wrapped
        model_to_compile = model.module if isinstance(model, DDP) else model
        if hasattr(model_to_compile, 'backbone'):
            model_to_compile.backbone = torch.compile(model_to_compile.backbone, mode='reduce-overhead', fullgraph=True)
        if args.model == 'ctm':
            if hasattr(model_to_compile, 'synapses'):
                model_to_compile.synapses = torch.compile(model_to_compile.synapses, mode='reduce-overhead', fullgraph=True)
        if world_size > 1: dist.barrier() # Sync after compilation
        if is_main_process(rank): print('Compilation finished.')


    # --- Training Loop ---
    model.train() # Ensure model is in train mode
    pbar = tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True, disable=not is_main_process(rank))

    iterator = iter(trainloader) 

    for bi in range(start_iter, args.training_iterations):

        # Set sampler epoch (important for shuffling in DistributedSampler)
        if not args.use_custom_sampler and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(bi)

        current_lr = optimizer.param_groups[-1]['lr']

        time_start_data = time.time()
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            # Reset iterator - set_epoch handles shuffling if needed
            iterator = iter(trainloader)
            inputs, targets = next(iterator)
        

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time_end_data = time.time()

        loss = None
        # Model-specific forward and loss calculation
        time_start_forward = time.time()
        with torch.autocast(device_type="cuda" if device.type == 'cuda' else "cpu", dtype=torch.float16, enabled=args.use_amp):
            if args.do_compile:
                 torch.compiler.cudagraph_mark_step_begin()

            if args.model == 'ctm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'lstm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'ff':
                predictions = model(inputs) # FF returns only predictions
                loss = nn.CrossEntropyLoss()(predictions, targets)
                where_most_certain = None # Not applicable for FF standard loss
        time_end_forward = time.time()
        time_start_backward = time.time()
        
        scaler.scale(loss).backward() # DDP handles gradient synchronization
        time_end_backward = time.time()

        if args.gradient_clipping!=-1:
            scaler.unscale_(optimizer)
            # Clip gradients across all parameters controlled by the optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # --- Aggregation and Logging (Rank 0) ---
        # Aggregate loss for logging
        loss_log = loss.detach() # Use detached loss for aggregation
        if world_size > 1: dist.all_reduce(loss_log, op=dist.ReduceOp.AVG)

        if is_main_process(rank):
             # Calculate accuracy locally on rank 0 for description (approximate)
             # Note: This uses rank 0's batch, not aggregated accuracy
             accuracy_local = 0.0
             if args.model in ['ctm', 'lstm']:
                accuracy_local = (predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain] == targets).float().mean().item()
                where_certain_tensor = where_most_certain.float() # Use rank 0's tensor for stats
                pbar_desc = f'Timing; d={(time_end_data-time_start_data):0.3f}, f={(time_end_forward-time_start_forward):0.3f}, b={(time_end_backward-time_start_backward):0.3f}. Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_local:.3f} LR={current_lr:.6f} WhereCert(loc)={where_certain_tensor.mean().item():.2f}'
             elif args.model == 'ff':
                accuracy_local = (predictions.argmax(1) == targets).float().mean().item()
                pbar_desc = f'Timing; d={(time_end_data-time_start_data):0.3f}, f={(time_end_forward-time_start_forward):0.3f}, b={(time_end_backward-time_start_backward):0.3f}. Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_local:.3f} LR={current_lr:.6f}'

             pbar.set_description(f'{args.model.upper()} {pbar_desc}')
        # --- End Aggregation and Logging ---


        # --- Evaluation and Plotting (Rank 0 + Aggregation) ---
        if bi % args.track_every == 0 and (bi != 0 or args.reload_model_only):
            
            model.eval()
            with torch.inference_mode():

                
                # --- Distributed Evaluation ---
                iters.append(bi)

                # TRAIN METRICS
                total_train_loss = torch.tensor(0.0, device=device)
                total_train_correct_certain = torch.tensor(0.0, device=device) # CTM/LSTM
                total_train_correct_standard = torch.tensor(0.0, device=device) # FF
                total_train_samples = torch.tensor(0.0, device=device)

                # Use a sampler for evaluation to ensure non-overlapping data if needed
                train_eval_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False)
                train_eval_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, sampler=train_eval_sampler, num_workers=1, pin_memory=True)

                pbar_inner_desc = 'Eval Train (Rank 0)' if is_main_process(rank) else None
                with tqdm(total=len(train_eval_loader), desc=pbar_inner_desc, leave=False, position=1, dynamic_ncols=True, disable=not is_main_process(rank)) as pbar_inner:
                    for inferi, (inputs, targets) in enumerate(train_eval_loader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        loss_eval = None
                        if args.model == 'ctm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                            total_train_correct_certain += (preds_eval == targets).sum()
                        elif args.model == 'lstm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                            total_train_correct_certain += (preds_eval == targets).sum()
                        elif args.model == 'ff':
                            predictions = model(inputs)
                            loss_eval = nn.CrossEntropyLoss()(predictions, targets)
                            preds_eval = predictions.argmax(1)
                            total_train_correct_standard += (preds_eval == targets).sum()

                        total_train_loss += loss_eval * inputs.size(0)
                        total_train_samples += inputs.size(0)

                        if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                        pbar_inner.update(1)

                # Aggregate Train Metrics
                if world_size > 1:
                    dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_correct_certain, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_correct_standard, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_train_samples, op=dist.ReduceOp.SUM)

                # Calculate final Train metrics on Rank 0
                if is_main_process(rank) and total_train_samples > 0:
                    avg_train_loss = total_train_loss.item() / total_train_samples.item()
                    train_losses.append(avg_train_loss)
                    if args.model in ['ctm', 'lstm']:
                        avg_train_acc_certain = total_train_correct_certain.item() / total_train_samples.item()
                        train_accuracies_most_certain.append(avg_train_acc_certain)
                    elif args.model == 'ff':
                        avg_train_acc_standard = total_train_correct_standard.item() / total_train_samples.item()
                        train_accuracies_standard.append(avg_train_acc_standard)
                    print(f"Iter {bi} Train Metrics (Agg): Loss={avg_train_loss:.4f}")

                # TEST METRICS
                total_test_loss = torch.tensor(0.0, device=device)
                total_test_correct_certain = torch.tensor(0.0, device=device) # CTM/LSTM
                total_test_correct_standard = torch.tensor(0.0, device=device) # FF
                total_test_samples = torch.tensor(0.0, device=device)

                pbar_inner_desc = 'Eval Test (Rank 0)' if is_main_process(rank) else None
                with tqdm(total=len(testloader), desc=pbar_inner_desc, leave=False, position=1, dynamic_ncols=True, disable=not is_main_process(rank)) as pbar_inner:
                    for inferi, (inputs, targets) in enumerate(testloader): # Testloader already uses sampler
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        loss_eval = None
                        if args.model == 'ctm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                            total_test_correct_certain += (preds_eval == targets).sum()
                        elif args.model == 'lstm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                            total_test_correct_certain += (preds_eval == targets).sum()
                        elif args.model == 'ff':
                            predictions = model(inputs)
                            loss_eval = nn.CrossEntropyLoss()(predictions, targets)
                            preds_eval = predictions.argmax(1)
                            total_test_correct_standard += (preds_eval == targets).sum()

                        total_test_loss += loss_eval * inputs.size(0)
                        total_test_samples += inputs.size(0)

                        if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                        pbar_inner.update(1)

                # Aggregate Test Metrics
                if world_size > 1:
                    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_correct_certain, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_correct_standard, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_test_samples, op=dist.ReduceOp.SUM)

                # Calculate and Plot final Test metrics on Rank 0
                if is_main_process(rank) and total_test_samples > 0:
                    avg_test_loss = total_test_loss.item() / total_test_samples.item()
                    test_losses.append(avg_test_loss)
                    acc_label = ''
                    acc_val = 0.0
                    if args.model in ['ctm', 'lstm']:
                        avg_test_acc_certain = total_test_correct_certain.item() / total_test_samples.item()
                        test_accuracies_most_certain.append(avg_test_acc_certain)
                        acc_label = f'Most certain ({avg_test_acc_certain:.3f})'
                        acc_val = avg_test_acc_certain
                    elif args.model == 'ff':
                        avg_test_acc_standard = total_test_correct_standard.item() / total_test_samples.item()
                        test_accuracies_standard.append(avg_test_acc_standard)
                        acc_label = f'Standard Acc ({avg_test_acc_standard:.3f})'
                        acc_val = avg_test_acc_standard
                    print(f"Iter {bi} Test Metrics (Agg): Loss={avg_test_loss:.4f}, Acc={acc_val:.4f}\n")


                    # --- Plotting ---
                    figacc = plt.figure(figsize=(10, 10))
                    axacc_train = figacc.add_subplot(211)
                    axacc_test = figacc.add_subplot(212)

                    if args.model in ['ctm', 'lstm']:
                        axacc_train.plot(iters, train_accuracies_most_certain, 'k-', alpha=0.9, label=f'Most certain ({train_accuracies_most_certain[-1]:.3f})')
                        axacc_test.plot(iters, test_accuracies_most_certain, 'k-', alpha=0.9, label=acc_label)
                    elif args.model == 'ff':
                        axacc_train.plot(iters, train_accuracies_standard, 'k-', alpha=0.9, label=f'Standard Acc ({train_accuracies_standard[-1]:.3f})')
                        axacc_test.plot(iters, test_accuracies_standard, 'k-', alpha=0.9, label=acc_label)

                    axacc_train.set_title('Train Accuracy (Aggregated)')
                    axacc_test.set_title('Test Accuracy (Aggregated)')
                    axacc_train.legend(loc='lower right')
                    axacc_test.legend(loc='lower right')
                    axacc_train.set_xlim([0, args.training_iterations])
                    axacc_test.set_xlim([0, args.training_iterations])

                    # Keep dataset specific ylim adjustments if needed
                    if args.dataset == 'imagenet':
                        # For easy comparison when training
                        train_ylim_set = False
                        if args.model in ['ctm', 'lstm'] and len(train_accuracies_most_certain)>0 and np.any(np.array(train_accuracies_most_certain)>0.4): train_ylim_set=True; axacc_train.set_ylim([0.4, 1])
                        if args.model == 'ff' and len(train_accuracies_standard)>0 and np.any(np.array(train_accuracies_standard)>0.4): train_ylim_set=True; axacc_train.set_ylim([0.4, 1])

                        test_ylim_set = False
                        if args.model in ['ctm', 'lstm'] and len(test_accuracies_most_certain)>0 and np.any(np.array(test_accuracies_most_certain)>0.3): test_ylim_set=True; axacc_test.set_ylim([0.3, 0.8])
                        if args.model == 'ff' and len(test_accuracies_standard)>0 and np.any(np.array(test_accuracies_standard)>0.3): test_ylim_set=True; axacc_test.set_ylim([0.3, 0.8])


                    figacc.tight_layout()
                    figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                    plt.close(figacc)

                    # Loss Plot
                    figloss = plt.figure(figsize=(10, 5))
                    axloss = figloss.add_subplot(111)
                    axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train (Aggregated): {train_losses[-1]:.4f}')
                    axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test (Aggregated): {test_losses[-1]:.4f}')
                    axloss.legend(loc='upper right')
                    axloss.set_xlabel("Iteration")
                    axloss.set_ylabel("Loss")
                    axloss.set_xlim([0, args.training_iterations])
                    axloss.set_ylim(bottom=0)
                    figloss.tight_layout()
                    figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                    plt.close(figloss)
                    # --- End Plotting ---

                # Visualization on Rank 0
                if is_main_process(rank) and args.model in ['ctm', 'lstm']:
                    try:
                        model_module = model.module if isinstance(model, DDP) else model # Get underlying model
                        # Simplified viz: use first batch from testloader
                        inputs_viz, targets_viz = next(iter(testloader))
                        inputs_viz = inputs_viz.to(device)
                        targets_viz = targets_viz.to(device)

                        pbar.set_description('Tracking (Rank 0): Viz Fwd Pass')
                        predictions_viz, certainties_viz, _, pre_activations_viz, post_activations_viz, attention_tracking_viz = model_module(inputs_viz, track=True)

                        att_shape = (model_module.kv_features.shape[2], model_module.kv_features.shape[3])
                        attention_tracking_viz = attention_tracking_viz.reshape(
                            attention_tracking_viz.shape[0], 
                            attention_tracking_viz.shape[1], -1, att_shape[0], att_shape[1])


                        pbar.set_description('Tracking (Rank 0): Dynamics Plot')
                        plot_neural_dynamics(post_activations_viz, 100, args.log_dir, axis_snap=True)

                        # Plot specific indices from test_data directly
                        pbar.set_description('Tracking (Rank 0): GIF Generation')
                        for plot_idx in args.plot_indices:
                            try:
                                if plot_idx < len(test_data):
                                    inputs_plot, target_plot = test_data.__getitem__(plot_idx)
                                    inputs_plot = inputs_plot.unsqueeze(0).to(device)

                                    preds_plot, certs_plot, _, _, posts_plot, atts_plot = model_module(inputs_plot, track=True)
                                    atts_plot = atts_plot.reshape(atts_plot.shape[0], atts_plot.shape[1], -1, att_shape[0], att_shape[1])
                                    

                                    img_gif = np.moveaxis(np.clip(inputs_plot[0].detach().cpu().numpy()*np.array(dataset_std).reshape(len(dataset_std), 1, 1) + np.array(dataset_mean).reshape(len(dataset_mean), 1, 1), 0, 1), 0, -1)

                                    make_classification_gif(img_gif, target_plot, preds_plot[0].detach().cpu().numpy(), certs_plot[0].detach().cpu().numpy(),
                                                        posts_plot[:,0], atts_plot[:,0] if atts_plot is not None else None, class_labels,
                                                        f'{args.log_dir}/idx{plot_idx}_attention.gif')
                                else:
                                    print(f"Warning: Plot index {plot_idx} out of range for test dataset size {len(test_data)}.")
                            except Exception as e_gif:
                                print(f"Rank 0 GIF generation failed for index {plot_idx}: {e_gif}")

                    except Exception as e_viz:
                        print(f"Rank 0 visualization failed: {e_viz}")



            if world_size > 1: dist.barrier() # Sync after evaluation block
            model.train() # Set back to train mode
        # --- End Evaluation Block ---


        # --- Checkpointing (Rank 0) ---
        if (bi % args.save_every == 0 or bi == args.training_iterations - 1) and bi != start_iter and is_main_process(rank):
            pbar.set_description('Rank 0: Saving checkpoint...')
            save_path = f'{args.log_dir}/checkpoint.pt'
            # Access underlying model state dict if DDP is used
            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

            save_dict = {
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'iteration': bi,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'iters': iters,
                    'args': args,
                    'torch_rng_state': torch.get_rng_state(), # CPU state
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    # Include conditional metrics
                    'train_accuracies': train_accuracies, # Placeholder
                    'test_accuracies': test_accuracies,   # Placeholder
                }
            if args.model in ['ctm', 'lstm']:
                save_dict['train_accuracies_most_certain'] = train_accuracies_most_certain
                save_dict['test_accuracies_most_certain'] = test_accuracies_most_certain
            elif args.model == 'ff':
                save_dict['train_accuracies_standard'] = train_accuracies_standard
                save_dict['test_accuracies_standard'] = test_accuracies_standard

            torch.save(save_dict , save_path)
            pbar.set_description(f"Rank 0: Checkpoint saved to {save_path}")
        # --- End Checkpointing ---


        if world_size > 1: dist.barrier() # Sync before next iteration

        # Update pbar on Rank 0
        if is_main_process(rank):
            pbar.update(1)
    # --- End Training Loop ---

    if is_main_process(rank):
        pbar.close()

    cleanup_ddp() # Cleanup DDP resources