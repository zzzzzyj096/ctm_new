# --- Core Libraries ---
import torch
import numpy as np
import os
import argparse
from tqdm.auto import tqdm
import torch.nn.functional as F # Used for interpolate

# --- Plotting & Visualization ---
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import patheffects
import seaborn as sns
import imageio
import cv2
from scipy.special import softmax
from tasks.image_classification.plotting import save_frames_to_mp4

# --- Data Handling & Model ---
from torchvision import transforms
from torchvision import datasets # Only used for CIFAR100 in debug mode
from scipy import ndimage # Used in find_island_centers
from data.custom_datasets import ImageNet 
from models.ctm import ContinuousThoughtMachine 
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES 
from tasks.image_classification.plotting import plot_neural_dynamics

# --- Global Settings ---
np.seterr(divide='ignore') 
mpl.use('Agg') 
sns.set_style('darkgrid')

# --- Helper Functions ---

def find_island_centers(array_2d, threshold):
    """
    Finds the center of mass of each island (connected component > threshold)
    in a 2D array, weighted by the array's values.
    Returns list of (y, x) centers and list of areas.
    """
    binary_image = array_2d > threshold
    labeled_image, num_labels = ndimage.label(binary_image)
    centers = []
    areas = []
    # Calculate center of mass for each labeled island (label 0 is background)
    for i in range(1, num_labels + 1):
        island_mask = (labeled_image == i)
        total_mass = np.sum(array_2d[island_mask])
        if total_mass > 0:
            # Get coordinates for this island
            y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
            # Calculate weighted average for center
            x_center = np.average(x_coords[island_mask], weights=array_2d[island_mask])
            y_center = np.average(y_coords[island_mask], weights=array_2d[island_mask])
            centers.append((round(y_center, 4), round(x_center, 4)))
            areas.append(np.sum(island_mask)) # Area is the count of pixels in the island
    return centers, areas

def parse_args():
    """Parses command-line arguments."""
    # Note: Original had two ArgumentParser instances, using the second one.
    parser = argparse.ArgumentParser(description="Visualize Continuous Thought Machine Attention")
    parser.add_argument('--actions', type=str, nargs='+', default=['videos'], choices=['plots', 'videos', 'demo'], help="Actions to take. Plots=results plots; videos=gifs/mp4s to watch attention; demo: last frame of internal ticks")
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help="GPU device index or -1 for CPU")
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/imagenet/ctm_clean.pt', help="Path to ATM checkpoint")
    parser.add_argument('--output_dir', type=str, default='tasks/image_classification/analysis/outputs/imagenet_viz', help="Directory for visualization outputs")
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=True, help='Debug mode: use CIFAR100 instead of ImageNet for debugging.')
    parser.add_argument('--plot_every', type=int, default=10, help="How often to plot.")
    
    parser.add_argument('--inference_iterations', type=int, default=50, help="Iterations to use during inference.")
    parser.add_argument('--data_indices', type=int, nargs='+', default=[], help="Use specific indices in validation data for demos, otherwise random.")
    parser.add_argument('--N_to_viz', type=int, default=5, help="When not supplying data_indices.")
    
    return parser.parse_args()


# --- Main Execution Block ---
if __name__=='__main__':

    # --- Setup ---
    args = parse_args()
    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # --- Load Checkpoint & Model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) # removed weights_only=False
    model_args = checkpoint['args']

    # Handle legacy arguments from checkpoint if necessary
    if not hasattr(model_args, 'backbone_type') and hasattr(model_args, 'resnet_type'):
        model_args.backbone_type = f'{model_args.resnet_type}-{getattr(model_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(model_args, 'neuron_select_type'):
        model_args.neuron_select_type = 'first-last'


    # Instantiate Model based on checkpoint args
    print("Instantiating CTM model...")
    model = ContinuousThoughtMachine(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.d_input,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=model_args.n_synch_action,
        synapse_depth=model_args.synapse_depth,
        memory_length=model_args.memory_length,
        deep_nlms=model_args.deep_memory,
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=model_args.do_normalisation,
        backbone_type=model_args.backbone_type,
        positional_embedding_type=model_args.positional_embedding_type,
        out_dims=model_args.out_dims,
        prediction_reshaper=[-1], # Kept fixed value from original code
        dropout=0, # No dropout for eval
        neuron_select_type=model_args.neuron_select_type,
        n_random_pairing_self=model_args.n_random_pairing_self,
    ).to(device)

    # Load weights into model
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
    model.eval() # Set model to evaluation mode

    # --- Prepare Dataset ---
    if args.debug:
        print("Debug mode: Using CIFAR100")
        # CIFAR100 specific normalization constants
        dataset_mean = [0.5070751592371341, 0.48654887331495067, 0.4409178433670344]
        dataset_std = [0.2673342858792403, 0.2564384629170882, 0.27615047132568393]
        img_size = 256 # Resize CIFAR images for consistency
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std), # Normalize
        ])
        validation_dataset = datasets.CIFAR100('data/', train=False, transform=transform, download=True)
        validation_dataset_centercrop = datasets.CIFAR100('data/', train=True, transform=transform, download=True)
    else:
        print("Using ImageNet")
        # ImageNet specific normalization constants
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        img_size = 256 # Resize ImageNet images
        # Note: Original comment mentioned no CenterCrop, this transform reflects that.
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std) # Normalize
        ])
        validation_dataset = ImageNet(which_split='validation', transform=transform)
        validation_dataset_centercrop = ImageNet(which_split='train', transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std) # Normalize
        ]))
    class_labels = list(IMAGENET2012_CLASSES.values()) # Load actual class names

    os.makedirs(f'{args.output_dir}', exist_ok=True)

    interp_mode = 'nearest'
    cmap_calib = sns.color_palette('viridis', as_cmap=True)
    loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    loader_crop = torch.utils.data.DataLoader(validation_dataset_centercrop, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    model.eval()

    figscale = 0.85
    topk = 5
    mean_certainties_correct, mean_certainties_incorrect = [],[]
    tracked_certainties = []
    tracked_targets = []
    tracked_predictions = []

    if model.iterations != args.inference_iterations:
        print('WARNING: you are setting inference iterations to a value not used during training!')

    model.iterations = args.inference_iterations

    if 'plots' in args.actions:
    
        with torch.inference_mode(): # Disable gradient calculations
            with tqdm(total=len(loader), initial=0, leave=False, position=0, dynamic_ncols=True) as pbar:
                imgi = 0
                for bi, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    if bi==0:
                        dynamics_inputs, _ = next(iter(loader_crop))  # Use this because of batching
                        _, _, _, _, post_activations_viz, _ = model(inputs, track=True)
                        plot_neural_dynamics(post_activations_viz, 15*10, args.output_dir, axis_snap=True, N_per_row=15)
                    predictions, certainties, synchronisation = model(inputs)

                    tracked_predictions.append(predictions.detach().cpu().numpy())
                    tracked_targets.append(targets.detach().cpu().numpy())
                    tracked_certainties.append(certainties.detach().cpu().numpy())

                    


                    pbar.set_description(f'Processing base image of size {inputs.shape}')
                    pbar.update(1)
                    if ((bi % args.plot_every == 0) or bi == len(loader)-1) and bi!=0: #

                        concatenated_certainties = np.concatenate(tracked_certainties, axis=0)
                        concatenated_targets = np.concatenate(tracked_targets, axis=0)
                        concatenated_predictions = np.concatenate(tracked_predictions, axis=0)
                        concatenated_predictions_argsorted = np.argsort(concatenated_predictions, 1)[:,::-1]



                        for topk in [1, 5]:
                            concatenated_predictions_argsorted_topk = concatenated_predictions_argsorted[:,:topk]

                            accs_instant, accs_avg, accs_certain = [], [], []
                            accs_avg_logits, accs_weighted_logits = [],[]
                            with tqdm(total=(concatenated_predictions.shape[-1]), initial=0, leave=False, position=1, dynamic_ncols=True) as pbarinner:
                                pbarinner.set_description('Acc types')
                                for stepi in np.arange(concatenated_predictions.shape[-1]):
                                    pred_avg = softmax(concatenated_predictions, 1)[:,:,:stepi+1].mean(-1).argsort(1)[:,-topk:]
                                    pred_instant = concatenated_predictions_argsorted_topk[:,:,stepi]
                                    pred_certain = concatenated_predictions_argsorted_topk[np.arange(concatenated_predictions.shape[0]),:, concatenated_certainties[:,1,:stepi+1].argmax(1)]
                                    pred_avg_logits = concatenated_predictions[:,:,:stepi+1].mean(-1).argsort(1)[:,-topk:]
                                    pred_weighted_logits = (concatenated_predictions[:,:,:stepi+1] * concatenated_certainties[:,1:,:stepi+1]).sum(-1).argsort(1)[:, -topk:]
                                    pbarinner.update(1)
                                    accs_instant.append(np.any(pred_instant==concatenated_targets[...,np.newaxis], -1).mean())
                                    accs_avg.append(np.any(pred_avg==concatenated_targets[...,np.newaxis], -1).mean())
                                    accs_avg_logits.append(np.any(pred_avg==concatenated_targets[...,np.newaxis], -1).mean())
                                    accs_weighted_logits.append(np.any(pred_weighted_logits==concatenated_targets[...,np.newaxis], -1).mean())
                                    accs_certain.append(np.any(pred_avg_logits==concatenated_targets[...,np.newaxis], -1).mean())
                            fig = plt.figure(figsize=(10*figscale, 4*figscale))
                            ax = fig.add_subplot(111)
                            cp = sns.color_palette("bright")
                            ax.plot(np.arange(concatenated_predictions.shape[-1])+1, 100*np.array(accs_instant), linestyle='-', color=cp[0], label='Instant')
                            # ax.plot(np.arange(concatenated_predictions.shape[-1])+1, 100*np.array(accs_avg), linestyle='--', color=cp[1], label='Based on average probability up to this step')
                            ax.plot(np.arange(concatenated_predictions.shape[-1])+1, 100*np.array(accs_certain), linestyle=':', color=cp[2], label='Most certain')
                            ax.plot(np.arange(concatenated_predictions.shape[-1])+1, 100*np.array(accs_avg_logits), linestyle='-.', color=cp[3], label='Average logits')
                            ax.plot(np.arange(concatenated_predictions.shape[-1])+1, 100*np.array(accs_weighted_logits), linestyle='--', color=cp[4], label='Logits weighted by certainty')
                            ax.set_xlim([0, concatenated_predictions.shape[-1]+1])
                            ax.set_ylim([75, 92])
                            ax.set_xlabel('Internal ticks')
                            ax.set_ylabel(f'Top-k={topk} accuracy')
                            ax.legend(loc='lower right')
                            fig.tight_layout(pad=0.1)
                            fig.savefig(f'{args.output_dir}/accuracy_types_{topk}.png', dpi=200)
                            fig.savefig(f'{args.output_dir}/accuracy_types_{topk}.pdf', dpi=200)
                            plt.close(fig)
                            print(f'k={topk}. Accuracy most certain at last internal tick={100*np.array(accs_certain)[-1]:0.4f}')  # Using certainty based approach


                        indices_over_80 = []
                        classes_80 = {}
                        corrects_80 = {}

                        topk = 5
                        concatenated_predictions_argsorted_topk = concatenated_predictions_argsorted[:,:topk]
                        for certainty_threshold in [0.5, 0.8, 0.9]:
                            # certainty_threshold = 0.6
                            percentage_corrects = []
                            percentage_incorrects = []
                            with tqdm(total=(concatenated_predictions.shape[-1]), initial=0, leave=False, position=1, dynamic_ncols=True) as pbarinner:
                                pbarinner.set_description(f'Certainty threshold={certainty_threshold}')
                                for stepi in np.arange(concatenated_predictions.shape[-1]):
                                    certainty_here = concatenated_certainties[:,1,stepi]
                                    certainty_mask = certainty_here>=certainty_threshold
                                    predictions_here = concatenated_predictions_argsorted_topk[:,:,stepi]
                                    is_correct_here = np.any(predictions_here==concatenated_targets[...,np.newaxis], axis=-1)
                                    percentage_corrects.append(is_correct_here[certainty_mask].sum()/predictions_here.shape[0])
                                    percentage_incorrects.append((~is_correct_here)[certainty_mask].sum()/predictions_here.shape[0])

                                    if certainty_threshold==0.8:
                                        indices_certain = np.where(certainty_mask)[0]
                                        for index in indices_certain:
                                            if index not in indices_over_80:
                                                indices_over_80.append(index)
                                                if concatenated_targets[index] not in classes_80:
                                                    classes_80[concatenated_targets[index]] = [stepi]
                                                    corrects_80[concatenated_targets[index]] = [is_correct_here[index]]
                                                else:
                                                    classes_80[concatenated_targets[index]] = classes_80[concatenated_targets[index]]+[stepi]
                                                    corrects_80[concatenated_targets[index]] = corrects_80[concatenated_targets[index]]+[is_correct_here[index]]


                                    pbarinner.update(1)
                            fig = plt.figure(figsize=(6.5*figscale, 4*figscale))
                            ax = fig.add_subplot(111)
                            ax.bar(np.arange(concatenated_predictions.shape[-1])+1, 
                                percentage_corrects, 
                                color='forestgreen', 
                                hatch='OO', 
                                width=0.9, 
                                label='Positive', 
                                alpha=0.9,
                                linewidth=1.0*figscale)
                            
                            ax.bar(np.arange(concatenated_predictions.shape[-1])+1, 
                                percentage_incorrects, 
                                bottom=percentage_corrects,
                                color='crimson', 
                                hatch='xx', 
                                width=0.9, 
                                label='Negative', 
                                alpha=0.9,
                                linewidth=1.0*figscale)
                            ax.set_xlim(-1, concatenated_predictions.shape[-1]+1)
                            ax.set_xlabel('Internal tick')
                            ax.set_ylabel('% of data')
                            ax.legend(loc='lower right')


                            fig.tight_layout(pad=0.1)
                            fig.savefig(f'{args.output_dir}/steps_versus_correct_{certainty_threshold}.png', dpi=200)
                            fig.savefig(f'{args.output_dir}/steps_versus_correct_{certainty_threshold}.pdf', dpi=200)
                            plt.close(fig)
                            

                        class_list = list(classes_80.keys())
                        mean_steps = [np.mean(classes_80[cls]) for cls in class_list]
                        std_steps = [np.std(classes_80[cls]) for cls in class_list]

                        
                        # Following code plots the class distribution over internal ticks
                        indices_to_show = np.arange(1000)

                        colours = cmap_diverse = plt.get_cmap('rainbow')(np.linspace(0, 1, 1000))
                        # np.random.shuffle(colours)
                        bottom = np.zeros(concatenated_predictions.shape[-1])

                        fig = plt.figure(figsize=(7*figscale, 4*figscale))
                        ax = fig.add_subplot(111)
                        for iii, idx in enumerate(indices_to_show):
                            if idx in classes_80:
                                steps = classes_80[idx]
                                colour = colours[iii]
                                vs, cts = np.unique(steps, return_counts=True)

                                bar = np.zeros(concatenated_predictions.shape[-1])
                                bar[vs] = cts 
                                ax.bar(np.arange(concatenated_predictions.shape[-1])+1, bar, bottom=bottom, color=colour, width=1, edgecolor='none')
                                bottom += bar 
                        ax.set_xlabel('Internal ticks')
                        ax.set_ylabel('Counts over 0.8 certainty')
                        fig.tight_layout(pad=0.1)
                        fig.savefig(f'{args.output_dir}/class_counts.png', dpi=200)
                        fig.savefig(f'{args.output_dir}/class_counts.pdf', dpi=200)
                        plt.close(fig)




                        
                        # The following code plots calibration
                        probability_space = np.linspace(0, 1, 10)
                        fig = plt.figure(figsize=(6*figscale, 4*figscale))
                        ax = fig.add_subplot(111)

                        
                        color_linspace = np.linspace(0, 1, concatenated_predictions.shape[-1])
                        with tqdm(total=(concatenated_predictions.shape[-1]), initial=0, leave=False, position=1, dynamic_ncols=True) as pbarinner:
                            pbarinner.set_description(f'Calibration')
                            for stepi in np.arange(concatenated_predictions.shape[-1]):
                                color = cmap_calib(color_linspace[stepi])
                                pred = concatenated_predictions[:,:,stepi].argmax(1)
                                is_correct = pred == concatenated_targets  # BxT
                                probabilities = softmax(concatenated_predictions[:,:,:stepi+1], axis=1)[np.arange(concatenated_predictions.shape[0]),pred].mean(-1)#softmax(concatenated_predictions[:,:,stepi], axis=1).max(1)
                                probability_space = np.linspace(0, 1, 10)
                                accuracies_per_bin = []
                                bin_centers = []
                                for pi in range(len(probability_space)-1):
                                    bin_low = probability_space[pi]
                                    bin_high = probability_space[pi+1]
                                    mask = ((probabilities >=bin_low) & (probabilities < bin_high)) if pi !=len(probability_space)-2 else ((probabilities >=bin_low) & (probabilities <= bin_high))
                                    accuracies_per_bin.append(is_correct[mask].mean())
                                    bin_centers.append(probabilities[mask].mean())

                                
                                if stepi==concatenated_predictions.shape[-1]-1: 
                                    ax.plot(bin_centers, accuracies_per_bin, linestyle='-', marker='.', color='#4050f7', alpha=1, label='After all ticks')
                                else: ax.plot(bin_centers, accuracies_per_bin, linestyle='-', marker='.', color=color, alpha=0.65)
                                pbarinner.update(1)
                        ax.plot(probability_space, np.linspace(0, 1, len(probability_space)), 'k--')

                        ax.legend(loc='upper left')
                        ax.set_xlim([-0.01, 1.01])
                        ax.set_ylim([-0.01, 1.01])

                        sm = plt.cm.ScalarMappable(cmap=cmap_calib, norm=plt.Normalize(vmin=0, vmax=concatenated_predictions.shape[-1] - 1))
                        sm.set_array([])  # Empty array for colormap
                        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
                        cbar.set_label('Internal ticks')

                        ax.set_xlabel('Mean predicted probabilities')
                        ax.set_ylabel('Ratio of positives')
                        fig.tight_layout(pad=0.1)
                        fig.savefig(f'{args.output_dir}/imagenet_calibration.png', dpi=200)
                        fig.savefig(f'{args.output_dir}/imagenet_calibration.pdf', dpi=200)
                        plt.close(fig)
    if 'videos' in args.actions:
        if not args.data_indices: # If list is empty
            n_samples = len(validation_dataset)
            num_to_sample = min(args.N_to_viz, n_samples)
            replace = n_samples < num_to_sample
            data_indices = np.random.choice(np.arange(n_samples), size=num_to_sample, replace=replace)
            print(f"Selected random indices: {data_indices}")
        else:
            data_indices = args.data_indices
            print(f"Using specified indices: {data_indices}")


        for di in data_indices:
            print(f'\nBuilding viz for dataset index {di}.')

            # --- Get Data & Run Inference ---
            # inputs_norm is already normalized by the transform
            inputs, ground_truth_target = validation_dataset.__getitem__(int(di))

            # Add batch dimension and send to device
            inputs = inputs.to(device).unsqueeze(0)

            # Run model inference
            predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking = model(inputs, track=True)
            # predictions: (B, Classes, Steps), attention_tracking: (Steps*B*Heads, SeqLen)
            n_steps = predictions.size(-1)

            # --- Reshape Attention ---
            # Infer feature map size from model internals (assuming B=1)
            h_feat, w_feat = model.kv_features.shape[-2:]
            
            n_heads = attention_tracking.shape[2] 
            # Reshape to (Steps, Heads, H_feat, W_feat) assuming B=1
            attention_tracking = attention_tracking.reshape(n_steps, n_heads, h_feat, w_feat)

            # --- Setup for Plotting ---
            step_linspace = np.linspace(0, 1, n_steps) # For step colors
            # Define color maps
            cmap_spectral = sns.color_palette("Spectral", as_cmap=True)
            cmap_attention = sns.color_palette('viridis', as_cmap=True)

            # Create output directory for this index
            index_output_dir = os.path.join(args.output_dir, str(di))
            os.makedirs(index_output_dir, exist_ok=True)

            frames = [] # Store frames for GIF
            head_routes = {h: [] for h in range(n_heads)} # Store (y,x) path points per head
            head_routes[-1] = []
            route_colours_step = [] # Store colors for each step's path segments

            # --- Loop Through Each Step ---
            for step_i in range(n_steps):

                # --- Prepare Image for Display ---
                # Denormalize the input tensor for visualization
                data_img_tensor = inputs[0].cpu() # Get first item in batch, move to CPU
                mean_tensor = torch.tensor(dataset_mean).view(3, 1, 1)
                std_tensor = torch.tensor(dataset_std).view(3, 1, 1)
                data_img_denorm = data_img_tensor * std_tensor + mean_tensor
                # Permute to (H, W, C) and convert to numpy, clip to [0, 1]
                data_img_np = data_img_denorm.permute(1, 2, 0).detach().numpy()
                data_img_np = np.clip(data_img_np, 0, 1)
                img_h, img_w = data_img_np.shape[:2]

                # --- Process Attention & Certainty ---
                # Average attention over last few steps (from original code)
                start_step = max(0, step_i - 5)
                attention_now = attention_tracking[start_step : step_i + 1].mean(0) # Avg over steps -> (Heads, H_feat, W_feat)
                # Get certainties up to current step
                certainties_now = certainties[0, 1, :step_i+1].detach().cpu().numpy() # Assuming index 1 holds relevant certainty

                # --- Calculate Attention Paths (using bilinear interp) ---
                # Interpolate attention to image size using bilinear for center finding
                attention_interp_bilinear = F.interpolate(
                    torch.from_numpy(attention_now).unsqueeze(0).float(), # Add batch dim, ensure float
                    size=(img_h, img_w),
                    mode=interp_mode,
                    # align_corners=False
                ).squeeze(0) # Remove batch dim -> (Heads, H, W)

                # Normalize each head's map to [0, 1]
                # Deal with mean
                attn_mean = attention_interp_bilinear.mean(0)
                attn_mean_min = attn_mean.min()
                attn_mean_max = attn_mean.max()
                attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)
                centers, areas = find_island_centers(attn_mean.detach().cpu().numpy(), threshold=0.7)

                if centers: # If islands found
                    largest_island_idx = np.argmax(areas)
                    current_center = centers[largest_island_idx] # (y, x)
                    head_routes[-1].append(current_center)
                elif head_routes[-1]: # If no center now, repeat last known center if history exists
                    head_routes[-1].append(head_routes[-1][-1])


                attn_min = attention_interp_bilinear.view(n_heads, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                attn_max = attention_interp_bilinear.view(n_heads, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                attention_interp_bilinear = (attention_interp_bilinear - attn_min) / (attn_max - attn_min + 1e-6)

                # Store step color
                current_colour = list(cmap_spectral(step_linspace[step_i]))
                route_colours_step.append(current_colour)

                # Find island center for each head
                for head_i in range(n_heads):
                    attn_head_np = attention_interp_bilinear[head_i].detach().cpu().numpy()
                    # Keep threshold=0.7 based on original call
                    centers, areas = find_island_centers(attn_head_np, threshold=0.7)

                    if centers: # If islands found
                        largest_island_idx = np.argmax(areas)
                        current_center = centers[largest_island_idx] # (y, x)
                        head_routes[head_i].append(current_center)
                    elif head_routes[head_i]: # If no center now, repeat last known center if history exists
                            head_routes[head_i].append(head_routes[head_i][-1])
                
                        

                # --- Plotting Setup ---
                mosaic = [['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                            ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                            ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                            ['head_mean', 'head_mean', 'head_mean', 'head_mean', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay', 'head_mean_overlay'],
                        ['head_0', 'head_0_overlay', 'head_1', 'head_1_overlay', 'head_2', 'head_2_overlay', 'head_3', 'head_3_overlay'],
                        ['head_4', 'head_4_overlay', 'head_5', 'head_5_overlay','head_6', 'head_6_overlay', 'head_7', 'head_7_overlay'],
                        ['head_8', 'head_8_overlay', 'head_9', 'head_9_overlay','head_10', 'head_10_overlay', 'head_11', 'head_11_overlay'],
                        ['head_12', 'head_12_overlay', 'head_13', 'head_13_overlay','head_14', 'head_14_overlay', 'head_15', 'head_15_overlay'],
                        ['probabilities', 'probabilities','probabilities', 'probabilities', 'certainty', 'certainty', 'certainty', 'certainty'],
                        ]

                img_aspect = data_img_np.shape[0] / data_img_np.shape[1]
                aspect_ratio = (8 * figscale, 9 * figscale * img_aspect) # W, H
                fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)

                for ax in axes.values():
                    ax.axis('off')

                # --- Plot Certainty ---
                ax_cert = axes['certainty']
                ax_cert.plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=figscale*1)
                # Add background color based on prediction correctness at each step
                for ii in range(len(certainties_now)):
                    is_correct = predictions[0, :, ii].argmax(-1).item() == ground_truth_target # .item() for scalar tensor
                    facecolor = 'limegreen' if is_correct else 'orchid'
                    ax_cert.axvspan(ii, ii + 1, facecolor=facecolor, edgecolor=None, lw=0, alpha=0.3)
                # Mark the last point
                ax_cert.plot(len(certainties_now)-1, certainties_now[-1], 'k.', markersize=figscale*4)
                ax_cert.axis('off')
                ax_cert.set_ylim([0.05, 1.05])
                ax_cert.set_xlim([0, n_steps]) # Use n_steps for consistent x-axis limit

                # --- Plot Probabilities ---
                ax_prob = axes['probabilities']
                # Get probabilities for the current step
                ps = torch.softmax(predictions[0, :, step_i], -1).detach().cpu()
                k = 15 # Top k predictions
                topk_probs, topk_indices = torch.topk(ps, k, dim=0, largest=True)
                topk_indices = topk_indices.numpy()
                topk_probs = topk_probs.numpy()

                top_classes = np.array(class_labels)[topk_indices]
                true_class_idx = ground_truth_target # Ground truth index

                # Determine bar colors (green if correct, blue otherwise - consistent with original)
                colours = ['g' if idx == true_class_idx else 'b' for idx in topk_indices]

                # Plot horizontal bars (inverted range for top-down display)
                ax_prob.barh(np.arange(k)[::-1], topk_probs, color=colours, alpha=1) # Use barh and inverted range
                ax_prob.set_xlim([0, 1])
                ax_prob.axis('off')

                # Add text labels for top classes
                for i, name_idx in enumerate(topk_indices):
                    name = class_labels[name_idx] # Get name from index
                    is_correct = name_idx == true_class_idx
                    fg_color = 'darkgreen' if is_correct else 'crimson' # Text colors from original
                    text_str = f'{name[:40]}' # Truncate long names
                    # Position text on the left side of the horizontal bars
                    ax_prob.text(
                        0.01, # Small offset from left edge
                        k - 1 - i, # Y-position corresponding to the bar
                        text_str,
                        #transform=ax_prob.transAxes, # Use data coordinates for Y
                        verticalalignment='center',
                        horizontalalignment='left',
                        fontsize=8,
                        color=fg_color,
                        alpha=0.9, # Slightly more visible than 0.5
                        path_effects=[
                            patheffects.Stroke(linewidth=2, foreground='white'), # Adjusted stroke
                            patheffects.Normal()
                        ])


                # --- Plot Attention Heads & Overlays (using nearest interp) ---
                # Re-interpolate attention using nearest neighbor for visual plotting
                attention_interp_plot = F.interpolate(
                    torch.from_numpy(attention_now).unsqueeze(0).float(),
                    size=(img_h, img_w),
                    mode=interp_mode, # 'nearest'
                ).squeeze(0)

                attn_mean = attention_interp_plot.mean(0)
                attn_mean_min = attn_mean.min()
                attn_mean_max = attn_mean.max()
                attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)


                # Normalize each head's map to [0, 1]
                attn_min_plot = attention_interp_plot.view(n_heads, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                attn_max_plot = attention_interp_plot.view(n_heads, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                attention_interp_plot = (attention_interp_plot - attn_min_plot) / (attn_max_plot - attn_min_plot + 1e-6)
                attention_interp_plot_np = attention_interp_plot.detach().cpu().numpy()
                


                


                for head_i in list(range(n_heads)) + [-1]:
                    axname = f'head_{head_i}' if head_i != -1 else 'head_mean'
                    if axname not in axes: continue # Skip if mosaic doesn't have this head

                    ax = axes[axname]
                    ax_overlay = axes[f'{axname}_overlay']

                    # Plot attention heatmap
                    this_attn = attention_interp_plot_np[head_i] if head_i != -1 else attn_mean
                    img_to_plot = cmap_attention(this_attn)
                    ax.imshow(img_to_plot)
                    ax.axis('off')

                    # Plot overlay: image + paths
                    these_route_steps = head_routes[head_i]
                    arrow_scale = 1.5 if head_i != -1 else 3

                    if these_route_steps: # Only plot if path exists
                        # Separate y and x coordinates
                        y_coords, x_coords = zip(*these_route_steps)
                        y_coords = np.array(y_coords)
                        x_coords = np.array(x_coords)

                        # Flip y-coordinates for correct plotting (imshow origin is top-left)
                        # NOTE: Original flip seemed complex, simplifying to standard flip
                        y_coords_flipped = img_h - 1 - y_coords

                        # Show original image flipped vertically to match coordinate system
                        ax_overlay.imshow(np.flipud(data_img_np), origin='lower')

                        # Draw arrows for path segments
                            # Arrow size scaling from original
                        for i in range(len(these_route_steps) - 1):
                            dx = x_coords[i+1] - x_coords[i]
                            dy = y_coords_flipped[i+1] - y_coords_flipped[i] # Use flipped y for delta

                            # Draw white background arrow (thicker)
                            ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                                linewidth=1.6 * arrow_scale * 1.3,
                                                head_width=1.9 * arrow_scale * 1.3,
                                                head_length=1.4 * arrow_scale * 1.45,
                                                fc='white', ec='white', length_includes_head=True, alpha=1)
                            # Draw colored foreground arrow
                            ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                                linewidth=1.6 * arrow_scale,
                                                head_width=1.9 * arrow_scale,
                                                head_length=1.4 * arrow_scale,
                                                fc=route_colours_step[i], ec=route_colours_step[i], # Use step color
                                                length_includes_head=True)

                    else: # If no path yet, just show the image
                            ax_overlay.imshow(np.flipud(data_img_np), origin='lower')


                    # Set limits and turn off axes for overlay
                    ax_overlay.set_xlim([0, img_w - 1])
                    ax_overlay.set_ylim([0, img_h - 1])
                    ax_overlay.axis('off')
                

                # --- Finalize and Save Frame ---
                fig.tight_layout(pad=0.1) # Adjust spacing

                # Render the plot to a numpy array
                canvas = fig.canvas
                canvas.draw()
                image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3] # Get RGB

                frames.append(image_numpy) # Add to list for GIF

                

                plt.close(fig) # Close figure to free memory

            # --- Save GIF ---
            gif_path = os.path.join(index_output_dir, f'{str(di)}_viz.gif')
            print(f"Saving GIF to {gif_path}...")
            imageio.mimsave(gif_path, frames, fps=15, loop=0) # loop=0 means infinite loop
            save_frames_to_mp4([fm[:,:,::-1] for fm in frames], os.path.join(index_output_dir, f'{str(di)}_viz.mp4'), fps=15, gop_size=1, preset='veryslow')
    if 'demo' in args.actions:


        
        # --- Select Data Indices ---
        if not args.data_indices: # If list is empty
            n_samples = len(validation_dataset)
            num_to_sample = min(args.N_to_viz, n_samples)
            replace = n_samples < num_to_sample
            data_indices = np.random.choice(np.arange(n_samples), size=num_to_sample, replace=replace)
            print(f"Selected random indices: {data_indices}")
        else:
            data_indices = args.data_indices
            print(f"Using specified indices: {data_indices}")


        for di in data_indices:
            
            index_output_dir = os.path.join(args.output_dir, str(di))
            os.makedirs(index_output_dir, exist_ok=True)

            print(f'\nBuilding viz for dataset index {di}.')

            inputs, ground_truth_target = validation_dataset.__getitem__(int(di))

            # Add batch dimension and send to device
            inputs = inputs.to(device).unsqueeze(0)
            predictions, certainties, synchronisations_over_time, pre_activations, post_activations, attention_tracking = model(inputs, track=True)

            # --- Reshape Attention ---
            # Infer feature map size from model internals (assuming B=1)
            h_feat, w_feat = model.kv_features.shape[-2:]
            n_steps = predictions.size(-1)
            n_heads = attention_tracking.shape[2] 
            # Reshape to (Steps, Heads, H_feat, W_feat) assuming B=1
            attention_tracking = attention_tracking.reshape(n_steps, n_heads, h_feat, w_feat)

            # --- Setup for Plotting ---
            step_linspace = np.linspace(0, 1, n_steps) # For step colors
            # Define color maps
            cmap_steps = sns.color_palette("Spectral", as_cmap=True)
            cmap_attention = sns.color_palette('viridis', as_cmap=True)

            # Create output directory for this index
            
            
            frames = [] # Store frames for GIF
            head_routes = [] # Store (y,x) path points per head
            route_colours_step = [] # Store colors for each step's path segments

            # --- Loop Through Each Step ---
            for step_i in range(n_steps):

                # Store step color
                current_colour = list(cmap_steps(step_linspace[step_i]))
                route_colours_step.append(current_colour)

                # --- Prepare Image for Display ---
                # Denormalize the input tensor for visualization
                data_img_tensor = inputs[0].cpu() # Get first item in batch, move to CPU
                mean_tensor = torch.tensor(dataset_mean).view(3, 1, 1)
                std_tensor = torch.tensor(dataset_std).view(3, 1, 1)
                data_img_denorm = data_img_tensor * std_tensor + mean_tensor
                # Permute to (H, W, C) and convert to numpy, clip to [0, 1]
                data_img_np = data_img_denorm.permute(1, 2, 0).detach().numpy()
                data_img_np = np.clip(data_img_np, 0, 1)
                img_h, img_w = data_img_np.shape[:2]

                # --- Process Attention & Certainty ---
                # Average attention over last few steps (from original code)
                start_step = max(0, step_i - 5)
                attention_now = attention_tracking[start_step : step_i + 1].mean(0) # Avg over steps -> (Heads, H_feat, W_feat)
                # Get certainties up to current step
                certainties_now = certainties[0, 1, :step_i+1].detach().cpu().numpy() # Assuming index 1 holds relevant certainty

                # --- Calculate Attention Paths (using bilinear interp) ---
                # Interpolate attention to image size using bilinear for center finding
                attention_interp_bilinear = F.interpolate(
                    torch.from_numpy(attention_now).unsqueeze(0).float(), # Add batch dim, ensure float
                    size=(img_h, img_w),
                    mode=interp_mode,
                ).squeeze(0) # Remove batch dim -> (Heads, H, W)

                attn_mean = attention_interp_bilinear.mean(0)
                attn_mean_min = attn_mean.min()
                attn_mean_max = attn_mean.max()
                attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)
                centers, areas = find_island_centers(attn_mean.detach().cpu().numpy(), threshold=0.7)

                if centers: # If islands found
                    largest_island_idx = np.argmax(areas)
                    current_center = centers[largest_island_idx] # (y, x)
                    head_routes.append(current_center)
                elif head_routes: # If no center now, repeat last known center if history exists
                    head_routes.append(head_routes[-1])
                    
                # --- Plotting Setup ---
                # if n_heads != 8: print(f"Warning: Plotting layout assumes 8 heads, found {n_heads}. Layout may be incorrect.")
                mosaic = [['head_0', 'head_1', 'head_2', 'head_3', 'head_mean', 'head_mean', 'head_mean', 'head_mean', 'overlay', 'overlay', 'overlay', 'overlay'],
                        ['head_4', 'head_5', 'head_6', 'head_7', 'head_mean', 'head_mean', 'head_mean', 'head_mean', 'overlay', 'overlay', 'overlay', 'overlay'],
                        ['head_8', 'head_9', 'head_10', 'head_11', 'head_mean', 'head_mean', 'head_mean', 'head_mean', 'overlay', 'overlay', 'overlay', 'overlay'],
                        ['head_12', 'head_13', 'head_14', 'head_15', 'head_mean', 'head_mean', 'head_mean', 'head_mean', 'overlay', 'overlay', 'overlay', 'overlay'],
                        ['probabilities', 'probabilities', 'probabilities', 'probabilities', 'probabilities', 'probabilities', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty'],
                        ['probabilities', 'probabilities', 'probabilities', 'probabilities', 'probabilities', 'probabilities', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty'],
                        ]

                img_aspect = data_img_np.shape[0] / data_img_np.shape[1]
                aspect_ratio = (12 * figscale, 6 * figscale * img_aspect) # W, H
                fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)
                for ax in axes.values():
                    ax.axis('off')

                # --- Plot Certainty ---
                ax_cert = axes['certainty']
                ax_cert.plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=figscale*1)
                # Add background color based on prediction correctness at each step
                for ii in range(len(certainties_now)):
                    is_correct = predictions[0, :, ii].argmax(-1).item() == ground_truth_target # .item() for scalar tensor
                    facecolor = 'limegreen' if is_correct else 'orchid'
                    ax_cert.axvspan(ii, ii + 1, facecolor=facecolor, edgecolor=None, lw=0, alpha=0.3)
                # Mark the last point
                ax_cert.plot(len(certainties_now)-1, certainties_now[-1], 'k.', markersize=figscale*4)
                ax_cert.axis('off')
                ax_cert.set_ylim([0.05, 1.05])
                ax_cert.set_xlim([0, n_steps]) # Use n_steps for consistent x-axis limit

                # --- Plot Probabilities ---
                ax_prob = axes['probabilities']
                # Get probabilities for the current step
                ps = torch.softmax(predictions[0, :, step_i], -1).detach().cpu()
                k = 15 # Top k predictions
                topk_probs, topk_indices = torch.topk(ps, k, dim=0, largest=True)
                topk_indices = topk_indices.numpy()
                topk_probs = topk_probs.numpy()

                top_classes = np.array(class_labels)[topk_indices]
                true_class_idx = ground_truth_target # Ground truth index

                # Determine bar colors (green if correct, blue otherwise - consistent with original)
                colours = ['g' if idx == true_class_idx else 'b' for idx in topk_indices]

                # Plot horizontal bars (inverted range for top-down display)
                ax_prob.barh(np.arange(k)[::-1], topk_probs, color=colours, alpha=1) # Use barh and inverted range
                ax_prob.set_xlim([0, 1])
                ax_prob.axis('off')

                # Add text labels for top classes
                for i, name_idx in enumerate(topk_indices):
                    name = class_labels[name_idx] # Get name from index
                    is_correct = name_idx == true_class_idx
                    fg_color = 'darkgreen' if is_correct else 'crimson' # Text colors from original
                    text_str = f'{name[:40]}' # Truncate long names
                    # Position text on the left side of the horizontal bars
                    ax_prob.text(
                        0.01, # Small offset from left edge
                        k - 1 - i, # Y-position corresponding to the bar
                        text_str,
                        #transform=ax_prob.transAxes, # Use data coordinates for Y
                        verticalalignment='center',
                        horizontalalignment='left',
                        fontsize=8,
                        color=fg_color,
                        alpha=0.7, # Slightly more visible than 0.5
                        path_effects=[
                            patheffects.Stroke(linewidth=2, foreground='white'), # Adjusted stroke
                            patheffects.Normal()
                        ])


                # --- Plot Attention Heads & Overlays (using nearest interp) ---
                # Re-interpolate attention using nearest neighbor for visual plotting
                attention_interp_plot = F.interpolate(
                    torch.from_numpy(attention_now).unsqueeze(0).float(),
                    size=(img_h, img_w),
                    mode=interp_mode # 'nearest'
                ).squeeze(0)


                attn_mean = attention_interp_plot.mean(0)
                attn_mean_min = attn_mean.min()
                attn_mean_max = attn_mean.max()
                attn_mean = (attn_mean - attn_mean_min) / (attn_mean_max - attn_mean_min)


                img_to_plot = cmap_attention(attn_mean)
                axes['head_mean'].imshow(img_to_plot)
                axes['head_mean'].axis('off')


                these_route_steps = head_routes
                ax_overlay = axes['overlay']

                if these_route_steps: # Only plot if path exists
                    # Separate y and x coordinates
                    y_coords, x_coords = zip(*these_route_steps)
                    y_coords = np.array(y_coords)
                    x_coords = np.array(x_coords)

                    # Flip y-coordinates for correct plotting (imshow origin is top-left)
                    # NOTE: Original flip seemed complex, simplifying to standard flip
                    y_coords_flipped = img_h - 1 - y_coords

                    # Show original image flipped vertically to match coordinate system
                    ax_overlay.imshow(np.flipud(data_img_np), origin='lower')

                    # Draw arrows for path segments
                    arrow_scale = 2 # Arrow size scaling from original
                    for i in range(len(these_route_steps) - 1):
                        dx = x_coords[i+1] - x_coords[i]
                        dy = y_coords_flipped[i+1] - y_coords_flipped[i] # Use flipped y for delta

                        # Draw white background arrow (thicker)
                        ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                            linewidth=1.6 * arrow_scale * 1.3,
                                            head_width=1.9 * arrow_scale * 1.3,
                                            head_length=1.4 * arrow_scale * 1.45,
                                            fc='white', ec='white', length_includes_head=True, alpha=1)
                        # Draw colored foreground arrow
                        ax_overlay.arrow(x_coords[i], y_coords_flipped[i], dx, dy,
                                            linewidth=1.6 * arrow_scale,
                                            head_width=1.9 * arrow_scale,
                                            head_length=1.4 * arrow_scale,
                                            fc=route_colours_step[i], ec=route_colours_step[i], # Use step color
                                            length_includes_head=True)
                    # Set limits and turn off axes for overlay
                    ax_overlay.set_xlim([0, img_w - 1])
                    ax_overlay.set_ylim([0, img_h - 1])
                    ax_overlay.axis('off')


                for head_i in range(n_heads):
                    if f'head_{head_i}' not in axes: continue # Skip if mosaic doesn't have this head

                    ax = axes[f'head_{head_i}']

                    # Plot attention heatmap
                    attn_up_to_now = attention_tracking[:step_i + 1, head_i].mean(0)
                    attn_up_to_now = (attn_up_to_now - attn_up_to_now.min())/(attn_up_to_now.max() - attn_up_to_now.min())
                    img_to_plot = cmap_attention(attn_up_to_now)
                    ax.imshow(img_to_plot)
                    ax.axis('off')



                    


                # --- Finalize and Save Frame ---
                fig.tight_layout(pad=0.1) # Adjust spacing

                # Render the plot to a numpy array
                canvas = fig.canvas
                canvas.draw()
                image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3] # Get RGB

                frames.append(image_numpy) # Add to list for GIF

                # Save individual frame if requested
                if step_i==model.iterations-1:
                    fig.savefig(os.path.join(index_output_dir, f'frame_{step_i}.png'), dpi=200)

                plt.close(fig) # Close figure to free memory
            outfilename = os.path.join(index_output_dir, f'{di}_demo.mp4')
            save_frames_to_mp4([fm[:,:,::-1] for fm in frames], outfilename, fps=15, gop_size=1, preset='veryslow')
        
                        