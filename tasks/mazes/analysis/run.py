import torch
import numpy as np
np.seterr(divide='ignore', invalid='warn') # Keep specific numpy error settings
import matplotlib as mpl
mpl.use('Agg') # Use Agg backend for matplotlib (important to set before importing pyplot)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid') # Keep seaborn style
import os
import argparse
import cv2
import imageio # Used for saving GIFs in viz

# Local imports
from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine
from tasks.mazes.plotting import draw_path # 
from tasks.image_classification.plotting import save_frames_to_mp4

def has_solved_checker(x_maze, route, valid_only=True, fault_tolerance=1, exclusions=[]):
    """Checks if a route solves a maze."""
    maze = np.copy(x_maze)
    H, W, _ = maze.shape
    start_coords = np.argwhere((maze == [1, 0, 0]).all(axis=2))
    end_coords = np.argwhere((maze == [0, 1, 0]).all(axis=2))

    if len(start_coords) == 0:
        return False, (-1, -1), 0  # Cannot start

    current_pos = tuple(start_coords[0])
    target_pos = tuple(end_coords[0]) if len(end_coords) > 0 else None

    mistakes_made = 0
    final_pos = current_pos
    path_taken_len = 0

    for step in route:
        if mistakes_made > fault_tolerance:
            break

        next_pos_candidate = list(current_pos) # Use a list for mutable coordinate calculation
        if step == 0: next_pos_candidate[0] -= 1
        elif step == 1: next_pos_candidate[0] += 1
        elif step == 2: next_pos_candidate[1] -= 1
        elif step == 3: next_pos_candidate[1] += 1
        elif step == 4: pass  # Stay in place
        else: continue # Invalid step action
        next_pos = tuple(next_pos_candidate)


        is_invalid_step = False
        # Check bounds first, then maze content if in bounds
        if not (0 <= next_pos[0] < H and 0 <= next_pos[1] < W):
            is_invalid_step = True
        elif np.all(maze[next_pos] == [0, 0, 0]):  # Wall
            is_invalid_step = True

        if is_invalid_step:
            mistakes_made += 1
            if valid_only:
                continue
        
        current_pos = next_pos
        path_taken_len += 1

        if target_pos and current_pos == target_pos:
            if mistakes_made <= fault_tolerance:
                return True, current_pos, path_taken_len

        if mistakes_made <= fault_tolerance:
            # Assuming exclusions is a list of tuples (as populated in the 'gen' action)
            if current_pos not in exclusions:
                final_pos = current_pos

    if target_pos and final_pos == target_pos and mistakes_made <= fault_tolerance: # Added mistakes_made check here
        return True, final_pos, path_taken_len
    return False, final_pos, path_taken_len


def parse_args():
    """Parses command-line arguments for maze analysis."""
    parser = argparse.ArgumentParser(description="Analyze Asynchronous Thought Machine on Maze Tasks")
    parser.add_argument('--actions', type=str, nargs='+', default=['gen'], help="Actions: 'viz', 'gen'")
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help="GPU device index or -1 for CPU")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt', help="Path to CTM checkpoint")
    parser.add_argument('--output_dir', type=str, default='tasks/mazes/analysis/outputs', help="Directory for analysis outputs")
    parser.add_argument('--dataset_for_viz', type=str, default='large', help="Dataset for 'viz' action")
    parser.add_argument('--dataset_for_gen', type=str, default='extralarge', help="Dataset for 'gen' action")
    parser.add_argument('--batch_size_test', type=int, default=32, help="Batch size for loading test data for 'viz'")
    parser.add_argument('--max_reapplications', type=int, default=20, help="When testing generalisation to extra large mazes")
    parser.add_argument('--legacy_scaling', action=argparse.BooleanOptionalAction, default=True, help='Legacy checkpoints scale between 0 and 1, new ones can scale -1 to 1.')
    return parser.parse_args()

def _load_ctm_model(checkpoint_path, device):
    """Loads the ContinuousThoughtMachine model from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']

    # Handle legacy arguments for model_args
    if not hasattr(model_args, 'backbone_type') and hasattr(model_args, 'resnet_type'):
        model_args.backbone_type = f'{model_args.resnet_type}-{getattr(model_args, "resnet_feature_scales", [4])[-1]}'
    
    # Ensure prediction_reshaper is derived correctly
    # Assuming out_dims exists and is used for this
    prediction_reshaper = [model_args.out_dims // 5, 5] if hasattr(model_args, 'out_dims') else None


    if not hasattr(model_args, 'neuron_select_type'):
        model_args.neuron_select_type = 'first-last'
    if not hasattr(model_args, 'n_random_pairing_self'):
        model_args.n_random_pairing_self = 0

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
        deep_nlms=model_args.deep_memory, # Mapping from model_args.deep_memory
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=model_args.do_normalisation, # Mapping from model_args.do_normalisation
        backbone_type=model_args.backbone_type,
        positional_embedding_type=model_args.positional_embedding_type,
        out_dims=model_args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=0, # Explicitly setting dropout to 0 as in original
        neuron_select_type=model_args.neuron_select_type,
        n_random_pairing_self=model_args.n_random_pairing_self,
    ).to(device)

    load_result = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"Loaded state_dict. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
    model.eval()
    return model

# --- Main Execution Block ---
if __name__=='__main__':
    args = parse_args()

    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    palette = sns.color_palette("husl", 8)
    cmap = plt.get_cmap('gist_rainbow')

    # --- Generalisation Action ('gen') ---
    if 'gen' in args.actions:
        model = _load_ctm_model(args.checkpoint, device)
        
        print(f"\n--- Running Generalisation Analysis ('gen'): {args.dataset_for_gen} ---")
        target_dataset_name = f'{args.dataset_for_gen}'
        data_root = f'data/mazes/{target_dataset_name}/test'
        max_target_route_len = 50 # Specific to 'gen' action
        
        test_data = MazeImageFolder(
            root=data_root, which_set='test', 
            maze_route_length=max_target_route_len, 
            expand_range=not args.legacy_scaling, # Legacy checkpoints need a [0, 1] range, but it might be better to default to [-1, 1] in the future
            trunc=True
        )
        # Load a single large batch for 'gen'
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=min(len(test_data), 2000), 
            shuffle=False, num_workers=1
        )
        inputs, targets = next(iter(testloader))

        actual_lengths = (targets != 4).sum(dim=-1)
        sorted_indices = torch.argsort(actual_lengths, descending=True)
        inputs, targets, actual_lengths = inputs[sorted_indices], targets[sorted_indices], actual_lengths[sorted_indices]

        test_how_many = min(1000, len(inputs))
        print(f"Processing {test_how_many} mazes sorted by length...")

        results = {}
        fault_tolerance = 2 # Specific to 'gen' analysis
        output_gen_dir = os.path.join(args.output_dir, 'gen', args.dataset_for_gen)
        os.makedirs(output_gen_dir, exist_ok=True)

        for n_tested in range(test_how_many):
            maze_actual_length = actual_lengths[n_tested].item()
            maze_idx_display = n_tested + 1 
            print(f"Testing maze {maze_idx_display}/{test_how_many} (Len: {maze_actual_length})...")

            initial_input_maze = inputs[n_tested:n_tested+1].clone().to(device)
            maze_output_dir = os.path.join(output_gen_dir, f"maze_{maze_idx_display}")
            
            re_applications = 0
            has_solved = False
            current_input_maze = initial_input_maze
            exclusions = []
            long_frames = []
            ongoing_solution_img = None

            while not has_solved and re_applications < args.max_reapplications:
                re_applications += 1
                with torch.no_grad():
                     predictions, certainties, _, _, _, attention_tracking = model(current_input_maze, track=True)

                h_feat, w_feat = model.kv_features.shape[-2:]
                attention_tracking = attention_tracking.reshape(attention_tracking.shape[0], -1, h_feat, w_feat) 

                n_steps_viz = predictions.shape[-1] # Use a different name to avoid conflict if n_steps is used elsewhere
                step_linspace = np.linspace(0, 1, n_steps_viz)
                current_maze_np = current_input_maze[0].permute(1,2,0).detach().cpu().numpy()

                for stepi in range(n_steps_viz):
                    pred_route = predictions[0, :, stepi].reshape(-1, 5).argmax(-1).detach().cpu().numpy()
                    frame = draw_path(current_maze_np, pred_route)
                    if attention_tracking is not None and stepi < attention_tracking.shape[0]:
                        try: 
                            attn = attention_tracking[stepi].mean(0) 
                            attn_resized = cv2.resize(attn, (current_maze_np.shape[1], current_maze_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                            if attn_resized.max() > attn_resized.min():
                                attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
                                attn_norm[attn_norm < np.percentile(attn_norm, 80)] = 0.0
                                frame = np.clip((np.copy(frame)*(1-attn_norm[:,:,np.newaxis])*1 + (attn_norm[:,:,np.newaxis]*0.8 * np.reshape(np.array(cmap(step_linspace[stepi]))[:3], (1, 1, 3)))), 0, 1)
                        except Exception: # Keep broad except for visualization robustness
                            pass 
                    frame_resized = cv2.resize(frame, (int(current_maze_np.shape[1]*4), int(current_maze_np.shape[0]*4)), interpolation=cv2.INTER_NEAREST) # Corrected shape[1]*4 for height
                    long_frames.append((np.clip(frame_resized, 0, 1) * 255).astype(np.uint8))
                
                where_most_certain = certainties[0, 1].argmax().item()
                chosen_pred_route = predictions[0, :, where_most_certain].reshape(-1, 5).argmax(-1).detach().cpu().numpy()
                current_start_loc_list = np.argwhere((current_maze_np == [1, 0, 0]).all(axis=2)).tolist()
                
                # Ensure current_start_loc_list is not empty before trying to access its elements
                if not current_start_loc_list:
                    print(f"Warning: Could not find start location in maze {maze_idx_display} during reapplication {re_applications}. Stopping reapplication.")
                    break # Cannot proceed without a start location

                solved_now, final_pos, _ = has_solved_checker(current_maze_np, chosen_pred_route, True, fault_tolerance, exclusions)

                path_img = draw_path(current_maze_np, chosen_pred_route, cmap=cmap, valid_only=True) 
                if ongoing_solution_img is None: 
                    ongoing_solution_img = path_img
                else:
                    mask = (np.any(ongoing_solution_img!=path_img, -1))&(~np.all(path_img==[1,1,1], -1))&(~np.all(ongoing_solution_img==[1,0,0], -1))
                    ongoing_solution_img[mask] = path_img[mask]

                if solved_now: 
                    has_solved = True
                    break

                if tuple(current_start_loc_list[0]) == final_pos: 
                    exclusions.append(tuple(current_start_loc_list[0]))
                
                next_input = current_input_maze.clone()
                old_start_idx = tuple(current_start_loc_list[0])
                next_input[0, :, old_start_idx[0], old_start_idx[1]] = 1.0 # Reset old start to path
                
                if 0 <= final_pos[0] < next_input.shape[2] and 0 <= final_pos[1] < next_input.shape[3]:
                    next_input[0, :, final_pos[0], final_pos[1]] = torch.tensor([1,0,0], device=device, dtype=next_input.dtype) # New start
                else:
                    print(f"Warning: final_pos {final_pos} out of bounds for maze {maze_idx_display}. Stopping reapplication.")
                    break 
                current_input_maze = next_input

            if has_solved:
                print(f'Solved maze of length {maze_actual_length}! Saving...')
                os.makedirs(maze_output_dir, exist_ok=True)
                if ongoing_solution_img is not None: 
                    cv2.imwrite(os.path.join(maze_output_dir, 'ongoing_solution.png'), (ongoing_solution_img * 255).astype(np.uint8)[:,:,::-1])
                if long_frames: 
                    save_frames_to_mp4([fm[:,:,::-1] for fm in long_frames], os.path.join(maze_output_dir, f'combined_process.mp4'), fps=45, gop_size=10, preset='veryslow', crf=20)
            else:
                print(f'Failed maze of length {maze_actual_length} after {re_applications} reapplications. Not saving visuals for this maze.')

            if maze_actual_length not in results: results[maze_actual_length] = []
            results[maze_actual_length].append((has_solved, re_applications))

            fig_success, ax_success = plt.subplots()
            fig_reapp, ax_reapp = plt.subplots()
            sorted_lengths = sorted(results.keys())
            if sorted_lengths: 
                success_rates = [np.mean([r[0] for r in results[l]]) * 100 for l in sorted_lengths]
                reapps_mean = [np.mean([r[1] for r in results[l] if r[0]]) if any(r[0] for r in results[l]) else np.nan for l in sorted_lengths]
                ax_success.plot(sorted_lengths, success_rates, linestyle='-', color=palette[0])
                ax_reapp.plot(sorted_lengths, reapps_mean, linestyle='-', color=palette[5])
            ax_success.set_xlabel('Route Length'); ax_success.set_ylabel('Success (%)')
            ax_reapp.set_xlabel('Route Length'); ax_reapp.set_ylabel('Re-applications (Avg on Success)')
            fig_success.tight_layout(pad=0.1); fig_reapp.tight_layout(pad=0.1)
            fig_success.savefig(os.path.join(output_gen_dir, f'{args.dataset_for_gen}-success_rate.png'), dpi=200)
            fig_success.savefig(os.path.join(output_gen_dir, f'{args.dataset_for_gen}-success_rate.pdf'), dpi=200)
            fig_reapp.savefig(os.path.join(output_gen_dir, f'{args.dataset_for_gen}-re-applications.png'), dpi=200)
            fig_reapp.savefig(os.path.join(output_gen_dir, f'{args.dataset_for_gen}-re-applications.pdf'), dpi=200)
            plt.close(fig_success); plt.close(fig_reapp)
            np.savez(os.path.join(output_gen_dir, f'{args.dataset_for_gen}_results.npz'), results=results)

        print("\n--- Generalisation Analysis ('gen') Complete ---")

    # --- Visualization Action ('viz') ---
    if 'viz' in args.actions:
        model = _load_ctm_model(args.checkpoint, device)

        print(f"\n--- Running Visualization ('viz'): {args.dataset_for_viz} ---")
        output_viz_dir = os.path.join(args.output_dir, 'viz')
        os.makedirs(output_viz_dir, exist_ok=True)

        target_dataset_name = f'{args.dataset_for_viz}'
        data_root = f'data/mazes/{target_dataset_name}/test'
        test_data = MazeImageFolder(
            root=data_root, which_set='test', 
            maze_route_length=100, # Max route length for viz data
            expand_range=not args.legacy_scaling, #  # Legacy checkpoints need a [0, 1] range, but it might be better to default to [-1, 1] in the future
            trunc=True
        )
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size_test, 
            shuffle=False, num_workers=1
        )

        all_inputs, all_targets, all_lengths = [], [], []
        for b_in, b_tgt in testloader:
            all_inputs.append(b_in)
            all_targets.append(b_tgt)
            all_lengths.append((b_tgt != 4).sum(dim=-1))
        
        if not all_inputs: 
            print("Error: No data in visualization loader. Exiting 'viz' action.")
            exit()
            
        all_inputs, all_targets, all_lengths = torch.cat(all_inputs), torch.cat(all_targets), torch.cat(all_lengths)
        
        num_viz_mazes = 10
        num_viz_mazes = min(num_viz_mazes, len(all_lengths))

        if num_viz_mazes == 0: 
            print("Error: No mazes found to visualize. Exiting 'viz' action.")
            exit()
            
        top_indices = torch.argsort(all_lengths, descending=True)[:num_viz_mazes]
        inputs_viz, targets_viz = all_inputs[top_indices].to(device), all_targets[top_indices]

        print(f"Visualizing {len(inputs_viz)} longest mazes...")

        with torch.no_grad():
            predictions, _, _, _, _, attention_tracking = model(inputs_viz, track=True)
        
        # Reshape attention: (Steps, Batch, Heads, H_feat, W_feat) assuming model.kv_features has H_feat, W_feat
        # The original reshape was slightly different, this tries to match the likely intended dimensions for per-step, per-batch item attention
        if attention_tracking is not None and hasattr(model, 'kv_features') and model.kv_features is not None:
             attention_tracking = attention_tracking.reshape(
                 attention_tracking.shape[0], # Iterations/Steps
                 inputs_viz.size(0), # Batch size (num_viz_mazes)
                 -1, # Heads (inferred)
                 model.kv_features.shape[-2], # H_feat
                 model.kv_features.shape[-1]  # W_feat
            )
        else:
            attention_tracking = None # Ensure it's None if it can't be reshaped
            print("Warning: Could not reshape attention_tracking. Visualizations may not include attention overlays.")


        for maze_i in range(inputs_viz.size(0)):
            maze_idx_display = maze_i + 1
            maze_output_dir = os.path.join(output_viz_dir, f"maze_{maze_idx_display}")
            os.makedirs(maze_output_dir, exist_ok=True)
            
            current_input_np_original = inputs_viz[maze_i].permute(1,2,0).detach().cpu().numpy()
            # Apply scaling for visualization based on legacy_scaling: Legacy checkpoints need a [0, 1] range, but it might be better to default to [-1, 1] in the future
            current_input_np_display = (current_input_np_original + 1) / 2 if not args.legacy_scaling else current_input_np_original

            current_target_route = targets_viz[maze_i].detach().cpu().numpy()
            print(f"Generating viz for maze {maze_idx_display}...")

            try:
                 solution_maze_img = draw_path(current_input_np_display, current_target_route, gt=True)
                 cv2.imwrite(os.path.join(maze_output_dir, 'solution_ground_truth.png'), (solution_maze_img * 255).astype(np.uint8)[:,:,::-1])
            except Exception: # Keep broad except for visualization robustness
                 print(f"Could not save ground truth solution for maze {maze_idx_display}")
                 pass

            frames = []
            n_steps_viz = predictions.shape[-1] # Use a different name
            step_linspace = np.linspace(0, 1, n_steps_viz)

            for stepi in range(n_steps_viz):
                pred_route = predictions[maze_i, :, stepi].reshape(-1, 5).argmax(-1).detach().cpu().numpy()
                frame = draw_path(current_input_np_display, pred_route)
                
                if attention_tracking is not None and stepi < attention_tracking.shape[0] and maze_i < attention_tracking.shape[1]:
                     
                    # Attention for current step (stepi) and current maze in batch (maze_i), average over heads
                    attn = attention_tracking[stepi, maze_i].mean(0)
                    attn_resized = cv2.resize(attn, (current_input_np_display.shape[1], current_input_np_display.shape[0]), interpolation=cv2.INTER_LINEAR)
                    if attn_resized.max() > attn_resized.min():
                            attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
                            attn_norm[attn_norm < np.percentile(attn_norm, 80)] = 0.0
                            frame = np.clip((np.copy(frame)*(1-attn_norm[:,:,np.newaxis])*0.9 + (attn_norm[:,:,np.newaxis]*1.2 * np.reshape(np.array(cmap(step_linspace[stepi]))[:3], (1, 1, 3)))), 0, 1)
                     

                frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                frames.append((np.clip(frame_resized, 0, 1) * 255).astype(np.uint8))

            if frames: 
                imageio.mimsave(os.path.join(maze_output_dir, 'attention_overlay.gif'), frames, fps=15, loop=0)
        
        print("\n--- Visualization Action ('viz') Complete ---")