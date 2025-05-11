import re
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from scipy.interpolate import interp1d
from collections import defaultdict
import csv
import multiprocessing

from tasks.rl.train import Agent, make_env_classic_control, make_env_minigrid, load_model
from utils.housekeeping import set_seed
from tasks.rl.utils import combine_tracking_data
from tasks.rl.plotting import make_rl_gif
from tasks.image_classification.plotting import plot_neural_dynamics

import seaborn as sns
sns.set_palette("hls")
sns.set_style('darkgrid')


def parse_args():
    parser = argparse.ArgumentParser(description='RL Analysis')
    parser.add_argument('--log_dir', type=str, default='checkpoints/rl', help='Directory to save logs.')
    parser.add_argument('--scale', type=float, default=0.5, help='Scaling factor for plot size')
    parser.add_argument('--num_eval_envs', type=int, default=10, help='Number of evaluation environments') 
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use') 
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')

    return parser.parse_args()

def get_checkpoint_paths_for_environment(environment, log_dir):
    checkpoint_files = []
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory '{log_dir}' does not exist.")
    for root, dirs, files in os.walk(log_dir):
        if environment in root:
            for file in files:
                if file == "checkpoint.pt":
                    checkpoint_files.append(os.path.join(root, file))
    return checkpoint_files

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint_path}.")
    return checkpoint

def get_global_steps_from_checkpoint(checkpoint):
    return checkpoint.get('global_steps_tracking', [])

def get_episode_rewards_from_checkpoint(checkpoint):
    return checkpoint.get('episode_rewards_tracking', [])

def get_episode_lengths_from_checkpoint(checkpoint):
    return checkpoint.get('episode_lengths_tracking', [])

def get_human_readable_name(checkpoint_path):
    name_map = {
        "lstm_1": "LSTM, 1 Iters.",
        "ctm_1": "CTM, 1 Iters.",
        "lstm_2": "LSTM, 2 Iters.",
        "ctm_2": "CTM, 2 Iters.",
        "lstm_5": "LSTM, 5 Iters.",
        "ctm_5": "CTM, 5 Iters.",
    }
    pattern = r"checkpoints/.+?/run\d+/([a-zA-Z0-9]+_[a-zA-Z0-9]+)/checkpoint\.pt"
    match = re.search(pattern, checkpoint_path)
    if match:
        config_key = match.group(1)
        human_name = name_map.get(config_key, config_key)
        return config_key, human_name
    else:
        raise ValueError(f"Could not extract human readable name from {checkpoint_path}")

def compute_mean_std_over_runs(steps_list, values_list, num_interpolation_points, smooth_window):
    common_steps = np.linspace(
        max(min(steps[0] for steps in steps_list), 1),
        min(max(steps[-1] for steps in steps_list), 400_000_000),
        num_interpolation_points
    )

    interpolated = [
        interp1d(steps, values, bounds_error=False, fill_value='extrapolate')(common_steps)
        for steps, values in zip(steps_list, values_list)
    ]

    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)

    if smooth_window > 0:
        # Compute how many points the smoothing window covers
        step_range = common_steps[-1] - common_steps[0]
        delta_step = step_range / (num_interpolation_points - 1)
        window_size = int((step_range * smooth_window) / delta_step)

        if window_size < 1:
            window_size = 1
        if window_size % 2 == 0:
            window_size += 1

        kernel = np.ones(window_size) / window_size
        mean = np.convolve(mean, kernel, mode='same')
        std = np.convolve(std, kernel, mode='same')

    return common_steps, mean, std

def extract_model_and_iters(name):
    """Extract model type and iteration number from name like 'CTM, 2 Iters.'"""
    parts = name.upper().split(",")
    model_type = parts[0].strip()
    num_iters = int(parts[1].strip().split()[0]) if len(parts) > 1 else 0
    return model_type, num_iters

def plot(grouped_data, scale=1, value_key='rewards', ylabel="Rewards", title="Episode Rewards", save_path="episode_rewards_avg.png"):

    fig, ax = plt.subplots(figsize=(10*scale, 5*scale))

    iter_groups = defaultdict(list)
    for key, data in grouped_data.items():
        model_type, num_iters = extract_model_and_iters(data['name'])
        iter_groups[num_iters].append((model_type, key, data))

    final_order = []
    for iters in sorted(iter_groups.keys()):
        for model_type in sorted(iter_groups[iters], key=lambda x: x[0]):
            final_order.append((model_type[1], model_type[2]))

    colors = sns.color_palette("hls", n_colors=len(final_order))
    max_steps = 0
    window_length = 0.01

    for i, (key, data) in enumerate(final_order):
        steps_list = data['steps']
        flat_steps = [step for run in steps_list for step in run]
        if len(flat_steps) > 0:
            run_max = max(flat_steps)
            if run_max > max_steps:
                max_steps = run_max

        values_list = data[value_key]
        label = data['name']

        num_interpolation_points = 200_000

        common_steps, mean, std = compute_mean_std_over_runs(steps_list, values_list, num_interpolation_points, window_length)
        color = colors[i]
        linestyle = '--' if 'LSTM' in label.upper() else '-'

        ax.plot(common_steps, mean, label=label, color=color, linestyle=linestyle)
        ax.fill_between(common_steps, mean - std, mean + std, alpha=0.1, color=color)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5)
    ax.set_xlim(0, int(max_steps*(1-window_length)))
    ax.legend()
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.replace(".png", ".pdf"), format="pdf")
    plt.close(fig)

def create_training_curves(save_dir, log_dir, device):
    for env_id in ("CartPole-v1", "Acrobot-v1", "MiniGrid-FourRooms-v0"):
        os.makedirs(f"{save_dir}/{env_id}", exist_ok=True)
        checkpoint_paths = get_checkpoint_paths_for_environment(env_id, log_dir)
        grouped_data = defaultdict(lambda: {
            'steps': [],
            'rewards': [],
            'lengths': [],
            'name': ''
        })

        for checkpoint_path in checkpoint_paths:
            checkpoint = load_checkpoint(checkpoint_path, device)
            global_steps = get_global_steps_from_checkpoint(checkpoint)
            episode_rewards = get_episode_rewards_from_checkpoint(checkpoint)
            episode_lengths = get_episode_lengths_from_checkpoint(checkpoint)
            config_key, human_readable_name = get_human_readable_name(checkpoint_path)

            if global_steps and episode_rewards and episode_lengths:
                grouped_data[config_key]['steps'].append(global_steps)
                grouped_data[config_key]['rewards'].append(episode_rewards)
                grouped_data[config_key]['lengths'].append(episode_lengths)
                grouped_data[config_key]['name'] = human_readable_name

        if grouped_data:
            plot(grouped_data, scale=args.scale, value_key='rewards', ylabel="Episode Reward", title="Avg Episode Rewards ± Std", save_path=f"{save_dir}/{env_id}/episode_rewards_avg.png")
            plot(grouped_data, scale=args.scale, value_key='lengths', ylabel="Episode Length", title="Avg Episode Lengths ± Std", save_path=f"{save_dir}/{env_id}/episode_lengths_avg.png")
        else:
            print("No valid checkpoint data found.")
    pass

def filter_checkpoints(checkpoint_paths, iters, arch, run):
    filtered_iters = filter_checkpoint_by_iters(checkpoint_paths, iters)
    filtered_arch = filter_checkpoint_by_arch(filtered_iters, arch)
    filtered_run = filter_checkpoint_by_run(filtered_arch, run)
    return filtered_run

def filter_checkpoint_by_iters(checkpoint_paths, iters):
    return [path for path in checkpoint_paths if f"_{iters}" in path]

def filter_checkpoint_by_arch(checkpoint_paths, arch):
    return [path for path in checkpoint_paths if f"{arch}" in path]

def filter_checkpoint_by_run(checkpoint_paths, run):
    return [path for path in checkpoint_paths if f"run{run}" in path]

def get_training_data_from_checkpoint_path(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    global_step = checkpoint.get('global_step', 0)
    training_iteration = checkpoint.get('training_iteration', 0)
    episode_rewards_tracking = checkpoint.get('episode_rewards_tracking', [])
    episode_lengths_tracking = checkpoint.get('episode_lengths_tracking', [])
    global_steps_tracking = checkpoint.get('global_steps_tracking', [])
    model_args = checkpoint.get('args', None)
    return global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, model_args

def prepare_csv(csv_filepath):
    if os.path.exists(csv_filepath):
        os.remove(csv_filepath)
    with open(csv_filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'iters', 'run', 'mean', 'std'])
    pass

def parse_env_id(env_id):
    if env_id == "CartPole-v1":
        return "cartpole"
    elif env_id == "Acrobot-v1":
        return "acrobot"
    elif env_id == "MiniGrid-FourRooms-v0":
        return "4rooms"
    else:
        raise ValueError(f"Environment {env_id} not supported.")

def get_size_action_space(env_id):
    if env_id == "CartPole-v1":
        return 2
    elif env_id == "Acrobot-v1":
        return 3
    elif env_id == "MiniGrid-FourRooms-v0":
        return 7
    else:
        raise ValueError(f"Environment {env_id} not supported.")

def prepare_env(env_id, max_environment_steps, mask_velocity, render_mode):
    if env_id in ("CartPole-v1", "Acrobot-v1"):
        return(make_env_classic_control(env_id, max_environment_steps, mask_velocity=mask_velocity, render_mode=render_mode)())
    elif "MiniGrid" in env_id:
        return(make_env_minigrid(env_id, max_environment_steps)())
    else:
        raise NotImplementedError(f"Environment {env_id} not supported.")

def create_episode_length_csv_and_activation_plots(save_dir, args):
    for env_id in ("CartPole-v1", "Acrobot-v1", "MiniGrid-FourRooms-v0"):
        size_action_space = get_size_action_space(env_id)
        csv_filepath = f'{save_dir}/{env_id}/episode_lengths.csv'
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        prepare_csv(csv_filepath)

        checkpoint_paths = get_checkpoint_paths_for_environment(env_id, args.log_dir)
        ARCHS_TO_TEST = ["ctm", "lstm"]
        ITERS_TO_TEST = [1, 2, 5]
        RUNS_TO_TEST = [1, 2, 3]
        for arch in ARCHS_TO_TEST:
            for iters in ITERS_TO_TEST:
                episode_lengths = []
                for run in RUNS_TO_TEST:
                    
                    activation_plots_path = f'{save_dir}/{env_id}/arch_{arch}_iters_{iters}_run_{run}'
                    os.makedirs(activation_plots_path, exist_ok=True)

                    checkpoints = filter_checkpoints(checkpoint_paths, iters, arch, run)
                    if not checkpoints:
                        print(f"Skipping: no checkpoint found for {env_id} | iters={iters} | arch={arch} | run={run}")
                        continue
                    
                    _, _, _, _, _, model_args = get_training_data_from_checkpoint_path(checkpoints[0], device)
                    model_args.log_dir = f"{args.log_dir}/{env_id}"

                    agent = Agent(size_action_space, model_args, device).to(device)
                    optimizer = optim.Adam(agent.parameters(), lr=model_args.lr, eps=1e-5)

                    _, _, _, _, _, _ = load_model(agent, optimizer, checkpoints[0], device)

                    eval_env = prepare_env(model_args.env_id, model_args.max_environment_steps, mask_velocity=model_args.mask_velocity, render_mode="rgb_array")

                    for idx in range(args.num_eval_envs):
                        eval_next_obs, _ = eval_env.reset(seed=idx)
                        eval_next_done = False
                        eval_state = agent.get_initial_state(1)
                        tracking_data_by_world_step = []
                        for environment_step in range(model_args.max_environment_steps):
                            with torch.no_grad():
                                action, _, _, value, eval_state, tracking_data, action_logits, action_probs = agent.get_action_and_value(
                                    torch.Tensor(eval_next_obs).to(device).unsqueeze(0),
                                    eval_state,
                                    torch.Tensor([eval_next_done]).to(device),
                                    track=True
                                )
                            eval_next_obs, reward, termination, truncation, _ = eval_env.step(action.cpu().numpy()[0])
                            eval_next_done = termination or truncation

                            tracking_data['actions'] = np.tile(action.detach().cpu().numpy(), (model_args.iterations)) # Shape T
                            tracking_data['values'] = np.tile(value.squeeze(-1).detach().cpu().numpy(), (model_args.iterations)) # Shape T
                            tracking_data['action_logits'] = np.tile(action_logits.detach().cpu().numpy(), (model_args.iterations, 1)) # Shape T, A
                            tracking_data['action_probs'] = np.tile(action_probs.detach().cpu().numpy(), (model_args.iterations, 1))# Shape T, A
                            tracking_data['rewards'] = np.tile(np.array(reward), (model_args.iterations)) # Shape T
                            tracking_data['inputs'] = np.tile(np.array(eval_env.render()), (model_args.iterations, 1, 1, 1)) # Shape T, H, W, C

                            tracking_data_by_world_step.append(tracking_data)

                            if eval_next_done:
                                break
                        
                        eval_env.close()

                        combined_tracking_data = combine_tracking_data(tracking_data_by_world_step)

                        n_to_plot = 80 if combined_tracking_data['post_activations'].shape[-1] < 100 else 100
                        if idx==0:
                            plot_neural_dynamics(combined_tracking_data['post_activations'], n_to_plot, activation_plots_path, axis_snap=True)
                            process = multiprocessing.Process(
                                target=make_rl_gif,
                                args=(
                                    combined_tracking_data['action_logits'],
                                    combined_tracking_data['action_probs'],
                                    combined_tracking_data['actions'],
                                    combined_tracking_data['values'],
                                    combined_tracking_data['rewards'],
                                    combined_tracking_data['pre_activations'],
                                    combined_tracking_data['post_activations'],
                                    combined_tracking_data['inputs'],
                                    f"{activation_plots_path}/eval_output_val_{idx}.gif"
                                )
                            )
                            process.start()

                        episode_lengths.append(environment_step+1)

                    if episode_lengths:
                        mean = np.mean(episode_lengths)
                        std = np.std(episode_lengths)

                        with open(csv_filepath, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([arch, iters, run, mean, std])

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    save_dir = "tasks/rl/analysis/outputs"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    create_training_curves(save_dir, args.log_dir, device)
    create_episode_length_csv_and_activation_plots(save_dir, args)
