import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.samplers import QAMNISTSampler
import argparse
import pandas as pd
import multiprocessing
from tqdm import tqdm

from utils.housekeeping import set_seed
from tasks.qamnist.plotting import make_qamnist_gif
from tasks.image_classification.plotting import plot_neural_dynamics
from tasks.parity.plotting import plot_training_curve_all_runs
from models.utils import load_checkpoint, get_all_log_dirs, get_model_args_from_checkpoint, get_latest_checkpoint_file
from tasks.parity.utils import reshape_attention_weights
from tasks.qamnist.utils import get_dataset, prepare_model


def parse_args():
    parser = argparse.ArgumentParser(description='QAMNIST Analysis')
    parser.add_argument('--log_dir', type=str, default='checkpoints/qamnist', help='Directory to save logs.')
    parser.add_argument('--scale_training_curve', type=float, default=0.5, help='Scaling factor for plots.')
    parser.add_argument('--generalization_heatmap_scale', type=float, default=0.5, help='Scaling for the generalization heatmap.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')
    return parser.parse_args()

def load_training_data_from_checkpoint(load_path, device="cpu"):
    """Loads only training data (losses and accuracies) from the checkpoint without loading the model."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}.")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    test_accuracies = checkpoint.get('test_accuracies', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies


def prepare_data_for_analysis(num_images, num_operations, args):
    _, test_data, _, _, _ = get_dataset(
        q_num_images=num_images,
        q_num_images_delta=0,
        q_num_repeats_per_input=args.q_num_repeats_per_input,
        q_num_operations=num_operations,
        q_num_operations_delta=0
        )
    
    test_sampler = QAMNISTSampler(test_data, batch_size=args.batch_size_test)
    testloader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_sampler=test_sampler)
    return testloader

def plot_accuracy_operations(accuracy_matrix, std_matrix, num_operations_to_test, num_digits_to_test, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("viridis")
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Training Regime')
    for i, digit in enumerate(num_digits_to_test):
        norm_digit = (digit - min(num_digits_to_test)) / (max(num_digits_to_test) - min(num_digits_to_test))
        color = cmap(norm_digit)
        ax.plot(num_operations_to_test, accuracy_matrix[:, i], color=color, alpha=0.7, linewidth=2)
        ax.fill_between(
            num_operations_to_test,
            accuracy_matrix[:, i] - std_matrix[:, i],
            accuracy_matrix[:, i] + std_matrix[:, i],
            color=color,
            alpha=0.2
        )
    norm = plt.Normalize(vmin=min(num_digits_to_test), vmax=max(num_digits_to_test))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(min(num_digits_to_test), max(num_digits_to_test) + 1))
    cbar.set_label("Number of Digits")
    ax.set_xlabel("Number of Operations")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Number of Operations for Different Number of Digits Shown")
    ax.set_xlim(min(num_operations_to_test), max(num_operations_to_test))
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.savefig(filename.replace(".png", ".pdf"), format="pdf")
    plt.show()


def plot_accuracy_grid(accuracy_matrix, num_digits_to_test, num_operations_to_test, scale, filename):
    fig, ax = plt.subplots(figsize=(scale*10, scale*8))
    im = ax.imshow(accuracy_matrix*100, interpolation='none', cmap='viridis', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(num_digits_to_test)))
    ax.set_yticks(np.arange(len(num_operations_to_test)))
    ax.set_xticklabels(num_digits_to_test)
    ax.set_yticklabels(num_operations_to_test)
    ax.set_xlabel("Number of Digits")
    ax.set_ylabel("Number of Operations")
    ax.plot([3.5, 3.5], [-0.5, 3.5], color='red', linestyle='--', linewidth=2)
    ax.plot([-0.5, 3.5], [3.5, 3.5], color='red', linestyle='--', linewidth=2)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.savefig(filename.replace(".png", ".pdf"), format="pdf")
    plt.show()
    
def plot_loss_all_runs(training_data, evaluate_every, save_path="train_loss_comparison_qamnist.png", window_size=1000):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_data)))
    legend_handles = []
    all_losses = [loss for data in training_data.values() for loss in data["train_losses"]]
    min_loss, max_loss = min(all_losses), max(all_losses)
    for (folder, data), color in zip(training_data.items(), colors):
        iters = range(len(data["train_losses"]))
        ax.plot(iters, data["train_losses"], alpha=0.2, color=color)
        smooth_losses = np.convolve(data["train_losses"], np.ones(window_size) / window_size, mode='same')
        smooth_line, = ax.plot(iters, smooth_losses, alpha=1.0, color=color)
        legend_handles.append((smooth_line, folder))
    ax.set_xlim([0, 200000])
    ax.set_ylim([min_loss * 0.9, max_loss * 1.1])
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Loss")
    ax.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles], loc="upper left", title="Runs")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace(".png", ".pdf"), format="pdf")
    plt.close(fig)

def contains_only_zero_zero_ops(q):
    matches = re.findall(r'\((\d) [\+\-] (\d)\)', q)
    return all(a == b == '0' for a, b in matches) and len(matches) > 0

def create_case_study_plots(model, model_args, save_dir):

    num_digits = 2
    num_operations = 4
    testloader = prepare_data_for_analysis(num_digits, num_operations, model_args)
    inputs, z, question_readable, targets = next(iter(testloader))
    inputs, targets = inputs.to(device), targets.to(device)
    z = torch.stack(z, 1).to(device)

    predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking, embedding_tracking = model(inputs, z, track=True)

    attention = reshape_attention_weights(attention_tracking)

    T = predictions.size(-1)
    B = predictions.size(0)
    gif_inputs = torch.zeros((T, B, 1, 32, 32), device=inputs.device)
    digits_input = inputs.permute(1, 0, 2, 3, 4)
    gif_inputs[:digits_input.size(0)] = digits_input

    T_embed = embedding_tracking.shape[0]
    pad_width = ((0, 0), (0, 0), (0, (32*32)-model_args.d_input))
    embedding_padded = np.pad(embedding_tracking, pad_width, mode='constant')
    reshaped = embedding_padded.reshape(T_embed,B, 1, 32, 32)
    embedding_input = np.zeros((T_embed, B, 1, 32, 32))
    embedding_input[:T_embed] = reshaped

    embedding_tensor = torch.from_numpy(embedding_input).to(gif_inputs.device)
    gif_inputs[digits_input.size(0):digits_input.size(0) + T_embed] = embedding_tensor[:T_embed]

    plot_neural_dynamics(post_activations, 100, save_dir, axis_snap=True)

    process = multiprocessing.Process(
        target=make_qamnist_gif,
        args=(
        predictions.detach().cpu().numpy(),
        certainties.detach().cpu().numpy(),
        targets.detach().cpu().numpy(),
        pre_activations,
        post_activations,
        attention,
        gif_inputs.detach().cpu().numpy(),
        f"{save_dir}/eval_output_val_{0}_iter_{0}.gif",
        question_readable
    ))
    process.start()

def get_accuracy(testloader, model, device, args):
    model.eval()
    with torch.inference_mode():
        n_total, n_correct_argmax = 0, 0
        inputs, z, _, targets = next(iter(testloader))
        z = torch.stack(z, 1).to(device)
        inputs = inputs.to(device)
        predictions, certainties, _ = model(inputs, z, track=False)
        if args.model_type == "lstm":
            predictions_argmax = predictions[:,:,-1].argmax(1)
        elif args.model_type == "ctm":
            predictions_argmax = predictions[:,:,-args.q_num_answer_steps:].argmax(1)[np.arange(inputs.size(0)),certainties[:,1,-args.q_num_answer_steps:].argmax(-1)]
        else:
            raise ValueError("Only CTM and LSTM models work with this accuracy calculation.")
        n_correct_argmax += (predictions_argmax.cpu() == targets).sum().item()
        n_total += targets.shape[0]
        acc = n_correct_argmax / n_total
    model.train()
    return acc

if __name__ == "__main__":

    args = parse_args()
    
    device = f'cuda:{args.device[0]}' if args.device[0] != -1 else 'cpu' 

    set_seed(args.seed)

    save_dir = "tasks/qamnist/analysis/outputs"
    os.makedirs(save_dir, exist_ok=True)

    num_repeats = 10
    num_digits_to_test = list(range(1, 11))
    num_operations_to_test = list(range(1, 11))

    # Plot training curves 
    all_runs_log_dirs = get_all_log_dirs(args.log_dir)
    plot_training_curve_all_runs(all_runs_log_dirs, save_dir, args.scale_training_curve, device, smooth=True, x_max=300_000, plot_individual_runs=False)

    progress_bar = tqdm(all_runs_log_dirs, desc="Analyzing Runs", dynamic_ncols=True)
    for folder in progress_bar:
        progress_bar.set_description(f"Analyzing Model at {folder}")

        run, model_name = folder.strip("/").split("/")[-2:]

        run_model_spefic_save_dir = f"{save_dir}/{model_name}/{run}"
        os.makedirs(run_model_spefic_save_dir, exist_ok=True)

        checkpoint_path = get_latest_checkpoint_file(folder)
        checkpoint = load_checkpoint(checkpoint_path, device)
        model_args = get_model_args_from_checkpoint(checkpoint)
        model_args.log_dir = args.log_dir

        model = prepare_model(model_args, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        create_case_study_plots(model, model_args, run_model_spefic_save_dir)

        accuracy_matrix = np.zeros((len(num_operations_to_test), len(num_digits_to_test)))
        std_matrix = np.zeros((len(num_operations_to_test), len(num_digits_to_test)))
        results = []
        for i, num_digits in enumerate(num_digits_to_test):
            for j, num_operations in enumerate(num_operations_to_test):
                testloader = prepare_data_for_analysis(num_digits, num_operations, model_args)
                acc_values = [get_accuracy(testloader, model, device, model_args) for _ in range(num_repeats)]
                mean_acc = np.mean(acc_values)
                std_acc = np.std(acc_values)
                accuracy_matrix[j, i] = mean_acc
                std_matrix[j, i] = std_acc
                results.append({
                    "model_type": model_args.model_type,
                    "num_repeats": num_repeats,
                    "num_digits": num_digits,
                    "num_operations": num_operations,
                    "accuracy": mean_acc,
                    "std": std_acc
                })

        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{run_model_spefic_save_dir}/accuracy_results_{model_args.model_type}_{model_args.q_num_repeats_per_input}.csv", index=False)

        plot_accuracy_grid(accuracy_matrix, num_digits_to_test, num_operations_to_test, args.generalization_heatmap_scale, filename=f"{run_model_spefic_save_dir}/accuracy_grid_{model_args.model_type}_{model_args.q_num_repeats_per_input}.png")
        plot_accuracy_operations(accuracy_matrix, std_matrix, num_operations_to_test, num_digits_to_test, filename=f"{run_model_spefic_save_dir}/accuracy_scatter_{model_args.model_type}_{model_args.q_num_repeats_per_input}.png")
