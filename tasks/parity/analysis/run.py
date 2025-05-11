import torch
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm
import math
import os
import csv
from utils.housekeeping import set_seed
from data.custom_datasets import ParityDataset
from tasks.parity.utils import prepare_model, reshape_attention_weights, reshape_inputs, get_where_most_certain
from tasks.parity.plotting import plot_attention_trajectory, plot_input, plot_target, plot_probabilities, plot_prediction, plot_accuracy_training, create_attentions_heatmap_gif, create_accuracies_heatmap_gif, create_stacked_gif, plot_training_curve_all_runs, plot_accuracy_thinking_time, make_parity_gif, plot_lstm_last_and_certain_accuracy
from models.utils import compute_normalized_entropy, reshape_predictions, get_latest_checkpoint_file, get_checkpoint_files, load_checkpoint, get_model_args_from_checkpoint, get_all_log_dirs
from tasks.image_classification.plotting import plot_neural_dynamics

import seaborn as sns
sns.set_palette("hls")
sns.set_style('darkgrid')

def parse_args():
    parser = argparse.ArgumentParser(description='Parity Analysis')
    parser.add_argument('--log_dir', type=str, default='checkpoints/parity', help='Directory to save logs.')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch size for testing')
    parser.add_argument('--scale_training_curve', type=float, default=0.6, help='Scaling factor for plots.')
    parser.add_argument('--scale_heatmap', type=float, default=0.4, help='Scaling factor for heatmap plots.')
    parser.add_argument('--scale_training_index_accuracy', type=float, default=0.4, help='Scaling factor for training index accuracy plots.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--model_type', type=str, choices=['ctm', 'lstm'], default='ctm', help='Type of model to analyze (ctm or lstm).')
    return parser.parse_args()

def calculate_corrects(predictions, targets):
    predicted_labels = predictions.argmax(2)
    accuracy = (predicted_labels == targets.unsqueeze(-1))
    return accuracy.detach().cpu().numpy()

def get_corrects_per_element_at_most_certain_time(predictions, certainty, targets):
    where_most_certain = get_where_most_certain(certainty)
    corrects = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device),:,where_most_certain] == targets).float()
    return corrects.detach().cpu().numpy()

def calculate_entropy_average_over_batch(normalized_entropy_per_elements):
    normalized_entropy_per_elements_avg_batch = normalized_entropy_per_elements.mean(axis=1)
    return normalized_entropy_per_elements_avg_batch

def calculate_thinking_time_average_over_batch(normalized_entropy_per_elements):
    first_occurrence = calculate_thinking_time(normalized_entropy_per_elements)
    average_thinking_time = np.mean(first_occurrence, axis=0)
    return average_thinking_time

def calculate_thinking_time(normalized_entropy_per_elements, finish_type="min", entropy_threshold=0.1):
    if finish_type == "min":
        min_entropy_time = np.argmin(normalized_entropy_per_elements, axis=0)
        return min_entropy_time
    elif finish_type == "threshold":
        T, B, S = normalized_entropy_per_elements.shape
        below_threshold = normalized_entropy_per_elements < entropy_threshold
        first_occurrence = np.argmax(below_threshold, axis=0)
        no_true = ~np.any(below_threshold, axis=0)
        first_occurrence[no_true] = T
        return first_occurrence

def test_handcrafted_examples(model, args, run_model_spefic_save_dir, device):
    test_cases = []
    all_even_input = torch.full((args.parity_sequence_length,), 1.0, dtype=torch.float32, device=device)
    all_even_target = torch.zeros_like(all_even_input, dtype=torch.long)
    test_cases.append((all_even_input, all_even_target))

    all_odd_input = torch.full((args.parity_sequence_length,), -1.0, dtype=torch.float32, device=device)
    all_odd_target = torch.cumsum((all_odd_input == -1).to(torch.long), dim=0) % 2
    test_cases.append((all_odd_input, all_odd_target))

    random_input = torch.randint(0, 2, (args.parity_sequence_length,), dtype=torch.float32, device=device) * 2 - 1
    random_target = torch.cumsum((random_input == -1).to(torch.long), dim=0) % 2
    test_cases.append((random_input, random_target))

    for i, (inputs, targets) in enumerate(test_cases):
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        filename = f"eval_handcrafted_{i}"
        extend_inference_time = False
        handcraft_dir = f"{run_model_spefic_save_dir}/handcrafted_examples/{i}"
        os.makedirs(handcraft_dir, exist_ok=True)

        model.eval()
        with torch.inference_mode():
            if extend_inference_time:
                model.iterations = model.iterations * 2
            predictions, certainties, synchronisation, pre_activations, post_activations, attention = model(inputs, track=True)
            predictions = reshape_predictions(predictions, prediction_reshaper=[args.parity_sequence_length, 2])
            input_images = reshape_inputs(inputs, args.iterations, grid_size=int(math.sqrt(args.parity_sequence_length)))

            plot_neural_dynamics(post_activations, 100, handcraft_dir, axis_snap=False)

            process = multiprocessing.Process(
                target=make_parity_gif,
                args=(
                predictions.detach().cpu().numpy(),
                certainties.detach().cpu().numpy(),
                targets.detach().cpu().numpy(),
                pre_activations,
                post_activations,
                reshape_attention_weights(attention),
                input_images,
                f"{handcraft_dir}/eval_output_val_{0}_iter_{0}.gif",
            ))
            process.start()


            input_images = input_images.squeeze(1).squeeze(1)
            attention = attention.squeeze(1)

            for h in range(args.heads):
                plot_attention_trajectory(attention[:, h, :, :], certainties, input_images, handcraft_dir, filename + f"_head_{h}", args)

            plot_attention_trajectory(attention.mean(1), certainties, input_images, handcraft_dir, filename, args)
            plot_input(input_images, handcraft_dir, filename)
            plot_target(targets, handcraft_dir, filename, args)
            plot_probabilities(predictions, certainties, handcraft_dir, filename, args)
            plot_prediction(predictions, certainties,handcraft_dir, filename, args)
        
        if extend_inference_time:
            model.iterations = model.iterations // 2
        model.train()
        pass

def build_model_from_checkpoint_path(checkpoint_path, model_type, device="cpu"):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_args = get_model_args_from_checkpoint(checkpoint)
    model = prepare_model([model_args.parity_sequence_length, 2], model_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model, model_args

def analyze_trained_model(run_model_spefic_save_dir, args, device):
    with torch.no_grad():

        latest_checkpoint_path = get_latest_checkpoint_file(args.log_dir)
        model, model_args = build_model_from_checkpoint_path(latest_checkpoint_path, args.model_type, device=device)
        model.eval()
        model_args.log_dir = args.log_dir
        test_data = ParityDataset(sequence_length=model_args.parity_sequence_length, length=10000)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0, drop_last=False)

        corrects, corrects_at_most_certain_times, entropys, attentions = [], [], [], []

        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions, certainties, synchronisation, pre_activations, post_activations, attention = model(inputs, track=True)
            predictions = reshape_predictions(predictions, prediction_reshaper=[model_args.parity_sequence_length, 2])
            corrects_batch = calculate_corrects(predictions, targets)
            corrects_at_most_certain_time_batch = get_corrects_per_element_at_most_certain_time(predictions, certainties, targets)
            corrects.append(corrects_batch)
            corrects_at_most_certain_times.append(corrects_at_most_certain_time_batch)
            attentions.append(attention)

        test_handcrafted_examples(model, model_args, run_model_spefic_save_dir, device)

        overall_mean_accuracy = np.mean(np.vstack(corrects_at_most_certain_times))
        overall_std_accuracy = np.std(np.mean(np.vstack(corrects_at_most_certain_times), axis=1))

    return overall_mean_accuracy, overall_std_accuracy, model_args.iterations

def analyze_training(run_model_spefic_save_dir, args, device):
    checkpoint_files = get_checkpoint_files(args.log_dir)
    all_accuracies = []
    all_accuracies_at_most_certain_time = []
    all_average_thinking_times = []
    all_std_thinking_times = []
    all_attentions = []
    for checkpoint_path in checkpoint_files:
        model, model_args = build_model_from_checkpoint_path(checkpoint_path, args.model_type, device=device)
        model_args.log_dir = run_model_spefic_save_dir
        test_data = ParityDataset(sequence_length=model_args.parity_sequence_length, length=1000)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0, drop_last=False)
        corrects = []
        corrects_at_most_certain_times = []
        thinking_times = []
        attentions = []

        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions, certainties, synchronisation, pre_activations, post_activations, attention = model(inputs, track=True)
            predictions = reshape_predictions(predictions, prediction_reshaper=[model_args.parity_sequence_length, 2])
            attention = reshape_attention_weights(attention)

            corrects_batch = calculate_corrects(predictions, targets)       
            corrects_at_most_certain_time_batch = get_corrects_per_element_at_most_certain_time(predictions, certainties, targets)
            entropy_per_element = compute_normalized_entropy(predictions.permute(0,3,1,2), reduction='none').detach().cpu().numpy()
            thinking_times_batch = np.argmin(entropy_per_element, axis=1)

            corrects.append(corrects_batch)
            corrects_at_most_certain_times.append(corrects_at_most_certain_time_batch)
            thinking_times.append(thinking_times_batch)
            attentions.append(attention)

        checkpoint_average_accuracies = np.mean(np.concatenate(corrects, axis=0), axis=0).transpose(1,0)
        all_accuracies.append(checkpoint_average_accuracies)

        stacked_corrects_at_most_certain_times = np.vstack(corrects_at_most_certain_times)
        checkpoint_average_accuracy_at_most_certain_time = np.mean(stacked_corrects_at_most_certain_times, axis=0)
        all_accuracies_at_most_certain_time.append(checkpoint_average_accuracy_at_most_certain_time)

        checkpoint_thinking_times = np.concatenate(thinking_times, axis=0)
        checkpoint_average_thinking_time = np.mean(checkpoint_thinking_times, axis=0)
        checkpoint_std_thinking_time = np.std(checkpoint_thinking_times, axis=0)
        all_average_thinking_times.append(checkpoint_average_thinking_time)
        all_std_thinking_times.append(checkpoint_std_thinking_time)

        checkpoint_average_attentions = np.mean(np.concatenate(attentions, axis=1), axis=1)
        all_attentions.append(checkpoint_average_attentions)

    plot_accuracy_training(all_accuracies_at_most_certain_time, args.scale_training_index_accuracy, run_model_spefic_save_dir, args=model_args)
    create_attentions_heatmap_gif(all_attentions, args.scale_heatmap, run_model_spefic_save_dir, model_args)
    create_accuracies_heatmap_gif(np.array(all_accuracies), all_average_thinking_times, all_std_thinking_times, args.scale_heatmap, run_model_spefic_save_dir, model_args)
    create_stacked_gif(run_model_spefic_save_dir)

def get_accuracy_and_loss_from_checkpoint(checkpoint):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies

if __name__ == "__main__":

    args = parse_args()

    device = f'cuda:{args.device[0]}' if args.device[0] != -1 else 'cpu' 

    set_seed(args.seed)

    save_dir = "tasks/parity/analysis/outputs"
    os.makedirs(save_dir, exist_ok=True)

    accuracy_csv_file_path = os.path.join(save_dir, "accuracy.csv")
    if os.path.exists(accuracy_csv_file_path):
        os.remove(accuracy_csv_file_path)

    all_runs_log_dirs = get_all_log_dirs(args.log_dir)

    plot_training_curve_all_runs(all_runs_log_dirs, save_dir, args.scale_training_curve, device, x_max=200_000)
    plot_lstm_last_and_certain_accuracy(all_folders=all_runs_log_dirs, save_path=f"{save_dir}/lstm_final_vs_certain_accuracy.png", scale=args.scale_training_curve)

    progress_bar = tqdm(all_runs_log_dirs, desc="Analyzing Runs", dynamic_ncols=True)
    for folder in progress_bar:

        run, model_name = folder.strip("/").split("/")[-2:]

        run_model_spefic_save_dir = f"{save_dir}/{model_name}/{run}"
        os.makedirs(run_model_spefic_save_dir, exist_ok=True)

        args.log_dir = folder
        progress_bar.set_description(f"Analyzing Trained Model at {folder}")

        accuracy_mean, accuracy_std, num_iterations = analyze_trained_model(run_model_spefic_save_dir, args, device)

        with open(accuracy_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)            
            if file.tell() == 0:
                writer.writerow(["Run", "Overall Mean Accuracy", "Overall Std Accuracy", "Num Iterations"])
            writer.writerow([folder, accuracy_mean, accuracy_std, num_iterations])

        progress_bar.set_description(f"Analyzing Training at {folder}")
        analyze_training(run_model_spefic_save_dir, args, device)

    plot_accuracy_thinking_time(accuracy_csv_file_path, scale=args.scale_training_curve, output_dir=save_dir)
