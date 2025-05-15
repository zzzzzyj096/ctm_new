import torch
import os
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import umap
from scipy.special import softmax
import seaborn as sns
import re
sns.set_style('darkgrid')
mpl.use('Agg')
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage[utf8]{inputenc}
        \usepackage{xcolor}
        \usepackage{amsmath}
        \renewcommand{\rmdefault}{cmr}
    """
})
from tasks.qamnist.utils import prepare_model
from models.utils import load_checkpoint, get_model_args_from_checkpoint
from tasks.qamnist.analysis.run import prepare_data_for_analysis
from tasks.parity.utils import reshape_attention_weights
from tasks.image_classification.plotting import save_frames_to_mp4

def compose_modular_expressions(input_string):
    lines = input_string.strip().split('\n')
    parsed = []

    for line in lines:
        match = re.match(r"(.*)mod\s+(\d+)\s*=\s*(-?\d+)", line)
        if not match:
            raise ValueError(f"Invalid format: '{line}'")
        expr, mod_val, result = match.groups()
        parsed.append((expr.strip(), int(mod_val), int(result)))

    # Start with first expression as value provider
    for i in range(len(parsed) - 1):
        outer_expr, outer_mod, outer_result = parsed[i]
        inner_expr, inner_mod, inner_result = parsed[i + 1]

        # Replace the outer result in inner expression
        new_inner_expr = re.sub(
            fr"\b{outer_result}\b",
            f"(({outer_expr}) mod {outer_mod})",
            inner_expr,
            count=1
        )

        # Update the next expression with substituted version
        parsed[i + 1] = (new_inner_expr, inner_mod, parsed[i + 1][2])

    # Final expression
    final_expr, final_mod, final_result = parsed[-1]
    final_composed = f"({final_expr}) mod {final_mod} = {final_result}"

    # Remove first '(' and last ')', regardless of position
    first_paren = final_composed.find('(')
    last_paren = final_composed.rfind(')')
    if first_paren != -1 and last_paren != -1 and first_paren < last_paren:
        final_composed = final_composed[:first_paren] + final_composed[first_paren+1:last_paren] + final_composed[last_paren+1:]

    return final_composed



def make_qamnist_gif(predictions, targets, post_activations, input_gates, inputs_to_model, save_path, question_readable, umap_positions, umap_point_size_scaler=2.5):

    # Config
    batch_index = 0
    figscale = 0.28
    n_steps = len(predictions)
    n_steps = predictions.shape[-1]
    heatmap_cmap = sns.color_palette("viridis", as_cmap=True)
    frames = []

    these_inputs = inputs_to_model[:, batch_index, :, :, :] # Shape: (T, C, H, W)
    these_input_gates = input_gates[:, batch_index, :, :] # Shape: (T, H, W)
    these_predictions = predictions[batch_index, :, :] # Shape: (C, T)
    this_target = targets[batch_index] # Shape: (C)
    these_post_activations = post_activations[:, batch_index, :] # Shape: (T, H)

    probs_min, probs_max = 0, 1

    class_labels = [str(i) for i in range(10)]
    pad = 0.1

    mosaic = [
        ["eq", "eq", "eq", "eq", "eq", "eq"],
        ["input", "input", "att", "att", "probs", "probs"],
        ["input", "input", "att", "att", "probs", "probs"],
    ]
    for _ in range(3, 8):
        mosaic.append(
            ["umap", "umap", "umap", "umap","umap", "umap"]
        )

            
    for stepi in range(n_steps):
        fig_gif, axes_gif = plt.subplot_mosaic(mosaic=mosaic, figsize=(31*figscale*8/4, 76*figscale))

        if question_readable:
            raw = compose_modular_expressions(question_readable)

            print(raw)
            raw = raw.replace("mod", r"\mod")
            full_text = r"$" + raw + r"$"
            axes_gif["eq"].text(
                0.5, 0.5, full_text,
                fontsize=32, ha="center", va="center",
                linespacing=1.5, multialignment="center"
            )
            axes_gif["eq"].axis("off")

        # Plot action log probs
        colors = [('g' if i == this_target else ('r')) for i, e in enumerate(these_predictions[:, stepi])]
        sort_idxs = np.arange(len(these_predictions[:, stepi]))

        # Add probability plot
        probs = softmax(these_predictions[:, stepi])
        bars_prob = axes_gif['probs'].bar(np.arange(len(probs)), probs[sort_idxs], 
                                        color=np.array(colors)[sort_idxs], width=0.9, alpha=0.5)
        axes_gif['probs'].axis('off')
        axes_gif['probs'].set_ylim([0, 1])
        for i, (bar, label) in enumerate(zip(bars_prob, class_labels)):
            x = bar.get_x() + bar.get_width() / 2
            axes_gif['probs'].text(
                x, -0.08, label,
                ha='center', va='top',
                fontsize=24,
                transform=axes_gif['probs'].get_xaxis_transform()  # ← anchors to axis, not data
            )

                                 
        axes_gif['probs'].set_ylim([probs_min, probs_max])

        # Show input image
        this_image = these_inputs[stepi].transpose(1, 2, 0)
        axes_gif['input'].imshow(this_image, cmap='binary', vmin=0, vmax=1)
        axes_gif['input'].grid(False) 
        axes_gif['input'].set_xticks([])
        axes_gif['input'].set_yticks([])

        # Create and show attention heatmap
        try:
            this_input_gate = these_input_gates[stepi]
        except (IndexError, TypeError):
            this_input_gate = np.zeros_like(these_input_gates[0])
        gate_min, gate_max = np.nanmin(this_input_gate), np.nanmax(this_input_gate)
        if not np.isclose(gate_min, gate_max):
            normalized_gate = (this_input_gate - gate_min) / (gate_max - gate_min + 1e-8)
        else:
            normalized_gate = np.zeros_like(this_input_gate)
        input_heatmap = heatmap_cmap(normalized_gate)[:,:,:3]
        # Show heatmaps
        axes_gif['att'].imshow(input_heatmap, vmin=0, vmax=1)
        axes_gif['att'].axis('off')

        z = these_post_activations[stepi]
        low, high = np.percentile(z, 5), np.percentile(z, 95)
        z_norm = np.clip((z - low) / (high - low), 0, 1)
        point_sizes = (np.abs(z_norm - 0.5) * 100 + 5) * umap_point_size_scaler
        cmap = plt.get_cmap("Spectral")
        axes_gif["umap"].scatter(
            umap_positions[:, 0],
            umap_positions[:, 1],
            s=point_sizes,
            c=cmap(z_norm),
            alpha=0.8
        )
        axes_gif["umap"].axis("off")


        # Save frames
        fig_gif.tight_layout(pad=pad)

        # Convert to frame
        canvas = fig_gif.canvas
        canvas.draw()
        image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3]
        frames.append(image_numpy)
        plt.close(fig_gif)

    imageio.mimsave(f"{save_path}/activations.gif", frames, fps=15, loop=100)

    # save mp4
    save_frames_to_mp4(
        [fm[:, :, ::-1] for fm in frames],  # RGB→BGR
        f"{save_path}/activation.mp4",
        fps=15,
        gop_size=1,
        preset="slow"
    )

    pass

def run_umap(model, model_args, device):
    num_digits= 4
    num_operations = 3
    model_args.batch_size_test = 1
    testloader = prepare_data_for_analysis(num_digits, num_operations, model_args)

    point_counts = 150
    all_post_activations = []
    sampled = 0
    with tqdm(total=point_counts, desc="Collecting UMAP data") as pbar:
        for (inputs, z, question_readable, targets) in testloader:

            inputs, targets = inputs.to(device), targets.to(device)
            z = torch.stack(z,1).to(device)
            B = inputs.shape[0]
            for b in range(B):
                if sampled >= point_counts:
                    break
                inputs_b = inputs[b].unsqueeze(0)
                targets_b = targets[b].unsqueeze(0)
                question_readable_b = question_readable[b]
                z_b = z[b].unsqueeze(0)
                predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking, embedding_tracking = model(inputs_b, z_b, track=True)
                all_post_activations.append(post_activations)
                sampled +=1
                pbar.update(1)


    final_tracking_data = {
        'predictions': predictions,
        'certainties': certainties,
        'synchronisation': synchronisation,
        'pre_activations': pre_activations,
        'post_activations': post_activations,
        'attention_tracking': attention_tracking,
        'embedding_tracking': embedding_tracking,
        'inputs_to_model': inputs_b,
        'targets': targets_b,
        'question_readable': question_readable_b
    }

    stacked = np.stack(all_post_activations, 1)
    umap_features = stacked.reshape(-1, stacked.shape[-1])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=1,
        spread=1,
        metric='cosine',
        local_connectivity=1
    )
    positions = reducer.fit_transform(umap_features.T)
    return final_tracking_data, positions

def run_model_and_make_gif(checkpoint_path, save_path, device):

    checkpoint = load_checkpoint(checkpoint_path, device)
    model_args = get_model_args_from_checkpoint(checkpoint)
    model = prepare_model(model_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    final_tracking_data, positions = run_umap(model, model_args, device)

    predictions = final_tracking_data['predictions'].detach().cpu().numpy()
    post_activations = final_tracking_data['post_activations']
    attention_tracking = final_tracking_data['attention_tracking']
    embedding_tracking = final_tracking_data['embedding_tracking']
    inputs = final_tracking_data['inputs_to_model']
    targets = final_tracking_data['targets'].detach().cpu().numpy()
    question_readable = final_tracking_data['question_readable']

    attention = reshape_attention_weights(attention_tracking)


    T = predictions.shape[-1]
    B = predictions.shape[0]

    gif_inputs = torch.zeros((T, B, 1, 32, 32), device=device)
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

    make_qamnist_gif(
        predictions=predictions,
        targets=targets,
        post_activations=post_activations,
        input_gates=attention,
        inputs_to_model=gif_inputs.detach().cpu().numpy(),
        save_path=save_path,
        question_readable=question_readable,
        umap_positions=positions
    )

    pass

if __name__ == "__main__":

    CHECKPOINT_PATH = "logs/qamnist/run1/ctm_10/checkpoint_300000.pt"
    SAVE_PATH = "tasks/qamnist/analysis/outputs/blog_gifs"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_model_and_make_gif(CHECKPOINT_PATH, SAVE_PATH, device)