
import torch
import os
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.special import softmax
import matplotlib.cm as cm
from data.custom_datasets import ParityDataset
import umap
from tqdm import tqdm


from models.utils import reshape_predictions
from tasks.parity.utils import reshape_inputs
from tasks.parity.analysis.run import build_model_from_checkpoint_path

from tasks.image_classification.plotting import save_frames_to_mp4


def make_parity_gif(
    predictions,
    targets,
    post_activations,
    attention_weights,
    inputs_to_model,
    save_path,
    umap_positions,
    umap_point_scaler=1.0,
):
    batch_index = 0
    figscale = 0.32 
    n_steps, n_heads, seqLen = attention_weights.shape[:3]
    grid_side = int(np.sqrt(seqLen))
    frames = []

    inputs_this_batch  = inputs_to_model[:, batch_index]
    preds_this_batch   = predictions[batch_index]
    targets_this_batch = targets[batch_index]
    post_act_this_batch = post_activations[:, batch_index]

    # build a flexible mosaic
    mosaic = [
        [f"att_0", f"in_0", "probs", "probs", "target", "target"],
        [f"att_1", f"in_1", "probs", "probs", "target", "target"],
    ]
    for h in range(2, n_heads):
        mosaic.append(
            [f"att_{h}", f"in_{h}", "umap", "umap",
             "umap", "umap"]
        )

    for t in range(n_steps):
        rows      = len(mosaic)
        cell_size = figscale * 4
        fig_h     = rows * cell_size

        fig, ax = plt.subplot_mosaic(
            mosaic,
            figsize=(6 * cell_size, fig_h),
            constrained_layout=False,
            gridspec_kw={'wspace': 0.05, 'hspace': 0.05},  # small gaps
        )
        # restore a little margin
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        # probabilities heatmap
        logits_t = preds_this_batch[:, :, t]
        probs_t  = softmax(logits_t, axis=1)[:, 0].reshape(grid_side, grid_side)
        ax["probs"].imshow(probs_t, cmap="gray", vmin=0, vmax=1)
        ax["probs"].axis("off")

        # target overlay
        ax["target"].imshow(
            targets_this_batch.reshape(grid_side, grid_side),
            cmap="gray_r", vmin=0, vmax=1
        )
        ax["target"].axis("off")
        ax["target"].grid(which="minor", color="black", linestyle="-", linewidth=0.5)

        z = post_act_this_batch[t]
        low, high = np.percentile(z, 5), np.percentile(z, 95)
        z_norm = np.clip((z - low) / (high - low), 0, 1)
        point_sizes = (np.abs(z_norm - 0.5) * 100 + 5) * umap_point_scaler
        cmap = plt.get_cmap("Spectral")
        ax["umap"].scatter(
            umap_positions[:, 0],
            umap_positions[:, 1],
            s=point_sizes,
            c=cmap(z_norm),
            alpha=0.8
        )
        ax["umap"].axis("off")


        # normalize attention
        att_t = attention_weights[t, :, :]
        a_min, a_max = att_t.min(), att_t.max()
        if not np.isclose(a_min, a_max):
            att_t = (att_t - a_min) / (a_max - a_min + 1e-8)
        else:
            att_t = np.zeros_like(att_t)

        # input image for arrows
        img_t = inputs_this_batch[t].transpose(1, 2, 0)

        if t == 0:
            route_history = [[] for _ in range(n_heads)]

        img_h, img_w = img_t.shape[:2]
        cell_h = img_h // grid_side
        cell_w = img_w // grid_side

        for h in range(n_heads):
            head_map = att_t[h].reshape(grid_side, grid_side)
            ax[f"att_{h}"].imshow(head_map, cmap="viridis", vmin=0, vmax=1)
            ax[f"att_{h}"].axis("off")
            ax[f"in_{h}"].imshow(img_t, cmap="gray", vmin=0, vmax=1)
            ax[f"in_{h}"].axis("off")

            # track argmax center
            flat_idx = np.argmax(head_map)
            gy, gx = divmod(flat_idx, grid_side)
            cx = int((gx + 0.5) * cell_w)
            cy = int((gy + 0.5) * cell_h)
            route_history[h].append((cx, cy))

            cmap_steps = plt.colormaps.get_cmap("Spectral")
            colors = [cmap_steps(i / (n_steps - 1)) for i in range(n_steps)]
            for i in range(len(route_history[h]) - 1):
                x0, y0 = route_history[h][i]
                x1, y1 = route_history[h][i + 1]
                color = colors[i]
                is_last = (i == len(route_history[h]) - 2)
                style   = '->' if is_last else '-'
                lw      = 2.0 if is_last else 1.6
                alpha   = 1.0 if is_last else 0.9
                scale   = 10  if is_last else 1

                # draw arrow
                arr = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle=style,
                    linewidth=lw,
                    mutation_scale=scale,
                    alpha=alpha,
                    facecolor=color,
                    edgecolor=color,
                    shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='round',
                    zorder=3 if is_last else 2,
                    clip_on=False,
                )
                ax[f"in_{h}"].add_patch(arr)

                ax[f"in_{h}"].scatter(
                    x1, y1,
                    marker='x',
                    s=40,
                    color=color,
                    linewidths=lw,
                    zorder=4
                )

        canvas = fig.canvas
        canvas.draw()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h   = canvas.get_width_height()
        frames.append(frame.reshape(h, w, 4)[..., :3])
        plt.close(fig)

    # save gif
    imageio.mimsave(f"{save_path}/activation.gif", frames, fps=15, loop=0)

    # save mp4
    save_frames_to_mp4(
        [fm[:, :, ::-1] for fm in frames],  # RGB→BGR
        f"{save_path}/activation.mp4",
        fps=15,
        gop_size=1,
        preset="slow"
    )

def run_umap(model, testloader):
    all_post_activations = []
    point_counts = 150
    sampled = 0
    with tqdm(total=point_counts, desc="Collecting UMAP data") as pbar:
        for inputs, _ in testloader:
            for i in range(inputs.size(0)):
                if sampled >= point_counts:
                    break
                input_i = inputs[i].unsqueeze(0).to(device)
                _, _, _, _, post_activations, _ = model(input_i, track=True)
                all_post_activations.append(post_activations)
                sampled += 1
                pbar.update(1)
            if sampled >= point_counts:
                break

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
    return positions


def run_model_and_make_gif(checkpoint_path, save_path, device):

    parity_sequence_length = 64
    iterations = 75

    test_data = ParityDataset(sequence_length=parity_sequence_length, length=10000)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0, drop_last=False)


    model, _ = build_model_from_checkpoint_path(checkpoint_path, "ctm", device=device)

    input = torch.randint(0, 2, (64,), dtype=torch.float32, device=device) * 2 - 1
    input = input.unsqueeze(0)

    target = torch.cumsum((input == -1).to(torch.long), dim=1) % 2
    target = target.unsqueeze(0)

    positions = run_umap(model, testloader)

    model.eval()
    with torch.inference_mode():
        predictions, _, _, _, post_activations, attention = model(input, track=True)
        predictons = reshape_predictions(predictions, prediction_reshaper=[parity_sequence_length, 2])
        input_images = reshape_inputs(input, iterations, grid_size=int(math.sqrt(parity_sequence_length)))

        make_parity_gif(
            predictions=predictons.detach().cpu().numpy(),
            targets=target.detach().cpu().numpy(),
            post_activations=post_activations,
            attention_weights=attention.squeeze(1).squeeze(2),
            inputs_to_model=input_images,
            save_path=save_path,
            umap_positions=positions,
            umap_point_scaler=1.0,  
        )



if __name__ == "__main__":
    
    CHECKPOINT_PATH = "checkpoints/parity/run1/ctm_75_25/checkpoint_200000.pt"
    SAVE_PATH = f"tasks/parity/analysis/outputs/blog_gifs/"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_model_and_make_gif(CHECKPOINT_PATH, SAVE_PATH, device)
