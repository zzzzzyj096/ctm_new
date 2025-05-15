import os
import torch
from tqdm import tqdm
import numpy as np
import umap
from matplotlib import pyplot as plt
import imageio
from scipy.special import softmax


from tasks.rl.train import Agent
from tasks.rl.analysis.run import get_training_data_from_checkpoint_path, get_size_action_space, prepare_env
from tasks.rl.utils import combine_tracking_data
from tasks.image_classification.plotting import save_frames_to_mp4 import save_frames_to_mp4


def load_model(agent, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])
    pass

def interpolate_post_activations(arrays, target_length):
    interpolated = []
    for arr in arrays:
        arr = arr.squeeze(1)
        T, D = arr.shape
        if T == target_length:
            interpolated.append(arr)
            continue
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_length)
        arr_interp = np.array([
            np.interp(x_new, x_old, arr[:, d]) for d in range(D)
        ]).T
        interpolated.append(arr_interp)
    return interpolated

def make_rl_gif(post_activations, inputs_to_model, action_probs, actions, save_path, umap_positions, umap_point_scaler=1.0):

    batch_index = 0
    figscale = 0.32
    n_steps = action_probs.shape[0]
    frames = []

    inputs_this_batch = inputs_to_model  # Already shape (T, H, W, C)

    class_labels = ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"]

    post_act_this_batch = post_activations[:, batch_index]

    mosaic = [
        [f"obs", f"obs", f"obs", f"obs", "probs", "probs", "probs", "probs"],
        [f"obs", f"obs", f"obs", f"obs", "probs", "probs","probs", "probs"],
    ]
    for _ in range(2, 8):
        mosaic.append(
            ["umap", "umap", "umap", "umap", "umap", "umap", "umap", "umap"]
        )

    for t in range(n_steps):
        rows      = len(mosaic)
        cell_size = figscale * 4
        fig_h     = rows * cell_size

        probs_t = action_probs[t]

        fig, ax = plt.subplot_mosaic(
            mosaic,
            figsize=(6 * cell_size, fig_h),
            constrained_layout=False,
            gridspec_kw={'wspace': 0.05, 'hspace': 0.05},  # small gaps
        )
        # restore a little margin
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        this_image = inputs_this_batch[t]
        if this_image.dtype != np.uint8:
            this_image = (np.clip(this_image, 0, 1) * 255).astype(np.uint8)
        if this_image.shape[-1] == 1:
            this_image = np.repeat(this_image, 3, axis=-1)


        ax["obs"].imshow(this_image)
        ax["obs"].axis("off")

        probs_t = action_probs[t]
        colors = ['black' if i == actions[t] else 'gray' for i in range(len(probs_t))]
        bars = ax["probs"].bar(np.arange(len(probs_t)), probs_t, color=colors, width=0.9, alpha=0.8)
        ax["probs"].axis('off')
        for bar, label in zip(bars, class_labels):
            x = bar.get_x() + bar.get_width() / 2
            ax["probs"].annotate(label, xy=(x, 0), xytext=(1, 0),
                                textcoords="offset points",
                                ha='center', va='bottom', rotation=90)
        ax["probs"].set_ylim([0, 1])

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


def run_umap(agent, model_args):

    all_post_activations = []
    point_counts = 150

    eval_env = prepare_env(model_args.env_id, model_args.max_environment_steps, mask_velocity=model_args.mask_velocity, render_mode="rgb_array")
    with tqdm(total=point_counts, desc="Collecting UMAP data") as pbar:
        for idx in range(point_counts):
            eval_next_obs, _ = eval_env.reset(seed=idx)
            eval_next_done = False
            eval_state = agent.get_initial_state(1)
            tracking_data_by_world_step = []
            for environment_step_i in range(model_args.max_environment_steps):
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
            all_post_activations.append(combined_tracking_data['post_activations'])
            pbar.update(1)

    all_post_activations = interpolate_post_activations(all_post_activations, all_post_activations[-1].shape[0])
    stacked = np.stack(all_post_activations, 1)
    umap_features = stacked.reshape(-1, stacked.shape[-1])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=40,
        min_dist=1,
        spread=1,
        metric='cosine',
        local_connectivity=1
    )
    positions = reducer.fit_transform(umap_features.T)
    return combined_tracking_data, positions

def run_model_and_make_gif(checkpoint_path, save_path, env_id, device):

    # Load the model
    _, _, _, _, _, model_args = get_training_data_from_checkpoint_path(checkpoint_path, device)
    agent = Agent(size_action_space=get_size_action_space(env_id), args=model_args, device=device).to(device)
    load_model(agent, checkpoint_path, device)

    # Run the umapping 
    tracking_data, positions = run_umap(agent, model_args)

    make_rl_gif(
        post_activations=tracking_data['post_activations'],
        inputs_to_model=tracking_data['inputs'],
        action_probs=tracking_data['action_probs'],
        actions=tracking_data['actions'],
        save_path=save_path,
        umap_positions=positions,
        umap_point_scaler=1.0,  
    )


    pass


if __name__ == "__main__":

    env_id = "MiniGrid-FourRooms-v0"

    CHECKPOINT_PATH = f"logs/rl/{env_id}/run1/ctm_2/checkpoint.pt"
    SAVE_PATH = f"tasks/rl/analysis/outputs/blog_gifs/{env_id}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_model_and_make_gif(CHECKPOINT_PATH, SAVE_PATH, env_id, device)
