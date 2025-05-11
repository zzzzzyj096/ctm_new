
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import imageio
from scipy.special import softmax
sns.set_style('darkgrid')
mpl.use('Agg')




def make_qamnist_gif(predictions, certainties, targets, pre_activations, post_activations, input_gates, inputs_to_model, filename, question_readable=None):

    # Config
    batch_index = 0
    n_neurons_to_visualise = 16
    figscale = 0.28
    n_steps = len(pre_activations)
    heatmap_cmap = sns.color_palette("viridis", as_cmap=True)
    frames = []

    these_pre_acts = pre_activations[:, batch_index, :] # Shape: (T, H)
    these_post_acts = post_activations[:, batch_index, :] # Shape: (T, H)
    these_inputs = inputs_to_model[:, batch_index, :, :, :] # Shape: (T, C, H, W)
    these_input_gates = input_gates[:, batch_index, :, :] # Shape: (T, H, W)
    these_predictions = predictions[batch_index, :, :] # Shape: (C, T)
    these_certainties = certainties[batch_index, :, :] # Shape: (C, T)
    this_target = targets[batch_index] # Shape: (C)

    logits_min, logits_max = np.min(these_predictions), np.max(these_predictions)
    probs_min, probs_max = 0, 1

    class_labels = [str(i) for i in range(10)]
    pad = 0.1
    if question_readable:
        this_question = question_readable[batch_index]
        pad = 1.6
        class_labels = ["" for i in range(len(these_predictions))]

    # Create mosaic layout
    mosaic = [['img_data', 'img_data', 'attention', 'attention', 'logits', 'logits', 'probs', 'probs'] for _ in range(2)] + \
             [['img_data', 'img_data', 'attention', 'attention', 'logits', 'logits', 'probs', 'probs'] for _ in range(2)] + \
             [['certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty']] + \
             [[f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}'] for ti in range(n_neurons_to_visualise)] 
             
    for stepi in range(n_steps):
        fig_gif, axes_gif = plt.subplot_mosaic(mosaic=mosaic, figsize=(31*figscale*8/4, 76*figscale))

        if question_readable:
            if this_question:
                fig_gif.suptitle(this_question, fontsize=24)

        # Plot action log probs
        colors = [('g' if i == this_target else ('b' if e >= 0 else 'r')) for i, e in enumerate(these_predictions[:, stepi])]
        sort_idxs = np.arange(len(these_predictions[:, stepi]))
        bars = axes_gif['logits'].bar(np.arange(len(these_predictions[:,stepi])), these_predictions[:,stepi][sort_idxs], color=np.array(colors)[sort_idxs], width=0.9, alpha=0.5)
        axes_gif['logits'].set_title('Logits')
        axes_gif['logits'].axis('off')
        for bar, label in zip(bars, class_labels):
            x = bar.get_x() + bar.get_width() / 2
            axes_gif['logits'].annotate(label, xy=(x, 0), xytext=(1, 0), 
                                            textcoords="offset points", 
                                            ha='center', va='bottom', rotation=90)
        axes_gif['logits'].set_ylim([logits_min - 0.1 * abs(logits_min), logits_max + 0.1 * abs(logits_max)])
    
        # Add probability plot
        probs = softmax(these_predictions[:, stepi])
        bars_prob = axes_gif['probs'].bar(np.arange(len(probs)), probs[sort_idxs], 
                                        color=np.array(colors)[sort_idxs], width=0.9, alpha=0.5)
        axes_gif['probs'].set_title('Probabilities')
        axes_gif['probs'].axis('off')
        axes_gif['probs'].set_ylim([0, 1])
        for bar, label in zip(bars_prob, class_labels):
            x = bar.get_x() + bar.get_width() / 2
            axes_gif['probs'].annotate(label, xy=(x, 0), xytext=(1, 0), textcoords="offset points", ha='center', va='bottom', rotation=90)
                                 
        axes_gif['probs'].set_ylim([probs_min, probs_max])

        # Add certainty plot
        axes_gif['certainty'].plot(np.arange(n_steps), these_certainties[1], 'k-', linewidth=2)
        axes_gif['certainty'].set_xlim([0, n_steps-1])
        axes_gif['certainty'].axvline(x=stepi, color='black', linewidth=1, alpha=0.5)
        axes_gif['certainty'].set_xticklabels([])
        axes_gif['certainty'].set_yticklabels([])
        axes_gif['certainty'].grid(False)

        # Plot neuron traces
        for neuroni in range(n_neurons_to_visualise):
            ax = axes_gif[f'trace_{neuroni}']

            pre_activation = these_pre_acts[:, neuroni]
            post_activation = these_post_acts[:, neuroni]
            
            ax_pre = ax.twinx()
            
            pre_min, pre_max = np.min(pre_activation), np.max(pre_activation)
            post_min, post_max = np.min(post_activation), np.max(post_activation)
            
            ax_pre.plot(np.arange(n_steps), pre_activation, 
                        color='grey', 
                        linestyle='--', 
                        linewidth=1, 
                        alpha=0.4,
                        label='Pre-activation')
            
            color = 'blue' if neuroni % 2 else 'red'
            ax.plot(np.arange(n_steps), post_activation,
                    color=color,
                    linestyle='-',
                    linewidth=2,
                    alpha=1.0,
                    label='Post-activation')

            ax.set_xlim([0, n_steps-1])
            ax_pre.set_xlim([0, n_steps-1])
            ax.set_ylim([post_min, post_max])
            ax_pre.set_ylim([pre_min, pre_max])

            ax.axvline(x=stepi, color='black', linewidth=1, alpha=0.5)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)

            ax_pre.set_xticklabels([])
            ax_pre.set_yticklabels([])
            ax_pre.grid(False)

        # Show input image
        this_image = these_inputs[stepi].transpose(1, 2, 0)
        # this_image = (this_image - this_image.min()) / (this_image.max() - this_image.min() + 1e-8)  # Normalize to [0,1]
        axes_gif['img_data'].imshow(this_image, cmap='binary', vmin=0, vmax=1)
        axes_gif['img_data'].grid(False) 
        axes_gif['img_data'].set_xticks([])
        axes_gif['img_data'].set_yticks([])

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
        axes_gif['attention'].imshow(input_heatmap, vmin=0, vmax=1)
        axes_gif['attention'].axis('off')
        axes_gif['attention'].set_title('Attention')

        # Save frames
        fig_gif.tight_layout(pad=pad)
        if stepi == 0:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame0.png', dpi=100)
        if stepi == 1:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame1.png', dpi=100)
        if stepi == n_steps-1:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame-1.png', dpi=100)

        # Convert to frame
        canvas = fig_gif.canvas
        canvas.draw()
        image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3]
        frames.append(image_numpy)
        plt.close(fig_gif)

    imageio.mimsave(filename, frames, fps=15, loop=100)

    pass