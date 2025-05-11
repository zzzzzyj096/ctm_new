
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')
import imageio


def make_rl_gif(action_logits, action_probs, actions, values, rewards, pre_activations, post_activations, inputs, filename):

    n_steps = len(pre_activations)
    pre_activations = pre_activations[:,0,:]
    post_activations = post_activations[:,0,:]

    if action_logits.shape[1] == 5:
        class_labels = ['W', 'U', 'D', 'L', 'R']
    elif action_logits.shape[1] == 2:
        class_labels = ['L', 'R']
    else:
        class_labels = [str(i) for i in range(action_logits.shape[1])]

    max_target = len(class_labels)

    figscale = 0.28
    frames = []
    n_neurons_to_visualise = 15
    
    # Create mosaic layout
    mosaic = [['img_data', 'img_data', 'img_data', 'img_data', 'action_logits', 'action_logits', 'action_log_probs', 'action_log_probs'] for _ in range(2)] + \
            [['img_data', 'img_data', 'img_data', 'img_data', 'action_logits', 'action_logits', 'action_log_probs', 'action_log_probs'] for _ in range(2)] + \
            [['value', 'value', 'value', 'value', 'value', 'value', 'value', 'value']] + \
            [['reward', 'reward', 'reward', 'reward', 'reward', 'reward', 'reward', 'reward']] + \
            [[f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}'] for ti in range(n_neurons_to_visualise)]


    # Main plotting loop
    for stepi in range(n_steps):
        fig_gif, axes_gif = plt.subplot_mosaic(mosaic=mosaic, figsize=(31*figscale*8/4, 76*figscale))

        # Plot action logits
        these_action_logits = np.array(action_logits)[:, :max_target]
        colors = ['black' if i == actions[stepi] else ('b' if e >= 0 else 'r') 
                for i, e in enumerate(these_action_logits[stepi])]
        sort_idxs = np.arange(len(these_action_logits[stepi]))
        bars = axes_gif['action_logits'].bar(np.arange(len(these_action_logits[stepi][sort_idxs])), these_action_logits[stepi][sort_idxs], color=np.array(colors)[sort_idxs],width=0.9, alpha=0.5)
        axes_gif['action_logits'].axis('off')
        for bar, label in zip(bars, class_labels):
            x = bar.get_x() + bar.get_width() / 2
            axes_gif['action_logits'].annotate(label, xy=(x, 0), xytext=(1, 0), 
                                            textcoords="offset points", 
                                            ha='center', va='bottom', rotation=90)
        axes_gif['action_logits'].set_ylim([np.min(these_action_logits), np.max(these_action_logits)])


        # Plot action probs
        these_action_log_probs = np.array(action_probs)[:, :max_target]
        colors = ['black' if i == actions[stepi] else ('b' if e >= 0 else 'r') 
                for i, e in enumerate(these_action_log_probs[stepi])]
        sort_idxs = np.arange(len(these_action_log_probs[stepi]))
        bars = axes_gif['action_log_probs'].bar(np.arange(len(these_action_log_probs[stepi][sort_idxs])), these_action_log_probs[stepi][sort_idxs], color=np.array(colors)[sort_idxs],width=0.9, alpha=0.5)
        axes_gif['action_log_probs'].axis('off')
        for bar, label in zip(bars, class_labels):
            x = bar.get_x() + bar.get_width() / 2
            axes_gif['action_log_probs'].annotate(label, xy=(x, 0), xytext=(1, 0), 
                                            textcoords="offset points", 
                                            ha='center', va='bottom', rotation=90)
        axes_gif['action_log_probs'].set_ylim([0,1])

        # Plot value trace
        ax_value = axes_gif['value']
        ax_value.plot(np.arange(n_steps), values, 'b-', linewidth=2)
        ax_value.axvline(x=stepi, color='k', linewidth=2, alpha=0.3)
        ax_value.set_xticklabels([])
        ax_value.set_yticklabels([])
        ax_value.grid(False)
        ax_value.set_xlim([0, n_steps-1])
        
        # Plot reward trace
        ax_reward = axes_gif['reward']
        ax_reward.plot(np.arange(n_steps), rewards, 'g-', linewidth=2)
        ax_reward.axvline(x=stepi, color='k', linewidth=2, alpha=0.3)
        ax_reward.set_xticklabels([])
        ax_reward.set_yticklabels([])
        ax_reward.grid(False)
        ax_reward.set_xlim([0, n_steps-1])

        # Plot neuron traces
        for neuroni in range(n_neurons_to_visualise):
            ax = axes_gif[f'trace_{neuroni}']
            
            pre_activation = pre_activations[:, neuroni]
            post_activation = post_activations[:, neuroni]
            
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

            
        ax.set_xlim([0, n_steps-1])
        ax.set_xticklabels([])
        ax.grid(False)

        # Show input image
        this_image = inputs[stepi]
        axes_gif['img_data'].imshow(this_image, cmap='binary', vmin=0, vmax=1)
        axes_gif['img_data'].grid(False) 
        axes_gif['img_data'].set_xticks([])
        axes_gif['img_data'].set_yticks([])

        # Save frames
        fig_gif.tight_layout(pad=0.1)
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

