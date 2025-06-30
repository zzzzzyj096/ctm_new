
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt 
import imageio

from tqdm.auto import tqdm

def find_center_of_mass(array_2d):
    """
    Alternative implementation using np.average and meshgrid.
    This version is generally faster and more concise.

    Args:
        array_2d: A 2D numpy array of values between 0 and 1.

    Returns:
        A tuple (x, y) representing the coordinates of the center of mass.
    """
    total_mass = np.sum(array_2d)
    if total_mass == 0:
      return (np.nan, np.nan)

    y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
    x_center = np.average(x_coords, weights=array_2d)
    y_center = np.average(y_coords, weights=array_2d)
    return (round(y_center, 4), round(x_center, 4))

def draw_path(x, route, valid_only=False, gt=False, cmap=None):
    """
    Draws a path on a maze image based on a given route.

    Args:
        maze: A numpy array representing the maze image.
        route: A list of integers representing the route, where 0 is up, 1 is down, 2 is left, and 3 is right.
        valid_only: A boolean indicating whether to only draw valid steps (i.e., steps that don't go into walls).

    Returns:
        A numpy array representing the maze image with the path drawn in blue.
    """
    x = np.copy(x)
    start = np.argwhere((x == [1, 0, 0]).all(axis=2))
    end = np.argwhere((x == [0, 1, 0]).all(axis=2))
    if cmap is None:
        cmap = plt.get_cmap('winter') if not valid_only else  plt.get_cmap('summer')

    # Initialize the current position
    current_pos = start[0]

    # Draw the path
    colors = cmap(np.linspace(0, 1, len(route)))
    si = 0
    for step in route:
        new_pos = current_pos
        if step == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif step == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif step == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif step == 3:  # Right
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif step == 4:  # Do nothing
            pass
        else:
            raise ValueError("Invalid step: {}".format(step))

        # Check if the new position is valid
        if valid_only:
            try:
                if np.all(x[new_pos] == [0,0,0]):  # Check if it's a wall
                    continue  # Skip this step if it's invalid
            except IndexError:
                continue  # Skip this step if it's out of bounds

        # Draw the step
        if new_pos[0] >= 0 and new_pos[0] < x.shape[0] and new_pos[1] >= 0 and new_pos[1] < x.shape[1]:
            if not ((x[new_pos] == [1,0,0]).all() or (x[new_pos] == [0,1,0]).all()):
                colour = colors[si][:3]
                si += 1
                x[new_pos] = x[new_pos]*0.5 + colour*0.5

        # Update the current position
        current_pos = new_pos
        # cv2.imwrite('maze2.png', x[:,:,::-1]*255)

    return x

def make_maze_gif(inputs, predictions, targets, attention_tracking, save_location, verbose=True):
    """
    Expect inputs, predictions, targets as numpy arrays
    """
    route_steps = []
    route_colours = []
    solution_maze = draw_path(np.moveaxis(inputs, 0, -1), targets)
    
    n_heads = attention_tracking.shape[1]
    mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_1', 'head_2', 'head_3', 'head_4', 'head_5', 'head_6', 'head_7'],
              ['head_8', 'head_9', 'head_10', 'head_11', 'head_12', 'head_13', 'head_14', 'head_15'],
              ]
    if n_heads == 8:
        mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_1', 'head_2', 'head_3', 'head_4', 'head_5', 'head_6', 'head_7'],
              ]
    elif n_heads == 4:
        mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_0', 'head_1', 'head_1', 'head_2', 'head_2', 'head_3', 'head_3'],
              ['head_0', 'head_0', 'head_1', 'head_1', 'head_2', 'head_2', 'head_3', 'head_3'],
              ]

    img_aspect = 1
    figscale = 1
    aspect_ratio = (len(mosaic[0]) * figscale, len(mosaic) * figscale * img_aspect) # W, H
    
    route_steps = [np.unravel_index(np.argmax((inputs == np.reshape(np.array([1, 0, 0]), (3, 1, 1))).all(0)), inputs.shape[1:])]  # Starting point
    frames = []
    cmap = plt.get_cmap('gist_rainbow')
    cmap_viridis = plt.get_cmap('viridis')
    step_linspace = np.linspace(0, 1, predictions.shape[-1])  # For sampling colours
    with tqdm(total=predictions.shape[-1], initial=0, leave=True, position=1, dynamic_ncols=True) as pbar:
        if verbose: pbar.set_description('Processing frames for maze plotting')
        for stepi in np.arange(0, predictions.shape[-1], 1):
            fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)
            for ax in axes.values():
                ax.axis('off')
            guess_maze = draw_path(np.moveaxis(inputs, 0, -1), predictions.argmax(1)[:,stepi], cmap=cmap)
            attention_now = attention_tracking[stepi]
            for hi in range(min((attention_tracking.shape[1], 16))):
                ax = axes[f'head_{hi}']
                attn = attention_tracking[stepi, hi]
                attn = (attn - attn.min())/(np.ptp(attn))
                ax.imshow(attn, cmap=cmap_viridis)
            # Upsample attention just for visualisation
            aggregated_attention = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), inputs.shape[-1], mode='bilinear')[0].mean(0).numpy()
            
            # Get approximate center of mass
            com_attn = np.copy(aggregated_attention)
            com_attn[com_attn < np.percentile(com_attn, 96)] = 0.0
            aggregated_attention[aggregated_attention < np.percentile(aggregated_attention, 80)] = 0.0
            route_steps.append(find_center_of_mass(com_attn))


            colour = list(cmap(step_linspace[stepi]))
            route_colours.append(colour)

            mapped_attention = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), inputs.shape[-1], mode='bilinear')[0].mean(0).numpy()
            mapped_attention = (mapped_attention - mapped_attention.min())/np.ptp(mapped_attention)
            # np.clip(guess_maze * (1-mapped_attention[...,np.newaxis]*0.5) + (cmap_viridis(mapped_attention)[:,:,:3] * mapped_attention[...,np.newaxis])*1.3, 0, 1)
            overlay_img = np.clip(guess_maze * (1-mapped_attention[...,np.newaxis]*0.6) + (cmap_viridis(mapped_attention)[:,:,:3] * mapped_attention[...,np.newaxis])*1.1, 0, 1)#np.clip((np.copy(guess_maze)*(1-aggregated_attention[:,:,np.newaxis])*0.7 + (aggregated_attention[:,:,np.newaxis]*3 * np.reshape(np.array(colour)[:3], (1, 1, 3)))), 0, 1)
            axes['overlay'].imshow(overlay_img)

            y_coords, x_coords = zip(*route_steps)
            y_coords = inputs.shape[-1] - np.array(list(y_coords))-1

            
            axes['route'].imshow(np.flip(np.moveaxis(inputs, 0, -1), axis=0), origin='lower')
            # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
            arrow_scale = 2
            for i in range(len(route_steps)-1):
                dx = x_coords[i+1] - x_coords[i]
                dy = y_coords[i+1] - y_coords[i]
                axes['route'].arrow(x_coords[i], y_coords[i], dx, dy, linewidth=2*arrow_scale, head_width=0.2*arrow_scale, head_length=0.3*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)
                
            fig.tight_layout(pad=0.1) # Adjust spacing

            # Render the plot to a numpy array
            canvas = fig.canvas
            canvas.draw()
            image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3] # Get RGB

            frames.append(image_numpy) # Add to list for GIF

            # fig.savefig(f'{save_location}/frame.png', dpi=200)

            plt.close(fig) 

            # # frame = np.clip((np.copy(guess_maze)*0.5 + (aggregated_attention[:,:,np.newaxis] * np.reshape(np.array(colour)[:3], (1, 1, 3)))), 0, 1)
            # frame = torch.nn.functional.interpolate(torch.from_numpy(frame).permute(2,0,1).unsqueeze(0), 256)[0].permute(1,2,0).detach().cpu().numpy()
            # frames.append((frame*255).astype(np.uint8))
            pbar.update(1)


    y_coords, x_coords = zip(*route_steps)
    y_coords = inputs.shape[-1] - np.array(list(y_coords))-1

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.flip(np.moveaxis(inputs, 0, -1), axis=0), origin='lower')
    # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
    arrow_scale = 2
    for i in range(len(route_steps)-1):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        plt.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=2*arrow_scale, head_width=0.2*arrow_scale, head_length=0.3*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)

    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(f'{save_location}/route_approximation.png', dpi=200)
    imageio.mimsave(f'{save_location}/prediction.gif', frames, fps=15, loop=100)
    plt.close(fig)
