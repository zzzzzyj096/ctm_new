import torch
import torch.nn.functional as F
import re
import os

def compute_decay(T, params, clamp_lims=(0, 15)):
    """
    This function computes exponential decays for learnable synchronisation 
    interactions between pairs of neurons. 
    """
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'
    
    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.

    Args:
        tensor: A PyTorch tensor of shape (B, D, H, W).

    Returns:
        A PyTorch tensor of shape (B, D, H, W, 2) with the last dimension
        representing the 2D coordinates within the HW dimensions.
    """
    B, H, W = x.shape
    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the 
    final dimension.

    Args:
      logits: A PyTorch tensor of logits. 

    Returns:
      A PyTorch tensor containing the normalized entropy values.
    """

    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy
    
def reshape_predictions(predictions, prediction_reshaper):
    B, T = predictions.size(0), predictions.size(-1)
    new_shape = [B] + prediction_reshaper + [T]
    rehaped_predictions = predictions.reshape(new_shape)
    return rehaped_predictions

def get_all_log_dirs(root_dir):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".pt") for f in filenames):
            folders.append(dirpath)
    return folders

def get_latest_checkpoint(log_dir):
    files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
    return os.path.join(log_dir, max(files, key=lambda f: int(re.search(r'\d+', f).group()))) if files else None

def get_latest_checkpoint_file(filepath, limit=300000):
    checkpoint_files = get_checkpoint_files(filepath)
    checkpoint_files = [
        f for f in checkpoint_files if int(re.search(r'checkpoint_(\d+)\.pt', f).group(1)) <= limit
    ]
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]

def get_checkpoint_files(filepath):
    regex = r'checkpoint_(\d+)\.pt'
    files = [f for f in os.listdir(filepath) if re.match(regex, f)]
    files = sorted(files, key=lambda f: int(re.search(regex, f).group(1)))
    return [os.path.join(filepath, f) for f in files]

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def get_model_args_from_checkpoint(checkpoint):
    if "args" in checkpoint:
        return(checkpoint["args"])
    else:
        raise ValueError("Checkpoint does not contain saved args.")

def get_accuracy_and_loss_from_checkpoint(checkpoint, device="cpu"):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies
