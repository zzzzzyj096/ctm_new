import torch
import torch.nn.functional as F

def decode_predictions(predictions, blank_label=0, return_wait_times=False):
    """
    Decodes the predictions using greedy decoding (best path), correctly handling duplicates.

    Args:
        predictions: A tensor of shape [B, C, L] representing the logits.
        blank_label: The index of the blank label.

    Returns:
        A list of tensors, where each tensor is the decoded sequence.
    """

    batch_size, num_classes, prediction_length = predictions.shape
    decoded_sequences = []
    wait_times_all = []
    probs = F.softmax(predictions, dim=1)  # Probabilities
    for b in range(batch_size):
        best_path = torch.argmax(probs[b], dim=0)  # Best path indices
        decoded = []
        wait_times = []
        
        prev_char = -1  # Keep track of the previous character
        wait_time_now = 0
        for t in range(prediction_length):
            char_idx = best_path[t].item()  # Get index as integer
            if char_idx != blank_label and char_idx != prev_char:  # Skip blanks and duplicates
                decoded.append(char_idx)
                prev_char = char_idx  # Update previous character
                wait_times.append(wait_time_now)
                wait_time_now = 0
            else:
                wait_time_now += 1
        decoded_sequences.append(torch.tensor(decoded, device=predictions.device))
        if return_wait_times: wait_times_all.append(torch.tensor(wait_times, device=predictions.device))

    if return_wait_times: return decoded_sequences, wait_times_all

    return decoded_sequences

def compute_ctc_accuracy(predictions, targets, blank_label=0):
    """
    Computes the accuracy of the predictions given the targets, considering CTC decoding.

    Args:
        predictions: A tensor of shape [B, C, L] representing the logits.
        targets: A list of tensors, each of shape [T_i], representing a target sequence.
        blank_label: The index of the blank label.

    Returns:
        The accuracy (a float).
    """

    batch_size, num_classes, prediction_length = predictions.shape
    total_correct = 0

    # 1. Get predicted sequences (decoded from logits):
    predicted_sequences = decode_predictions(predictions, blank_label)

    # 2. Compare predicted sequences to targets:
    for i in range(batch_size):
        target = targets[i]
        predicted = predicted_sequences[i]

        if torch.equal(predicted, target):  # Direct comparison of tensors
            total_correct += 1

    accuracy = total_correct / batch_size if batch_size > 0 else 0.0
    return accuracy