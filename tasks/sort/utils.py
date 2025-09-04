import torch
import torch.nn.functional as F
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.simpleRNN_sort import SimpleNetSORT
from models.simpleEIRNN_sort import NetSORT
def prepare_model(prediction_reshaper, args, device):
    if args.model_type == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            memory_length=args.memory_length,
            memory_hidden_dims=args.memory_hidden_dims,
            backbone_type = args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            synapse_depth =args.synapse_depth,  #在ContinuousThoughtMachineSIMPLE要注释掉
            deep_nlms=args.deep_memory, #在ContinuousThoughtMachineSIMPLE要注释掉
            do_layernorm_nlm=args.do_normalisation, #在ContinuousThoughtMachineSIMPLE要注释掉
        ).to(device)
    elif args.model_type == 'lstm':
        model = LSTMBaseline(
            iterations=args.iterations,
            d_model=args.d_model,
            num_layers=1,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'simplernn':  # 添加新的模型类型
        model = SimpleNetSORT(
            num_layers=1,  # 固定为1层
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'eirnn':
        model = NetSORT(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            num_layers=1,  # EIRNN 是单层的
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Model must be either ctm or lstm, not {args.model_type}")

    return model
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