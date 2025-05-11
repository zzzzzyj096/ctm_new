import os
import re
import math
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline

def prepare_model(prediction_reshaper, args, device):
    if args.model_type == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,  
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,  
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,  
            do_layernorm_nlm=args.do_normalisation,  
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,          
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
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
    else:
        raise ValueError(f"Model must be either ctm or lstm, not {args.model_type}")

    return model

def reshape_attention_weights(attention_weights):
    T, B = attention_weights.shape[0], attention_weights.shape[1]
    grid_size = math.sqrt(attention_weights.shape[-1])
    assert grid_size.is_integer(), f'Grid size should be a perfect square, but got {attention_weights.shape[-1]}'
    H_ATTENTION = W_ATTENTION = int(grid_size)
    attn_weights_reshaped = attention_weights.reshape(T, B, -1, H_ATTENTION, W_ATTENTION)
    return attn_weights_reshaped.mean(2)

def reshape_inputs(inputs, iterations, grid_size):
    reshaped_inputs = inputs.reshape(-1, grid_size, grid_size).unsqueeze(0).repeat(iterations, 1, 1, 1).unsqueeze(2).detach().cpu().numpy()
    return reshaped_inputs

def get_where_most_certain(certainties):
    return certainties[:,1].argmax(-1)

def parse_folder_name(folder_path):
    folder = os.path.basename(folder_path)

    lstm_match = re.match(r"lstm_(\d+)", folder)
    if lstm_match:
        model_type = "LSTM"
        iters = int(lstm_match.group(1))
        return f"{model_type}, {iters} Iters.", model_type, iters

    ctm_full_match = re.match(r"ctm(\d+)_(\d+)", folder)
    if ctm_full_match:
        model_type = "CTM"
        iters = int(ctm_full_match.group(1))
        mem_len = int(ctm_full_match.group(2))
        return f"{model_type}, {iters} Iters., {mem_len} Mem. Len.", model_type, iters

    ctm_partial_match = re.match(r"ctm_(\d+)", folder)
    if ctm_partial_match:
        model_type = "CTM"
        iters = int(ctm_partial_match.group(1))
        return f"{model_type}, {iters} Iters.", model_type, iters

    return "Unknown", None, None
