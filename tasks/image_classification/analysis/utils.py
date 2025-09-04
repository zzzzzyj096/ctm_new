import torch
import torch.nn.functional as F
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.simpleRNN import SimpleNet
from models.simpleEIRNN import Net
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
        model = SimpleNet(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
            prediction_reshaper=prediction_reshaper,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model_type == 'eirnn':
        model = Net(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
            prediction_reshaper=prediction_reshaper,
            num_layers=args.num_layers,
        ).to(device)
    else:
        raise ValueError(f"Model must be either ctm or lstm, not {args.model_type}")

    return model