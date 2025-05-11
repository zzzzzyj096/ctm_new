from models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
from models.lstm_qamnist import LSTMBaseline
from data.custom_datasets import QAMNISTDataset
from torchvision import datasets
from torchvision import transforms
import numpy as np

def get_dataset(q_num_images, q_num_images_delta, q_num_repeats_per_input, q_num_operations, q_num_operations_delta):
    dataset_mean = 0.1307
    dataset_std = 0.3081    
    transform = transforms.Compose(
        [transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
        ])
    train_data = QAMNISTDataset(datasets.MNIST("data/", train=True, transform=transform, download=True), num_images=q_num_images, num_images_delta=q_num_images_delta, num_repeats_per_input=q_num_repeats_per_input, num_operations=q_num_operations, num_operations_delta=q_num_operations_delta)
    test_data = QAMNISTDataset(datasets.MNIST("data/", train=False, transform=transform, download=True), num_images=q_num_images, num_images_delta=q_num_images_delta, num_repeats_per_input=q_num_repeats_per_input, num_operations=q_num_operations, num_operations_delta=q_num_operations_delta)
    class_labels = [str(i) for i in np.arange(train_data.output_range[0], train_data.output_range[1]+1)]        
    return train_data, test_data, class_labels, dataset_mean, dataset_std

def prepare_model(args, device):
    if args.model_type == 'ctm':
        model = ContinuousThoughtMachineQAMNIST(
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
        out_dims=args.out_dims,
        prediction_reshaper=[-1],
        dropout=args.dropout,          
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
        iterations_per_digit=args.q_num_repeats_per_input,
        iterations_per_question_part=args.q_num_repeats_per_input,
        iterations_for_answering=args.q_num_answer_steps,
        ).to(device)
    elif args.model_type == 'lstm':
            model = LSTMBaseline(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,  
            heads=args.heads,
            out_dims=args.out_dims,
            prediction_reshaper=[-1],
            iterations_per_digit=args.q_num_repeats_per_input,
            iterations_per_question_part=args.q_num_repeats_per_input,
            iterations_for_answering=args.q_num_answer_steps,
        ).to(device)
    else:
        raise ValueError(f"Model must be either ctm or lstm, not {args.model_type}")

    return model
