import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import Squeeze

from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class Identity(nn.Module):
    """Identity Module."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """Squeeze Module."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    """SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM."""
    def __init__(self, in_dims, out_dims, N, dropout=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))

    def forward(self, x):
            out = self.dropout(x)
            out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1
            out = out.squeeze(-1)
            return out

class ContinuousThoughtMachineSIMPLE(nn.Module):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 memory_length,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 out_dims,
                 memory_hidden_dims,
                 dropout=0.0,
                 neuron_select_type='random'
                 ):
        super(ContinuousThoughtMachineSIMPLE, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.heads = 0
        self.n_synch_out = n_synch_out
        self.n_synch_action = 0
        self.out_dims = out_dims
        self.memory_length = memory_length
        self.memory_hidden_dims = memory_hidden_dims
        self.neuron_select_type = neuron_select_type

        # --- Input Processing  ---
        #self.backbone = nn.Sequential(
            #nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(d_input),
            #nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            #nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(d_input),
            #nn.ReLU(),
            #nn.MaxPool2d(2, 2),
        #)
        self.attention = None
        self.kv_proj = None
        self.q_proj = None

        # --- Core CTM Modules ---
        self.synapses = nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        self.trace_processor = nn.Sequential(
            SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model, dropout=dropout),
            nn.GLU(),
            SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model, dropout=dropout),
            nn.GLU(),
            Squeeze(-1)
        )

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(
                torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))),
                requires_grad=True
            ))

        self.register_parameter('start_trace', nn.Parameter(
            torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length))),
            requires_grad=True
        ))

        # --- Synchronisation ---
        self.synch_representation_size_action = None
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)

        for synch_type, size in [('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)]:
            print(f"Synch representation size {synch_type}: {size}")

        self.set_synchronisation_parameters('out', self.n_synch_out)


        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def set_synchronisation_parameters(self, synch_type: str):
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))


    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == 'action': # Get action parameters
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out': # Get input parameters
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right

        if self.neuron_select_type in ('first-last', 'random'):
            # For first-last and random, we compute the pairwise sync between all selected neurons
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action': # Use last n_synch neurons for action
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out': # Use first n_synch neurons for out
                    selected_left = selected_right = activated_state[:, :n_synch]
            else: # Use the randomly selected neurons
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]

            # Compute outer product of selected neurons
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            # Resulting matrix is symmetric, so we only need the upper triangle
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]

        elif self.neuron_select_type == 'random-pairing':
            # For random-pairing, we compute the sync between specific pairs of neurons
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        input_features = self.backbone(x)
        kv = self.kv_proj(input_features.flatten(2).transpose(1, 2))
        return kv

    def compute_certainty(self, current_prediction):
        ne = compute_normalized_entropy(current_prediction)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):

        assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
        left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):

        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):

        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    def verify_args(self):

        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"

        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"

        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"

        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):

        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size
    ##--------------------------------------------------------------------------------------
    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        attention_tracking = []

        # --- For SORT: no need to featurise data ---

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)  # Shape: (B, H)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Initialise Recurrent Synch Values  ---
        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out,
                                                                          synch_type='out')
        # Compute learned weighting for synchronisation

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            pre_synapse_input = torch.concatenate((x, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)
            # One would also keep an 'activated_state_trace' as the history of outgoing post-activations
            # BUT, this is unnecessary because the synchronisation calculation is fully linear and can be
            # done using only the currect activated state (see compute_synchronisation method for explanation)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state,
                                                                                                decay_alpha_out,
                                                                                                decay_beta_out, r_out,
                                                                                                synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, np.array(synch_out_tracking), np.array(pre_activations_tracking), np.array(
                post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out