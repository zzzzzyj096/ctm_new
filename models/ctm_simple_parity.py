import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy
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
                 heads,
                 n_synch_out,
                 n_synch_action,
                 memory_length,
                 memory_hidden_dims,
                 positional_embedding_type,
                 out_dims,
                 backbone_type,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 n_random_pairing_self = 0,
                 neuron_select_type='random-pairing',
                 ):
        super(ContinuousThoughtMachineSIMPLE,self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.memory_length = memory_length
        self.memory_hidden_dims = memory_hidden_dims
        self.neuron_select_type = neuron_select_type
        self.positional_embedding_type = positional_embedding_type
        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
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
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input))
        self.q_proj = nn.LazyLinear(self.d_input)

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
            torch.zeros((d_model)).uniform_(-math.sqrt(1 / (d_model)), math.sqrt(1 / (d_model))),
            requires_grad=True
        ))

        self.register_parameter('start_trace', nn.Parameter(
            torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1 / (d_model + memory_length)),
                                                           math.sqrt(1 / (d_model + memory_length))),
            requires_grad=True
        ))

        # --- Synchronisation ---
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)

        for synch_type, size in [('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)]:
            print(f"Synch representation size {synch_type}: {size}")

        self.set_synchronisation_parameters('out')
        self.set_synchronisation_parameters('action')

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def set_initial_rgb(self):
        """
        This is largely to accommodate training on grayscale images and is legacy, but it
        doesn't hurt the model in any way that we can tell.
        """
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type:
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        Get the positional embedding module.

        For Imagenet and mazes we used NO positional embedding, and largely don't think
        that it is necessary as the CTM can build up its own internal world model when
        observing.

        LearnableFourierPositionalEncoding:
            Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
            Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
            Provides positional information for 2D feature maps.

            (MultiLearnableFourierPositionalEncoding uses multiple feature scales)

        CustomRotationalEmbedding:
            Simple sinusoidal embedding to encourage interpretability
        """
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")
        #pass
    def set_synchronisation_parameters(self, synch_type: str):
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        n_synch = self.n_synch_action if synch_type == 'action' else self.n_synch_out
        # 初始化神经元索引
        if self.neuron_select_type in ['first-last', 'random', 'random-pairing']:
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch)
            self.register_buffer(f'{synch_type}_neuron_indices_left', left)
            self.register_buffer(f'{synch_type}_neuron_indices_right', right)

        self.register_parameter(f'decay_params_{synch_type}',
                                nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))


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

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):

        if self.neuron_select_type == 'first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model - n_synch, d_model)

        elif self.neuron_select_type == 'random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type == 'random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(
                np.random.choice(np.arange(d_model), size=n_synch - n_random_pairing_self))))

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

    def calculate_synch_representation_size(self, n_synch):

        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)


        # --- Storage for outputs per iteration
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        decay_alpha_action, decay_beta_action = None, None
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            #print(f"state_trace shape: {state_trace.shape}")
            #--- Activate ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty


            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out