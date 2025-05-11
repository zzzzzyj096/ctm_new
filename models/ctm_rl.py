import torch
import torch.nn as nn
import numpy as np
import math
from models.ctm import ContinuousThoughtMachine
from models.modules import MiniGridBackbone, ClassicControlBackbone, SynapseUNET
from models.utils import compute_decay
from models.constants import VALID_NEURON_SELECT_TYPES

class ContinuousThoughtMachineRL(ContinuousThoughtMachine):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 n_synch_out,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 prediction_reshaper=[-1],
                 dropout=0,
                 neuron_select_type='first-last',
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=0,  # Set heads to 0 will return None
            n_synch_out=n_synch_out,
            n_synch_action=0,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            out_dims=0,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            backbone_type=backbone_type,
            n_random_pairing_self=0,
            positional_embedding_type='none',
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None

        # --- Start dynamics with a learned activated state trace ---
        self.register_parameter('start_activated_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length))), requires_grad=True)) 
        self.start_activated_state = None

        self.register_buffer('diagonal_mask_out', torch.triu(torch.ones(self.n_synch_out, self.n_synch_out, dtype=torch.bool)))

        self.attention = None  # Should already be None because super(... heads=0... ) 
        self.q_proj = None  # Should already be None because super(... heads=0... ) 
        self.kv_proj = None  # Should already be None because super(... heads=0... ) 
        self.output_projector = None

    # --- Core CTM Methods ---

    def compute_synchronisation(self, activated_state_trace):
        """Compute the synchronisation between neurons."""
        assert self.neuron_select_type == "first-last", "only fisrst-last neuron selection is supported here"
        # For RL tasks we track a sliding window of activations from which we compute synchronisation
        S = activated_state_trace.permute(0, 2, 1)
        diagonal_mask = self.diagonal_mask_out.to(S.device)
        decay = compute_decay(S.size(1), self.decay_params_out, clamp_lims=(0, 4))
        synchronisation = ((decay.unsqueeze(0) *(S[:,:,-self.n_synch_out:].unsqueeze(-1) * S[:,:,-self.n_synch_out:].unsqueeze(-2))[:,:,diagonal_mask]).sum(1))/torch.sqrt(decay.unsqueeze(0).sum(1,))
        return synchronisation

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Set the initial RGB values for the backbone."""
        return None

    def get_d_backbone(self):
        """Get the dimensionality of the backbone output."""
        return self.d_input
    
    def set_backbone(self):
        """Set the backbone module based on the specified type."""
        if self.backbone_type == 'navigation-backbone':
            self.backbone = MiniGridBackbone(self.d_input)
        elif self.backbone_type == 'classic-control-backbone':
            self.backbone = ClassicControlBackbone(self.d_input)
        else:
            raise NotImplemented('The only backbone supported for RL are for navigation (symbolic C x H x W inputs) and classic control (vectors of length D).')
        pass

    def get_positional_embedding(self, d_backbone):
        """Get the positional embedding module."""
        return None


    def get_synapses(self, synapse_depth, d_model, dropout):
        """
        Get the synapse module.

        We found in our early experimentation that a single Linear, GLU and LayerNorm block performed worse than two blocks. 
        For that reason we set the default synapse depth to two blocks. 
        
        TODO: This is legacy and needs further experimentation to iron out.        
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model*2),
                nn.GLU(),
                nn.LayerNorm(d_model),
                nn.LazyLinear(d_model*2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
        """Set the parameters for the synchronisation of neurons."""
        if synch_type == 'action':
            pass
        elif synch_type == 'out':
            left, right = self.initialize_left_right_neurons("out", self.d_model, n_synch, n_random_pairing_self)
            self.register_buffer(f'out_neuron_indices_left', left)
            self.register_buffer(f'out_neuron_indices_right', right)
            self.register_parameter(f'decay_params_out', nn.Parameter(torch.zeros(self.synch_representation_size_out), requires_grad=True))
            pass
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")

    # --- Utilty Methods ---

    def verify_args(self):
        """Verify the validity of the input arguments."""
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        assert self.neuron_select_type != 'random-pairing', \
            f"Random pairing is not supported for RL."
        assert self.backbone_type in ('navigation-backbone', 'classic-control-backbone'), \
            f"Invalid backbone_type: {self.backbone_type}"
        assert self.d_model >= (self.n_synch_out), \
            "d_model must be >= n_synch_out for neuron subsets"
        pass

    


    def forward(self, x, hidden_states, track=False):
        
        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []

        # --- Featurise Input Data ---
        features = self.backbone(x)

        # ---  Get Recurrent State ---
        state_trace, activated_state_trace = hidden_states

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):
            
            pre_synapse_input = torch.concatenate((features.reshape(x.size(0), -1), activated_state_trace[:,:,-1]), -1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply NLMs ---
            activated_state = self.trace_processor(state_trace)
            activated_state_trace = torch.concatenate((activated_state_trace[:,:,1:], activated_state.unsqueeze(-1)), -1)

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())

        hidden_states = (
            state_trace,
            activated_state_trace,
        )

        # --- Calculate Output Synchronisation ---
        synchronisation_out = self.compute_synchronisation(activated_state_trace)

        # --- Return Values ---
        if track:
            return synchronisation_out, hidden_states, np.array(pre_activations_tracking), np.array(post_activations_tracking)
        return synchronisation_out, hidden_states