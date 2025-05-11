import torch.nn as nn
import torch
import torch.nn.functional as F # Used for GLU if not in modules
import numpy as np
import math

# Local imports (Assuming these contain necessary custom modules)
from models.modules import *
from models.utils import * # Assuming compute_decay, compute_normalized_entropy are here


class LSTMBaseline(nn.Module):
    """

    LSTM Baseline

    Args:
        iterations (int): Number of internal 'thought' steps (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space (D, in paper).
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 backbone_type,
                 ):
        super(LSTMBaseline, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.backbone_type = backbone_type

        # --- Input Assertions ---
        assert backbone_type in ('navigation-backbone', 'classic-control-backbone'), f"Invalid backbone_type: {backbone_type}"

        # --- Backbone / Feature Extraction ---
        if self.backbone_type == 'navigation-backbone':
            grid_size = 7
            self.backbone = MiniGridBackbone(d_input=d_input, grid_size=grid_size)
            lstm_cell_input_dim = grid_size * grid_size * d_input

        elif self.backbone_type == 'classic-control-backbone':
            self.backbone = ClassicControlBackbone(d_input=d_input)
            lstm_cell_input_dim = d_input

        else:
            raise NotImplemented('The only backbone supported for RL are for navigation (symbolic C x H x W inputs) and classic control (vectors of length D).')

        # --- Core LSTM Modules ---
        self.lstm_cell = nn.LSTMCell(lstm_cell_input_dim, d_model)
        self.register_parameter('start_hidden_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))), requires_grad=True))
        self.register_parameter('start_cell_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))), requires_grad=True))
    
    def compute_features(self, x):
        """Applies backbone and positional embedding to input."""
        return self.backbone(x)


    def forward(self, x, hidden_states, track=False):
        """
        Forward pass - Reverted to structure closer to user's working version.
        Executes T=iterations steps.
        """

        # --- Tracking Initialization ---
        activations_tracking = []

        # --- Featurise Input Data ---
        features = self.compute_features(x)

        hidden_state = hidden_states[0]
        cell_state = hidden_states[1]

        # --- Recurrent Loop ---
        for stepi in range(self.iterations):

            lstm_input = features.reshape(x.size(0), -1)
            hidden_state, cell_state = self.lstm_cell(lstm_input.squeeze(1), (hidden_state, cell_state))

            # --- Tracking ---
            if track:
                activations_tracking.append(hidden_state.squeeze(1).detach().cpu().numpy())

        hidden_states = (
            hidden_state,
            cell_state
        )

        # --- Return Values ---
        if track:
            return hidden_state, hidden_states, np.array(activations_tracking), np.array(activations_tracking)
        return hidden_state, hidden_states