import torch.nn as nn
import torch
import numpy as np
import math
from models.simpleRNN import SimpleNet, simpleRNN
from models.modules import ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)


#————————————————————————————————————————————————————————————————————————————————————————————
class SimpleNetSORT(SimpleNet):
    """
    LSTM Baseline

    Args:
        iterations (int): Number of internal 'thought' steps (T, in paper).
        d_model (int): Core dimensionality of the latent space.
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        heads (int): Number of attention heads.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        positional_embedding_type (str): Type of positional embedding for backbone features.
        out_dims (int): Dimensionality of the final output projection.
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
        dropout (float): Dropout rate.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 out_dims,
                 heads,
                 prediction_reshaper=[-1],
                 backbone_type='none',
                 positional_embedding_type='none',
                 num_layers=1,
                 dropout=0,
                 ):
        super(SimpleNetSORT, self).__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            out_dims=out_dims,
            heads=heads,
            prediction_reshaper=prediction_reshaper,
            backbone_type=backbone_type,
            positional_embedding_type=positional_embedding_type,
            num_layers=num_layers,
            dropout=dropout,
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None

        self.attention = None  # Should already be None because super(... heads=0... )
        self.q_proj = None  # Should already be None because super(... heads=0... )
        self.kv_proj = None  # Should already be None because super(... heads=0... )

    def forward(self, x, track=False):
        """
        Forward pass - Reverted to structure closer to user's working version.
        Executes T=iterations steps.
        """
        B = x.size(0)
        device = x.device
        # 初始化跟踪变量
        if track:
            activations_tracking = []
            attention_tracking = []
        # Sort 任务直接处理输入，不需要复杂的特征提取
        kv = x.unsqueeze(1)
        # --- Initialise Recurrent State ---
        hn = self.start_hidden_state.repeat(B, 1)  # (B, d_model)
        cn = self.start_cell_state.repeat(B, 1)  # (B, d_model)
        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):
            lstm_input = kv
            # --- Apply simpleNet ---
            lstm_output, (hn, cn) = self.lstm(lstm_input, (hn, cn))
            hn = hn.squeeze(0) if hn.dim() == 3 else hn

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(hn)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

        # 跟踪激活
            if track:
                activations_tracking.append(hn.detach().cpu().numpy())
                # 由于没有真正的注意力，创建虚拟的注意力权重
                attention_tracking.append(torch.ones(B, 1, 1).detach().cpu().numpy())

        if track:
        # 返回6个值以匹配训练脚本的期望
            pre_activations = np.zeros_like(activations_tracking)  # 虚拟的pre-activations
            post_activations = np.array(activations_tracking)
            attention_weights = np.array(attention_tracking)
            return predictions, certainties, None, pre_activations, post_activations, attention_weights
        else:
            return predictions, certainties, None