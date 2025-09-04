import torch.nn as nn
import torch
import numpy as np
import math
from models.lstm import LSTMBaseline
from models.modules import ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)


#————————————————————————————————————————————————————————————————————————————————————————————
class LSTMSORT(LSTMBaseline):
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
                 num_layers,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 backbone_type='none',
                 positional_embedding_type='none',
                 out_dims=1,
                 prediction_reshaper=[-1],
                 dropout=0,
                 ):
        super(LSTMSORT, self).__init__(
            num_layers=num_layers,
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            backbone_type=backbone_type,
            positional_embedding_type=positional_embedding_type,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None
        #不使用attation时
        #self.attention = None  # Should already be None because super(... heads=0... )
        #self.q_proj = None  # Should already be None because super(... heads=0... )
        #self.kv_proj = None  # Should already be None because super(... heads=0... )

    def compute_features(self, x):
        """Sort任务优化的特征处理 - 跳过复杂的backbone处理"""
        """Applies backbone and positional embedding to input."""
        x = self.initial_rgb(x)
        if x.dim() == 2:
            # (B, sequence_length) -> (B, 1, sequence_length)
            kv = x.unsqueeze(1)
        else:
            kv = x

        if hasattr(self, 'kv_proj') and self.kv_proj is not None:
            kv = self.kv_proj(kv)

        return kv

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

        #kv = x.unsqueeze(1) # Sort 任务不使用attation直接处理输入，不需要复杂的特征提取
        kv = self.compute_features(x)  # 使用attation时

        # --- Initialise Recurrent State ---
        hn = torch.repeat_interleave(self.start_hidden_state.unsqueeze(1), x.size(0), 1)
        cn = torch.repeat_interleave(self.start_cell_state.unsqueeze(1), x.size(0), 1)
        state_trace = [hn.squeeze(1)]

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            #lstm_input = kv #不使用attation时
            #使用attation时
            q = self.q_proj(hn[-1]).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            lstm_input = attn_out
            # --- Apply LSTM ---
            hidden_state, (hn,cn) = self.lstm(lstm_input, (hn, cn))
            hidden_state = hidden_state.squeeze(1)
            state_trace.append(hidden_state)

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(hidden_state)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

        # 跟踪激活
            if track:
                activations_tracking.append(hn.detach().cpu().numpy())
                #attention_tracking.append(torch.ones(B, 1, 1).detach().cpu().numpy()) # 不使用attation由于没有真正的注意力，创建虚拟的注意力权重
                attention_tracking.append(attn_weights.detach().cpu().numpy())  # 使用attation使用真实的注意力权重

        if track:
        # 返回6个值以匹配训练脚本的期望
            pre_activations = np.zeros_like(activations_tracking)  # 虚拟的pre-activations
            post_activations = np.array(activations_tracking)
            attention_weights = np.array(attention_tracking)
            return predictions, certainties, None, pre_activations, post_activations, attention_weights
        else:
            return predictions, certainties, None