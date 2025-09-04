import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)
from torch.nn import functional as F
from models.simpleEIRNN import Net

class NetSORT(Net):

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
        super(NetSORT, self).__init__(
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

        if heads > 0:  # 使用attention时正确初始化attention组件
            self.q_proj = nn.LazyLinear(d_input)
            self.kv_proj = nn.Sequential(nn.LazyLinear(d_input), nn.LayerNorm(d_input))
            self.attention = nn.MultiheadAttention(d_input, heads, dropout, batch_first=True)
        else:  #不使用attation时
            self.attention = None
            self.q_proj = None
            self.kv_proj = None

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
        kv = self.compute_features(x)   #使用attation时
        # --- Initialise Recurrent State ---
        hn = self.start_hidden_state.repeat(B, 1)  # (B, d_model)
        cn = self.start_cell_state.repeat(B, 1)  # (B, d_model)
        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Interact with Data via Attention ---
            #lstm_input = kv #不使用attation时
            #使用attation时
            q = self.q_proj(hn.unsqueeze(1))
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            lstm_input = attn_out

            # --- Apply EIRNNNet ---
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