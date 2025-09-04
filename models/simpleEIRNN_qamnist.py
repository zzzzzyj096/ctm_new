import torch.nn as nn
import torch
import torch.nn.functional as F # Used for GLU if not in modules
import numpy as np
import math

# Local imports (Assuming these contain necessary custom modules)
from models.modules import *
from models.utils import * # Assuming compute_decay, compute_normalized_entropy are here
from models.simpleEIRNN import Net,EISepLSTM

class NetQA(Net):

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 out_dims,
                 iterations_per_digit,
                 iterations_per_question_part,
                 iterations_for_answering,
                 prediction_reshaper=[-1],
                 dropout=0,
                 ):
        super(NetQA, self).__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            backbone_type='none',
            num_layers=1,
            positional_embedding_type='none',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
        )

        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.total_iterations_for_answering = iterations_for_answering

        # --- Backbone / Feature Extraction ---
        self.backbone_digit = MNISTBackbone(d_input)
        self.index_backbone = QAMNISTIndexEmbeddings(50, d_input)
        self.operator_backbone = QAMNISTOperatorEmbeddings(2, d_input)

        # Attention
        self.q_proj = nn.LazyLinear(d_input)
        self.kv_proj = nn.Sequential(nn.LazyLinear(d_input), nn.LayerNorm(d_input))
        self.attention = nn.MultiheadAttention(d_input, heads, dropout, batch_first=True)

        self.lstm = EISepLSTM(
            input_size=d_input,
            hidden_size=d_model,
            output_size=d_model
        )
        self.output_projector = nn.Sequential(nn.LazyLinear(out_dims))

        # 初始状态参数
        self.register_parameter('start_hidden_state',nn.Parameter(torch.zeros(d_model).uniform_(-math.sqrt(1 / d_model), math.sqrt(1 / d_model))))
        self.register_parameter('start_cell_state',nn.Parameter(torch.zeros(d_model).uniform_(-math.sqrt(1 / d_model), math.sqrt(1 / d_model))))

    def compute_certainty(self, current_prediction):
        """Compute the certainty of the current prediction."""
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] +self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def get_kv_for_step(self, stepi, x, z, thought_steps, prev_input=None, prev_kv=None):
        is_digit_step, is_question_step, is_answer_step = thought_steps.determine_step_type(stepi)

        if is_digit_step:
            current_input = x[:, stepi]
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            kv = self.kv_proj(self.backbone_digit(current_input).flatten(2).permute(0, 2, 1))

        elif is_question_step:
            offset = stepi - thought_steps.total_iterations_for_digits
            current_input = z[:, offset].squeeze(0)
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            is_index_step, is_operator_step = thought_steps.determine_answer_step_type(stepi)
            if is_index_step:
                kv = self.kv_proj(self.index_backbone(current_input))
            elif is_operator_step:
                kv = self.kv_proj(self.operator_backbone(current_input))
            else:
                raise ValueError("Invalid step type for question processing.")

        elif is_answer_step:
            current_input = None
            kv = torch.zeros((x.size(0), self.d_input), device=x.device)

        else:
            raise ValueError("Invalid step type.")

        return kv, current_input

    def forward(self, x, z, track=False):
        """
        Forward pass - Reverted to structure closer to user's working version.
        Executes T=iterations steps.
        """
        B = x.size(0) # Batch size

        # --- Tracking Initialization ---
        activations_tracking = []
        attention_tracking = [] # Note: reshaping this correctly requires knowing num_heads
        embedding_tracking = []

        thought_steps = ThoughtSteps(self.iterations_per_digit, self.iterations_per_question_part, self.total_iterations_for_answering, x.size(1), z.size(1))

        # --- Step 2: Initialise Recurrent State ---
        hn = self.start_hidden_state.repeat(B, 1)  # (B, d_model)
        cn = self.start_cell_state.repeat(B, 1)  # (B, d_model)

        state_trace = [hn]

        device = hn.device

        # Storage for outputs per iteration
        predictions = torch.empty(B, self.out_dims, thought_steps.total_iterations, device=device, dtype=x.dtype) # Adjust dtype if needed
        certainties = torch.empty(B, 2, thought_steps.total_iterations, device=device, dtype=x.dtype) # Adjust dtype if needed

        prev_input = None
        prev_kv = None

        # --- Recurrent Loop (T=iterations steps) ---
        for stepi in range(thought_steps.total_iterations):

            is_digit_step, is_question_step, is_answer_step = thought_steps.determine_step_type(stepi)
            kv, prev_input = self.get_kv_for_step(stepi, x, z, thought_steps, prev_input, prev_kv)
            prev_kv = kv

            # --- Interact with Data via Attention ---
            attn_weights = None
            if is_digit_step:
                q = self.q_proj(hn).unsqueeze(1)
                attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
                lstm_input = attn_out
            else:
                lstm_input = kv.unsqueeze(1) if kv.dim() == 2 else kv

            lstm_output, (hn, cn) = self.lstm(lstm_input, (hn, cn))
            hn = hn.squeeze(0) if hn.dim() == 3 else hn
            state_trace.append(hn)

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(hn)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                activations_tracking.append(hn.squeeze(1).detach().cpu().numpy())
                if attn_weights is not None:
                    attention_tracking.append(attn_weights.detach().cpu().numpy())
                if is_question_step:
                    embedding_tracking.append(kv.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, None, np.array(activations_tracking), np.array(activations_tracking), np.array(attention_tracking), np.array(embedding_tracking)
        return predictions, certainties, None