import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import MNISTBackbone,ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)
from torch.nn import functional as F

def activation_derivative(act_module, postact, preact=None):
    """ Compute the derivative of the activation function.
    preact: (B, H)
    returns: (B, H) act'(preact)
    """
    if isinstance(act_module, torch.nn.ReLU):
        return torch.where(preact > 0, 1.0, 0.0)
    if isinstance(act_module, torch.nn.Tanh):
        return 1 - postact ** 2
    if isinstance(act_module, torch.nn.Softplus):
        return 1 - torch.exp(-postact)
    raise NotImplementedError(f"Derivative for {type(act_module)} not implemented.")
class EISepLSTM(nn.Module):
    """
    Custom multi-layer sequence LSTM with separate excitatory/inhibitory neurons.
    Input x: (seq_len, batch_size, feature_dim)
    Outputs:
      - output_seq: (seq_len, batch_size, hidden_size)
      - (h_n, c_n): each (num_layers, batch_size, hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # per-layer input → hidden
        std = 1/ np.sqrt(hidden_size)
        # std = 0.1
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        # print('Wrec_free1:', self.Wrec_free.device)
        # self.Wrec_free = nn.ReLU()(self.Wrec_free)  # Ensure non-negative recurrent weights
        # print('Wrec_free after ReLU1:', self.Wrec_free.device)
        ### test exponential init
        # self.Wrec_free = nn.Parameter(torch.FloatTensor(np.random.exponential(scale=1/np.sqrt(hidden_size), size=(hidden_size, hidden_size))))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)
        self.eimask_logit.requires_grad = True
        ### test fixed sign
        # self.eimask_logit = nn.Parameter(torch.zeros(hidden_size, 2))
        # self.eimask_logit.requires_grad = False # test for fixed sign
        # exc_idx = torch.randperm(hidden_size)  # 80% excitatory
        # self.eimask_logit[exc_idx[:int(hidden_size*0.8)], 0] = 1.0  # excitatory
        # self.eimask_logit[exc_idx[int(hidden_size*0.8):], 1] = 1.0  # inhibitory


        # map top-layer hidden → output
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act          = nn.Softplus()
        self.rec_filter   = nn.ReLU()

        self.training = True
        ###
        # nn.init.orthogonal_(self.Wrec_free)  # Initialize recurrent weights orthogonally
        nn.init.normal_(self.eimask_logit)
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hx=None):
        # 修改输入形状处理：期望 (B, 1, D_in)
        if x.dim() == 3 and x.size(1) == 1:
            # 来自注意力的输入：(B, 1, D_in) -> (B, D_in)
            x = x.squeeze(1)
            T = 1
            B = x.size(0)
            single_step = True
        else:
            # 原始序列输入：(T, B, D_in)
            T, B, _ = x.shape
            single_step = False
        device  = x.device

        # init or unpack hidden/cell: (L, B, H)
        if hx is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = hx

        outputs = torch.zeros(T, B, self.output_size, device=device)  # (T, B, H)
        hs = torch.zeros(T, B, self.hidden_size, device=device)  # (T, B, H)
        cs = torch.zeros(T, B, self.hidden_size, device=device)  # (T, B, H)
        rec_ins = torch.zeros(T, B, self.hidden_size, device=device)

        if single_step:
            # 单步处理
            input_gate = self.input_layer(x)  # (B, H)

            # EI mask 计算
            if self.training:
                mask = F.gumbel_softmax(self.eimask_logit, tau=0.1, hard=False, dim=-1)
                ei_code = mask[:, 0] - mask[:, 1]
            else:
                ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)

            Wrec = self.rec_filter(self.Wrec_free)
            actual = Wrec @ torch.diag(ei_code)
            rec_in = h @ actual.T + self.bias
            new_h = self.act(input_gate + rec_in)
            output = self.output_layer(new_h)
            # 返回格式匹配训练脚本期望
            return output.unsqueeze(0), (new_h, c)
        else:
                # loop over time
            for t in range(T):
                layer_in = x[t]  # (B, D_in) or (B, H)
                # no loop over layers
                input_gate = self.input_layer(layer_in)  # (B, H)
                ###
                if self.training:
                    mask = F.gumbel_softmax(self.eimask_logit,
                                        tau=0.1, hard=False, dim=-1)
                    ei_code = mask[:, 0] - mask[:, 1]
                else:
                    ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
                # if temperature < 0:
                #     ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
                # else:
                #     mask = (self.eimask_logit/temperature).softmax(dim=-1)  # (H, 2)
                #     ei_code = mask[:, 0] - mask[:, 1]
                Wrec = self.rec_filter(self.Wrec_free)  # (H, H)
                actual = Wrec @ torch.diag(ei_code)  # (H, H)
                # 3) rec input
                rec_in = h @ actual.T + self.bias  # (B, H)
                # 4) new hidden
                new_h = self.act(input_gate + rec_in)  # (B, H)

                # new_h.append(h_i)
                # new_c.append(c[i])   # 如果以后要更新 c_i，在这里替换
                # layer_in = h_i       # feed to next layer
                new_c = c

                # stack per-layer states
                # h = torch.stack(new_h, dim=0)  # (L, B, H)
                # c = torch.stack(new_c, dim=0)
                hs[t] = new_h  # (B, H)
                cs[t] = new_c  # (B, H)
                rec_ins[t] = rec_in
                h = new_h  # (B, H)
                c = new_c  # (B, H)
                # project top-layer hidden to output
                # outputs.append(self.output_layer(new_h))  # (B, H)
                outputs[t] = self.output_layer(new_h)

            # (T, B, H)
            # output_seq = torch.stack(outputs, dim=0)
            # h_seq = torch.stack(hs, dim=0)  # (T, B, H)
            # c_seq = torch.stack(cs, dim=0)  # (T, B, H)
            output_seq = outputs
            h_seq = hs
            c_seq = cs

            return output_seq, (h_seq[-1], c_seq[-1])  # 返回最后的状态
            # return outputs, (new_h, new_c)

    def eval(self):
        self.training = False
    def train(self, mode=True):
        self.training = mode

    #def compute_lyapunov_spectrum(self, x, hx=None):
        #"""
        #Return:
        #lambda_i: (H,) # average Lyapunov exponent for each hidden unit
        #lambda_j: (B, H) # Lyapunov exponent for each hidden unit in each sample
        #"""
        ## x: (T, B, D_in)
        #T, B, _ = x.shape
        #device  = x.device

        ## init or unpack hidden/cell: (L, B, H)
        #if hx is None:
            #h = x.new_zeros(B, self.hidden_size)
            #c = x.new_zeros(B, self.hidden_size)
        #else:
            #h, c = hx


        ## 初始化正交基 Q: (B, H, H)
        #Q = torch.eye(self.hidden_size, device=device).unsqueeze(0).repeat(B, 1, 1)  # 每个 sample 一组基
        ## 累计 gamma: (B, H)
        #gamma = torch.zeros(B, self.hidden_size, device=device)

        ## 逐步推进
        #for t in range(T):
            #layer_in = x[t]  # (B, D_in) or (B, H)
            ## no loop over layers
            #input_gate = self.input_layer(layer_in)  # (B, H)

            ## 3) rec input
            #rec_in = h @ self.Wrec_free + self.bias          # (B, H)
            ## 4) new hidden
            #new_h    = self.act(input_gate + rec_in)                   # (B, H)



            ## 计算雅可比作用在当前正交基上：J = diag(phi'(preact)) @ actual.T
            ## 先算 A = actual.T @ Q  -> (B, H, H)
            #A = torch.einsum('ij,bjk->bik', self.Wrec_free.T, Q)  # (B, H, H)

            ## 再算 act'（每个 batch, 每个维度）
            #act_prime = activation_derivative(self.act, new_h, input_gate+rec_in)  # (B, H)
            ## JQ = diag(act_prime) @ A; diag multiplies每一行
            #JQ = act_prime.unsqueeze(-1) * A  # (B, H, H)

            ## QR 重正交化（batch 版）
            ## torch.linalg.qr 在新版 PyTorch 支持 batch
            #Q_new, R = torch.linalg.qr(JQ)  # Q_new: (B,H,H), R: (B,H,H)
            ## 累加 log 伸缩因子
            #diag_R = torch.diagonal(R, dim1=1, dim2=2)  # (B, H)
            #gamma += torch.log(torch.abs(diag_R) + 1e-10)  # 防止 0

            ## 更新状态 / 基
            #Q = Q_new
            #h = new_h
            ## c 保持不变（和你的 forward 一样）

        ## 该 batch 每个 sample 的有限时间 Lyapunov 指数
        #lambda_j = gamma / float(T)  # (B, H)
        ## 平均得到谱
        #lambda_i = lambda_j.mean(dim=0)  # (H,)

        #return lambda_i, lambda_j

#————————————————————————————————————————————————————————————————————————————————————————————
class Net(nn.Module):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 backbone_type,
                 num_layers,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 ):
        super(Net, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.prediction_reshaper = prediction_reshaper
        self.backbone_type = backbone_type
        self.positional_embedding_type = positional_embedding_type
        self.out_dims = out_dims

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()

        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = self.get_kv_proj()
        self.lstm = EISepLSTM(
            input_size=d_input,
            hidden_size=d_model,
            output_size=d_model  # 输出维度，通常等于隐藏层维度
        )
        self.q_proj = self.get_q_proj()
        self.attention = self.get_attention(heads, dropout)
        self.output_projector = nn.Sequential(nn.LazyLinear(out_dims))

        #  --- Start States ---
        self.register_parameter('start_hidden_state', nn.Parameter(torch.zeros(d_model).uniform_(-math.sqrt(1/d_model), math.sqrt(1/d_model))))
        self.register_parameter('start_cell_state', nn.Parameter(torch.zeros(d_model).uniform_(-math.sqrt(1/d_model), math.sqrt(1/d_model))))

    # --- Core LSTM Methods ---

    def compute_features(self, x):
        """Applies backbone and positional embedding to input."""
        x = self.initial_rgb(x)
        self.kv_features = self.backbone(x)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """Compute the certainty of the current prediction."""
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] +self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Set the initial RGB processing module based on the backbone type."""
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
        elif self.backbone_type == 'MNISTBackbone':
            return self.d_input
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
        """Set the backbone module based on the specified type."""
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif self.backbone_type == 'MNISTBackbone':
            d_backbone = self.get_d_backbone()
            self.backbone = MNISTBackbone(self.d_input)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """Get the positional embedding module."""
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

    def get_attention(self, heads, dropout):
        """Get the attention module."""
        print(f"DEBUG: self.d_input = {self.d_input}, heads = {heads}")
        print(f"DEBUG: self.d_input % heads = {self.d_input % heads}")
        return nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True)

    def get_kv_proj(self):
        """Get the key-value projection module."""
        return nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input))

    def get_q_proj(self):
        """Get the query projection module."""
        return nn.LazyLinear(self.d_input)


    def verify_args(self):
        """Verify the validity of the input arguments."""

        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

        pass

    def forward(self, x, track=False):
        """
        Forward pass - Reverted to structure closer to user's working version.
        Executes T=iterations steps.
        """
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        activations_tracking = []
        attention_tracking = []
        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        hn = self.start_hidden_state.repeat(B, 1)  # (B, d_model)
        cn = self.start_cell_state.repeat(B, 1)  # (B, d_model)
        state_trace = [hn]
        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Interact with Data via Attention ---
            q = self.q_proj(hn.unsqueeze(1))  # (B, 1, d_input)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            lstm_input = attn_out  # 使用注意力输出作为LSTM输入

            # --- Apply EIRNNNet ---
            lstm_output, (new_hn, new_cn) = self.lstm(lstm_input, (hn, cn))
            hn = new_hn.squeeze(0) if new_hn.dim() == 3 else new_hn  # 处理维度
            cn = new_cn if isinstance(new_cn, torch.Tensor) else cn
            state_trace.append(hn)

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(hn)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                activations_tracking.append(hn.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, None, np.zeros_like(activations_tracking), np.array(activations_tracking), np.array(attention_tracking)
        return predictions, certainties, None