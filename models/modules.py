import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np

# Assuming 'add_coord_dim' is defined in models.utils
from models.utils import add_coord_dim

# --- Basic Utility Modules ---

class Identity(nn.Module):
    """
    Identity Module.

    Returns the input tensor unchanged. Useful as a placeholder or a no-op layer
    in nn.Sequential containers or conditional network parts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Squeeze(nn.Module):
    """
    Squeeze Module.

    Removes a specified dimension of size 1 from the input tensor.
    Useful for incorporating tensor dimension squeezing within nn.Sequential.

    Args:
      dim (int): The dimension to squeeze.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

# --- Core CTM Component Modules ---

class SynapseUNET(nn.Module):
    """
    UNET-style architecture for the Synapse Model (f_theta1 in the paper).

    This module implements the connections between neurons in the CTM's latent
    space. It processes the combined input (previous post-activation state z^t
    and attention output o^t) to produce the pre-activations (a^t) for the
    next internal tick (Eq. 1 in the paper).

    While a simpler Linear or MLP layer can be used, the paper notes
    that this U-Net structure empirically performed better, suggesting benefit
    from more flexible synaptic connections[cite: 79, 80]. This implementation
    uses `depth` points in linspace and creates `depth-1` down/up blocks.

    Args:
      in_dims (int): Number of input dimensions (d_model + d_input).
      out_dims (int): Number of output dimensions (d_model).
      depth (int): Determines structure size; creates `depth-1` down/up blocks.
      minimum_width (int): Smallest channel width at the U-Net bottleneck.
      dropout (float): Dropout rate applied within down/up projections.
    """
    def __init__(self,
                 out_dims,
                 depth,
                 minimum_width=16,
                 dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth # Store depth just for reference if needed

        # Define UNET structure based on depth
        # Creates `depth` width values, leading to `depth-1` blocks
        widths = np.linspace(out_dims, minimum_width, depth)

        # Initial projection layer
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])), # Project to the first width
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        # Downward path (encoding layers)
        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1 # Number of down/up blocks created

        for i in range(num_blocks):
            # Down block: widths[i] -> widths[i+1]
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            # Up block: widths[i+1] -> widths[i]
            # Note: Up blocks are added in order matching down blocks conceptually,
            # but applied in reverse order in the forward pass.
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            # Skip connection LayerNorm operates on width[i]
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        # Initial projection
        out_first = self.first_projection(x)

        # Downward path, storing outputs for skip connections
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))
        # outs_down contains [level_0, level_1, ..., level_depth-1=bottleneck] outputs

        # Upward path, starting from the bottleneck output
        outs_up = outs_down[-1] # Bottleneck activation
        num_blocks = len(self.up_projections) # Should be depth - 1

        for i in range(num_blocks):
            # Apply up projection in reverse order relative to down blocks
            # up_projection[num_blocks - 1 - i] processes deeper features first
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)

            # Get corresponding skip connection from downward path
            # skip_connection index = num_blocks - 1 - i (same as up_layer_idx)
            # This matches the output width of the up_projection[up_layer_idx]
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]

            # Add skip connection and apply LayerNorm corresponding to this level
            # skip_lns index also corresponds to the level = skip_idx
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        # The final output after all up-projections
        return outs_up


class SuperLinear(nn.Module):
    """
    SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM.

    This layer is the core component enabling Neuron-Level Models (NLMs),
    referred to as g_theta_d in the paper (Eq. 3). It applies N independent
    linear transformations (or small MLPs when used sequentially) to corresponding
    slices of the input tensor along a specified dimension (typically the neuron
    or feature dimension).

    How it works for NLMs:
    - The input `x` is expected to be the pre-activation history for each neuron,
      shaped (batch_size, n_neurons=N, history_length=in_dims).
    - This layer holds unique weights (`w1`) and biases (`b1`) for *each* of the `N` neurons.
      `w1` has shape (in_dims, out_dims, N), `b1` has shape (1, N, out_dims).
    - `torch.einsum('bni,iog->bno', x, self.w1)` performs N independent matrix
      multiplications in parallel (mapping from dim `i` to `o` for each neuron `n`):
        - For each neuron `n` (from 0 to N-1):
        - It takes the neuron's history `x[:, n, :]` (shape B, in_dims).
        - Multiplies it by the neuron's unique weight matrix `self.w1[:, :, n]` (shape in_dims, out_dims).
        - Resulting in `out[:, n, :]` (shape B, out_dims).
    - The unique bias `self.b1[:, n, :]` is added.
    - The result is squeezed on the last dim (if out_dims=1) and scaled by `T`.

    This allows each neuron `d` to process its temporal history `A_d^t` using
    its private parameters `theta_d` to produce the post-activation `z_d^{t+1}`,
    enabling the fine-grained temporal dynamics central to the CTM[cite: 7, 30, 85].
    It's typically used within the `trace_processor` module of the main CTM class.

    Args:
      in_dims (int): Input dimension (typically `memory_length`).
      out_dims (int): Output dimension per neuron.
      N (int): Number of independent linear models (typically `d_model`).
      T (float): Initial value for learnable temperature/scaling factor applied to output.
      do_norm (bool): Apply Layer Normalization to the input history before linear transform.
      dropout (float): Dropout rate applied to the input.
    """
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        super().__init__()
        # N is the number of neurons (d_model), in_dims is the history length (memory_length)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims # Corresponds to memory_length
        # LayerNorm applied across the history dimension for each neuron independently
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        # Initialize weights and biases
        # w1 shape: (memory_length, out_dims, d_model)
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        # b1 shape: (1, d_model, out_dims)
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        # Learnable temperature/scaler T
        self.register_parameter('T', nn.Parameter(torch.Tensor([T]))) 

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape (B, N, in_dims)
                              where B=batch, N=d_model, in_dims=memory_length.
        Returns:
            torch.Tensor: Output tensor, shape (B, N) after squeeze(-1).
        """
        # Input shape: (B, D, M) where D=d_model=N neurons in CTM, M=history/memory length
        out = self.dropout(x)
        # LayerNorm across the memory_length dimension (dim=-1)
        out = self.layernorm(out) # Shape remains (B, N, M)

        # Apply N independent linear models using einsum
        # einsum('BDM,MHD->BDH', ...)
        # x: (B=batch size, D=N neurons, one NLM per each of these, M=history/memory length)
        # w1: (M, H=hidden dims if using MLP, otherwise output, D=N neurons, parallel)
        # b1: (1, D=N neurons, H)
        # einsum result: (B, D, H)
        # Applying bias requires matching shapes, b1 is broadcasted.
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        # Squeeze the output dimension (assumed to be 1 usually) and scale by T
        # This matches the original code's structure exactly.
        out = out.squeeze(-1) / self.T
        return out


# --- Backbone Modules ---

class ParityBackbone(nn.Module):
    def __init__(self, n_embeddings, d_embedding):
        super(ParityBackbone, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        """
        Maps -1 (negative parity) to 0 and 1 (positive) to 1
        """
        x = (x == 1).long() 
        return self.embedding(x.long()).transpose(1, 2) # Transpose for compatibility with other backbones

class QAMNISTOperatorEmbeddings(nn.Module):
    def __init__(self, num_operator_types, d_projection):
        super(QAMNISTOperatorEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_operator_types, d_projection)

    def forward(self, x):
        # -1 for plus and -2 for minus
        return self.embedding(-x - 1)

class QAMNISTIndexEmbeddings(torch.nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        embedding = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return self.embedding[x]
    
class ThoughtSteps:
    """
    Helper class for managing "thought steps" in the ctm_qamnist pipeline.

    Args:
        iterations_per_digit (int): Number of iterations for each digit.
        iterations_per_question_part (int): Number of iterations for each question part.
        total_iterations_for_answering (int): Total number of iterations for answering.
        total_iterations_for_digits (int): Total number of iterations for digits.
        total_iterations_for_question (int): Total number of iterations for question.
    """
    def __init__(self, iterations_per_digit, iterations_per_question_part, total_iterations_for_answering, total_iterations_for_digits, total_iterations_for_question):
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.total_iterations_for_digits = total_iterations_for_digits
        self.total_iterations_for_question = total_iterations_for_question
        self.total_iterations_for_answering = total_iterations_for_answering
        self.total_iterations = self.total_iterations_for_digits + self.total_iterations_for_question + self.total_iterations_for_answering

    def determine_step_type(self, stepi: int):
        is_digit_step = stepi < self.total_iterations_for_digits
        is_question_step = self.total_iterations_for_digits <= stepi < self.total_iterations_for_digits + self.total_iterations_for_question
        is_answer_step = stepi >= self.total_iterations_for_digits + self.total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_answer_step_type(self, stepi: int):
        step_within_questions = stepi - self.total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

class MNISTBackbone(nn.Module):
    """
    Simple backbone for MNIST feature extraction.
    """
    def __init__(self, d_input):
        super(MNISTBackbone, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.layers(x)


class MiniGridBackbone(nn.Module):
    def __init__(self, d_input, grid_size=7, num_objects=11, num_colors=6, num_states=3, embedding_dim=8):
        super().__init__()
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        
        self.position_embedding = nn.Embedding(grid_size * grid_size, embedding_dim)

        self.project_to_d_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        x = x.long()
        B, H, W, C = x.size()

        object_idx = x[:,:,:, 0]
        color_idx =  x[:,:,:, 1]
        state_idx =  x[:,:,:, 2]

        obj_embed = self.object_embedding(object_idx)
        color_embed = self.color_embedding(color_idx)
        state_embed = self.state_embedding(state_idx)
        
        pos_idx = torch.arange(H * W, device=x.device).view(1, H, W).expand(B, -1, -1)
        pos_embed = self.position_embedding(pos_idx)

        out = self.project_to_d_projection(torch.cat([obj_embed, color_embed, state_embed, pos_embed], dim=-1))
        return out

class ClassicControlBackbone(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        return self.input_projector(x)


class ShallowWide(nn.Module):
    """
    Simple, wide, shallow convolutional backbone for image feature extraction.

    Alternative to ResNet, uses grouped convolutions and GLU activations.
    Fixed structure, useful for specific experiments.
    """
    def __init__(self):
        super(ShallowWide, self).__init__()
        # LazyConv2d infers input channels
        self.layers = nn.Sequential(
            nn.LazyConv2d(4096, kernel_size=3, stride=2, padding=1), # Output channels = 4096
            nn.GLU(dim=1), # Halves channels to 2048
            nn.BatchNorm2d(2048),
            # Grouped convolution maintains width but processes groups independently
            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1, groups=32),
            nn.GLU(dim=1), # Halves channels to 2048
            nn.BatchNorm2d(2048)
        )
    def forward(self, x):
        return self.layers(x)


class PretrainedResNetWrapper(nn.Module):
    """
    Wrapper to use standard pre-trained ResNet models from torchvision.

    Loads a specified ResNet architecture pre-trained on ImageNet, removes the
    final classification layer (fc), average pooling, and optionally later layers
    (e.g., layer4), allowing it to be used as a feature extractor backbone.

    Args:
        resnet_type (str): Name of the ResNet model (e.g., 'resnet18', 'resnet50').
        fine_tune (bool): If False, freezes the weights of the pre-trained backbone.
    """
    def __init__(self, resnet_type, fine_tune=True):
        super(PretrainedResNetWrapper, self).__init__()
        self.resnet_type = resnet_type
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', resnet_type, pretrained=True)

        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove final layers to use as feature extractor
        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()
        # Keep layer4 by default, user can modify instance if needed
        # self.backbone.layer4 = Identity()

    def forward(self, x):
        # Get features from the modified ResNet
        out = self.backbone(x)

        # Reshape output to (B, C, H, W) - This is heuristic based on original comment.
        # User might need to adjust this based on which layers are kept/removed.
        # Infer C based on ResNet type (example values)
        nc = 256 if ('18' in self.resnet_type or '34' in self.resnet_type) else 512 if '50' in self.resnet_type else 1024 if '101' in self.resnet_type else 2048 # Approx for layer3/4 output channel numbers
        # Infer H, W assuming output is flattened C * H * W
        num_features = out.shape[-1]
        # This calculation assumes nc is correct and feature map is square
        wh_squared = num_features / nc
        if wh_squared < 0 or not float(wh_squared).is_integer():
             print(f"Warning: Cannot reliably reshape PretrainedResNetWrapper output. nc={nc}, num_features={num_features}")
             # Return potentially flattened features if reshape fails
             return out
        wh = int(np.sqrt(wh_squared))

        return out.reshape(x.size(0), nc, wh, wh)

# --- Positional Encoding Modules ---

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Feature Positional Encoding.

    Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
    Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
    Provides positional information for 2D feature maps.

    Args:
        d_model (int): The output dimension of the positional encoding (D).
        G (int): Positional groups (default 1).
        M (int): Dimensionality of input coordinates (default 2 for H, W).
        F_dim (int): Dimension of the Fourier features.
        H_dim (int): Hidden dimension of the MLP.
        gamma (float): Initialization scale for the Fourier projection weights (Wr).
    """
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma=1/2.5,
                 ):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = d_model
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GLU(), # Halves H_dim
            nn.Linear(self.H_dim // 2, self.D // self.G),
            nn.LayerNorm(self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Computes positional encodings for the input feature map x.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            torch.Tensor: Positional encoding tensor, shape (B, D, H, W).
        """
        B, C, H, W = x.shape
        # Creates coordinates based on (H, W) and repeats for batch B.
        # Takes x[:,0] assuming channel dim isn't needed for coords.
        x_coord = add_coord_dim(x[:,0]) # Expects (B, H, W) -> (B, H, W, 2)

        # Compute Fourier features
        projected = self.Wr(x_coord) # (B, H, W, F_dim // 2)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = (1.0 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1) # (B, H, W, F_dim)

        # Project features through MLP
        Y = self.mlp(F) # (B, H, W, D // G)

        # Reshape to (B, D, H, W)
        PEx = Y.permute(0, 3, 1, 2) # Assuming G=1
        return PEx


class MultiLearnableFourierPositionalEncoding(nn.Module):
    """
    Combines multiple LearnableFourierPositionalEncoding modules with different
    initialization scales (gamma) via a learnable weighted sum.

    Allows the model to learn an optimal combination of positional frequencies.

    Args:
        d_model (int): Output dimension of the encoding.
        G, M, F_dim, H_dim: Parameters passed to underlying LearnableFourierPositionalEncoding.
        gamma_range (list[float]): Min and max gamma values for the linspace.
        N (int): Number of parallel embedding modules to create.
    """
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma_range=[1.0, 0.1], # Default range
                 N=10,
                 ):
        super().__init__()
        self.embedders = nn.ModuleList()
        for gamma in np.linspace(gamma_range[0], gamma_range[1], N):
            self.embedders.append(LearnableFourierPositionalEncoding(d_model, G, M, F_dim, H_dim, gamma))

        # Renamed parameter from 'combination' to 'combination_weights' for clarity only in comments
        # Actual registered name remains 'combination' as in original code
        self.register_parameter('combination', torch.nn.Parameter(torch.ones(N), requires_grad=True))
        self.N = N


    def forward(self, x):
        """
        Computes combined positional encoding.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            torch.Tensor: Combined positional encoding tensor, shape (B, D, H, W).
        """
        # Compute embeddings from all modules and stack: (N, B, D, H, W)
        pos_embs = torch.stack([emb(x) for emb in self.embedders], dim=0)

        # Compute combination weights using softmax
        # Use registered parameter name 'combination'
        # Reshape weights for broadcasting: (N,) -> (N, 1, 1, 1, 1)
        weights = F.softmax(self.combination, dim=-1).view(self.N, 1, 1, 1, 1)

        # Compute weighted sum over the N dimension
        combined_emb = (pos_embs * weights).sum(0) # (B, D, H, W)
        return combined_emb


class CustomRotationalEmbedding(nn.Module):
    """
    Custom Rotational Positional Embedding.

    Generates 2D positional embeddings based on rotating a fixed start vector.
    The rotation angle for each grid position is determined primarily by its
    horizontal position (width dimension). The resulting rotated vectors are
    concatenated and projected.

    Note: The current implementation derives angles only from the width dimension (`x.size(-1)`).

    Args:
        d_model (int): Dimensionality of the output embeddings.
    """
    def __init__(self, d_model):
        super(CustomRotationalEmbedding, self).__init__()
        # Learnable 2D start vector
        self.register_parameter('start_vector', nn.Parameter(torch.Tensor([0, 1]), requires_grad=True))
        # Projects the 4D concatenated rotated vectors to d_model
        # Input size 4 comes from concatenating two 2D rotated vectors
        self.projection = nn.Sequential(nn.Linear(4, d_model))

    def forward(self, x):
        """
        Computes rotational positional embeddings based on input width.

        Args:
            x (torch.Tensor): Input tensor (used for shape and device),
                              shape (batch_size, channels, height, width).
        Returns:
            Output tensor containing positional embeddings,
            shape (1, d_model, height, width) - Batch dim is 1 as PE is same for all.
        """
        B, C, H, W = x.shape
        device = x.device

        # --- Generate rotations based only on Width ---
        # Angles derived from width dimension
        theta_rad = torch.deg2rad(torch.linspace(0, 180, W, device=device)) # Angle per column
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)

        # Create rotation matrices: Shape (W, 2, 2)
        # Use unsqueeze(1) to allow stacking along dim 1
        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1), # Shape (W, 2)
            torch.stack([sin_theta, cos_theta], dim=-1)  # Shape (W, 2)
        ], dim=1) # Stacks along dim 1 -> Shape (W, 2, 2)

        # Rotate the start vector by column angle: Shape (W, 2)
        rotated_vectors = torch.einsum('wij,j->wi', rotation_matrices, self.start_vector)

        # --- Create Grid Key ---
        # Original code uses repeats based on rotated_vectors.shape[0] (which is W) for both dimensions.
        # This creates a (W, W, 4) key tensor.
        key = torch.cat((
            torch.repeat_interleave(rotated_vectors.unsqueeze(1), W, dim=1), # (W, 1, 2) -> (W, W, 2)
            torch.repeat_interleave(rotated_vectors.unsqueeze(0), W, dim=0)  # (1, W, 2) -> (W, W, 2)
        ), dim=-1) # Shape (W, W, 4)

        # Project the 4D key vector to d_model: Shape (W, W, d_model)
        pe_grid = self.projection(key)

        # Reshape to (1, d_model, W, W) and then select/resize to target H, W?
        # Original code permutes to (d_model, W, W) and unsqueezes to (1, d_model, W, W)
        pe = pe_grid.permute(2, 0, 1).unsqueeze(0)

        # If H != W, this needs adjustment. Assuming H=W or cropping/padding happens later.
        # Let's return the (1, d_model, W, W) tensor as generated by the original logic.
        # If H != W, downstream code must handle the mismatch or this PE needs modification.
        if H != W:
            # Simple interpolation/cropping could be added, but sticking to original logic:
            # Option 1: Interpolate
            # pe = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
            # Option 2: Crop/Pad (e.g., crop if W > W_target, pad if W < W_target)
            # Sticking to original: return shape (1, d_model, W, W)
            pass

        return pe

class CustomRotationalEmbedding1D(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding1D, self).__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x):
        start_vector = torch.tensor([0., 1.], device=x.device, dtype=torch.float)
        theta_rad = torch.deg2rad(torch.linspace(0, 180, x.size(2), device=x.device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_theta = cos_theta.unsqueeze(1)  # Shape: (height, 1)
        sin_theta = sin_theta.unsqueeze(1)  # Shape: (height, 1)

        # Create rotation matrices
        rotation_matrices = torch.stack([
        torch.cat([cos_theta, -sin_theta], dim=1),
        torch.cat([sin_theta, cos_theta], dim=1)
        ], dim=1)  # Shape: (height, 2, 2)

        # Rotate the start vector
        rotated_vectors = torch.einsum('bij,j->bi', rotation_matrices, start_vector)

        pe = self.projection(rotated_vectors)
        pe = torch.repeat_interleave(pe.unsqueeze(0), x.size(0), 0)
        return pe.transpose(1, 2) # Transpose for compatibility with other backbones
    
