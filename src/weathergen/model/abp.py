"""
Attention-Based Pooling for atmospheric chemistry data reduction.
Minimal working example - production version in implementation_guide.md
"""

import torch
import torch.nn as nn
import math


class AttentionBasedPooling(nn.Module):
    """
    Reduces n tokens to single vector via learned inducing points.
    
    Math:
    1. Attend input tokens to m inducing points: (B, n, d) -> (B, m, d)
    2. Pool inducing points to scalar: (B, m, d) -> (B, d)
    3. Project: (B, d) -> (B, d_out)
    
    Complexity: O(nmd + m²d) vs O(n²d) for full attention.
    """
    
    def __init__(self, d_model: int, n_inducing: int = 64, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_inducing = n_inducing
        
        # Learnable inducing points
        self.inducing = nn.Parameter(
            torch.randn(1, n_inducing, d_model) / math.sqrt(d_model)
        )
        
        # Encode: tokens -> inducing
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        
        # Decode: inducing -> global
        self.to_k_dec = nn.Linear(d_model, d_model)
        self.to_v_dec = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_tokens, d_model) -> output: (B, d_model)"""
        B, N, D = x.shape
        
        # Inducing points
        inducing = self.inducing.expand(B, -1, -1)  # (B, m, D)
        
        # Stage 1: Attend tokens to inducing points
        # Using simplified scaled dot-product attention
        Q = self.to_q(inducing)  # (B, m, D)
        K = self.to_k(x)         # (B, n, D)
        V = self.to_v(x)         # (B, n, D)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)  # (B, m, n)
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights, V)  # (B, m, D)
        
        # Stage 2: Pool inducing points to scalar
        # Create global token that attends to all inducing points
        global_token = torch.zeros(B, 1, D, device=x.device)
        
        Q_dec = global_token  # (B, 1, D)
        K_dec = self.to_k_dec(attended)  # (B, m, D)
        V_dec = self.to_v_dec(attended)  # (B, m, D)
        
        scores_dec = torch.bmm(Q_dec, K_dec.transpose(1, 2)) / math.sqrt(D)  # (B, 1, m)
        attn_weights_dec = torch.softmax(scores_dec, dim=-1)
        pooled = torch.bmm(attn_weights_dec, V_dec).squeeze(1)  # (B, D)
        
        return pooled


class ChannelStackedABP(nn.Module):
    """
    Strategy:
    1. Stack all chemistry species in channel dimension: (B, H, W, C_in)
    2. Expand channels to high dim: (B, H, W, 1800)
    3. Flatten spatial: (B, H*W, 1800)
    4. Apply ABP: (B, H*W, 1800) -> (B, 1800)
    """
    
    def __init__(
        self,
        n_species: int,
        n_levels: int,
        n_emissions: int = 10,
        spatial_h: int = 100,
        spatial_w: int = 100,
        d_model: int = 1800,
        n_inducing: int = 64,
    ):
        super().__init__()
        
        # Input: species*levels + emissions
        c_in = n_species * n_levels + n_emissions
        
        # Expand to d_model
        self.channel_expand = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # ABP pooling
        self.abp = AttentionBasedPooling(d_model, n_inducing)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, c_in) - stacked chemistry data
        output: (B, d_model) - pooled representation
        """
        B, H, W, C_in = x.shape
        
        # Expand channels
        x = self.channel_expand(x)  # (B, H, W, d_model)
        
        # Flatten spatial dimensions
        x = x.reshape(B, H * W, -1)  # (B, n_tokens, d_model)
        
        # Apply ABP
        pooled = self.abp(x)  # (B, d_model)
        
        return pooled
