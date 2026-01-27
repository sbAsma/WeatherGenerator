"""
Chemistry stream embedding with ABP pre-reduction.
Integrates into WeatherGenerator embedding pipeline.
"""

import torch
import torch.nn as nn
from weathergen.model.abp import ChannelStackedABP


class ChemistryStreamEmbedding(nn.Module):
    """
    Converts CAMS analysis data to fixed embedding.
    
    Input: (B, H, W, n_species*levels+emissions)
    Output: (B, d_embedding)
    """
    
    def __init__(
        self,
        n_species: int = 50,
        n_levels: int = 25,
        n_emissions: int = 10,
        spatial_h: int = 100,
        spatial_w: int = 100,
        d_embedding: int = 512,
        d_intermediate: int = 1800,
        n_inducing: int = 64,
    ):
        super().__init__()
        
        self.abp = ChannelStackedABP(
            n_species=n_species,
            n_levels=n_levels,
            n_emissions=n_emissions,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            d_model=d_intermediate,
            n_inducing=n_inducing
        )
        
        # Project to final embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(d_intermediate, d_embedding),
            nn.LayerNorm(d_embedding)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C_in)
        output: (B, d_embedding)
        """
        reduced = self.abp(x)  # (B, d_intermediate)
        embedded = self.proj(reduced)  # (B, d_embedding)
        return embedded
