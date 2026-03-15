"""Chemistry stream embedding with GNN-based pre-reduction."""

import torch
import torch.nn as nn
from weathergen.model.gnn_reduction import GridGNNChemistryReducer


class ChemistryStreamEmbedding(nn.Module):
    """
    Converts CAMS analysis data to fixed embedding.
    
    Input: (B, H, W, n_species*levels+emissions)
    Output: (B, d_embedding)
    """
    
    def __init__(
        self,
        n_species: int,
        n_levels: int,
        n_emissions: int,
        d_embedding: int,
        gnn_hidden_dim: int,
        gnn_num_layers: int,
        gnn_edge_k: int,
        gnn_dropout_rate: float,
        gnn_pool: str,
    ):
        super().__init__()

        c_in = n_species * n_levels + n_emissions
        self.reducer = GridGNNChemistryReducer(
            c_in=c_in,
            d_hidden=gnn_hidden_dim,
            d_out=d_embedding,
            num_layers=gnn_num_layers,
            edge_k=gnn_edge_k,
            dropout_rate=gnn_dropout_rate,
            pool=gnn_pool,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C_in)
        output: (B, d_embedding)
        """
        return self.reducer(x)
