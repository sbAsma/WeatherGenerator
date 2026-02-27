"""
GNN-based channel reducer for high-dimensional streams (e.g. CAMS chemistry).

Treats the channel dimension as a graph: each channel is a node with features
from the token_size spatial dimension. A k-NN graph is built over learned channel
embeddings, and message-passing layers reduce n_channels → latent_dim before the
standard StreamEmbedTransformer sees the data.

Input:  (N, token_size, n_channels)   — raw source tokens
Output: (N, token_size, latent_dim)   — reduced source tokens
"""

import torch
import torch.nn as nn


class GNNChannelReducer(nn.Module):
    """
    Reduce the channel dimension of source tokens via a lightweight GNN.

    Graph construction:
        Nodes = channels (n_channels).  Features per node = token_size values.
        Edges = k-nearest neighbours in a *learned* channel embedding space.

    Message passing:
        Simple GraphSAGE-style: aggregate neighbour features with mean, concat
        with self, project.  Repeated for ``n_layers``.

    Read-out:
        A linear projection from ``n_channels × hidden_dim`` →
        ``latent_dim × token_size`` reshapes back to token format so the rest
        of the pipeline is unaware of the reduction.

    Parameters from ``gnn_reducer`` config block:
        n_channels  – number of input channels (= sources_size for that stream)
        token_size  – spatial token size (must match stream token_size)
        latent_dim  – reduced channel count output
        hidden_dim  – width of GNN hidden layers
        n_layers    – number of message-passing rounds
        k_neighbors – k for the kNN graph
    """

    def __init__(
        self,
        n_channels: int,
        token_size: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
        k_neighbors: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.token_size = token_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.k = min(k_neighbors, n_channels - 1)  # can't exceed #channels-1

        # Learnable channel embeddings used to build the kNN graph
        self.channel_embed = nn.Parameter(
            torch.randn(n_channels, hidden_dim) * 0.02
        )

        # Input projection: token_size → hidden_dim per channel-node
        self.input_proj = nn.Linear(token_size, hidden_dim)

        # Message-passing layers (GraphSAGE-style)
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            )

        # Read-out: project aggregated graph features → reduced token
        self.readout = nn.Sequential(
            nn.Linear(n_channels * hidden_dim, latent_dim * token_size),
        )

    # ------------------------------------------------------------------
    def _build_knn_edges(self) -> torch.Tensor:
        """Return (2, n_channels * k) edge_index from channel_embed."""
        # pairwise distances among channel embeddings
        with torch.no_grad():
            dists = torch.cdist(self.channel_embed, self.channel_embed)  # (C, C)
            # set self-distance to inf so a node is not its own neighbour
            dists.fill_diagonal_(float("inf"))
            # k nearest neighbours per node
            _, topk_idx = dists.topk(self.k, dim=-1, largest=False)  # (C, k)

        src = (
            torch.arange(self.n_channels, device=topk_idx.device)
            .unsqueeze(1)
            .expand_as(topk_idx)
            .reshape(-1)
        )
        dst = topk_idx.reshape(-1)
        return torch.stack([src, dst], dim=0)  # (2, C*k)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., token_size, n_channels)  — batch of source tokens.
               Leading dimensions are flattened and restored automatically.
        Returns:
            (..., token_size, latent_dim)
        """
        # Flatten arbitrary leading dims into a single batch dimension
        leading_shape = x.shape[:-2]
        N = x[..., 0, 0].numel()  # product of leading dims
        x = x.reshape(N, self.token_size, self.n_channels)

        # --- node features: treat each channel as a node -----------------
        # x is (N, token_size, C);  transpose → (N, C, token_size)
        h = x.transpose(-2, -1)                        # (N, C, token_size)
        h = self.input_proj(h)                          # (N, C, hidden_dim)

        # --- build kNN graph from channel embeddings (shared across batch)
        edge_index = self._build_knn_edges()            # (2, E)

        # --- message passing (batched) -----------------------------------
        for mp in self.mp_layers:
            # gather neighbour features
            src_idx = edge_index[0]                     # (E,)
            dst_idx = edge_index[1]                     # (E,)
            neigh_feats = h[:, dst_idx]                 # (N, E, hidden_dim)

            # mean-aggregate per node
            # scatter-mean: for each src node, average its neighbours' features
            agg = torch.zeros_like(h)                   # (N, C, hidden_dim)
            counts = torch.zeros(
                self.n_channels, device=h.device, dtype=h.dtype
            )
            agg.scatter_add_(1, src_idx.unsqueeze(0).unsqueeze(-1).expand(N, -1, self.hidden_dim), neigh_feats)
            counts.scatter_add_(0, src_idx, torch.ones_like(src_idx, dtype=h.dtype))
            counts = counts.clamp(min=1.0)
            agg = agg / counts.unsqueeze(0).unsqueeze(-1)

            # SAGE update: concat(self, agg) → project
            h = mp(torch.cat([h, agg], dim=-1)) + h     # residual

        # --- read-out: flatten graph → reduced token ----------------------
        h_flat = h.reshape(N, -1)                       # (N, C * hidden_dim)
        out = self.readout(h_flat)                       # (N, latent_dim * token_size)
        out = out.reshape(N, self.latent_dim, self.token_size)  # (N, latent_dim, token_size)
        out = out.transpose(-2, -1)                     # (N, token_size, latent_dim)

        # Restore leading dimensions
        out = out.reshape(*leading_shape, self.token_size, self.latent_dim)

        return out
