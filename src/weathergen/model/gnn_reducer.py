"""
GNN-based channel reducer for CAMS data streams.

Compresses high-dimensional CAMS chemical species channels into a compact latent
representation using message passing on a KNN graph built over HEALPix cell
centroids on the unit sphere.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def build_healpix_graph(healpix_level: int, k_neighbors: int):
    """Construct a KNN graph over HEALPix cell centroids on the unit sphere.

    Parameters
    ----------
    healpix_level : int
        HEALPix nside = 2**healpix_level.
    k_neighbors : int
        Number of nearest neighbours per node.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E)
        Source-target edge indices.
    positions : np.ndarray, shape (N, 3)
        3-D Cartesian positions of cell centroids on the unit sphere.
    """
    import astropy_healpix as hp

    nside = 2**healpix_level
    n_cells = 12 * nside**2
    ipix = np.arange(n_cells)

    # HEALPix cell centres → (theta, phi) → Cartesian xyz
    theta, phi = hp.healpix_to_lonlat(ipix, nside, order="nested")
    # astropy returns Longitude/Latitude objects; convert to radians
    lon = np.asarray(theta.rad, dtype=np.float64)
    lat = np.asarray(phi.rad, dtype=np.float64)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    positions = np.stack([x, y, z], axis=-1)

    # KNN on unit sphere
    tree = cKDTree(positions)
    # k+1 because query includes self
    _, indices = tree.query(positions, k=k_neighbors + 1)
    # exclude self-loop (first column)
    neighbors = indices[:, 1:]

    src = np.repeat(np.arange(n_cells), k_neighbors)
    dst = neighbors.flatten()
    edge_index = np.stack([src, dst], axis=0)

    return edge_index, positions


def _check_finite(tensor: torch.Tensor, tag: str) -> None:
    """Raise immediately if tensor contains NaN or Inf."""
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{tag} contains NaN or Inf")


class GNNBlock(nn.Module):
    """Single message-passing layer: edge update → scatter-aggregate → node update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, D)  node features
        edge_index : (2, E)  long tensor

        Returns
        -------
        x_out : (N, D)
        """
        src, dst = edge_index[0], edge_index[1]

        # edge update
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        edge_msg = self.edge_mlp(edge_feat)
        edge_msg = self.edge_norm(edge_msg)
        _check_finite(edge_msg, "GNNBlock.edge_msg")

        # scatter-mean aggregate (use edge_msg dtype to handle mixed precision)
        agg = torch.zeros(x.shape[0], edge_msg.shape[1], device=x.device, dtype=edge_msg.dtype)
        count = torch.zeros(x.shape[0], 1, device=x.device, dtype=edge_msg.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=x.device, dtype=edge_msg.dtype))
        count = count.clamp(min=1.0)
        agg = agg / count

        # node update with residual
        node_inp = torch.cat([x.to(agg.dtype), agg], dim=-1)
        x_out = x.to(agg.dtype) + self.node_mlp(node_inp)
        x_out = self.node_norm(x_out)
        _check_finite(x_out, "GNNBlock.node_update")

        return x_out


class CAMSGraphReducer(nn.Module):
    """GNN-based channel reducer for CAMS streams.

    Flattens per-cell tokens ``(N, token_size, n_channels)`` →
    ``(N, token_size × n_channels)``, projects into hidden space, adds learnable
    positional encodings from 3-D xyz coordinates, runs ``n_layers`` GNN blocks,
    then projects back to ``(N, token_size, latent_dim)``.

    Graph tensors are plain attributes (not buffers) recomputed once per rank
    via :meth:`precompute_graph`, so they are excluded from ``state_dict``.
    """

    def __init__(
        self,
        healpix_level: int,
        in_features: int,
        token_size: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        k_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.healpix_level = healpix_level
        self.in_features = in_features
        self.token_size = token_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors

        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.pos_enc = nn.Linear(3, hidden_dim, bias=False)

        self.blocks = nn.ModuleList([GNNBlock(hidden_dim) for _ in range(n_layers)])

        self.output_proj = nn.Linear(hidden_dim, token_size * latent_dim)

        # Populated by precompute_graph(); not stored in state_dict
        self.edge_index: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None

    def precompute_graph(self, device: torch.device) -> None:
        """Build the HEALPix KNN graph and move tensors to ``device``.

        Should be called once after the model is moved to the target device.
        """
        edge_index_np, positions_np = build_healpix_graph(
            self.healpix_level, self.k_neighbors
        )
        self.edge_index = torch.from_numpy(edge_index_np).long().to(device)
        self.positions = torch.from_numpy(positions_np).float().to(device)
        logger.info(
            "CAMSGraphReducer: graph precomputed on %s  "
            "(cells=%d, edges=%d, k=%d)",
            device,
            self.positions.shape[0],
            self.edge_index.shape[1],
            self.k_neighbors,
        )

    @staticmethod
    def pool_to_cells(
        tokens: torch.Tensor,
        cell_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter-mean variable-length tokens into one vector per cell.

        Parameters
        ----------
        tokens : (total_tokens, token_size, n_channels)
        cell_lens : (n_cells,)  number of tokens per cell

        Returns
        -------
        pooled : (n_cells, token_size, n_channels)
        """
        n_cells = cell_lens.shape[0]
        cell_ids = torch.repeat_interleave(
            torch.arange(n_cells, device=tokens.device), cell_lens
        )
        # (total_tokens, 1, 1) → broadcast
        cell_ids_exp = cell_ids.unsqueeze(1).unsqueeze(2).expand_as(tokens)

        pooled = torch.zeros(
            n_cells, tokens.shape[1], tokens.shape[2],
            dtype=tokens.dtype, device=tokens.device,
        )
        pooled.scatter_add_(0, cell_ids_exp, tokens)

        counts = cell_lens.float().clamp(min=1.0).unsqueeze(1).unsqueeze(2)
        pooled = pooled / counts

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, token_size, n_channels) or (B, N, token_size, n_channels)

        Returns
        -------
        out : same leading dims, last two are (token_size, latent_dim)
        """
        if self.edge_index is None:
            raise RuntimeError(
                "CAMSGraphReducer.precompute_graph() must be called before forward()"
            )

        batched = x.dim() == 4
        if batched:
            B, N, T, C = x.shape
            # replicate graph for block-diagonal batching
            offsets = (
                torch.arange(B, device=x.device).unsqueeze(1) * N
            )  # (B, 1)
            ei = self.edge_index.unsqueeze(0) + offsets.unsqueeze(0)  # (2, E) + (B, 1) needs broadcast
            ei = (
                self.edge_index.unsqueeze(0).expand(2, B, -1)
                + offsets.unsqueeze(0).expand(2, B, -1).reshape(2, B, -1)
            )
            # Proper block-diagonal edge index
            ei_list = []
            for b in range(B):
                ei_list.append(self.edge_index + b * N)
            edge_index = torch.cat(ei_list, dim=1)

            x = x.reshape(B * N, T, C)
            pos = self.positions.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 3)
        else:
            N, T, C = x.shape
            edge_index = self.edge_index
            pos = self.positions

        _check_finite(x, "CAMSGraphReducer.input")

        # flatten tokens per cell
        h = x.reshape(x.shape[0], T * C)  # (N_total, T*C)
        h = self.input_proj(h)  # (N_total, hidden_dim)
        h = h + self.pos_enc(pos)  # add positional encoding
        _check_finite(h, "CAMSGraphReducer.after_input_proj")

        for i, block in enumerate(self.blocks):
            h = block(h, edge_index)
            _check_finite(h, f"CAMSGraphReducer.block_{i}")

        out = self.output_proj(h)  # (N_total, T*latent_dim)
        out = out.reshape(x.shape[0], T, self.latent_dim)
        _check_finite(out, "CAMSGraphReducer.output")

        if batched:
            out = out.reshape(B, N, T, self.latent_dim)

        return out
