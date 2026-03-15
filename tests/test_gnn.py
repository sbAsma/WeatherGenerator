"""Unit tests for the GNN-based CAMS channel reducer."""

import pytest
import torch

from weathergen.model.gnn_reducer import CAMSGraphReducer, build_healpix_graph


def test_build_healpix_graph_small():
    """Graph construction at HEALPix level 1 produces valid edge indices and 3-D positions."""
    edge_index, positions = build_healpix_graph(healpix_level=1, k_neighbors=4)

    n_cells = 12 * (2**1) ** 2  # 48
    assert positions.shape == (n_cells, 3), f"Expected ({n_cells}, 3), got {positions.shape}"
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == n_cells * 4  # 4 neighbors per cell

    # all indices must be valid cell ids
    assert edge_index.min() >= 0
    assert edge_index.max() < n_cells

    # positions should be on the unit sphere
    norms = (positions**2).sum(axis=1) ** 0.5
    assert pytest.approx(norms, abs=1e-6) == [1.0] * n_cells


def test_cams_graph_reducer_forward_finite():
    """A tiny CAMSGraphReducer produces finite output of the correct shape."""
    level = 1
    n_cells = 12 * (2**level) ** 2
    token_size = 4
    n_channels = 8
    latent_dim = 3
    hidden_dim = 8
    n_layers = 2
    k = 4

    reducer = CAMSGraphReducer(
        healpix_level=level,
        in_features=token_size * n_channels,
        token_size=token_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        k_neighbors=k,
    )
    reducer.precompute_graph(torch.device("cpu"))

    x = torch.randn(n_cells, token_size, n_channels)
    out = reducer(x)

    assert out.shape == (n_cells, token_size, latent_dim)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_cams_graph_reducer_batched():
    """Batched (B, N, T, C) input produces correct output shape."""
    level = 1
    n_cells = 12 * (2**level) ** 2
    token_size = 4
    n_channels = 8
    latent_dim = 3
    B = 2

    reducer = CAMSGraphReducer(
        healpix_level=level,
        in_features=token_size * n_channels,
        token_size=token_size,
        latent_dim=latent_dim,
        hidden_dim=8,
        n_layers=1,
        k_neighbors=4,
    )
    reducer.precompute_graph(torch.device("cpu"))

    x = torch.randn(B, n_cells, token_size, n_channels)
    out = reducer(x)

    assert out.shape == (B, n_cells, token_size, latent_dim)
    assert torch.isfinite(out).all()


def test_cams_graph_reducer_detects_nan():
    """Injecting a NaN into the input triggers a RuntimeError."""
    level = 1
    n_cells = 12 * (2**level) ** 2

    reducer = CAMSGraphReducer(
        healpix_level=level,
        in_features=2 * 4,
        token_size=2,
        latent_dim=2,
        hidden_dim=8,
        n_layers=1,
        k_neighbors=4,
    )
    reducer.precompute_graph(torch.device("cpu"))

    x = torch.randn(n_cells, 2, 4)
    x[0, 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        reducer(x)


def test_cams_graph_reducer_no_precompute_raises():
    """Calling forward without precompute_graph raises RuntimeError."""
    reducer = CAMSGraphReducer(
        healpix_level=1,
        in_features=2 * 4,
        token_size=2,
        latent_dim=2,
        hidden_dim=8,
        n_layers=1,
        k_neighbors=4,
    )

    x = torch.randn(48, 2, 4)
    with pytest.raises(RuntimeError, match="precompute_graph"):
        reducer(x)
