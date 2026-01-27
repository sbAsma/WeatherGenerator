"""
Quick test that ABP shapes are correct.
Run: python test_abp.py
"""

import torch
from weathergen.model.chemistry_embedding import ChemistryStreamEmbedding


def test_shapes():
    """Test ABP output shapes."""
    print("Testing ABP shapes...")
    
    # Create embedding
    embed = ChemistryStreamEmbedding(
        n_species=50,
        n_levels=25,
        n_emissions=10,
        spatial_h=100,
        spatial_w=100,
        d_embedding=512,
        d_intermediate=1800,
        n_inducing=64
    )
    
    # Random CAMS data
    B, H, W = 2, 100, 100
    C_in = 50 * 25 + 10  # species*levels + emissions
    x = torch.randn(B, H, W, C_in)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = embed(x)
    print(f"Output shape: {output.shape}")
    
    # Check shape
    expected = (B, 512)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    
    # Check no NaN
    assert not torch.isnan(output).any(), "Output contains NaN!"
    
    print(f"✅ Test passed! Output shape: {output.shape}")
    print(f"Memory per batch: {output.element_size() * output.nelement() / 1e6:.1f} MB")


def test_gradient_flow():
    """Test gradients flow through ABP."""
    print("\nTesting gradient flow...")
    
    embed = ChemistryStreamEmbedding()
    x = torch.randn(2, 100, 100, 1260, requires_grad=True)
    
    output = embed(x)
    loss = output.mean()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input!"
    print(f"✅ Gradients flow! Input grad shape: {x.grad.shape}")


def test_inference_speed():
    """Time ABP inference."""
    print("\nTesting inference speed...")
    
    import time
    
    embed = ChemistryStreamEmbedding()
    x = torch.randn(32, 100, 100, 1260)
    
    # Warmup
    for _ in range(3):
        _ = embed(x)
    
    # Time
    start = time.time()
    for _ in range(10):
        _ = embed(x)
    elapsed = time.time() - start
    
    per_batch = elapsed / (10 * 32)
    print(f"✅ Inference time: {per_batch*1000:.2f} ms per sample")
    print(f"   Throughput: {1/per_batch:.1f} samples/sec")


if __name__ == "__main__":
    test_shapes()
    test_gradient_flow()
    test_inference_speed()
    print("\n✅ All tests passed!")
