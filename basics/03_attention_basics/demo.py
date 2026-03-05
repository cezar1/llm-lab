
"""
Self-Attention & Multihead Attention Demo
Hand-wire dot-product attention, visualize per-head weights, test causality, ablate heads
"""

from attention import DotProductAttention, MultiheadAttention, create_causal_mask, test_causality, ablate_num_heads
import torch

def demo_single_token_attention():
    """Demonstrate single-token attention"""
    print("=" * 60)
    print("1. SINGLE-TOKEN ATTENTION")
    print("=" * 60)

    embed_dim = 16
    seq_len = 5

    # Create random query, key, value
    q = torch.randn(1, seq_len, embed_dim)
    k = torch.randn(1, seq_len, embed_dim)
    v = torch.randn(1, seq_len, embed_dim)

    # Initialize attention
    attn = DotProductAttention(embed_dim)

    # Forward pass
    output, weights = attn.forward(q, k, v)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Visualize
    print("\nVisualizing attention weights...")
    attn.visualize_weights(weights, title="Single-Token Attention Weights")


def demo_multihead_attention():
    """Demonstrate multihead attention"""
    print("\n" + "=" * 60)
    print("2. MULTIHEAD ATTENTION")
    print("=" * 60)

    embed_dim = 64
    num_heads = 8
    seq_len = 10

    # Create input
    x = torch.randn(2, seq_len, embed_dim)

    # Initialize multihead attention
    attn = MultiheadAttention(embed_dim, num_heads)
    attn.eval()

    # Forward pass
    output, weights = attn(x, x, x)

    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Head dim: {embed_dim // num_heads}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")

    # Visualize
    print("\nVisualizing all attention heads...")
    attn.visualize_all_heads(title="Multihead Attention: All Heads")

    print("\nComparing head characteristics...")
    attn.visualize_head_comparison()


def demo_causal_attention():
    """Demonstrate causal (autoregressive) attention"""
    print("\n" + "=" * 60)
    print("3. CAUSAL (AUTOREGRESSIVE) ATTENTION")
    print("=" * 60)

    embed_dim = 64
    num_heads = 8
    seq_len = 10

    # Create input
    x = torch.randn(1, seq_len, embed_dim)

    # Create multihead attention
    attn = MultiheadAttention(embed_dim, num_heads)
    attn.eval()

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask (first 5x5):\n{causal_mask[:5, :5].int()}")

    # Forward pass with causal mask
    output, weights = attn(x, x, x, mask=causal_mask)

    # Visualize
    print("\nVisualizing causal attention (should be lower triangular)...")
    attn.visualize_all_heads(title="Causal Multihead Attention")

    # Test causality
    print("\nTesting causality constraints...")
    is_causal = test_causality(attn, seq_len)

    if is_causal:
        print("\n✓ Causality test PASSED: No attention to future tokens")
    else:
        print("\n✗ Causality test FAILED: Attention to future tokens detected")


def demo_attention_ablation():
    """Demonstrate ablation studies"""
    print("\n" + "=" * 60)
    print("4. ABLATION: VARYING NUMBER OF HEADS")
    print("=" * 60)

    ablate_num_heads()


def demo_attention_masking_patterns():
    """Explore different masking patterns"""
    print("\n" + "=" * 60)
    print("5. ATTENTION MASKING PATTERNS")
    print("=" * 60)

    seq_len = 10
    embed_dim = 32
    num_heads = 4

    x = torch.randn(1, seq_len, embed_dim)
    attn = MultiheadAttention(embed_dim, num_heads)
    attn.eval()

    # Test different masks
    masks = {
        'No Mask': torch.ones(seq_len, seq_len),
        'Causal': torch.tril(torch.ones(seq_len, seq_len)),
        'Local Window (±2)': torch.diag(torch.ones(seq_len)) +
                             torch.diag(torch.ones(seq_len-1), diagonal=1) +
                             torch.diag(torch.ones(seq_len-1), diagonal=-1) +
                             torch.diag(torch.ones(seq_len-2), diagonal=2) +
                             torch.diag(torch.ones(seq_len-2), diagonal=-2),
    }

    for mask_name, mask in masks.items():
        print(f"\n--- Testing {mask_name} ---")
        output, weights = attn(x, x, x, mask=mask)
        attn.visualize_all_heads(title=f"Attention Pattern: {mask_name}")


def demo_attention_head_diversity():
    """Show that different heads specialize"""
    print("\n" + "=" * 60)
    print("6. HEAD SPECIALIZATION & DIVERSITY")
    print("=" * 60)

    embed_dim = 64
    num_heads = 8
    seq_len = 12

    # Create structured input (to see if heads specialize)
    x = torch.randn(1, seq_len, embed_dim)

    attn = MultiheadAttention(embed_dim, num_heads)
    attn.eval()

    output, weights = attn(x, x, x)

    w = weights[0].detach().numpy()

    # Analyze head patterns
    import numpy as np

    print("\nHead Analysis:")
    print("-" * 60)

    for head_idx in range(num_heads):
        w_head = w[head_idx]

        # Compute statistics
        entropy = -np.sum(w_head * np.log(w_head + 1e-10), axis=1).mean()
        max_weight = w_head.max()
        min_weight = w_head.min()

        # Check if attending to few positions (peaked) or many (distributed)
        num_attending_to_multiple = np.sum((w_head > 0.1).sum(axis=1) > 3)
        coverage = (w_head > 0.01).sum(axis=1).mean()

        print(f"Head {head_idx:2d}: Entropy={entropy:.3f}, Max={max_weight:.3f}, "
              f"Coverage={coverage:.1f} positions")

    print("\nVisualization:")
    attn.visualize_all_heads(title="Head Specialization Patterns")


if __name__ == "__main__":
    demo_single_token_attention()
    demo_multihead_attention()
    demo_causal_attention()
    demo_attention_ablation()
    demo_attention_masking_patterns()
    demo_attention_head_diversity()

    print("\n" + "=" * 60)
    print("ATTENTION BASICS DEMO COMPLETE")
    print("=" * 60)
    print("\nKey insights:")
    print("- Attention weights are normalized using softmax")
    print("- Multiple heads allow different attention patterns simultaneously")
    print("- Causal masking prevents looking at future tokens")
    print("- Different heads specialize in different types of relationships")
    print("- More heads can capture more diverse patterns (with diminishing returns)")
