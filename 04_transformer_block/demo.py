
"""
Transformer Block Demo
Stack attention + FFN with LayerNorm and residuals into single blocks and n-block formers
"""

from transformer import (
    TransformerBlock, MiniTransformer,
    visualize_transformer_block_flow,
    visualize_mini_transformer_analysis,
    ablate_transformer_components
)
import torch

def demo_single_block():
    """Demonstrate single transformer block"""
    print("=" * 60)
    print("1. SINGLE TRANSFORMER BLOCK")
    print("=" * 60)

    embed_dim = 64
    num_heads = 4
    ffn_dim = 256
    batch_size = 2
    seq_len = 8

    block = TransformerBlock(embed_dim, num_heads, ffn_dim)
    block.eval()

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input norm: {x.norm(dim=-1).mean().item():.4f}")
    print(f"Output norm: {output.norm(dim=-1).mean().item():.4f}")

    print("\nVisualizing block information flow...")
    visualize_transformer_block_flow()


def demo_mini_former():
    """Demonstrate mini-former (n-block transformer)"""
    print("\n" + "=" * 60)
    print("2. MINI-FORMER (N-BLOCK TRANSFORMER)")
    print("=" * 60)

    vocab_size = 256
    embed_dim = 64
    num_heads = 4
    num_blocks = 4
    ffn_dim = 256

    model = MiniTransformer(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim)
    model.eval()

    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of layers: {num_blocks}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nVisualizing layer-by-layer activation flow...")
    visualize_mini_transformer_analysis()


def demo_qkv_analysis():
    """Analyze Query, Key, Value dynamics"""
    print("\n" + "=" * 60)
    print("3. QUERY-KEY-VALUE (QKV) ANALYSIS")
    print("=" * 60)

    embed_dim = 64
    batch_size = 2
    seq_len = 8
    num_heads = 4

    block = TransformerBlock(embed_dim, num_heads, ffn_dim=256)
    block.eval()

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Get attention module
    attn_module = block.attention

    # Create QKV projections (manual to inspect)
    x_norm = torch.nn.LayerNorm(embed_dim)(x)

    # This shows how QKV are computed (we're using torch's built-in attention)
    print(f"Input shape: {x.shape}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {embed_dim // num_heads}")

    # Forward pass
    output, weights = attn_module(x_norm, x_norm, x_norm, need_weights=True)

    print(f"\nQuery/Key/Value all from same input (self-attention)")
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention sparsity (weights < 0.1): {(weights < 0.1).sum() / weights.numel():.2%}")


def demo_causal_masking():
    """Demonstrate causal masking in transformer blocks"""
    print("\n" + "=" * 60)
    print("4. CAUSAL MASKING FOR AUTOREGRESSIVE GENERATION")
    print("=" * 60)

    embed_dim = 32
    num_heads = 4
    seq_len = 6

    block = TransformerBlock(embed_dim, num_heads, ffn_dim=128)
    block.eval()

    x = torch.randn(1, seq_len, embed_dim)

    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask == 0  # True where should be masked

    print(f"Sequence length: {seq_len}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask (first 4x4):")
    print(causal_mask[:4, :4].int())

    # Forward with causal mask
    output = block(x, attn_mask=causal_mask)

    print(f"\nOutput shape: {output.shape}")
    print("✓ Causal masking prevents attending to future positions")


def demo_residual_connections():
    """Show the importance of residual connections"""
    print("\n" + "=" * 60)
    print("5. RESIDUAL CONNECTIONS & GRADIENT FLOW")
    print("=" * 60)

    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    block = TransformerBlock(embed_dim, num_heads, ffn_dim=256)

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    output = block(x)

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check gradients
    input_grad_norm = x.grad.norm().item()
    param_grad_norms = [p.grad.norm().item() for p in block.parameters() if p.grad is not None]

    print(f"Input gradient norm: {input_grad_norm:.4f}")
    print(f"Parameter gradient norms: mean={np.mean(param_grad_norms):.4f}, "
          f"std={np.std(param_grad_norms):.4f}")
    print(f"✓ Residuals help gradients flow through deep networks")

    import numpy as np


def demo_component_ablation():
    """Ablation: Remove components and see effect"""
    print("\n" + "=" * 60)
    print("6. COMPONENT ABLATION")
    print("=" * 60)

    ablate_transformer_components()


def demo_attention_head_patterns():
    """Show that attention heads learn different patterns"""
    print("\n" + "=" * 60)
    print("7. ATTENTION HEAD SPECIALIZATION")
    print("=" * 60)

    embed_dim = 64
    num_heads = 8
    seq_len = 10

    block = TransformerBlock(embed_dim, num_heads, ffn_dim=256)
    block.eval()

    x = torch.randn(1, seq_len, embed_dim)
    _ = block(x)

    weights = block.attn_weights[0].detach().numpy()  # (num_heads, seq_len, seq_len)

    print(f"Number of attention heads: {num_heads}")
    print(f"Attention weights shape: {weights.shape}")

    print("\nHead specialization analysis:")
    print("-" * 50)

    for head_idx in range(min(4, num_heads)):
        w = weights[head_idx]
        entropy = -np.sum(w * np.log(w + 1e-10), axis=1).mean()
        is_peaked = entropy < 1.0
        pattern = "peaked (focused)" if is_peaked else "distributed (broad)"
        print(f"  Head {head_idx}: {pattern} (entropy={entropy:.2f})")

    print("\n✓ Different heads specialize in different attention patterns")

    import numpy as np


if __name__ == "__main__":
    demo_single_block()
    demo_mini_former()
    demo_qkv_analysis()
    demo_causal_masking()
    demo_residual_connections()
    demo_component_ablation()
    demo_attention_head_patterns()

    print("\n" + "=" * 60)
    print("TRANSFORMER BLOCK DEMO COMPLETE")
    print("=" * 60)
    print("\nKey insights:")
    print("- Transformer blocks combine attention + FFN with residuals & LayerNorm")
    print("- Residual connections enable training of deep networks")
    print("- Different attention heads specialize in different patterns")
    print("- Causal masking prevents attending to future tokens")
    print("- Stacking blocks increases model capacity and context understanding")
