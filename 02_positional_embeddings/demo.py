
"""
Positional Embeddings Demo
Compare sinusoidal, learned, RoPE, and ALiBi methods
Animate position encoding in 3D, ablate dimensions
"""

from positional_embeddings import PositionalEmbeddingAnalyzer
import matplotlib.pyplot as plt

def demo_all_methods():
    """Visualize all four positional encoding methods"""
    print("=" * 60)
    print("POSITIONAL EMBEDDINGS: ALL FOUR METHODS")
    print("=" * 60)

    analyzer = PositionalEmbeddingAnalyzer(embed_dim=64, seq_len=20)

    print("\n1. Heatmap comparison of encoding methods...")
    analyzer.visualize_all_methods()

    print("\n2. Position similarity matrices...")
    analyzer.plot_position_distance_matrix()

    print("\n3. Embedding norm evolution...")
    analyzer.plot_position_norm_evolution()


def demo_position_interpolation():
    """Show how encodings handle position interpolation"""
    print("\n" + "=" * 60)
    print("INTERPOLATION: EXTRAPOLATION BEYOND TRAINING LENGTH")
    print("=" * 60)

    from positional_embeddings import SinusoidalPE, LearnedPE, RoPEPE
    import torch

    seq_lens = [10, 50, 100, 500]
    embed_dim = 64

    sin_pe = SinusoidalPE(embed_dim, max_seq_len=1000)
    learned_pe = LearnedPE(embed_dim, max_seq_len=1000)
    rope_pe = RoPEPE(embed_dim, max_seq_len=1000)

    # Test extrapolation
    print("\nTesting position encoding extrapolation...")
    print("(Can they handle sequences longer than training?)")

    for seq_len in seq_lens:
        x = torch.randn(1, seq_len, embed_dim)

        try:
            sin_enc = sin_pe(x)
            sin_ok = True
        except:
            sin_ok = False

        try:
            learned_enc = learned_pe(x)
            learned_ok = True
        except:
            learned_ok = False

        try:
            rope_enc = rope_pe(x)
            rope_ok = True
        except:
            rope_ok = False

        print(f"  Seq Len {seq_len:3d}: Sinusoidal={sin_ok}, Learned={learned_ok}, RoPE={rope_ok}")

    print("\nKey insight: RoPE & Sinusoidal handle extrapolation better than Learned PE")


def demo_alibi_attention_bias():
    """Visualize ALiBi attention bias patterns"""
    print("\n" + "=" * 60)
    print("ALiBi: ATTENTION WITH LINEAR BIASES")
    print("=" * 60)

    from positional_embeddings import ALiBiPE
    import torch

    alibi = ALiBiPE(num_heads=8)
    seq_len = 16

    bias = alibi(seq_len, torch.device('cpu'))

    print(f"ALiBi bias shape: {bias.shape}")
    print(f"Number of heads: {bias.shape[0]}")
    print(f"Sequence length: {seq_len}")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for head_idx in range(8):
        head_bias = bias[head_idx].numpy()
        im = axes[head_idx].imshow(head_bias, cmap='RdBu_r', aspect='auto')
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[head_idx])

    plt.suptitle('ALiBi Bias Patterns Across Heads')
    plt.tight_layout()
    plt.show()

    print("\nKey insight: Different heads have different slopes (bias strengths)")
    print(f"Slopes: {alibi.slopes.tolist()}")


def demo_ablation_embedding_dim():
    """Ablation: embedding dimension impact"""
    print("\n" + "=" * 60)
    print("ABLATION: EMBEDDING DIMENSION IMPACT")
    print("=" * 60)

    analyzer = PositionalEmbeddingAnalyzer(embed_dim=64, seq_len=20)
    analyzer.ablate_embedding_dim()

    print("\nKey insight: Larger embedding dims help differentiate positions")


def demo_comparison_table():
    """Create a comparison table of all methods"""
    print("\n" + "=" * 60)
    print("COMPARISON TABLE: POSITIONAL ENCODING METHODS")
    print("=" * 60)

    methods = {
        'Sinusoidal': {
            'Learnable': 'No',
            'Extrapolation': 'Excellent',
            'Frequency': 'Fixed',
            'Complexity': 'Low',
            'Used in': 'Original Transformer'
        },
        'Learned': {
            'Learnable': 'Yes',
            'Extrapolation': 'Poor',
            'Frequency': 'Dynamic',
            'Complexity': 'Very Low',
            'Used in': 'BERT, most modern models'
        },
        'RoPE': {
            'Learnable': 'No',
            'Extrapolation': 'Excellent',
            'Frequency': 'Fixed',
            'Complexity': 'Medium',
            'Used in': 'LLaMA, Falcon'
        },
        'ALiBi': {
            'Learnable': 'Partial',
            'Extrapolation': 'Excellent',
            'Frequency': 'N/A (bias)',
            'Complexity': 'Low',
            'Used in': 'BLOOM, T5'
        }
    }

    print("\n{:<15} {:<12} {:<15} {:<15} {:<15} {:<20}".format(
        "Method", "Learnable", "Extrapolation", "Frequency", "Complexity", "Used In"
    ))
    print("-" * 92)

    for method, props in methods.items():
        print("{:<15} {:<12} {:<15} {:<15} {:<15} {:<20}".format(
            method,
            props['Learnable'],
            props['Extrapolation'],
            props['Frequency'],
            props['Complexity'],
            props['Used in']
        ))

    print("\nChoosing your PE method:")
    print("  - Fixed-length sequences → Use Learned PE (simple, effective)")
    print("  - Long-context or extrapolation needed → Use RoPE or Sinusoidal")
    print("  - No position embeddings overhead → Use ALiBi (add to attention)")
    print("  - Balanced approach → Use RoPE (extrapolates well, computationally efficient)")


if __name__ == "__main__":
    demo_all_methods()
    demo_position_interpolation()
    demo_alibi_attention_bias()
    demo_ablation_embedding_dim()
    demo_comparison_table()

    print("\n" + "=" * 60)
    print("POSITIONAL EMBEDDINGS DEMO COMPLETE")
    print("=" * 60)
