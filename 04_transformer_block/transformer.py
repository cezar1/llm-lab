
"""
Transformer Block: LayerNorm + Residuals + Attention + FFN
Stack to build mini-former, dissect QKV, ablate components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class TransformerBlock(nn.Module):
    """Single transformer block: Attention + FFN with LayerNorm and residuals"""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Track activations for analysis
        self.attn_weights = None
        self.activations = {}

    def forward(self, x, attn_mask=None, need_weights=True):
        """
        x: (batch, seq_len, embed_dim)
        attn_mask: optional attention mask
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=need_weights
        )
        x = x + attn_out  # Residual connection
        self.attn_weights = attn_weights.detach()

        # FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual connection

        return x

    def get_activation_stats(self, x):
        """Analyze activation statistics"""
        stats = {}

        # Norm stats
        stats['input_norm'] = x.norm(dim=-1).mean().item()
        stats['input_std'] = x.std(dim=-1).mean().item()

        # Attention contribution
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=True)
        stats['attn_contribution'] = attn_out.norm(dim=-1).mean().item()

        # FFN contribution
        x_with_attn = x + attn_out
        x_norm2 = self.norm2(x_with_attn)
        ffn_out = self.ffn(x_norm2)
        stats['ffn_contribution'] = ffn_out.norm(dim=-1).mean().item()

        return stats


class MiniTransformer(nn.Module):
    """Stack multiple transformer blocks into a mini-former"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, ffn_dim, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attn_mask=None):
        """
        input_ids: (batch, seq_len) LongTensor
        attn_mask: optional attention mask
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.token_embed(input_ids)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Pass through blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # Final normalization
        x = self.norm(x)

        # Predictions
        logits = self.head(x)

        return logits

    def analyze_layer_outputs(self, input_ids):
        """Analyze activations at each layer"""
        batch_size, seq_len = input_ids.shape
        x = self.token_embed(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        layer_stats = []

        for layer_idx, block in enumerate(self.blocks):
            stats = block.get_activation_stats(x)
            stats['layer'] = layer_idx
            layer_stats.append(stats)
            x = block(x)

        return layer_stats


def visualize_transformer_block_flow():
    """Visualize information flow through a transformer block"""
    embed_dim = 64
    batch_size = 2
    seq_len = 8

    block = TransformerBlock(embed_dim, num_heads=4, ffn_dim=256)

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)

    # Visualize norms at each stage
    x_norm = torch.nn.LayerNorm(embed_dim)(x)
    attn_out, _ = block.attention(x_norm, x_norm, x_norm, need_weights=True)
    x_after_attn = x + attn_out

    x_norm2 = torch.nn.LayerNorm(embed_dim)(x_after_attn)
    ffn_out = block.ffn(x_norm2)
    x_after_ffn = x_after_attn + ffn_out

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def plot_activation(ax, tensor, title):
        data = tensor[0].detach().numpy()  # First batch
        ax.imshow(data, aspect='auto', cmap='RdBu')
        ax.set_title(title)
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Sequence Position')
        plt.colorbar(ax.images[0], ax=ax)

    plot_activation(axes[0, 0], x, 'Input (x)')
    plot_activation(axes[0, 1], attn_out, 'Attention Output')
    plot_activation(axes[0, 2], x_after_attn, 'After Attention + Residual')
    plot_activation(axes[1, 0], ffn_out, 'FFN Output')
    plot_activation(axes[1, 1], x_after_ffn, 'Final Output')

    # Activation norms
    norms = {
        'Input': x.norm(dim=-1).mean().item(),
        'Attn Out': attn_out.norm(dim=-1).mean().item(),
        'After Attn': x_after_attn.norm(dim=-1).mean().item(),
        'FFN Out': ffn_out.norm(dim=-1).mean().item(),
        'Final': x_after_ffn.norm(dim=-1).mean().item()
    }

    ax = axes[1, 2]
    ax.bar(norms.keys(), norms.values())
    ax.set_ylabel('Mean Norm')
    ax.set_title('Activation Norms')
    ax.grid(True, alpha=0.3, axis='y')
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.tight_layout()
    plt.show()


def visualize_mini_transformer_analysis():
    """Analyze activation flow through mini-transformer"""
    model = MiniTransformer(
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_blocks=4,
        ffn_dim=256
    )
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, 256, (2, 8))

    # Analyze
    layer_stats = model.analyze_layer_outputs(input_ids)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    layers = [s['layer'] for s in layer_stats]
    input_norms = [s['input_norm'] for s in layer_stats]
    attn_contribs = [s['attn_contribution'] for s in layer_stats]
    ffn_contribs = [s['ffn_contribution'] for s in layer_stats]

    axes[0].plot(layers, input_norms, 'o-', label='Input Norm')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Norm')
    axes[0].set_title('Input Norm by Layer')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, attn_contribs, 'o-', label='Attention')
    axes[1].plot(layers, ffn_contribs, 's-', label='FFN')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Contribution Norm')
    axes[1].set_title('Component Contributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Ratio of contributions
    ratios = [a / (f + 1e-6) for a, f in zip(attn_contribs, ffn_contribs)]
    axes[2].plot(layers, ratios, 'o-')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Attn / FFN Ratio')
    axes[2].set_title('Attention vs FFN Balance')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def ablate_transformer_components():
    """Ablation: What happens without attention, FFN, residuals, layernorm?"""
    print("\n" + "=" * 60)
    print("ABLATION: TRANSFORMER COMPONENTS")
    print("=" * 60)

    embed_dim = 64
    batch_size = 4
    seq_len = 10
    num_heads = 4
    ffn_dim = 256

    x = torch.randn(batch_size, seq_len, embed_dim)

    results = {}

    # Full block
    full_block = TransformerBlock(embed_dim, num_heads, ffn_dim)
    full_out = full_block(x)
    results['Full Block'] = full_out.norm(dim=-1).mean().item()

    # Without residuals (manual)
    norm1 = nn.LayerNorm(embed_dim)
    attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    attn_out, _ = attn(norm1(x), norm1(x), norm1(x), need_weights=False)
    norm2 = nn.LayerNorm(embed_dim)
    ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim))
    ffn_out = ffn(norm2(attn_out))
    results['No Residuals'] = ffn_out.norm(dim=-1).mean().item()

    # Without LayerNorm
    attn_no_norm, _ = attn(x, x, x, need_weights=False)
    x_no_norm = x + attn_no_norm
    ffn_no_norm = ffn(x_no_norm)
    no_norm_out = x_no_norm + ffn_no_norm
    results['No LayerNorm'] = no_norm_out.norm(dim=-1).mean().item()

    # Just attention
    attn_only, _ = attn(norm1(x), norm1(x), norm1(x), need_weights=False)
    attn_res = x + attn_only
    results['Attn Only'] = attn_res.norm(dim=-1).mean().item()

    # Just FFN
    ffn_only = ffn(norm2(x))
    ffn_res = x + ffn_only
    results['FFN Only'] = ffn_res.norm(dim=-1).mean().item()

    # Print results
    print("\nOutput norms (higher = more signal):")
    for config, norm in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {config:<20}: {norm:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values())
    plt.ylabel('Output Norm')
    plt.title('Effect of Removing Components')
    plt.grid(True, alpha=0.3, axis='y')
    for label in plt.gca().get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()
    plt.show()
