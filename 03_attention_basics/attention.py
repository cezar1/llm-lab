
"""
Self-Attention & Multihead Attention
Hand-wire dot-product attention, visualize weight heatmaps, test causality
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class DotProductAttention:
    """Hand-wired single-token attention (Query-Key-Value)"""

    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

    def forward(self, q, k, v, mask=None):
        """
        q: (batch, seq_len, embed_dim)
        k: (batch, seq_len, embed_dim)
        v: (batch, seq_len, embed_dim)
        mask: (seq_len, seq_len) optional causal mask
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        weights = torch.nan_to_num(weights, 0)  # Handle -inf cases

        # Apply attention weights to values
        output = torch.matmul(weights, v)  # (batch, seq_len, embed_dim)

        return output, weights

    def visualize_weights(self, weights, title="Attention Weights"):
        """Visualize attention weight matrix"""
        w = weights[0].detach().numpy()  # First batch, remove batch dim

        plt.figure(figsize=(10, 8))
        plt.imshow(w, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(title)
        plt.tight_layout()
        plt.show()


class MultiheadAttention(torch.nn.Module):
    """Multihead attention with learnable projections"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        # Projections
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.attention_weights = None  # Store for visualization

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, seq_len, embed_dim)
        mask: (seq_len, seq_len) optional
        """
        batch_size, seq_len, _ = q.shape

        # Project
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape for multihead: (batch, seq_len, embed_dim) → (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len, seq_len)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, 0)
        self.attention_weights = weights.detach()  # Store for visualization

        # Apply weights
        output = torch.matmul(weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, embed_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Final projection
        output = self.out_proj(output)

        return output, weights

    def visualize_all_heads(self, title="Multihead Attention Weights"):
        """Visualize attention weights for all heads"""
        if self.attention_weights is None:
            print("No attention weights stored. Run forward pass first.")
            return

        weights = self.attention_weights[0].detach().numpy()  # (num_heads, seq_len, seq_len)
        num_heads = weights.shape[0]

        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 12))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            w = weights[head_idx]
            im = axes[head_idx].imshow(w, cmap='viridis', aspect='auto')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key Pos')
            axes[head_idx].set_ylabel('Query Pos')
            plt.colorbar(im, ax=axes[head_idx])

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def visualize_head_comparison(self):
        """Compare attention patterns across heads"""
        if self.attention_weights is None:
            print("No attention weights stored.")
            return

        weights = self.attention_weights[0].detach().numpy()  # (num_heads, seq_len, seq_len)

        # Compute entropy and sparsity for each head
        entropies = []
        sparsities = []

        for head_idx in range(self.num_heads):
            w = weights[head_idx]
            # Entropy: how spread out is attention?
            entropy = -np.sum(w * np.log(w + 1e-10), axis=1).mean()
            entropies.append(entropy)

            # Sparsity: what fraction of weights are near-zero?
            sparsity = np.sum(w < 0.1) / w.size
            sparsities.append(sparsity)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(range(self.num_heads), entropies)
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Attention Entropy by Head')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(range(self.num_heads), sparsities)
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Sparsity')
        ax2.set_title('Attention Sparsity by Head')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()


def create_causal_mask(seq_len, device='cpu'):
    """Create causal mask (prevent attention to future tokens)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def test_causality(attn_module, seq_len=8):
    """Test that causal attention prevents looking at future tokens"""
    print(f"\nTesting causality for sequence length {seq_len}...")

    x = torch.randn(1, seq_len, attn_module.embed_dim)
    causal_mask = create_causal_mask(seq_len)

    # Forward pass with causal mask
    _, weights = attn_module(x, x, x, mask=causal_mask)
    weights = weights[0].detach().numpy()  # (num_heads, seq_len, seq_len)

    # Check causality: for each position i, attention to j should be 0 if j > i
    print("Checking causality (no future attention):")
    all_causal = True

    for head_idx in range(attn_module.num_heads):
        w = weights[head_idx]
        # Sum weights above diagonal (future)
        future_weight = np.triu(w, k=1).sum()
        is_causal = future_weight < 1e-5  # Should be near-zero

        if not is_causal:
            all_causal = False
            print(f"  Head {head_idx}: FAIL (future weight = {future_weight:.4f})")
        else:
            print(f"  Head {head_idx}: PASS")

    return all_causal


def ablate_num_heads():
    """Ablation: how does number of heads affect attention?"""
    print("\n" + "=" * 60)
    print("ABLATION: NUMBER OF HEADS")
    print("=" * 60)

    embed_dim = 64
    seq_len = 10
    x = torch.randn(1, seq_len, embed_dim)

    head_counts = [1, 2, 4, 8]
    metrics = {
        'avg_entropy': [],
        'avg_sparsity': [],
        'output_norm': []
    }

    for num_heads in head_counts:
        attn = MultiheadAttention(embed_dim, num_heads)
        attn.eval()

        output, weights = attn(x, x, x)

        # Compute metrics
        w = weights[0].detach().numpy()
        entropy = -np.sum(w * np.log(w + 1e-10), axis=(1, 2)).mean()
        sparsity = np.sum(w < 0.1) / w.size
        out_norm = output.norm().item()

        metrics['avg_entropy'].append(entropy)
        metrics['avg_sparsity'].append(sparsity)
        metrics['output_norm'].append(out_norm)

        print(f"Heads={num_heads}: Entropy={entropy:.4f}, Sparsity={sparsity:.4f}, Out Norm={out_norm:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(head_counts, metrics['avg_entropy'], 'o-')
    axes[0].set_xlabel('Number of Heads')
    axes[0].set_ylabel('Average Entropy')
    axes[0].set_title('Attention Entropy vs Heads')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(head_counts, metrics['avg_sparsity'], 'o-')
    axes[1].set_xlabel('Number of Heads')
    axes[1].set_ylabel('Sparsity')
    axes[1].set_title('Attention Sparsity vs Heads')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(head_counts, metrics['output_norm'], 'o-')
    axes[2].set_xlabel('Number of Heads')
    axes[2].set_ylabel('Output Norm')
    axes[2].set_title('Output Norm vs Heads')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
