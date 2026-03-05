
"""
Positional Embeddings: Sinusoidal, Learned, RoPE, ALiBi
Animate and ablate all four methods to see how they encode position.
"""

import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SinusoidalPE(nn.Module):
    """Classic sinusoidal positional encoding"""

    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Pre-compute position encodings
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """x shape: (batch, seq_len, embed_dim)"""
        seq_len = x.shape[1]
        return self.pe[:seq_len, :].unsqueeze(0)


class LearnedPE(nn.Module):
    """Learnable positional embeddings"""

    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        """x shape: (batch, seq_len, embed_dim)"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pe(positions)


class RoPEPE(nn.Module):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Compute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """Apply rotary position encoding to query/key
        x shape: (batch, seq_len, embed_dim) or (batch, heads, seq_len, embed_dim)
        """
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # Compute angles
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Apply rotation
        cos_cached = emb.cos()
        sin_cached = emb.sin()

        # Reshape for rotation
        if x.dim() == 3:  # (batch, seq_len, embed_dim)
            x_rot = torch.stack([
                x[..., :self.embed_dim // 2] * cos_cached[..., :self.embed_dim // 2] -
                x[..., self.embed_dim // 2:] * sin_cached[..., self.embed_dim // 2:],
                x[..., :self.embed_dim // 2] * sin_cached[..., :self.embed_dim // 2] +
                x[..., self.embed_dim // 2:] * cos_cached[..., self.embed_dim // 2:]
            ], dim=-1)
            return x_rot.flatten(-2)
        else:
            return emb


class ALiBiPE(nn.Module):
    """Attention with Linear Biases (ALiBi) - no position embeddings, just attention bias"""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        # Pre-compute head-specific slopes
        slopes = torch.tensor([2 ** (-8 * h / num_heads) for h in range(num_heads)])
        self.register_buffer('slopes', slopes)

    def forward(self, seq_len, device):
        """Generate position bias for attention mechanism"""
        # Relative position matrix
        arange = torch.arange(seq_len, device=device)
        context_position = arange.unsqueeze(1)
        memory_position = arange.unsqueeze(0)
        relative_pos = memory_position - context_position  # (seq_len, seq_len)

        # Apply slopes
        bias = relative_pos.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        return bias  # (num_heads, seq_len, seq_len)


class PositionalEmbeddingAnalyzer:
    """Analyze and visualize positional encodings"""

    def __init__(self, embed_dim=64, seq_len=20):
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.sin_pe = SinusoidalPE(embed_dim)
        self.learned_pe = LearnedPE(embed_dim)
        self.rope_pe = RoPEPE(embed_dim)
        self.alibi_pe = ALiBiPE(num_heads=8)

    def get_encodings(self):
        """Get position encodings from all methods"""
        x = torch.randn(1, self.seq_len, self.embed_dim)

        sin_enc = self.sin_pe(x)[0].detach()  # (seq_len, embed_dim)
        learned_enc = self.learned_pe(x)[0].detach()  # (seq_len, embed_dim)
        rope_enc = self.rope_pe(x)
        if rope_enc.dim() == 2:
            rope_enc = rope_enc[:self.seq_len, :]
        alibi_enc = self.alibi_pe(self.seq_len, x.device)  # (heads, seq_len, seq_len)

        return sin_enc, learned_enc, rope_enc, alibi_enc

    def plot_3d_animation_frame(self, encoding, pos_idx, ax, title):
        """Plot 3D embedding for a specific position"""
        emb = encoding[pos_idx, :3] if encoding.dim() == 2 else encoding[pos_idx, :3]

        ax.scatter([emb[0]], [emb[1]], [emb[2]], s=100, color='red')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.set_zlabel('Dim 2')
        ax.set_title(f'{title} (pos={pos_idx})')

    def visualize_all_methods(self):
        """Heatmap comparison of all positional encoding methods"""
        sin_enc, learned_enc, rope_enc, alibi_enc = self.get_encodings()

        # Clip for visualization
        def clip_for_viz(x):
            return torch.clamp(x, -1, 1)

        sin_enc_viz = clip_for_viz(sin_enc).numpy()
        learned_enc_viz = clip_for_viz(learned_enc).numpy()
        rope_enc_viz = clip_for_viz(rope_enc).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sinusoidal
        axes[0, 0].imshow(sin_enc_viz, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[0, 0].set_title('Sinusoidal PE')
        axes[0, 0].set_xlabel('Embedding Dimension')
        axes[0, 0].set_ylabel('Position')

        # Learned
        axes[0, 1].imshow(learned_enc_viz, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[0, 1].set_title('Learned PE')
        axes[0, 1].set_xlabel('Embedding Dimension')
        axes[0, 1].set_ylabel('Position')

        # RoPE
        axes[1, 0].imshow(rope_enc_viz, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('RoPE')
        axes[1, 0].set_xlabel('Embedding Dimension')
        axes[1, 0].set_ylabel('Position')

        # ALiBi (show first head's attention bias)
        alibi_viz = alibi_enc[0].numpy()
        axes[1, 1].imshow(alibi_viz, cmap='RdBu', aspect='auto')
        axes[1, 1].set_title('ALiBi (First Head)')
        axes[1, 1].set_xlabel('Key Position')
        axes[1, 1].set_ylabel('Query Position')

        plt.suptitle('Positional Encoding Methods Comparison')
        plt.tight_layout()
        plt.show()

    def plot_position_norm_evolution(self):
        """Show how position norm evolves with position"""
        sin_enc, learned_enc, rope_enc, alibi_enc = self.get_encodings()

        sin_norms = torch.norm(sin_enc, dim=1).numpy()
        learned_norms = torch.norm(learned_enc, dim=1).numpy()
        rope_norms = torch.norm(rope_enc, dim=1).numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(sin_norms, label='Sinusoidal', marker='o')
        plt.plot(learned_norms, label='Learned', marker='s')
        plt.plot(rope_norms, label='RoPE', marker='^')
        plt.xlabel('Position')
        plt.ylabel('Embedding Norm')
        plt.title('Embedding Norm vs Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_position_distance_matrix(self):
        """Show cosine similarity between position embeddings"""
        sin_enc, learned_enc, rope_enc, _ = self.get_encodings()

        from sklearn.metrics.pairwise import cosine_similarity

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, (enc, title) in enumerate([
            (sin_enc.numpy(), 'Sinusoidal'),
            (learned_enc.numpy(), 'Learned'),
            (rope_enc.numpy(), 'RoPE')
        ]):
            sim = cosine_similarity(enc)
            axes[i].imshow(sim, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            axes[i].set_title(f'{title} Cosine Similarity')
            axes[i].set_xlabel('Position')
            axes[i].set_ylabel('Position')
            plt.colorbar(axes[i].images[0], ax=axes[i])

        plt.suptitle('Position-to-Position Similarity')
        plt.tight_layout()
        plt.show()

    def ablate_embedding_dim(self):
        """Ablation: How does embedding dimension affect position encoding quality?"""
        dims = [8, 16, 32, 64, 128]
        results = {
            'sinusoidal': [],
            'learned': [],
            'rope': []
        }

        for dim in dims:
            sin_pe = SinusoidalPE(dim)
            learned_pe = LearnedPE(dim)
            rope_pe = RoPEPE(dim)

            x = torch.randn(1, self.seq_len, dim)

            sin_enc = sin_pe(x)[0]
            learned_enc = learned_pe(x)[0]
            rope_enc = rope_pe(x)
            if rope_enc.dim() > 2:
                rope_enc = rope_enc[:self.seq_len, :]

            # Measure "position orthogonality": how different are encodings at different positions?
            sin_orth = (torch.pdist(sin_enc).mean()).item()
            learned_orth = (torch.pdist(learned_enc).mean()).item()
            rope_orth = (torch.pdist(rope_enc).mean()).item()

            results['sinusoidal'].append(sin_orth)
            results['learned'].append(learned_orth)
            results['rope'].append(rope_orth)

        plt.figure(figsize=(10, 5))
        for method, values in results.items():
            plt.plot(dims, values, marker='o', label=method)

        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position Differentiation')
        plt.title('Ablation: Embedding Dimension Impact')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
