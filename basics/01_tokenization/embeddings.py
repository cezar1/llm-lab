
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class TokenEmbedder:
    """Explore one-hot vs learned embeddings and cosine distances"""

    def __init__(self, vocab_size, embed_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # One-hot embeddings (baseline)
        self.one_hot = np.eye(vocab_size)
        # Learned embeddings (random initialization)
        self.learned = np.random.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)
        self.learned = self.learned / np.linalg.norm(self.learned, axis=1, keepdims=True)

    def cosine_distances(self, token_ids, embedding='learned'):
        """Compute pairwise cosine distances for token sequences"""
        embeds = self.learned if embedding == 'learned' else self.one_hot
        token_embeds = embeds[token_ids]
        sim = cosine_similarity(token_embeds)
        return sim

    def plot_embedding_comparison(self, token_ids, embedding_type='learned'):
        """Plot cosine similarity heatmap"""
        sim = self.cosine_distances(token_ids, embedding_type)

        plt.figure(figsize=(10, 8))
        plt.imshow(sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title(f'Token Cosine Similarity ({embedding_type} embedding)')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        plt.tight_layout()
        plt.show()

    def visualize_embeddings_2d(self, token_ids):
        """Project learned embeddings to 2D using PCA"""
        embeds = self.learned[token_ids]

        if len(set(token_ids)) < 3:
            print("Need at least 3 unique tokens for visualization")
            return

        pca = PCA(n_components=2)
        embeds_2d = pca.fit_transform(embeds)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1],
                            c=token_ids, cmap='tab20', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Token ID')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.title('Token Embeddings (PCA 2D)')
        for i, tid in enumerate(token_ids):
            plt.annotate(str(tid), embeds_2d[i], fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_embeddings(self, token_ids):
        """Compare one-hot vs learned embeddings side-by-side"""
        one_hot_sim = self.cosine_distances(token_ids, 'one_hot')
        learned_sim = self.cosine_distances(token_ids, 'learned')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # One-hot
        im1 = ax1.imshow(one_hot_sim, cmap='RdYlBu_r', vmin=-0.1, vmax=1.1)
        ax1.set_title('One-Hot Embeddings')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Token Position')
        plt.colorbar(im1, ax=ax1)

        # Learned
        im2 = ax2.imshow(learned_sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax2.set_title('Learned Embeddings')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Token Position')
        plt.colorbar(im2, ax=ax2)

        plt.suptitle('Cosine Similarity: One-Hot vs Learned Embeddings')
        plt.tight_layout()
        plt.show()
