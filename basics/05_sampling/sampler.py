
"""
Sampling Parameters: Temperature, Top-K, Top-P
Interactively tune parameters, plot entropy vs diversity, ablate each method
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class TokenSampler:
    """Sample tokens with various strategies"""

    @staticmethod
    def temperature_sampling(logits, temperature=1.0):
        """Temperature scaling: lower=sharper, higher=softer"""
        if temperature == 0:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_k_sampling(logits, k=10, temperature=1.0):
        """Keep top-k most likely tokens"""
        scaled_logits = logits / temperature
        top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)

        # Zero out non-top-k
        scaled_logits_masked = torch.full_like(scaled_logits, float('-inf'))
        scaled_logits_masked.scatter_(-1, top_k_indices, top_k_logits)

        probs = F.softmax(scaled_logits_masked, dim=-1)
        probs = torch.nan_to_num(probs, 0)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_p_sampling(logits, p=0.9, temperature=1.0):
        """Keep tokens with cumulative probability >= p"""
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        sorted_indices_to_remove = cumsum_probs > p
        sorted_indices_to_remove[..., 0] = False  # Keep at least one

        # Create mask
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0

        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def compute_entropy(logits):
        """Compute entropy of probability distribution"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()

    @staticmethod
    def compute_sparsity(logits, threshold=0.01):
        """Compute sparsity (fraction of probs below threshold)"""
        probs = F.softmax(logits, dim=-1)
        sparsity = (probs < threshold).float().mean().item()
        return sparsity


class SamplingAnalyzer:
    """Analyze and visualize sampling methods"""

    def __init__(self, vocab_size=1000, seed=42):
        self.vocab_size = vocab_size
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_logits(self, distribution='normal'):
        """Generate synthetic logits"""
        if distribution == 'normal':
            logits = torch.randn(self.vocab_size)
        elif distribution == 'power_law':
            logits = torch.tensor([1.0 / (i + 1) ** 0.5 for i in range(self.vocab_size)])
        else:
            logits = torch.randn(self.vocab_size)

        return logits

    def ablate_temperature(self):
        """Ablate temperature parameter"""
        logits = self.generate_logits()
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

        results = {'temp': [], 'entropy': [], 'max_prob': [], 'sparsity': []}

        for temp in temperatures:
            if temp == 0:
                probs = F.softmax(logits * 1e6, dim=-1)  # Approximate argmax
            else:
                probs = F.softmax(logits / temp, dim=-1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_prob = probs.max().item()
            sparsity = (probs < 0.01).float().mean().item()

            results['temp'].append(temp)
            results['entropy'].append(entropy)
            results['max_prob'].append(max_prob)
            results['sparsity'].append(sparsity)

        return results

    def ablate_top_k(self):
        """Ablate top-k parameter"""
        logits = self.generate_logits()
        k_values = [1, 5, 10, 50, 100, 500]

        results = {'k': [], 'entropy': [], 'coverage': []}

        for k in k_values:
            if k >= self.vocab_size:
                continue

            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            logits_masked = torch.full_like(logits, float('-inf'))
            logits_masked.scatter_(0, top_k_indices, top_k_logits)

            probs = F.softmax(logits_masked, dim=-1)
            probs = torch.nan_to_num(probs, 0)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            coverage = probs[probs > 0].sum().item()

            results['k'].append(k)
            results['entropy'].append(entropy)
            results['coverage'].append(coverage)

        return results

    def ablate_top_p(self):
        """Ablate top-p parameter"""
        logits = self.generate_logits()
        p_values = [0.5, 0.7, 0.9, 0.95, 0.99]

        results = {'p': [], 'entropy': [], 'num_tokens': []}

        for p in p_values:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumsum_probs > p
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            num_tokens = (probs > 0).sum().item()

            results['p'].append(p)
            results['entropy'].append(entropy)
            results['num_tokens'].append(num_tokens)

        return results

    def plot_ablations(self):
        """Plot all ablation studies"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # Temperature
        temp_results = self.ablate_temperature()
        axes[0, 0].plot(temp_results['temp'], temp_results['entropy'], 'o-')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].set_title('Temperature vs Entropy')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(temp_results['temp'], temp_results['max_prob'], 'o-')
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Max Probability')
        axes[0, 1].set_title('Temperature vs Max Probability')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(temp_results['temp'], temp_results['sparsity'], 'o-')
        axes[0, 2].set_xlabel('Temperature')
        axes[0, 2].set_ylabel('Sparsity')
        axes[0, 2].set_title('Temperature vs Sparsity (P<0.01)')
        axes[0, 2].grid(True, alpha=0.3)

        # Top-K
        topk_results = self.ablate_top_k()
        axes[1, 0].plot(topk_results['k'], topk_results['entropy'], 'o-')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('K (log scale)')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Top-K vs Entropy')
        axes[1, 0].grid(True, alpha=0.3)

        # Top-P
        topp_results = self.ablate_top_p()
        axes[1, 1].plot(topp_results['p'], topp_results['entropy'], 'o-')
        axes[1, 1].set_xlabel('P')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Top-P vs Entropy')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(topp_results['p'], topp_results['num_tokens'], 'o-')
        axes[1, 2].set_xlabel('P')
        axes[1, 2].set_ylabel('Number of Tokens')
        axes[1, 2].set_title('Top-P vs Active Tokens')
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle('Sampling Parameter Ablations')
        plt.tight_layout()
        plt.show()

    def compare_sampling_distributions(self):
        """Compare distribution of samples under different strategies"""
        logits = self.generate_logits()
        num_samples = 5000

        sampler = TokenSampler()

        # Sample using different methods
        samples_temp01 = []
        samples_temp10 = []
        samples_topk10 = []
        samples_topp09 = []

        for _ in range(num_samples):
            samples_temp01.append(sampler.temperature_sampling(logits, temperature=0.1).item())
            samples_temp10.append(sampler.temperature_sampling(logits, temperature=1.0).item())
            samples_topk10.append(sampler.top_k_sampling(logits, k=10, temperature=1.0).item())
            samples_topp09.append(sampler.top_p_sampling(logits, p=0.9, temperature=1.0).item())

        # Plot histograms
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        for ax, samples, title in [
            (axes[0, 0], samples_temp01, 'Temperature=0.1'),
            (axes[0, 1], samples_temp10, 'Temperature=1.0'),
            (axes[1, 0], samples_topk10, 'Top-K=10'),
            (axes[1, 1], samples_topp09, 'Top-P=0.9'),
        ]:
            ax.hist(samples, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Token ID')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Distribution of Sampled Tokens')
        plt.tight_layout()
        plt.show()
