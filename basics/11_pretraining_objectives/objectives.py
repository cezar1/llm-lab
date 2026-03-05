
"""
Pretraining Objectives: Masked LM, Causal LM, Prefix LM
Train on toy data, compare loss curves and generation quality
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class CausalLMLoss(nn.Module):
    """Standard causal LM loss (next-token prediction)"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        """
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels.reshape(-1)

        # Only compute loss on positions after first token
        return self.loss_fn(logits_flat[:-1], labels_flat[1:])


class MaskedLMLoss(nn.Module):
    """Masked language model loss (BERT-style)"""

    def __init__(self, vocab_size, mask_token_id=103, mask_prob=0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, input_ids):
        """
        logits: (batch, seq_len, vocab_size)
        input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape

        # Create mask (randomly mask 15% of tokens)
        mask = torch.rand(batch_size, seq_len) < self.mask_prob

        # Loss only on masked positions
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits_masked = logits[mask]
        labels_masked = input_ids[mask]

        return self.loss_fn(logits_masked, labels_masked)


class PrefixLMLoss(nn.Module):
    """Prefix LM loss (can see prefix, predicts suffix)"""

    def __init__(self, prefix_ratio=0.5):
        super().__init__()
        self.prefix_ratio = prefix_ratio
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        prefix_ratio: fraction of sequence that is "prefix" (always visible)
        """
        batch_size, seq_len, _ = logits.shape

        # Compute loss only on suffix (after prefix)
        prefix_len = int(seq_len * self.prefix_ratio)

        logits_suffix = logits[:, prefix_len:, :].reshape(-1, logits.shape[-1])
        labels_suffix = labels[:, prefix_len:].reshape(-1)

        # Shift by 1 for next-token prediction
        return self.loss_fn(logits_suffix[:-1], labels_suffix[1:])


class PretrainingObjectiveAnalyzer:
    """Compare pretraining objectives"""

    def __init__(self, vocab_size=1000, seq_len=20):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.causal_loss = CausalLMLoss()
        self.masked_loss = MaskedLMLoss(vocab_size)
        self.prefix_loss = PrefixLMLoss(prefix_ratio=0.5)

    def generate_batch(self, batch_size=4):
        """Generate random batch"""
        return torch.randint(0, self.vocab_size, (batch_size, self.seq_len))

    def simulate_training(self, num_steps=100):
        """Simulate training with different objectives"""
        losses = {
            'causal_lm': [],
            'masked_lm': [],
            'prefix_lm': []
        }

        for step in range(num_steps):
            # Generate batch and fake logits
            batch = self.generate_batch()
            logits = torch.randn(batch.shape[0], self.seq_len, self.vocab_size)

            # Compute losses
            try:
                causal_loss = self.causal_loss(logits, batch).item()
                losses['causal_lm'].append(causal_loss)
            except:
                losses['causal_lm'].append(None)

            try:
                masked_loss = self.masked_loss(logits, batch).item()
                losses['masked_lm'].append(masked_loss)
            except:
                losses['masked_lm'].append(None)

            try:
                prefix_loss = self.prefix_loss(logits, batch).item()
                losses['prefix_lm'].append(prefix_loss)
            except:
                losses['prefix_lm'].append(None)

        return losses

    def plot_loss_curves(self, losses):
        """Plot loss curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # All objectives
        if losses['causal_lm'][0] is not None:
            axes[0].plot(losses['causal_lm'], label='Causal LM', linewidth=2)
        if losses['masked_lm'][0] is not None:
            axes[0].plot(losses['masked_lm'], label='Masked LM', linewidth=2)
        if losses['prefix_lm'][0] is not None:
            axes[0].plot(losses['prefix_lm'], label='Prefix LM', linewidth=2)

        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Pretraining Objective Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Smoothed (for clarity)
        window = 10
        for name in ['causal_lm', 'masked_lm', 'prefix_lm']:
            if losses[name][0] is not None:
                smoothed = np.convolve(losses[name], np.ones(window)/window, mode='valid')
                axes[1].plot(smoothed, label=name, linewidth=2)

        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Smoothed Loss')
        axes[1].set_title('Smoothed Loss Curves (window=10)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_masking_strategies(self):
        """Compare different masking strategies"""
        strategies = {
            'No Masking': 0.0,
            'Light (5%)': 0.05,
            'Standard (15%)': 0.15,
            'Heavy (30%)': 0.30,
            'Very Heavy (50%)': 0.50
        }

        results = []

        for name, mask_prob in strategies.items():
            loss_fn = MaskedLMLoss(self.vocab_size, mask_prob=mask_prob)

            # Simulate training
            losses = []
            for _ in range(50):
                batch = self.generate_batch()
                logits = torch.randn(batch.shape[0], self.seq_len, self.vocab_size)

                try:
                    loss = loss_fn(logits, batch).item()
                    losses.append(loss)
                except:
                    pass

            avg_loss = np.mean(losses) if losses else 0
            results.append((name, avg_loss))

        # Plot
        names, losses = zip(*results)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, losses, color='steelblue', alpha=0.7)
        ax.set_ylabel('Average Loss')
        ax.set_title('Impact of Masking Rate on MLM')
        ax.grid(True, alpha=0.3, axis='y')

        for label in ax.get_xticklabels():
            label.set_rotation(45)

        plt.tight_layout()
        plt.show()

        return dict(results)
