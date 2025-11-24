
"""
Finetuning, Instruction Tuning, and RLHF
Implement reward model and simple PPO for learning from human feedback
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class RewardModel(nn.Module):
    """Simple reward model for RLHF"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, hidden_state):
        """hidden_state: (batch, seq_len, hidden_dim) -> (batch, 1)"""
        pooled = hidden_state.mean(dim=1)  # Average pooling
        return self.mlp(pooled)


class FinetuningSampler:
    """Simulate different finetuning approaches"""

    def __init__(self):
        self.history = {
            'supervised': [],
            'instruction': [],
            'rlhf': []
        }

    def supervised_finetuning_step(self, step):
        """Simulate supervised finetuning on specific task"""
        # Loss decreases over time
        loss = 2.0 * np.exp(-step / 50) + 0.5
        self.history['supervised'].append(loss)
        return loss

    def instruction_tuning_step(self, step):
        """Simulate instruction tuning on multiple task formats"""
        # Slightly slower convergence but better generalization
        loss = 2.0 * np.exp(-step / 80) + 0.3
        self.history['instruction'].append(loss)
        return loss

    def rlhf_step(self, step):
        """Simulate RLHF training"""
        # More volatile but eventually better alignment
        base_loss = 2.0 * np.exp(-step / 100) + 0.2
        noise = np.sin(step / 5) * 0.2
        loss = base_loss + noise
        self.history['rlhf'].append(max(loss, 0.1))
        return loss

    def simulate_finetuning(self, num_steps=200):
        """Simulate all finetuning approaches"""
        for step in range(num_steps):
            self.supervised_finetuning_step(step)
            self.instruction_tuning_step(step)
            self.rlhf_step(step)

        return self.history

    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw curves
        for name, losses in self.history.items():
            axes[0].plot(losses, label=name, linewidth=2, alpha=0.7)

        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Finetuning Approaches: Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Smoothed curves
        window = 10
        for name, losses in self.history.items():
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            axes[1].plot(smoothed, label=name, linewidth=2)

        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Smoothed Loss')
        axes[1].set_title('Smoothed Loss Curves (window=10)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_final_performance(self):
        """Compare final performance metrics"""
        final_losses = {
            'Supervised': self.history['supervised'][-1],
            'Instruction': self.history['instruction'][-1],
            'RLHF': self.history['rlhf'][-1]
        }

        print("\nFinal Loss Values:")
        print("-" * 40)
        for method, loss in sorted(final_losses.items(), key=lambda x: x[1]):
            print(f"  {method:<15}: {loss:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(final_losses.keys())
        losses = list(final_losses.values())

        bars = ax.bar(methods, losses, color=['steelblue', 'orange', 'green'], alpha=0.7)
        ax.set_ylabel('Final Loss')
        ax.set_title('Finetuning Method Comparison: Final Performance')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def analyze_instruction_tuning_formats():
    """Analyze different instruction tuning formats"""
    formats = {
        'No Format': 0.4,  # Poor performance without structure
        'QA Format': 0.25,  # Q: ... A: ...
        'Task+Input Format': 0.18,  # Task: ... Input: ... Output: ...
        'COT Format': 0.12,  # Chain-of-thought
        'Multi-task Format': 0.15,  # Task: X, Instance: Y, Output: Z
    }

    print("\nInstruction Tuning Format Comparison:")
    print("-" * 50)

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(formats.keys())
    scores = list(formats.values())

    bars = ax.barh(methods, scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Task Performance Score')
    ax.set_title('Instruction Format Effectiveness')
    ax.set_xlim(0, 0.5)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return formats


def analyze_rlhf_dynamics():
    """Analyze RLHF reward and policy dynamics"""
    steps = np.arange(0, 200)

    # Simulated reward over RLHF steps
    reward = 0.5 + 0.3 * (1 - np.exp(-steps / 100)) + 0.05 * np.sin(steps / 20)

    # Policy divergence from base model
    divergence = 0.1 + 0.05 * steps / 200 + 0.02 * np.sin(steps / 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, reward, linewidth=2, color='green')
    ax1.fill_between(steps, reward - 0.02, reward + 0.02, alpha=0.3, color='green')
    ax1.set_xlabel('RLHF Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Signal during RLHF')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, divergence, linewidth=2, color='orange')
    ax2.set_xlabel('RLHF Step')
    ax2.set_ylabel('KL Divergence from Base')
    ax2.set_title('Policy Divergence during RLHF')
    ax2.axhline(y=0.2, color='r', linestyle='--', label='Typical KL Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('RLHF Dynamics: Reward vs Divergence')
    plt.tight_layout()
    plt.show()
