
"""Pretraining Objectives Demo"""

from objectives import PretrainingObjectiveAnalyzer
import torch


def demo():
    print("=" * 60)
    print("PRETRAINING OBJECTIVES")
    print("=" * 60)

    analyzer = PretrainingObjectiveAnalyzer(vocab_size=1000, seq_len=20)

    # Compare loss curves
    print("\nComparing causal LM, masked LM, and prefix LM...")
    losses = analyzer.simulate_training(num_steps=100)
    analyzer.plot_loss_curves(losses)

    # Analyze masking strategies
    print("\nAnalyzing masking strategies for MLM...")
    mask_results = analyzer.analyze_masking_strategies()

    print("\nMasking strategy results:")
    for strategy, loss in mask_results.items():
        print(f"  {strategy:<20}: loss = {loss:.4f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("Causal LM: Next-token prediction, suited for generation")
    print("Masked LM: Bidirectional context, suits encoding tasks")
    print("Prefix LM: Hybrid - see all prefix, predict suffix")
    print("=" * 60)


if __name__ == "__main__":
    demo()
