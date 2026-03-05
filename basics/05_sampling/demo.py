
"""
Sampling Parameters Demo
Tune temperature, top-k, top-p interactively, plot entropy vs diversity
"""

from sampler import TokenSampler, SamplingAnalyzer
import torch


def demo_temperature_sampling():
    """Show temperature effects"""
    print("=" * 60)
    print("1. TEMPERATURE SAMPLING")
    print("=" * 60)

    logits = torch.randn(100)
    sampler = TokenSampler()

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("\nTemperature effects on probability distribution:")
    print("-" * 60)

    for temp in temperatures:
        if temp == 0:
            probs = torch.nn.functional.softmax(logits * 1e6, dim=-1)
        else:
            probs = torch.nn.functional.softmax(logits / temp, dim=-1)

        entropy = sampler.compute_entropy(logits / temp if temp > 0 else logits * 1e6)
        max_prob = probs.max().item()

        print(f"T={temp:3.1f}: Entropy={entropy:6.3f}, Max Prob={max_prob:6.3f}, "
              f"Top-5 concentration={(probs.topk(5)[0].sum()):6.3f}")

    print("\nVisualization:")
    analyzer = SamplingAnalyzer()
    results = analyzer.ablate_temperature()

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(results['temp'], results['entropy'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Entropy vs Temperature')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results['temp'], results['max_prob'], 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Max Probability')
    ax2.set_title('Max Probability vs Temperature')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_top_k_sampling():
    """Show top-k effects"""
    print("\n" + "=" * 60)
    print("2. TOP-K SAMPLING")
    print("=" * 60)

    logits = torch.randn(500)
    sampler = TokenSampler()

    k_values = [1, 5, 10, 50, 100, 250, 500]

    print("\nTop-K effects:")
    print("-" * 60)

    for k in k_values:
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        logits_masked = torch.full_like(logits, float('-inf'))
        logits_masked.scatter_(0, top_k_indices, top_k_logits)

        probs = torch.nn.functional.softmax(logits_masked, dim=-1)
        probs = torch.nan_to_num(probs, 0)

        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        print(f"K={k:3d}: Entropy={entropy:6.3f}, Active tokens={k:3d}")

    analyzer = SamplingAnalyzer()
    print("\nVisualizing top-k ablation...")
    analyzer.plot_ablations()


def demo_top_p_sampling():
    """Show top-p effects"""
    print("\n" + "=" * 60)
    print("3. TOP-P (NUCLEUS) SAMPLING")
    print("=" * 60)

    logits = torch.randn(500)
    sampler = TokenSampler()

    p_values = [0.5, 0.7, 0.9, 0.95, 0.99]

    print("\nTop-P (Nucleus) effects:")
    print("-" * 60)

    for p in p_values:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumsum_probs > p
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()

        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        num_active = (probs > 0).sum().item()

        print(f"P={p:4.2f}: Entropy={entropy:6.3f}, Active tokens={num_active:3d}, "
              f"Coverage={probs[probs > 0].sum():.3f}")

    analyzer = SamplingAnalyzer()
    print("\nVisualizing sampling ablations...")
    analyzer.plot_ablations()


def demo_sampling_distributions():
    """Compare actual sample distributions"""
    print("\n" + "=" * 60)
    print("4. SAMPLE DISTRIBUTION COMPARISON")
    print("=" * 60)

    analyzer = SamplingAnalyzer()
    analyzer.compare_sampling_distributions()


def demo_sampling_recommendations():
    """Provide recommendations for different use cases"""
    print("\n" + "=" * 60)
    print("5. SAMPLING RECOMMENDATIONS")
    print("=" * 60)

    recommendations = {
        "Deterministic (e.g., translation)": {
            "temperature": 0.0,
            "top_k": "N/A",
            "top_p": "N/A",
            "note": "Use argmax for consistent, deterministic output"
        },
        "Focused (e.g., summarization)": {
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.9,
            "note": "Low entropy, focused sampling"
        },
        "Balanced (e.g., dialogue)": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "note": "Standard settings, good quality & diversity"
        },
        "Creative (e.g., story writing)": {
            "temperature": 1.5,
            "top_k": 100,
            "top_p": 0.99,
            "note": "Higher entropy, more diverse/creative output"
        },
        "Unpredictable (e.g., brainstorming)": {
            "temperature": 2.0,
            "top_k": 200,
            "top_p": 1.0,
            "note": "Very high entropy, maximum diversity"
        }
    ]

    for use_case, params in recommendations.items():
        print(f"\n{use_case}:")
        print(f"  Temperature: {params['temperature']}")
        print(f"  Top-K: {params['top_k']}")
        print(f"  Top-P: {params['top_p']}")
        print(f"  Note: {params['note']}")

    print("\n" + "-" * 60)
    print("Pro tips:")
    print("  - Temperature=0 for deterministic/greedy decoding")
    print("  - Use either top-k OR top-p, rarely both")
    print("  - Top-k: good for fixed vocabulary control")
    print("  - Top-p: better for variable-length diversity")
    print("  - Lower temp + higher k/p = more stable")
    print("  - Higher temp + lower k/p = more creative")


if __name__ == "__main__":
    demo_temperature_sampling()
    demo_top_k_sampling()
    demo_top_p_sampling()
    demo_sampling_distributions()
    demo_sampling_recommendations()

    print("\n" + "=" * 60)
    print("SAMPLING PARAMETERS DEMO COMPLETE")
    print("=" * 60)
