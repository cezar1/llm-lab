
"""Scaling Laws Demo"""

from scaling import ScalingLawSimulator


def demo():
    print("=" * 60)
    print("SCALING LAWS & MODEL CAPACITY")
    print("=" * 60)

    simulator = ScalingLawSimulator()

    print("\n1. Sweeping over model sizes...")
    results = simulator.sweep_model_sizes()

    print("\n2. Plotting scaling laws...")
    simulator.plot_scaling_laws(results)

    print("\n3. Analyzing Pareto frontier (time/VRAM constraints)...")
    simulator.analyze_pareto_frontier(results)
    simulator.plot_pareto_frontier(results)

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("1. Loss follows power law: loss ≈ a + b/N^0.07")
    print("2. Chinchilla scaling: match compute between model & data")
    print("3. Bigger models are more sample-efficient")
    print("4. Training time and VRAM scale roughly O(N)")
    print("5. Optimal model size depends on compute budget")
    print("=" * 60)


if __name__ == "__main__":
    demo()
