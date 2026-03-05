
"""Finetuning & RLHF Demo"""

from finetuning import (
    FinetuningSampler, analyze_instruction_tuning_formats,
    analyze_rlhf_dynamics
)


def demo():
    print("=" * 60)
    print("FINETUNING, INSTRUCTION TUNING & RLHF")
    print("=" * 60)

    sampler = FinetuningSampler()

    print("\n1. Simulating finetuning approaches...")
    sampler.simulate_finetuning(num_steps=200)
    sampler.plot_training_curves()

    print("\n2. Final performance comparison...")
    sampler.compare_final_performance()

    print("\n3. Instruction tuning format analysis...")
    analyze_instruction_tuning_formats()

    print("\n4. RLHF reward and policy dynamics...")
    analyze_rlhf_dynamics()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("Supervised FT: Fast convergence, task-specific")
    print("Instruction TG: Better generalization, multiple tasks")
    print("RLHF: Better alignment with human preferences")
    print("=" * 60)


if __name__ == "__main__":
    demo()
