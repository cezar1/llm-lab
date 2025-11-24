
"""Inference Stacks Demo"""

from inference import InferenceProfiler


def demo():
    print("=" * 60)
    print("INFERENCE STACKS: HF, vLLM, ExLlama")
    print("=" * 60)

    profiler = InferenceProfiler()

    print("\n1. Comparing inference stacks across batch sizes...")
    results = profiler.compare_stacks(batch_sizes=[1, 4, 16, 32, 64])

    print("\nThroughput (tokens/sec):")
    print("-" * 60)
    for i, bs in enumerate(results['batch_size']):
        print(f"Batch {bs:2d}: HF={results['hf_throughput'][i]:6.0f}  "
              f"vLLM={results['vllm_throughput'][i]:6.0f}  "
              f"ExLlama={results['exllama_throughput'][i]:6.0f}")

    print("\n2. Plotting comparative analysis...")
    profiler.plot_comparison(results)

    print("\n3. Recommendations...")
    profiler.print_recommendations()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("1. HF: Most flexible, easiest to use, moderate performance")
    print("2. vLLM: 5-10x faster, excellent batching, production-ready")
    print("3. ExLlama: Fastest for Llama models, limited architecture")
    print("4. Choose based on: flexibility vs performance tradeoff")
    print("5. For production: vLLM for most use cases")
    print("=" * 60)


if __name__ == "__main__":
    demo()
