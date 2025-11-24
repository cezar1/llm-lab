
"""
KV Cache Demo: Fast inference with cached keys and values
"""

from cache import (
    KVCache, benchmark_with_without_cache, plot_cache_benchmarks,
    visualize_cache_memory, analyze_cache_efficiency
)
import torch


def demo_kv_cache():
    """Demonstrate KV cache"""
    print("=" * 60)
    print("KV CACHE: FASTER INFERENCE")
    print("=" * 60)

    batch_size = 4
    seq_len = 20
    num_heads = 8
    head_dim = 64

    cache = KVCache(batch_size, seq_len, num_heads, head_dim)

    print(f"Cache configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max sequence length: {seq_len}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Memory usage: {cache.memory_usage():.2f} MB")

    # Simulate generation
    print(f"\nSimulating generation with cache:")
    for step in range(5):
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)

        cache.update(k, v)
        k_cached, v_cached = cache.get()

        print(f"  Step {step}: cache size = {cache.cur_len}, k_cached shape = {k_cached.shape}")


def demo_cache_speedup():
    """Benchmark cache speedup"""
    print("\n" + "=" * 60)
    print("CACHE SPEEDUP BENCHMARK")
    print("=" * 60)

    results = benchmark_with_without_cache(seq_lengths=[10, 20, 40, 80])
    plot_cache_benchmarks(results)


def demo_cache_memory():
    """Visualize cache memory usage"""
    print("\n" + "=" * 60)
    print("CACHE MEMORY USAGE")
    print("=" * 60)

    visualize_cache_memory()


def demo_cache_efficiency():
    """Analyze cache efficiency"""
    print("\n" + "=" * 60)
    print("CACHE EFFICIENCY ANALYSIS")
    print("=" * 60)

    analyze_cache_efficiency()


def demo_cache_insights():
    """Key insights about KV cache"""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS: KV CACHE")
    print("=" * 60)

    print("""
1. WITHOUT KV CACHE:
   - Each generation step computes attention over all previous + new token
   - Complexity: O(seq_len^2) attention computations
   - Lots of redundant computation

2. WITH KV CACHE:
   - K, V from all previous tokens are stored
   - New token only attends to cached K, V
   - No redundant attention computation
   - Complexity: O(seq_len) for generation

3. SPEEDUP:
   - Linear in sequence length
   - 10x faster at seq_len=100
   - 100x+ faster at seq_len=1000

4. TRADEOFFS:
   - Requires extra memory: ~(2 * seq_len * d_model)
   - At seq_len=1024, embed_dim=2048: ~8 MB per batch item
   - Worth it for generation (almost always faster)

5. ADVANCED CACHE STRATEGIES:
   - Sliding window cache: keep only recent tokens
   - Sparse cache: cache every N-th token
   - Quantized cache: compress K, V to lower precision
   - Multi-query/grouped-query: share cache across heads
""")


if __name__ == "__main__":
    demo_kv_cache()
    demo_cache_speedup()
    demo_cache_memory()
    demo_cache_efficiency()
    demo_cache_insights()

    print("\n" + "=" * 60)
    print("KV CACHE DEMO COMPLETE")
    print("=" * 60)
