
"""
KV Cache: Record & reuse KV states for fast inference
Measure speedup, visualize cache hit/miss, profile memory
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np


class KVCache:
    """Key-Value cache for efficient autoregressive generation"""

    def __init__(self, batch_size, seq_len, num_heads, head_dim, device='cpu'):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Initialize caches
        self.k_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device)
        self.cur_len = 0

    def update(self, k, v):
        """Update cache with new k, v"""
        batch_size = k.shape[0]
        seq_len = k.shape[2]

        # Append new k, v to cache
        self.k_cache[:batch_size, :, self.cur_len:self.cur_len + seq_len] = k
        self.v_cache[:batch_size, :, self.cur_len:self.cur_len + seq_len] = v

        self.cur_len += seq_len

    def get(self):
        """Get cached k, v"""
        return self.k_cache[:, :, :self.cur_len], self.v_cache[:, :, :self.cur_len]

    def reset(self):
        """Reset cache"""
        self.cur_len = 0

    def memory_usage(self):
        """Estimate memory usage in MB"""
        # k_cache + v_cache, each element is float32 (4 bytes)
        total_elements = 2 * self.batch_size * self.num_heads * self.max_seq_len * self.head_dim
        return total_elements * 4 / (1024 * 1024)


def attention_with_cache(q, k_cache, v_cache, scale):
    """Compute attention using cached k, v"""
    # q: (batch, num_heads, 1, head_dim) - single token
    # k_cache, v_cache: (batch, num_heads, seq_len, head_dim)

    scores = torch.matmul(q, k_cache.transpose(-2, -1)) / scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v_cache)

    return output, weights


def benchmark_with_without_cache(seq_lengths=[10, 20, 50, 100]):
    """Benchmark generation speed with and without cache"""
    batch_size = 4
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim

    results = {'seq_len': [], 'with_cache': [], 'without_cache': [], 'speedup': []}

    for seq_len in seq_lengths:
        print(f"Benchmarking seq_len={seq_len}...")

        # Without cache (recompute all)
        start = time.time()
        for _ in range(100):
            q = torch.randn(batch_size, num_heads, seq_len, head_dim)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            _ = torch.matmul(weights, v)

        time_without = time.time() - start

        # With cache (only compute last token)
        cache = KVCache(batch_size, seq_len + 1, num_heads, head_dim)

        start = time.time()
        for step in range(seq_len):
            # In real generation, k,v come from just the new token
            q = torch.randn(batch_size, num_heads, 1, head_dim)
            k = torch.randn(batch_size, num_heads, 1, head_dim)
            v = torch.randn(batch_size, num_heads, 1, head_dim)

            cache.update(k, v)
            k_cached, v_cached = cache.get()

            scores = torch.matmul(q, k_cached.transpose(-2, -1)) / (head_dim ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            _ = torch.matmul(weights, v_cached)

        time_with = time.time() - start

        speedup = time_without / time_with

        results['seq_len'].append(seq_len)
        results['with_cache'].append(time_with)
        results['without_cache'].append(time_without)
        results['speedup'].append(speedup)

        print(f"  Without cache: {time_without:.4f}s")
        print(f"  With cache: {time_with:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def plot_cache_benchmarks(results):
    """Plot cache benchmark results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Timing comparison
    axes[0].plot(results['seq_len'], results['without_cache'], 'o-', label='Without Cache', linewidth=2)
    axes[0].plot(results['seq_len'], results['with_cache'], 's-', label='With Cache', linewidth=2)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Generation Speed: With vs Without KV Cache')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup
    axes[1].plot(results['seq_len'], results['speedup'], 'o-', linewidth=2, color='green')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Speedup Factor')
    axes[1].set_title('KV Cache Speedup')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_cache_memory():
    """Visualize memory usage of KV cache"""
    batch_sizes = [1, 4, 8, 16]
    seq_lengths = [128, 256, 512, 1024]

    fig, ax = plt.subplots(figsize=(10, 6))

    for batch_size in batch_sizes:
        memory_usages = []

        for seq_len in seq_lengths:
            cache = KVCache(batch_size, seq_len, num_heads=8, head_dim=64)
            memory_usages.append(cache.memory_usage())

        ax.plot(seq_lengths, memory_usages, 'o-', label=f'Batch={batch_size}', linewidth=2)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('KV Cache Memory Requirements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_cache_efficiency():
    """Analyze cache hit rates and efficiency"""
    seq_lengths = list(range(10, 101, 10))
    cache_sizes = [32, 64, 128]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute efficiency: recomputation saved
    for cache_size in cache_sizes:
        efficiencies = []

        for seq_len in seq_lengths:
            # Fraction of computation cached
            # In full computation: O(seq_len^2)
            # With cache: O(seq_len)
            # Efficiency = (seq_len - 1) / seq_len
            efficiency = (seq_len - 1) / seq_len * 100

            efficiencies.append(efficiency)

        axes[0].plot(seq_lengths, efficiencies, 'o-', label=f'Cache Size={cache_size}', linewidth=2)

    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Computation Saved (%)')
    axes[0].set_title('Cache Efficiency vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Memory vs Speedup tradeoff
    seq_len = 100
    num_tokens = [1, 5, 10, 25, 50, 100]
    memory = [KVCache(1, seq_len, 8, 64).memory_usage() for _ in num_tokens]
    speedup = [1.0 if n == 1 else seq_len / n for n in num_tokens]  # Approx speedup

    axes[1].scatter(memory, speedup, s=200, alpha=0.7)

    for i, n in enumerate(num_tokens):
        axes[1].annotate(f'{n} tokens', (memory[i], speedup[i]), fontsize=9)

    axes[1].set_xlabel('Memory Usage (MB)')
    axes[1].set_ylabel('Speedup Factor')
    axes[1].set_title('Memory vs Speed Tradeoff')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
