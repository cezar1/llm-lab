
"""
Scaling Laws: Train models of different sizes, plot loss vs parameters
Measure VRAM, wall-clock time, extrapolate scaling curve
"""

import numpy as np
import matplotlib.pyplot as plt


class ScalingLawSimulator:
    """Simulate training models of different sizes"""

    @staticmethod
    def estimate_loss(num_params, tokens=1e8):
        """Estimate loss using Chinchilla scaling law"""
        # Loss ≈ A + B / (N^α) + C / (D^β)
        # Simplified: Loss ≈ a + b / N^0.07
        a = 0.5  # Irreducible loss
        b = 100
        alpha = 0.07

        loss = a + b / (num_params ** alpha)
        return loss

    @staticmethod
    def estimate_compute(num_params, tokens=1e8):
        """Estimate FLOPs needed"""
        # FLOPs ≈ 6 * N * D (where D is dataset size)
        flops = 6 * num_params * tokens
        return flops

    @staticmethod
    def estimate_vram(num_params, batch_size=32):
        """Estimate VRAM needed (in GB)"""
        # Rough estimate: 4 bytes per param + activations + optimizer states
        # Typical: ~20 bytes per param for full training
        vram_per_param = 20
        total_bytes = num_params * vram_per_param * batch_size
        return total_bytes / (1024 ** 3)

    @staticmethod
    def estimate_wall_clock(num_params, tokens=1e8, tflops=100):
        """Estimate training time"""
        flops = ScalingLawSimulator.estimate_compute(num_params, tokens)
        seconds = flops / (tflops * 1e12)  # Convert to seconds
        hours = seconds / 3600
        return hours

    def sweep_model_sizes(self):
        """Sweep over model sizes and collect metrics"""
        model_sizes = np.logspace(5, 11, 20)  # 100K to 100B parameters

        results = {
            'params': [],
            'loss': [],
            'vram': [],
            'hours': [],
            'flops': []
        }

        for num_params in model_sizes:
            results['params'].append(num_params)
            results['loss'].append(self.estimate_loss(num_params))
            results['vram'].append(self.estimate_vram(num_params, batch_size=16))
            results['hours'].append(self.estimate_wall_clock(num_params))
            results['flops'].append(self.estimate_compute(num_params))

        return results

    def plot_scaling_laws(self, results):
        """Plot scaling law results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        params = np.array(results['params'])
        loss = np.array(results['loss'])
        vram = np.array(results['vram'])
        hours = np.array(results['hours'])

        # Loss vs Parameters
        axes[0, 0].loglog(params, loss, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Model Parameters')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Scaling Law: Loss vs Model Size')
        axes[0, 0].grid(True, alpha=0.3, which='both')

        # Loss improvement (difference from baseline)
        loss_improvement = loss[0] - loss
        axes[0, 1].semilogx(params, loss_improvement, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Model Parameters')
        axes[0, 1].set_ylabel('Loss Improvement')
        axes[0, 1].set_title('Loss Improvement vs Baseline')
        axes[0, 1].grid(True, alpha=0.3)

        # VRAM vs Parameters
        axes[1, 0].loglog(params, vram, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_xlabel('Model Parameters')
        axes[1, 0].set_ylabel('VRAM (GB)')
        axes[1, 0].set_title('Memory Requirement vs Model Size')
        axes[1, 0].grid(True, alpha=0.3, which='both')

        # Training Time vs Parameters
        axes[1, 1].loglog(params, hours, 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_xlabel('Model Parameters')
        axes[1, 1].set_ylabel('Training Time (hours)')
        axes[1, 1].set_title('Wall-Clock Time vs Model Size')
        axes[1, 1].grid(True, alpha=0.3, which='both')

        plt.suptitle('Scaling Laws: Model Size vs Performance & Resources')
        plt.tight_layout()
        plt.show()

        return fig

    def analyze_pareto_frontier(self, results):
        """Find optimal model sizes given constraints"""
        params = np.array(results['params'])
        loss = np.array(results['loss'])
        hours = np.array(results['hours'])
        vram = np.array(results['vram'])

        print("\nModel Size Recommendations:")
        print("-" * 60)

        # Find best loss for given time budget
        time_budgets = [1, 10, 100, 1000]  # hours
        for budget in time_budgets:
            valid_indices = hours <= budget
            if valid_indices.sum() > 0:
                best_idx = np.argmin(loss[valid_indices])
                best_params = params[valid_indices][best_idx]
                best_loss = loss[valid_indices][best_idx]
                print(f"  Time budget {budget:4.0f}h → {best_params/1e6:6.1f}M params, loss={best_loss:.4f}")

        # Find best loss for given VRAM budget
        vram_budgets = [1, 10, 40, 80]  # GB
        print("\nVRAM-constrained models:")
        for budget in vram_budgets:
            valid_indices = vram <= budget
            if valid_indices.sum() > 0:
                best_idx = np.argmin(loss[valid_indices])
                best_params = params[valid_indices][best_idx]
                best_loss = loss[valid_indices][best_idx]
                print(f"  VRAM budget {budget:3.0f}GB → {best_params/1e6:6.1f}M params, loss={best_loss:.4f}")

    def plot_pareto_frontier(self, results):
        """Plot performance vs resource tradeoffs"""
        params = np.array(results['params'])
        loss = np.array(results['loss'])
        hours = np.array(results['hours'])
        vram = np.array(results['vram'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss vs Time
        scatter1 = ax1.scatter(hours, loss, c=params, cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Training Time (hours, log scale)')
        ax1.set_ylabel('Loss')
        ax1.set_xscale('log')
        ax1.set_title('Pareto Frontier: Loss vs Time')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Parameters')

        # Loss vs VRAM
        scatter2 = ax2.scatter(vram, loss, c=params, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('VRAM (GB, log scale)')
        ax2.set_ylabel('Loss')
        ax2.set_xscale('log')
        ax2.set_title('Pareto Frontier: Loss vs VRAM')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Parameters')

        plt.tight_layout()
        plt.show()
