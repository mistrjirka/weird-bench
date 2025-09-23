#!/usr/bin/env python3
"""
Plot results from all benchmarks.
"""
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Any, List, Optional


class BenchmarkPlotter:
    """Generate plots for benchmark results."""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "result_plots"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_all_results(self):
        """Plot results from all available benchmarks."""
        print("🎨 Generating plots for all benchmarks...")
        
        # Try to load combined results first
        combined_file = os.path.join(self.results_dir, "all_benchmarks_results.json")
        if os.path.exists(combined_file):
            self._plot_from_combined_results(combined_file)
        else:
            # Fall back to individual result files
            self._plot_from_individual_results()
        
        print("✅ All plots generated successfully!")
    
    def _plot_from_combined_results(self, combined_file: str):
        """Plot from combined results file."""
        with open(combined_file, 'r') as f:
            data = json.load(f)
        
        benchmarks = data.get("benchmarks", {})
        
        if "reversan" in benchmarks and not benchmarks["reversan"].get("failed", False):
            self._plot_reversan_results(benchmarks["reversan"])
        
        if "llama" in benchmarks and not benchmarks["llama"].get("failed", False):
            self._plot_llama_results(benchmarks["llama"])
    
    def _plot_from_individual_results(self):
        """Plot from individual result files."""
        # Reversan results
        reversan_file = os.path.join(self.results_dir, "reversan_results.json")
        if os.path.exists(reversan_file):
            with open(reversan_file, 'r') as f:
                reversan_data = json.load(f)
            self._plot_reversan_results(reversan_data)
        
        # Llama results
        llama_file = os.path.join(self.results_dir, "llama_results.json")
        if os.path.exists(llama_file):
            with open(llama_file, 'r') as f:
                llama_data = json.load(f)
            self._plot_llama_results(llama_data)
    
    def _plot_reversan_results(self, results: Dict[str, Any]):
        """Plot Reversan benchmark results."""
        print("Plotting Reversan results...")
        
        # Depth performance
        if "runs_depth" in results:
            self._plot_reversan_depth_performance(results["runs_depth"])
        
        # Thread performance
        if "runs_threads" in results:
            self._plot_reversan_thread_performance(results["runs_threads"])
        
        # Combined summary for Reversan
        self._plot_reversan_summary(results)
    
    def _plot_llama_results(self, results: Dict[str, Any]):
        """Plot Llama.cpp benchmark results."""
        print("Plotting Llama.cpp results...")
        
        # CPU vs GPU performance comparison
        if "runs_cpu" in results or "runs_gpu" in results:
            self._plot_llama_performance_comparison(results)
        
        # Performance by prompt/generation size
        if "runs_cpu" in results:
            self._plot_llama_performance_matrix(results)
        
        # Build time comparison
        if "build" in results:
            self._plot_llama_build_times(results["build"])
    
    def _extract_metrics_from_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from a run, handling both single and multi-run formats."""
        if 'average_metrics' in run:
            return run['average_metrics']
        else:
            return run['metrics']
    
    def _plot_reversan_depth_performance(self, depth_data: List[Dict[str, Any]]):
        """Plot Reversan depth sweep performance."""
        depths = [run['depth'] for run in depth_data]
        times = [self._extract_metrics_from_run(run)['elapsed_seconds'] for run in depth_data]
        
        # Memory data (filter out None values)
        memory_data = []
        depths_with_memory = []
        for run in depth_data:
            metrics = self._extract_metrics_from_run(run)
            if metrics.get('max_rss_kb') is not None:
                memory_data.append(metrics['max_rss_kb'] / 1024.0)  # Convert KB to MB
                depths_with_memory.append(run['depth'])
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot time on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Search Depth')
        ax1.set_ylabel('Execution Time (seconds)', color=color1)
        line1 = ax1.plot(depths, times, 'bo-', color=color1, linewidth=2, markersize=8, label='Execution Time')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot memory on secondary y-axis (if available)
        if memory_data:
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Memory Usage (MB)', color=color2)
            line2 = ax2.plot(depths_with_memory, memory_data, 'rs-', color=color2, linewidth=2, markersize=8, label='Memory Usage')
            ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if memory_data:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        plt.title('Reversan Engine: Performance vs Search Depth', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'reversan_depth_performance.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reversan_thread_performance(self, thread_data: List[Dict[str, Any]]):
        """Plot Reversan thread performance."""
        # Filter out skipped runs
        valid_runs = [run for run in thread_data if not run.get('skipped', False)]
        
        if not valid_runs:
            print("No valid thread performance data to plot")
            return
        
        threads = [run['threads'] for run in valid_runs]
        times = [self._extract_metrics_from_run(run)['elapsed_seconds'] for run in valid_runs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(threads, times, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Reversan Engine: Thread Scaling Performance (Depth 11)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Calculate speedup
        if len(times) > 1:
            baseline_time = times[0]  # Single thread time
            speedups = [baseline_time / t for t in times]
            
            # Add speedup annotation
            for i, (thread, speedup) in enumerate(zip(threads, speedups)):
                ax.annotate(f'{speedup:.1f}x', (thread, times[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'reversan_threads_performance.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reversan_summary(self, results: Dict[str, Any]):
        """Plot Reversan summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Depth performance (top left)
        if "runs_depth" in results:
            depth_data = results["runs_depth"]
            depths = [run['depth'] for run in depth_data]
            times = [self._extract_metrics_from_run(run)['elapsed_seconds'] for run in depth_data]
            ax1.semilogy(depths, times, 'bo-', linewidth=2, markersize=6)
            ax1.set_xlabel('Depth')
            ax1.set_ylabel('Time (s)')
            ax1.set_title('Performance vs Depth')
            ax1.grid(True, alpha=0.3)
        
        # Thread performance (top right)
        if "runs_threads" in results:
            thread_data = [run for run in results["runs_threads"] if not run.get('skipped', False)]
            if thread_data:
                threads = [run['threads'] for run in thread_data]
                times = [self._extract_metrics_from_run(run)['elapsed_seconds'] for run in thread_data]
                ax2.plot(threads, times, 'go-', linewidth=2, markersize=6)
                ax2.set_xlabel('Threads')
                ax2.set_ylabel('Time (s)')
                ax2.set_title('Thread Scaling (Depth 11)')
                ax2.grid(True, alpha=0.3)
        
        # Build info (bottom left)
        build_info = results.get("build", {})
        if build_info:
            labels = []
            values = []
            if "compile_time_seconds" in build_info:
                labels.append(f'Compile Time\n({build_info["compile_time_seconds"]:.1f}s)')
                values.append(build_info["compile_time_seconds"])
            ax3.pie([1], labels=labels if labels else ['Build Info'], autopct='', startangle=90)
            ax3.set_title('Build Information')
        
        # System info (bottom right)
        meta = results.get("meta", {})
        info_text = f"Host: {meta.get('host', 'Unknown')}\n"
        info_text += f"Platform: {meta.get('platform', 'Unknown')}\n"
        info_text += f"GNU Time: {'Yes' if meta.get('gnu_time') else 'No'}\n"
        if "test_runs_per_config" in meta:
            info_text += f"Runs per test: {meta['test_runs_per_config']}"
        
        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('System Information')
        
        plt.suptitle('Reversan Engine Benchmark Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'reversan_benchmark_summary.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_llama_performance_comparison(self, results: Dict[str, Any]):
        """Plot Llama.cpp CPU vs GPU performance comparison."""
        cpu_runs = results.get("runs_cpu", [])
        gpu_runs = results.get("runs_gpu", [])
        
        if not cpu_runs:
            print("No CPU performance data to plot")
            return
        
        # Prepare data
        configs = []
        cpu_tokens_per_sec = []
        gpu_tokens_per_sec = []
        
        for cpu_run in cpu_runs:
            config = f"P{cpu_run['prompt_size']}/G{cpu_run['generation_size']}"
            configs.append(config)
            
            # Extract tokens per second (fallback to calculated from elapsed time)
            cpu_tps = cpu_run.get("metrics", {}).get("tokens_per_second")
            if cpu_tps is None and cpu_run.get("elapsed_seconds"):
                # Rough estimate: generation_size / elapsed_time
                cpu_tps = cpu_run["generation_size"] / cpu_run["elapsed_seconds"]
            cpu_tokens_per_sec.append(cpu_tps or 0)
            
            # Find corresponding GPU run
            gpu_tps = 0
            for gpu_run in gpu_runs:
                if (gpu_run["prompt_size"] == cpu_run["prompt_size"] and 
                    gpu_run["generation_size"] == cpu_run["generation_size"]):
                    gpu_tps = gpu_run.get("metrics", {}).get("tokens_per_second")
                    if gpu_tps is None and gpu_run.get("elapsed_seconds"):
                        gpu_tps = gpu_run["generation_size"] / gpu_run["elapsed_seconds"]
                    break
            gpu_tokens_per_sec.append(gpu_tps or 0)
        
        # Create comparison plot
        x = np.arange(len(configs))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, cpu_tokens_per_sec, width, label='CPU (ngl=0)', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpu_tokens_per_sec, width, label='GPU (ngl=99)', alpha=0.8)
        
        ax.set_xlabel('Configuration (Prompt/Generation size)')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Llama.cpp: CPU vs GPU Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'llama_cpu_gpu_comparison.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_llama_performance_matrix(self, results: Dict[str, Any]):
        """Plot Llama.cpp performance as a matrix/heatmap."""
        cpu_runs = results.get("runs_cpu", [])
        
        if not cpu_runs:
            return
        
        # Extract unique prompt and generation sizes
        prompt_sizes = sorted(set(run["prompt_size"] for run in cpu_runs))
        gen_sizes = sorted(set(run["generation_size"] for run in cpu_runs))
        
        # Create performance matrix
        perf_matrix = np.zeros((len(gen_sizes), len(prompt_sizes)))
        
        for run in cpu_runs:
            p_idx = prompt_sizes.index(run["prompt_size"])
            g_idx = gen_sizes.index(run["generation_size"])
            
            tps = run.get("metrics", {}).get("tokens_per_second")
            if tps is None and run.get("elapsed_seconds"):
                tps = run["generation_size"] / run["elapsed_seconds"]
            
            perf_matrix[g_idx, p_idx] = tps or 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(perf_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(prompt_sizes)))
        ax.set_yticks(np.arange(len(gen_sizes)))
        ax.set_xticklabels(prompt_sizes)
        ax.set_yticklabels(gen_sizes)
        
        ax.set_xlabel('Prompt Size')
        ax.set_ylabel('Generation Size')
        ax.set_title('Llama.cpp CPU Performance Matrix (Tokens/Second)')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Tokens per Second')
        
        # Add text annotations
        for i in range(len(gen_sizes)):
            for j in range(len(prompt_sizes)):
                text = ax.text(j, i, f'{perf_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontweight="bold")
        
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'llama_performance_matrix.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_llama_build_times(self, build_data: Dict[str, Any]):
        """Plot Llama.cpp build time comparison with detailed breakdown."""
        # Extract timing data from new structure
        regular_timing = build_data.get("regular_build_timing", {})
        vulkan_timing = build_data.get("vulkan_build_timing", {})
        
        # Fallback to old structure for backward compatibility
        if not regular_timing and not vulkan_timing:
            regular_time = build_data.get("regular_build_time_seconds", 0)
            vulkan_time = build_data.get("vulkan_build_time_seconds", 0)
            regular_timing = {"total_time_seconds": regular_time}
            vulkan_timing = {"total_time_seconds": vulkan_time}
        
        regular_total = regular_timing.get("total_time_seconds", 0)
        vulkan_total = vulkan_timing.get("total_time_seconds", 0)
        
        if regular_total == 0 and vulkan_total == 0:
            print("No build timing data to plot")
            return
        
        # Create detailed breakdown plot if we have the data
        if regular_timing.get("config_time_seconds") and vulkan_timing.get("config_time_seconds"):
            self._plot_detailed_build_times(regular_timing, vulkan_timing)
        else:
            self._plot_simple_build_times(regular_total, vulkan_total)
    
    def _plot_simple_build_times(self, regular_time: float, vulkan_time: float):
        """Plot simple build time comparison."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        builds = ['Regular', 'Vulkan']
        times = [regular_time, vulkan_time]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax.bar(builds, times, color=colors, alpha=0.8)
        
        ax.set_ylabel('Build Time (seconds)')
        ax.set_title('Llama.cpp Build Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            if time > 0:
                ax.annotate(f'{time:.1f}s',
                           xy=(bar.get_x() + bar.get_width() / 2, time),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'llama_build_times.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_build_times(self, regular_timing: Dict[str, float], vulkan_timing: Dict[str, float]):
        """Plot detailed build time breakdown."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Data for stacked bar chart
        builds = ['Regular', 'Vulkan']
        config_times = [
            regular_timing.get("config_time_seconds", 0),
            vulkan_timing.get("config_time_seconds", 0)
        ]
        build_times = [
            regular_timing.get("build_time_seconds", 0),
            vulkan_timing.get("build_time_seconds", 0)
        ]
        
        # Stacked bar chart (left)
        width = 0.6
        bars1 = ax1.bar(builds, config_times, width, label='Configuration', color='lightblue', alpha=0.8)
        bars2 = ax1.bar(builds, build_times, width, bottom=config_times, label='Compilation', color='lightcoral', alpha=0.8)
        
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Build Time Breakdown')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (build, config, compile) in enumerate(zip(builds, config_times, build_times)):
            total = config + compile
            if total > 0:
                # Config time label
                if config > 0:
                    ax1.annotate(f'{config:.1f}s', xy=(i, config/2), ha='center', va='center', fontweight='bold', color='darkblue')
                # Build time label
                if compile > 0:
                    ax1.annotate(f'{compile:.1f}s', xy=(i, config + compile/2), ha='center', va='center', fontweight='bold', color='darkred')
                # Total time label
                ax1.annotate(f'Total: {total:.1f}s', xy=(i, total + max(total)*0.02), ha='center', va='bottom', fontweight='bold')
        
        # Comparison chart (right)
        total_times = [sum(pair) for pair in zip(config_times, build_times)]
        colors = ['skyblue', 'lightcoral']
        bars = ax2.bar(builds, total_times, color=colors, alpha=0.8)
        
        ax2.set_ylabel('Total Build Time (seconds)')
        ax2.set_title('Total Build Time Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add speedup/slowdown annotation
        if total_times[0] > 0 and total_times[1] > 0:
            ratio = total_times[1] / total_times[0]
            if ratio > 1:
                ax2.text(0.5, max(total_times) * 0.9, f'Vulkan build is {ratio:.1f}x slower', 
                        ha='center', va='center', transform=ax2.transData, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            else:
                ax2.text(0.5, max(total_times) * 0.9, f'Vulkan build is {1/ratio:.1f}x faster', 
                        ha='center', va='center', transform=ax2.transData,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Add value labels on bars
        for bar, time in zip(bars, total_times):
            if time > 0:
                ax2.annotate(f'{time:.1f}s',
                           xy=(bar.get_x() + bar.get_width() / 2, time),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.output_dir, f'llama_build_times.{ext}'), 
                       dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="result_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    plotter = BenchmarkPlotter(args.results_dir, args.output_dir)
    plotter.plot_all_results()


if __name__ == "__main__":
    main()