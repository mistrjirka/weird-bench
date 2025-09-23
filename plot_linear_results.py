#!/usr/bin/env python3
"""
Plot results from benchmark results with linear scaling for better readability.
Supports both Reversan and Llama benchmarks.
"""
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from typing import Dict, Any, List, Optional

# Configuration
OUTPUT_DIR = "result_plots"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results(filename: str):
    """Load benchmark results from JSON file."""
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def extract_metrics_from_run(run):
    """Extract metrics from a run, handling both single and multi-run formats."""
    if 'average_metrics' in run:
        # Multi-run format - use averages
        return run['average_metrics']
    else:
        # Single-run format - use direct metrics
        return run['metrics']

def plot_reversan_results(results, output_dir):
    """Plot Reversan benchmark results."""
    print("📊 Plotting Reversan results...")
    
    # Plot depth sweep results
    if 'runs_depth' in results:
        plot_reversan_depth_results(results, output_dir)
    
    # Plot threads results
    if 'runs_threads' in results:
        plot_reversan_threads_results(results, output_dir)
    
    # Create summary plot
    create_reversan_summary_plot(results, output_dir)

def plot_reversan_depth_results(results, output_dir):
    """Plot Reversan depth sweep results."""
    depth_data = results['runs_depth']
    depths = [run['depth'] for run in depth_data]
    times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in depth_data]
    
    # Convert memory from KB to MB and filter out None values
    memory_data = []
    depths_with_memory = []
    for run in depth_data:
        metrics = extract_metrics_from_run(run)
        if metrics['max_rss_kb'] is not None:
            memory_data.append(metrics['max_rss_kb'] / 1024.0)  # Convert KB to MB
            depths_with_memory.append(run['depth'])
    
    # Create figure with subplots
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot time on primary y-axis (LINEAR scale)
    color1 = 'tab:blue'
    ax1.set_xlabel('Search Depth', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', color=color1, fontsize=12)
    line1 = ax1.plot(depths, times, marker='o', color=color1, linewidth=2, markersize=6, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for memory (LINEAR scale)
    if memory_data:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Memory Usage (MB)', color=color2, fontsize=12)
        line2 = ax2.plot(depths_with_memory, memory_data, marker='s', color=color2, linewidth=2, markersize=6, label='Memory Usage')
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    plt.title('Reversan Engine Performance vs Search Depth\n(Linear Scale)', fontsize=14, fontweight='bold')
    
    # Create legend
    lines = line1
    labels = ['Execution Time']
    if memory_data:
        lines += line2
        labels += ['Memory Usage']
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reversan_depth_performance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'reversan_depth_performance.svg'), bbox_inches='tight')
    plt.close()

def plot_reversan_threads_results(results, output_dir):
    """Plot Reversan threads results."""
    threads_data = results['runs_threads']
    threads = [run['threads'] for run in threads_data]
    times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in threads_data]
    
    # Calculate speedup (relative to single thread)
    baseline_time = times[0]  # Assuming first entry is single thread
    speedups = [baseline_time / time for time in times]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot execution time (LINEAR scale)
    ax1.plot(threads, times, marker='o', color='tab:blue', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Thread Count\n(Linear Scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)
    
    # Plot speedup
    ax2.plot(threads, speedups, marker='s', color='tab:green', linewidth=2, markersize=6, label='Actual Speedup')
    ax2.plot(threads, threads, linestyle='--', color='tab:red', alpha=0.7, label='Ideal Speedup')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Speedup vs Thread Count', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(threads)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reversan_threads_performance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'reversan_threads_performance.svg'), bbox_inches='tight')
    plt.close()

def create_reversan_summary_plot(results, output_dir):
    """Create a summary plot for Reversan results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Depth results
    if 'runs_depth' in results:
        depth_data = results['runs_depth']
        depths = [run['depth'] for run in depth_data]
        times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in depth_data]
        ax1.plot(depths, times, marker='o', color='tab:blue', linewidth=2)
        ax1.set_xlabel('Search Depth')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Performance vs Depth')
        ax1.grid(True, alpha=0.3)
    
    # Threads results
    if 'runs_threads' in results:
        threads_data = results['runs_threads']
        threads = [run['threads'] for run in threads_data]
        times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in threads_data]
        ax2.plot(threads, times, marker='s', color='tab:green', linewidth=2)
        ax2.set_xlabel('Thread Count')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Performance vs Threads')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(threads)
    
    # Memory usage by depth
    if 'runs_depth' in results:
        memory_data = []
        depths_with_memory = []
        for run in results['runs_depth']:
            metrics = extract_metrics_from_run(run)
            if metrics['max_rss_kb'] is not None:
                memory_data.append(metrics['max_rss_kb'] / 1024.0)
                depths_with_memory.append(run['depth'])
        
        if memory_data:
            ax3.plot(depths_with_memory, memory_data, marker='^', color='tab:red', linewidth=2)
            ax3.set_xlabel('Search Depth')
            ax3.set_ylabel('Memory (MB)')
            ax3.set_title('Memory Usage vs Depth')
            ax3.grid(True, alpha=0.3)
    
    # Build timing if available
    if 'meta' in results and 'build_info' in results['meta']:
        build_info = results['meta']['build_info']
        ax4.text(0.1, 0.8, f"Build Time: {build_info.get('total_time', 'N/A')} seconds", transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Compiler: {build_info.get('compiler', 'N/A')}", transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Optimization: {build_info.get('optimization', 'N/A')}", transform=ax4.transAxes)
        ax4.set_title('Build Information')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    plt.suptitle('Reversan Engine Benchmark Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reversan_benchmark_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'reversan_benchmark_summary.svg'), bbox_inches='tight')
    plt.close()

def plot_llama_results(results, output_dir):
    """Plot Llama benchmark results."""
    print("📊 Plotting Llama results...")
    
    # Plot CPU vs GPU performance
    plot_llama_performance_comparison(results, output_dir)
    
    # Plot build timing
    plot_llama_build_timing(results, output_dir)
    
    # Plot detailed metrics if available
    plot_llama_detailed_metrics(results, output_dir)
    
    # Create summary plot
    create_llama_summary_plot(results, output_dir)

def plot_llama_performance_comparison(results, output_dir):
    """Plot CPU vs GPU performance comparison."""
    cpu_runs = results.get('runs_cpu', [])
    gpu_runs = results.get('runs_gpu', [])
    
    if not cpu_runs and not gpu_runs:
        print("⚠️  No benchmark runs found for Llama")
        return
    
    # Extract data for plotting
    configurations = []
    cpu_times = []
    gpu_times = []
    cpu_tokens_per_sec = []
    gpu_tokens_per_sec = []
    
    # Process CPU runs
    for run in cpu_runs:
        config = f"P{run['prompt_size']}/G{run['generation_size']}"
        configurations.append(config)
        cpu_times.append(run['elapsed_seconds'])
        
        # Extract tokens per second if available
        tokens_per_sec = run.get('metrics', {}).get('tokens_per_second', 0)
        cpu_tokens_per_sec.append(tokens_per_sec)
    
    # Process GPU runs (match with CPU configurations)
    for i, cpu_run in enumerate(cpu_runs):
        gpu_time = 0
        gpu_tokens = 0
        
        # Find matching GPU run
        for gpu_run in gpu_runs:
            if (gpu_run['prompt_size'] == cpu_run['prompt_size'] and 
                gpu_run['generation_size'] == cpu_run['generation_size']):
                gpu_time = gpu_run['elapsed_seconds']
                gpu_tokens = gpu_run.get('metrics', {}).get('tokens_per_second', 0)
                break
        
        gpu_times.append(gpu_time)
        gpu_tokens_per_sec.append(gpu_tokens)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(configurations))
    width = 0.35
    
    # Plot execution times (LINEAR scale)
    ax1.bar(x - width/2, cpu_times, width, label='CPU', color='tab:blue', alpha=0.8)
    if any(t > 0 for t in gpu_times):
        ax1.bar(x + width/2, gpu_times, width, label='GPU (Vulkan)', color='tab:red', alpha=0.8)
    
    ax1.set_xlabel('Configuration (Prompt/Generation tokens)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Llama.cpp Performance: CPU vs GPU\n(Linear Scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configurations, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot tokens per second if available
    if any(t > 0 for t in cpu_tokens_per_sec + gpu_tokens_per_sec):
        bars1 = ax2.bar(x - width/2, cpu_tokens_per_sec, width, label='CPU', color='tab:blue', alpha=0.8)
        if any(t > 0 for t in gpu_tokens_per_sec):
            bars2 = ax2.bar(x + width/2, gpu_tokens_per_sec, width, label='GPU (Vulkan)', color='tab:red', alpha=0.8)
        
        ax2.set_xlabel('Configuration (Prompt/Generation tokens)')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Llama.cpp Throughput: CPU vs GPU')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configurations, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (cpu_val, gpu_val) in enumerate(zip(cpu_tokens_per_sec, gpu_tokens_per_sec)):
            if cpu_val > 0:
                ax2.text(i - width/2, cpu_val + max(cpu_tokens_per_sec + gpu_tokens_per_sec) * 0.01,
                        f'{cpu_val:.1f}', ha='center', va='bottom', fontsize=9)
            if gpu_val > 0:
                ax2.text(i + width/2, gpu_val + max(cpu_tokens_per_sec + gpu_tokens_per_sec) * 0.01,
                        f'{gpu_val:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No throughput data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Throughput Data Not Available')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llama_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'llama_performance_comparison.svg'), bbox_inches='tight')
    plt.close()

def plot_llama_build_timing(results, output_dir):
    """Plot Llama build timing results."""
    build_data = results.get('build', {})
    
    # Extract build timings
    cpu_timing = build_data.get('cpu_build_timing', {})
    vulkan_timing = build_data.get('vulkan_build_timing', {})
    
    if not cpu_timing and not vulkan_timing:
        print("⚠️  No build timing data found for Llama")
        return
    
    # Prepare data
    build_types = []
    config_times = []
    build_times = []
    total_times = []
    
    if cpu_timing:
        build_types.append('CPU Build')
        config_times.append(cpu_timing.get('config_time_seconds', 0))
        build_times.append(cpu_timing.get('build_time_seconds', 0))
        total_times.append(cpu_timing.get('total_time_seconds', 0))
    
    if vulkan_timing:
        build_types.append('Vulkan Build')
        config_times.append(vulkan_timing.get('config_time_seconds', 0))
        build_times.append(vulkan_timing.get('build_time_seconds', 0))
        total_times.append(vulkan_timing.get('total_time_seconds', 0))
    
    # Create build timing plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(build_types))
    width = 0.35
    
    # Stacked bar chart for build phases
    if config_times and build_times:
        ax1.bar(x, config_times, width, label='Configuration', color='tab:blue', alpha=0.8)
        ax1.bar(x, build_times, width, bottom=config_times, label='Compilation', color='tab:orange', alpha=0.8)
        
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Llama.cpp Build Time Breakdown')
        ax1.set_xticks(x)
        ax1.set_xticklabels(build_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add total time labels on bars
        for i, total in enumerate(total_times):
            ax1.text(i, total + max(total_times) * 0.01, f'{total:.1f}s', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Total build times comparison
    if total_times:
        bars = ax2.bar(build_types, total_times, color=['tab:blue', 'tab:red'] if len(build_types) == 2 else ['tab:blue'], alpha=0.8)
        ax2.set_ylabel('Total Time (seconds)')
        ax2.set_title('Total Build Time Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, total_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_times) * 0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llama_build_timing.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'llama_build_timing.svg'), bbox_inches='tight')
    plt.close()

def plot_llama_detailed_metrics(results, output_dir):
    """Plot detailed metrics from JSON output if available."""
    cpu_runs = results.get('runs_cpu', [])
    gpu_runs = results.get('runs_gpu', [])
    
    # Check if we have detailed JSON metrics
    has_detailed_metrics = False
    for run in cpu_runs + gpu_runs:
        if 'raw_json' in run and run['raw_json']:
            has_detailed_metrics = True
            break
    
    if not has_detailed_metrics:
        print("⚠️  No detailed JSON metrics available for advanced plotting")
        return
    
    print("📊 Creating detailed metrics plots...")
    
    # Create detailed performance breakdown
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Prompt vs Generation Performance
    prompt_speeds = {"CPU": [], "GPU": [], "configs": []}
    gen_speeds = {"CPU": [], "GPU": [], "configs": []}
    
    for run in cpu_runs:
        if 'metrics' in run and 'prompt_processing' in run['metrics'] and 'generation' in run['metrics']:
            config = f"P{run['prompt_size']}/G{run['generation_size']}"
            prompt_speeds["configs"].append(config)
            prompt_speeds["CPU"].append(run['metrics']['prompt_processing'].get('avg_tokens_per_sec', 0))
            gen_speeds["CPU"].append(run['metrics']['generation'].get('avg_tokens_per_sec', 0))
    
    # Match GPU runs
    for cpu_run in cpu_runs:
        gpu_prompt_speed = 0
        gpu_gen_speed = 0
        for gpu_run in gpu_runs:
            if (gpu_run['prompt_size'] == cpu_run['prompt_size'] and 
                gpu_run['generation_size'] == cpu_run['generation_size']):
                if 'metrics' in gpu_run and 'prompt_processing' in gpu_run['metrics']:
                    gpu_prompt_speed = gpu_run['metrics']['prompt_processing'].get('avg_tokens_per_sec', 0)
                    gpu_gen_speed = gpu_run['metrics']['generation'].get('avg_tokens_per_sec', 0)
                break
        prompt_speeds["GPU"].append(gpu_prompt_speed)
        gen_speeds["GPU"].append(gpu_gen_speed)
    
    if prompt_speeds["configs"]:
        x = np.arange(len(prompt_speeds["configs"]))
        width = 0.35
        
        ax1.bar(x - width/2, prompt_speeds["CPU"], width, label='CPU', color='tab:blue', alpha=0.8)
        if any(s > 0 for s in prompt_speeds["GPU"]):
            ax1.bar(x + width/2, prompt_speeds["GPU"], width, label='GPU', color='tab:red', alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Tokens per Second')
        ax1.set_title('Prompt Processing Speed')
        ax1.set_xticks(x)
        ax1.set_xticklabels(prompt_speeds["configs"], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x - width/2, gen_speeds["CPU"], width, label='CPU', color='tab:blue', alpha=0.8)
        if any(s > 0 for s in gen_speeds["GPU"]):
            ax2.bar(x + width/2, gen_speeds["GPU"], width, label='GPU', color='tab:red', alpha=0.8)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Text Generation Speed')
        ax2.set_xticks(x)
        ax2.set_xticklabels(gen_speeds["configs"], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: System Information
    ax3.axis('off')
    ax3.set_title('System Information')
    
    # Extract system info from first run with detailed metrics
    system_info = None
    for run in cpu_runs + gpu_runs:
        if 'metrics' in run and 'system_info' in run['metrics']:
            system_info = run['metrics']['system_info']
            break
    
    if system_info:
        info_text = f"CPU: {system_info.get('cpu_info', 'Unknown')}\n"
        info_text += f"GPU: {system_info.get('gpu_info', 'Unknown')}\n"
        info_text += f"Backend: {system_info.get('backends', 'Unknown')}\n"
        info_text += f"Model: {system_info.get('model_type', 'Unknown')}\n"
        info_text += f"Parameters: {system_info.get('model_n_params', 0):,}\n"
        info_text += f"Threads: {system_info.get('n_threads', 0)}\n"
        info_text += f"GPU Layers: {system_info.get('n_gpu_layers', 0)}"
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    # Plot 4: Performance variability (standard deviation)
    if prompt_speeds["configs"]:
        cpu_stddevs = []
        gpu_stddevs = []
        
        for i, config in enumerate(prompt_speeds["configs"]):
            cpu_stddev = 0
            gpu_stddev = 0
            
            # Find corresponding runs
            for run in cpu_runs:
                if f"P{run['prompt_size']}/G{run['generation_size']}" == config:
                    if 'metrics' in run and 'generation' in run['metrics']:
                        cpu_stddev = run['metrics']['generation'].get('stddev_tokens_per_sec', 0)
                    break
            
            for run in gpu_runs:
                if f"P{run['prompt_size']}/G{run['generation_size']}" == config:
                    if 'metrics' in run and 'generation' in run['metrics']:
                        gpu_stddev = run['metrics']['generation'].get('stddev_tokens_per_sec', 0)
                    break
            
            cpu_stddevs.append(cpu_stddev)
            gpu_stddevs.append(gpu_stddev)
        
        if any(s > 0 for s in cpu_stddevs + gpu_stddevs):
            ax4.bar(x - width/2, cpu_stddevs, width, label='CPU', color='tab:blue', alpha=0.8)
            if any(s > 0 for s in gpu_stddevs):
                ax4.bar(x + width/2, gpu_stddevs, width, label='GPU', color='tab:red', alpha=0.8)
            
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Standard Deviation (tokens/s)')
            ax4.set_title('Performance Variability')
            ax4.set_xticks(x)
            ax4.set_xticklabels(prompt_speeds["configs"], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No variability data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Variability')
    
    plt.suptitle('Llama.cpp Detailed Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llama_detailed_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'llama_detailed_metrics.svg'), bbox_inches='tight')
    plt.close()

def create_llama_summary_plot(results, output_dir):
    """Create a summary plot for Llama results."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])
    
    # Performance comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    cpu_runs = results.get('runs_cpu', [])
    gpu_runs = results.get('runs_gpu', [])
    
    if cpu_runs:
        configs = [f"P{run['prompt_size']}/G{run['generation_size']}" for run in cpu_runs]
        cpu_times = [run['elapsed_seconds'] for run in cpu_runs]
        ax1.bar(configs, cpu_times, color='tab:blue', alpha=0.8, label='CPU')
        
        if gpu_runs:
            gpu_times = []
            for cpu_run in cpu_runs:
                gpu_time = 0
                for gpu_run in gpu_runs:
                    if (gpu_run['prompt_size'] == cpu_run['prompt_size'] and 
                        gpu_run['generation_size'] == cpu_run['generation_size']):
                        gpu_time = gpu_run['elapsed_seconds']
                        break
                gpu_times.append(gpu_time)
            
            x = np.arange(len(configs))
            ax1.bar(x + 0.4, gpu_times, 0.4, color='tab:red', alpha=0.8, label='GPU')
        
        ax1.set_title('Performance Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
    
    # Build timing (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    build_data = results.get('build', {})
    
    build_times = []
    build_labels = []
    
    cpu_timing = build_data.get('cpu_build_timing', {})
    if cpu_timing:
        build_times.append(cpu_timing.get('total_time_seconds', 0))
        build_labels.append('CPU Build')
    
    vulkan_timing = build_data.get('vulkan_build_timing', {})
    if vulkan_timing:
        build_times.append(vulkan_timing.get('total_time_seconds', 0))
        build_labels.append('Vulkan Build')
    
    if build_times:
        ax2.bar(build_labels, build_times, color=['tab:blue', 'tab:red'][:len(build_times)], alpha=0.8)
        ax2.set_title('Build Times')
        ax2.set_ylabel('Time (seconds)')
    
    # System info (bottom span)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    
    # Display system and build information
    meta = results.get('meta', {})
    info_text = f"Model: {meta.get('model_url', 'N/A').split('/')[-1] if meta.get('model_url') else 'N/A'}\n"
    info_text += f"Repository: {meta.get('repo', 'N/A')}\n"
    
    build_info = build_data
    if build_info.get('vulkan_supported'):
        info_text += "Vulkan: Supported\n"
        vulkan_devices = build_info.get('vulkan_devices', [])
        if vulkan_devices:
            info_text += f"Vulkan Devices: {len(vulkan_devices)} detected"
    else:
        info_text += "Vulkan: Not supported"
    
    ax3.text(0.05, 0.8, info_text, transform=ax3.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Llama.cpp Benchmark Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llama_benchmark_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'llama_benchmark_summary.svg'), bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Single file specified
        results_file = sys.argv[1]
        print(f"📊 Loading results from: {results_file}")
        results = load_results(results_file)
        
        if not results:
            print(f"❌ Could not load results from {results_file}")
            sys.exit(1)
        
        ensure_output_dir()
        process_single_results_file(results)
        
    else:
        # Auto-detect and process multiple individual benchmark files
        individual_files = [
            ("reversan_results.json", "Reversan"),
            ("llama_results.json", "Llama"),
            ("results/reversan_results.json", "Reversan"),
            ("results/llama_results.json", "Llama")
        ]
        
        found_files = []
        for file_path, benchmark_name in individual_files:
            if os.path.exists(file_path):
                found_files.append((file_path, benchmark_name))
        
        # Remove duplicates (prefer files in results/ directory)
        unique_files = {}
        for file_path, benchmark_name in found_files:
            if benchmark_name not in unique_files or file_path.startswith("results/"):
                unique_files[benchmark_name] = file_path
        
        if not unique_files:
            # Fall back to combined file
            combined_files = [
                "all_benchmarks_results.json",
                "results/all_benchmarks_results.json"
            ]
            
            combined_file = None
            for file in combined_files:
                if os.path.exists(file):
                    combined_file = file
                    break
            
            if combined_file:
                print(f"📊 Loading combined results from: {combined_file}")
                results = load_results(combined_file)
                if results:
                    ensure_output_dir()
                    process_single_results_file(results)
                else:
                    print(f"❌ Could not load results from {combined_file}")
                    sys.exit(1)
            else:
                print("❌ No results files found. Please run benchmarks first.")
                print("Usage: python plot_linear_results.py [results_file.json]")
                sys.exit(1)
        else:
            # Process multiple individual files
            ensure_output_dir()
            print(f"📊 Found {len(unique_files)} benchmark result files")
            
            for benchmark_name, file_path in unique_files.items():
                print(f"\n📊 Processing {benchmark_name} results from: {file_path}")
                results = load_results(file_path)
                
                if results:
                    process_single_results_file(results)
                else:
                    print(f"⚠️  Could not load {benchmark_name} results from {file_path}")
    
    print(f"✅ All plots saved to {OUTPUT_DIR}/ directory")

def process_single_results_file(results):
    """Process a single results file and generate appropriate plots."""
    # Determine what type of results we have
    if "benchmarks" in results:
        # Combined results format
        benchmarks = results["benchmarks"]
        
        if "reversan" in benchmarks and not benchmarks["reversan"].get("failed", False):
            plot_reversan_results(benchmarks["reversan"], OUTPUT_DIR)
        
        if "llama" in benchmarks and not benchmarks["llama"].get("failed", False):
            plot_llama_results(benchmarks["llama"], OUTPUT_DIR)
            
    elif "runs_depth" in results or "runs_threads" in results:
        # Reversan results format
        plot_reversan_results(results, OUTPUT_DIR)
        
    elif "runs_cpu" in results or "runs_gpu" in results or "build" in results:
        # Llama results format
        plot_llama_results(results, OUTPUT_DIR)
        
    else:
        print("⚠️  Unknown results format, skipping...")

if __name__ == "__main__":
    main()