#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

# Configuration
INPUT_JSON = "reversan_results.json"
OUTPUT_DIR = "result_plots"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    """Load benchmark results from JSON file."""
    with open(INPUT_JSON, 'r') as f:
        return json.load(f)

def extract_metrics_from_run(run):
    """Extract metrics from a run, handling both single and multi-run formats."""
    if 'average_metrics' in run:
        # Multi-run format - use averages
        return run['average_metrics']
    else:
        # Single-run format - use direct metrics
        return run['metrics']

def plot_depth_results(results):
    """Plot depth sweep results."""
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
    
    # Plot time on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Search Depth')
    ax1.set_ylabel('Elapsed Time (seconds)', color=color1)
    line1 = ax1.plot(depths, times, 'o-', color=color1, linewidth=2, markersize=6, label='Elapsed Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    # Use normal scale instead of log scale for better readability
    
    # Add memory on secondary y-axis if available
    if memory_data:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Memory Usage (MB)', color=color2)
        line2 = ax2.plot(depths_with_memory, memory_data, 's-', color=color2, linewidth=2, markersize=6, label='Memory Usage')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Create combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    plt.title('Reversi Engine Performance vs Search Depth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save as both SVG and PNG
    base_name = 'depth_performance'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Depth performance plot saved to {OUTPUT_DIR}/{base_name}.(svg|png)")

def plot_threads_results(results):
    """Plot threads sweep results."""
    threads_data = results['runs_threads']
    
    # Filter out skipped tests
    valid_runs = [run for run in threads_data if not run.get('skipped', False)]
    
    if not valid_runs:
        print("⚠ No valid thread test results to plot")
        return
    
    threads = [run['threads'] for run in valid_runs]
    times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in valid_runs]
    
    # Convert memory from KB to MB and collect data
    memory_data = []
    threads_with_memory = []
    for run in valid_runs:
        metrics = extract_metrics_from_run(run)
        if metrics['max_rss_kb'] is not None:
            memory_data.append(metrics['max_rss_kb'] / 1024.0)  # Convert KB to MB
            threads_with_memory.append(run['threads'])
    
    # Calculate speedup relative to single thread
    if times and threads[0] == 1:
        baseline_time = times[0]
        speedups = [baseline_time / time for time in times]
    else:
        speedups = None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute time and memory
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Elapsed Time (seconds)', color=color1)
    line1 = ax1.plot(threads, times, 'o-', color=color1, linewidth=2, markersize=6, label='Elapsed Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)
    
    # Add memory on secondary y-axis if available
    if memory_data:
        ax1_mem = ax1.twinx()
        color2 = 'tab:red'
        ax1_mem.set_ylabel('Memory Usage (MB)', color=color2)
        line2 = ax1_mem.plot(threads_with_memory, memory_data, 's-', color=color2, linewidth=2, markersize=6, label='Memory Usage')
        ax1_mem.tick_params(axis='y', labelcolor=color2)
        
        # Create combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    ax1.set_title('Threading Performance (Absolute)', fontweight='bold')
    
    # Plot 2: Speedup and efficiency
    if speedups:
        color3 = 'tab:green'
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Speedup Factor', color=color3)
        line3 = ax2.plot(threads, speedups, 'o-', color=color3, linewidth=2, markersize=6, label='Speedup')
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(threads)
        
        # Add ideal speedup line
        ax2.plot(threads, threads, '--', color='gray', alpha=0.7, label='Ideal Speedup')
        
        # Calculate and plot efficiency
        ax2_eff = ax2.twinx()
        color4 = 'tab:orange'
        efficiency = [speedup / thread_count * 100 for speedup, thread_count in zip(speedups, threads)]
        ax2_eff.set_ylabel('Efficiency (%)', color=color4)
        line4 = ax2_eff.plot(threads, efficiency, '^-', color=color4, linewidth=2, markersize=6, label='Efficiency')
        ax2_eff.tick_params(axis='y', labelcolor=color4)
        ax2_eff.set_ylim(0, 110)
        
        # Create combined legend
        lines = line3 + [ax2.lines[1]] + line4  # Include ideal speedup line
        labels = ['Speedup', 'Ideal Speedup', 'Efficiency']
        ax2.legend(lines, labels, loc='upper right')
        
        ax2.set_title('Threading Efficiency', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No speedup data available\n(missing single-thread baseline)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Threading Efficiency (N/A)', fontweight='bold')
    
    plt.suptitle('Reversi Engine Threading Performance (Depth 11)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save as both SVG and PNG
    base_name = 'threads_performance'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Threading performance plot saved to {OUTPUT_DIR}/{base_name}.(svg|png)")

def plot_summary(results):
    """Create a summary plot with key metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Depth vs Time (normal scale)
    depth_data = results['runs_depth']
    depths = [run['depth'] for run in depth_data]
    times = [extract_metrics_from_run(run)['elapsed_seconds'] for run in depth_data]
    
    ax1.plot(depths, times, 'o-', linewidth=2, markersize=6, color='tab:blue')
    ax1.set_xlabel('Search Depth')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Search Time Growth', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(depths)
    
    # 2. Threading speedup
    threads_data = [run for run in results['runs_threads'] if not run.get('skipped', False)]
    if threads_data:
        threads = [run['threads'] for run in threads_data]
        thread_times = [run['metrics']['elapsed_seconds'] for run in threads_data]
        if thread_times and threads[0] == 1:
            baseline = thread_times[0]
            speedups = [baseline / time for time in thread_times]
            ax2.plot(threads, speedups, 'o-', linewidth=2, markersize=6, color='tab:green', label='Actual')
            ax2.plot(threads, threads, '--', color='gray', alpha=0.7, label='Ideal')
            ax2.set_xlabel('Number of Threads')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Threading Speedup', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(threads)
    else:
        ax2.text(0.5, 0.5, 'No threading data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Threading Speedup (N/A)', fontweight='bold')
    
    # 3. Memory growth by depth
    memory_data_mb = []
    depths_for_memory = []
    for run in depth_data:
        metrics = extract_metrics_from_run(run)
        if metrics['max_rss_kb'] is not None:
            memory_data_mb.append(metrics['max_rss_kb'] / 1024.0)
            depths_for_memory.append(run['depth'])
    
    if memory_data_mb:
        ax3.plot(depths_for_memory, memory_data_mb, 'o-', linewidth=2, markersize=6, color='tab:red')
        ax3.set_xlabel('Search Depth')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Growth', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(depths_for_memory)
    else:
        ax3.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Memory Growth (N/A)', fontweight='bold')
    
    # 4. System info and metadata
    meta = results['meta']
    build = results['build']
    
    info_text = f"""
System Information:
• Host: {meta['host']}
• Platform: {meta['platform'].split('-')[0]} {meta['platform'].split('-')[1]}
• Python: {meta['python']}
• GNU Time: {'Available' if meta['gnu_time'] else 'Not available'}

Build Information:
• Binary: {os.path.basename(build['binary'])}
• Compile Time: {build.get('compile_time_seconds', 'N/A'):.2f}s
• Time Measurer: {build['time_measurer']}
• Threads Supported: {build['threads_supported']}

Test Results:
• Depth Tests: {len(results['runs_depth'])} runs (depth 1-{max(run['depth'] for run in results['runs_depth'])})
• Thread Tests: {len([r for r in results['runs_threads'] if not r.get('skipped', False)])} runs
• Max Time: {max(extract_metrics_from_run(run)['elapsed_seconds'] for run in results['runs_depth']):.1f}s (depth {max(results['runs_depth'], key=lambda x: extract_metrics_from_run(x)['elapsed_seconds'])['depth']})

Memory Usage:
• Min Memory: {min(extract_metrics_from_run(run)['max_rss_kb'] for run in results['runs_depth'] if extract_metrics_from_run(run)['max_rss_kb'])//1024:.1f}MB (depth {min([r for r in results['runs_depth'] if extract_metrics_from_run(r)['max_rss_kb']], key=lambda x: extract_metrics_from_run(x)['max_rss_kb'])['depth']})
• Max Memory: {max(extract_metrics_from_run(run)['max_rss_kb'] for run in results['runs_depth'] if extract_metrics_from_run(run)['max_rss_kb'])//1024:.1f}MB (depth {max([r for r in results['runs_depth'] if extract_metrics_from_run(r)['max_rss_kb']], key=lambda x: extract_metrics_from_run(x)['max_rss_kb'])['depth']})
    """.strip()
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Benchmark Summary', fontweight='bold')
    
    plt.suptitle('Reversi Engine Benchmark Results Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save as both SVG and PNG
    base_name = 'benchmark_summary'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary plot saved to {OUTPUT_DIR}/{base_name}.(svg|png)")

def main():
    """Main plotting function."""
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: {INPUT_JSON} not found. Run the benchmark first.")
        return 1
    
    print("📊 Loading benchmark results...")
    results = load_results()
    
    print("📁 Creating output directory...")
    ensure_output_dir()
    
    print("📈 Generating plots...")
    
    # Generate individual plots
    plot_depth_results(results)
    plot_threads_results(results)
    plot_summary(results)
    
    print(f"\n🎉 All plots saved to '{OUTPUT_DIR}/' directory")
    print("   Generated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   • {filename}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())