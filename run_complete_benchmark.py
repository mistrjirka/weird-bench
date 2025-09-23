#!/usr/bin/env python3
"""
Complete benchmark runner that executes all benchmarks and generates plots.
"""
import subprocess
import sys
import os
import argparse

def run_benchmarks(benchmark_type="all", runs=1):
    """Run the benchmark scripts."""
    print("🔄 Running benchmarks...")
    cmd = [sys.executable, "run_benchmarks.py", "--benchmark", benchmark_type, "--runs", str(runs)]
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("❌ Benchmarks failed!")
        return False
    print("✅ Benchmarks completed successfully!")
    return True

def generate_plots():
    """Generate performance plots."""
    print("🎨 Generating plots...")
    result = subprocess.run([sys.executable, "plot_all_results.py"], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("❌ Plot generation failed!")
        return False
    print("✅ Plots generated successfully!")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Complete benchmark suite runner with plotting. "
                   "Runs benchmarks with efficient model caching and generates comprehensive plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        choices=["reversan", "llama", "all"],
        default="all",
        help="Which benchmark to run"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of times to run each test (applies to Reversan benchmark)"
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting complete benchmark suite with plotting...")
    
    if not run_benchmarks(args.benchmark, args.runs):
        return 1
    
    if not args.skip_plots:
        if not generate_plots():
            return 1
    else:
        print("📈 Skipping plot generation as requested")
    
    print("\n🎉 Complete benchmark suite finished!")
    print("📊 Results saved to: results/")
    if not args.skip_plots:
        print("📈 Plots saved to:   result_plots/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())