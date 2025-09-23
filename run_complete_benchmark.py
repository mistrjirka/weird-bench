#!/usr/bin/env python3
"""
Complete benchmark runner that executes the benchmark and generates plots.
"""
import subprocess
import sys
import os

def run_benchmark():
    """Run the benchmark script."""
    print("🔄 Running benchmark...")
    result = subprocess.run([sys.executable, "bench.py"], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("❌ Benchmark failed!")
        return False
    print("✅ Benchmark completed successfully!")
    return True

def generate_plots():
    """Generate performance plots."""
    print("🎨 Generating plots...")
    result = subprocess.run([sys.executable, "plot_results.py"], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print("❌ Plot generation failed!")
        return False
    print("✅ Plots generated successfully!")
    return True

def main():
    """Main function."""
    print("🚀 Starting complete benchmark run with plotting...")
    
    if not run_benchmark():
        return 1
    
    if not generate_plots():
        return 1
    
    print("\n🎉 Complete benchmark run finished!")
    print("📊 Results saved to: reversan_results.json")
    print("📈 Plots saved to:   result_plots/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())