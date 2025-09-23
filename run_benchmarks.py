#!/usr/bin/env python3
"""
Main benchmark runner that orchestrates all benchmarks.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

# Import benchmark implementations
from benchmarks.reversan import ReversanBenchmark
from benchmarks.llama import LlamaBenchmark
from benchmarks.sevenzip import SevenZipBenchmark
from benchmarks.blender import BlenderBenchmark


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.benchmarks = {
            "reversan": ReversanBenchmark,
            "llama": LlamaBenchmark,
            "7zip": SevenZipBenchmark,
            "blender": BlenderBenchmark   
        }
        os.makedirs(output_dir, exist_ok=True)
    
    def run_benchmark(self, benchmark_name: str, args: Any = None) -> Dict[str, Any]:
        """Run a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(self.benchmarks.keys())}")
        
        benchmark_class = self.benchmarks[benchmark_name]
        benchmark = benchmark_class(self.output_dir)
        
        return benchmark.run(args)
    
    def run_all_benchmarks(self, args: Any = None) -> Dict[str, Any]:
        """Run all available benchmarks."""
        print("🚀 Starting complete benchmark suite...")
        start_time = time.time()
        
        all_results = {
            "meta": {
                "suite_start_time": start_time,
                "benchmarks_run": [],
            },
            "benchmarks": {}
        }
        
        for benchmark_name in self.benchmarks.keys():
            try:
                print(f"\n{'='*60}")
                print(f"Starting {benchmark_name} benchmark...")
                print(f"{'='*60}")
                
                results = self.run_benchmark(benchmark_name, args)
                all_results["benchmarks"][benchmark_name] = results
                all_results["meta"]["benchmarks_run"].append(benchmark_name)
                
                print(f"✅ {benchmark_name} benchmark completed successfully!")
                
            except Exception as e:
                print(f"❌ {benchmark_name} benchmark failed: {e}")
                all_results["benchmarks"][benchmark_name] = {
                    "error": str(e),
                    "failed": True
                }
        
        end_time = time.time()
        all_results["meta"]["suite_end_time"] = end_time
        all_results["meta"]["total_duration_seconds"] = end_time - start_time
        
        # Save combined results
        combined_results_file = os.path.join(self.output_dir, "all_benchmarks_results.json")
        with open(combined_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 Benchmark suite completed!")
        print(f"📊 Combined results saved to: {combined_results_file}")
        print(f"⏱️  Total duration: {end_time - start_time:.2f} seconds")
        
        return all_results
    
    def list_benchmarks(self) -> List[str]:
        """List available benchmarks."""
        return list(self.benchmarks.keys())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-benchmark runner for performance testing. "
                   "Supports Reversan Engine (depth/thread scaling) and Llama.cpp (CPU/GPU inference with build timing). "
                   "Models are cached in shared directory to avoid re-downloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        choices=["reversan", "llama", "7zip", "blender", "all"],
        default="all",
        help="Which benchmark to run: 'reversan' (game engine depth/thread scaling), "
             "'llama' (LLM inference with build timing), '7zip' (compression performance), "
             "'blender' (3D rendering with device comparison), or 'all' (all benchmarks)"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of times to run each test configuration (applies to Reversan benchmark only). "
             "Llama.cpp uses clean builds for accurate timing."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory for benchmark results. Models are cached separately in 'models/' directory."
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks with details and exit"
    )
    
    args = parser.parse_args()
    
    if args.runs < 1:
        print("Error: Number of runs must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    runner = BenchmarkRunner(args.output_dir)
    
    if args.list:
        print("Available benchmarks:")
        print()
        print("  reversan - Reversan Engine benchmark")
        print("    • Tests game engine performance at different search depths (1-12)")
        print("    • Tests thread scaling performance (1 to max CPU threads)")
        print("    • Measures execution time and memory usage")
        print("    • Supports multiple test runs for averaging")
        print()
        print("  llama - Llama.cpp benchmark")
        print("    • Downloads Qwen3-4B model once to shared 'models/' directory")
        print("    • Tests both regular and Vulkan builds with detailed timing")
        print("    • CPU inference tests (ngl=0) with various prompt/generation sizes")
        print("    • GPU inference tests (ngl=99) if Vulkan is available")
        print("    • Each build gets clean environment, model copied from cache")
        print()
        print("  7zip - 7-Zip compression benchmark")
        print("    • Tests compression performance with multi-threading")
        print("    • Creates test data (text and binary files) for realistic compression")
        print("    • Measures compression time, ratio, and thread scaling efficiency")
        print("    • Tests thread counts from 1 to maximum CPU threads")
        print("    • Uses system-installed 7-Zip (p7zip-full package)")
        print()
        print("  blender - Blender 3D rendering benchmark")
        print("    • Downloads official Blender benchmark suite automatically")
        print("    • Tests all available devices (CPU, GPU with different frameworks)")
        print("    • Measures samples per minute for: monster, junkshop, classroom scenes")
        print("    • Automatic device detection and comparative analysis")
        print("    • Uses Blender 4.5.0 with JSON output for reliable parsing")
        print()
        print("  all - Run all benchmarks (default)")
        print("    • Efficient execution with shared model caching")
        print("    • Combined results and plotting")
        return 0
    
    try:
        if args.benchmark == "all":
            runner.run_all_benchmarks(args)
        else:
            runner.run_benchmark(args.benchmark, args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())