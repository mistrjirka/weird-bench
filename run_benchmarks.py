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
            # Skip GPU-related benchmarks if --no-gpu is specified
            if getattr(args, 'no_gpu', False):
                if benchmark_name == "llama":
                    print(f"⏭️  Skipping Vulkan build in {benchmark_name} benchmark (--no-gpu)")
                elif benchmark_name == "blender":
                    print(f"🖥️  Running {benchmark_name} benchmark in CPU-only mode (--no-gpu)")
            
            try:
                print(f"\n{'='*60}")
                print(f"Starting {benchmark_name} benchmark...")
                print(f"{'='*60}")
                
                results = self.run_benchmark(benchmark_name, args)
                all_results["benchmarks"][benchmark_name] = results
                all_results["meta"]["benchmarks_run"].append(benchmark_name)
                
                # Save individual benchmark results
                individual_results_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
                with open(individual_results_file, "w") as f:
                    json.dump({
                        "benchmark_name": benchmark_name,
                        "timestamp": time.time(),
                        "results": results
                    }, f, indent=2)
                print(f"💾 Individual results saved to: {individual_results_file}")
                
                print(f"✅ {benchmark_name} benchmark completed successfully!")
                
            except Exception as e:
                print(f"❌ {benchmark_name} benchmark failed: {e}")
                error_result = {
                    "error": str(e),
                    "failed": True
                }
                all_results["benchmarks"][benchmark_name] = error_result
                
                # Save individual error results too
                individual_results_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
                with open(individual_results_file, "w") as f:
                    json.dump({
                        "benchmark_name": benchmark_name,
                        "timestamp": time.time(),
                        "results": error_result
                    }, f, indent=2)
        
        end_time = time.time()
        all_results["meta"]["suite_end_time"] = end_time
        all_results["meta"]["total_duration_seconds"] = end_time - start_time
        
        # Save combined results
        combined_results_file = os.path.join(self.output_dir, "all_benchmarks_results.json")
        with open(combined_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 Benchmark suite completed!")
        print(f"📊 Combined results saved to: {combined_results_file}")
        print(f"📁 Individual benchmark files available in {self.output_dir}/")
        print(f"⏱️  Total duration: {end_time - start_time:.2f} seconds")
        
        return all_results
    
    def combine_results_from_files(self) -> Dict[str, Any]:
        """Combine individual benchmark result files into a single JSON."""
        print("🔄 Combining individual benchmark results...")
        
        combined_results = {
            "meta": {
                "suite_start_time": time.time(),
                "benchmarks_run": [],
                "combined_from_individual_files": True
            },
            "benchmarks": {}
        }
        
        benchmark_files = []
        for benchmark_name in self.benchmarks.keys():
            individual_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
            if os.path.exists(individual_file):
                benchmark_files.append((benchmark_name, individual_file))
        
        if not benchmark_files:
            print("❌ No individual benchmark result files found")
            return combined_results
        
        for benchmark_name, file_path in benchmark_files:
            try:
                with open(file_path, "r") as f:
                    benchmark_data = json.load(f)
                    combined_results["benchmarks"][benchmark_name] = benchmark_data.get("results", {})
                    combined_results["meta"]["benchmarks_run"].append(benchmark_name)
                print(f"✅ Loaded {benchmark_name} results from {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {benchmark_name} from {file_path}: {e}")
        
        combined_results["meta"]["suite_end_time"] = time.time()
        combined_results["meta"]["total_benchmarks_found"] = len(benchmark_files)
        
        # Save combined results
        combined_results_file = os.path.join(self.output_dir, "all_benchmarks_results.json")
        with open(combined_results_file, "w") as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"💾 Combined results saved to: {combined_results_file}")
        print(f"📊 Combined {len(benchmark_files)} benchmark result files")
        
        return combined_results
    
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
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU benchmarks: no Vulkan build in Llama benchmark and CPU-only mode for Blender"
    )
    
    parser.add_argument(
        "--combine-results",
        action="store_true",
        help="Combine individual benchmark result files into a single all_benchmarks_results.json"
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
        if args.combine_results:
            runner.combine_results_from_files()
        elif args.benchmark == "all":
            runner.run_all_benchmarks(args)
        else:
            result = runner.run_benchmark(args.benchmark, args)
            # Save individual result file even for single benchmark runs
            individual_results_file = os.path.join(args.output_dir, f"{args.benchmark}_results.json")
            with open(individual_results_file, "w") as f:
                json.dump({
                    "benchmark_name": args.benchmark,
                    "timestamp": time.time(),
                    "results": result
                }, f, indent=2)
            print(f"💾 Results saved to: {individual_results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())