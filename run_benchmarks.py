#!/usr/bin/env python3
"""
Main benchmark runner that orchestrates benchmarks.
"""
import argparse
import json
import os
import sys
import time
import requests
import platform
import subprocess
from typing import Dict, Any, List

# Import benchmark implementations
from benchmarks.reversan import ReversanBenchmark
from benchmarks.llama import LlamaBenchmark
from benchmarks.sevenzip import SevenZipBenchmark
from benchmarks.blender import BlenderBenchmark


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(self, output_dir: str = "results", api_url: str = "https://weirdbench.eu/api"):
        self.output_dir = output_dir
        self.api_url = api_url
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

        # Apply GPU selection for llama benchmark if specified
        if benchmark_name == "llama" and hasattr(benchmark, 'set_gpu_selection'):
            gpu_device = getattr(args, 'gpu_device', None)
            vk_driver = getattr(args, 'vk_driver', None)
            if gpu_device is not None or vk_driver:
                benchmark.set_gpu_selection(gpu_device, vk_driver)

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
                error_result = { "error": str(e), "failed": True }
                all_results["benchmarks"][benchmark_name] = error_result

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

        combined_results_file = os.path.join(self.output_dir, "all_benchmarks_results.json")
        with open(combined_results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n🎉 Benchmark suite completed!")
        print(f"📊 Combined results saved to: {combined_results_file}")
        print(f"📁 Individual benchmark files available in {self
