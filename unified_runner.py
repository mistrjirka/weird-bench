#!/usr/bin/env python3
"""
Unified benchmark runner with hardware-aware benchmarking and single output file.
"""

import argparse
import sys
import time
import os
import json
import subprocess
from typing import Dict, Any, List, Optional

# Import unified models and hardware detection
from unified_models import (
    UnifiedBenchmarkResult, SystemInfo, HardwareDevice,
    LlamaBenchmarkResult, LlamaRunResult,
    ReversanBenchmarkResult, ReversanDepthResult, ReversanThreadResult,
    SevenZipBenchmarkResult, 
    BlenderBenchmarkResult, BlenderSceneResult, BlenderDeviceResult
)
from hardware_detector import GlobalHardwareDetector

# Import benchmark implementations
from benchmarks.reversan import ReversanBenchmark
from benchmarks.llama import LlamaBenchmark  
from benchmarks.sevenzip import SevenZipBenchmark
from benchmarks.blender import BlenderBenchmark


class UnifiedBenchmarkRunner:
    """Unified benchmark runner with hardware detection and single output file."""

    def __init__(self, output_dir: str = "results", api_url: str = "https://weirdbench.eu/api"):
        self.output_dir = output_dir
        self.api_url = api_url
        self.hardware_detector = GlobalHardwareDetector()
        self.system_info = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for GNU time (required for detailed benchmarking)
        self.gnu_time = self._find_gnu_time()
        if not self.gnu_time:
            print("⚠️  WARNING: GNU time not found. Some benchmarks may not provide detailed timing information.")
            print("   Ubuntu/Debian: sudo apt install time")
            print("   Fedora/RHEL: already included")
            print("   Arch: already included") 

    def _find_gnu_time(self) -> Optional[str]:
        """Find GNU time command."""
        import shutil
        for cmd in ["/usr/bin/time", shutil.which("time"), shutil.which("gtime")]:
            if cmd and os.path.exists(cmd):
                try:
                    result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and "GNU time" in result.stderr:
                        return cmd
                except:
                    continue
        return None

    def detect_hardware(self, cpu_only: bool = False) -> SystemInfo:
        """Detect system hardware and create SystemInfo."""
        print("🔍 Detecting system hardware...")
        
        hardware = self.hardware_detector.detect_all_hardware()
        
        self.system_info = SystemInfo(
            platform=f"{os.uname().sysname}-{os.uname().release}-{os.uname().machine}",
            host=os.uname().nodename,
            timestamp=time.time(),
            cpu_only=cpu_only,
            hardware=hardware
        )
        
        return self.system_info

    def run_benchmark(self, benchmark_name: str, args: Any = None) -> Any:
        """Run a specific benchmark using the new unified approach."""
        if not self.system_info:
            raise RuntimeError("Hardware detection must be run first. Call detect_hardware().")
        
        print(f"\n🚀 Running {benchmark_name} benchmark...")
        
        if benchmark_name == "reversan":
            return self._run_reversan_benchmark(args)
        elif benchmark_name == "llama":
            return self._run_llama_benchmark(args)
        elif benchmark_name == "sevenzip":
            return self._run_7zip_benchmark(args)
        elif benchmark_name == "blender":
            return self._run_blender_benchmark(args)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def _run_reversan_benchmark(self, args: Any = None) -> ReversanBenchmarkResult:
        """Run Reversan benchmark with unified output."""
        benchmark = ReversanBenchmark(self.output_dir)
        benchmark.setup()
        benchmark.build()
        
        # Run the legacy benchmark to get the data
        legacy_result = benchmark.run()
        
        # Convert to unified format
        depth_benchmarks = []
        for run in legacy_result.get("runs_depth", []):
            depth_benchmarks.append({
                "depth": run["depth"],
                "time_seconds": run["metrics"]["elapsed_seconds"],
                "memory_kb": run["metrics"]["max_rss_kb"]
            })
        
        thread_benchmarks = []
        for run in legacy_result.get("runs_threads", []):
            thread_benchmarks.append({
                "threads": run["threads"], 
                "time_seconds": run["metrics"]["elapsed_seconds"],
                "memory_kb": run["metrics"]["max_rss_kb"]
            })
        
        return ReversanBenchmarkResult(
            compile_time=legacy_result.get("build", {}).get("compile_time_seconds", 0.0),
            depth_benchmarks=[ReversanDepthResult(**d) for d in depth_benchmarks],
            thread_benchmarks=[ReversanThreadResult(**t) for t in thread_benchmarks]
        )

    def _run_llama_benchmark(self, args: Any = None) -> LlamaBenchmarkResult:
        """Run Llama benchmark with unified output."""
        benchmark = LlamaBenchmark(self.output_dir)
        benchmark.setup(skip_build=getattr(args, 'skip_build', False))
        
        # Set GPU selection if specified
        if hasattr(args, 'gpu_device') and args.gpu_device is not None:
            benchmark.set_gpu_selection(gpu_device_index=args.gpu_device)
        if hasattr(args, 'vk_driver') and args.vk_driver:
            benchmark.set_gpu_selection(vk_driver_files=args.vk_driver)
        
        build_result = benchmark.build()
        
        # Run benchmarks
        cpu_benchmark = None
        gpu_benchmarks = []
        
        # Always run CPU benchmark if CPU build was successful
        if "cpu_bench_binary" in build_result:
            cpu_binary = build_result["cpu_bench_binary"]
            cpu_result = benchmark._run_benchmark_command(
                [cpu_binary, "-m", benchmark.project_model_path, "-p", "512", "-n", "64"],
                "cpu", 512, 64, 0
            )
            
            if cpu_result.get("returncode") == 0 and not cpu_result.get("failed"):
                cpu_hw_id = self.hardware_detector.find_matching_device(
                    cpu_result["metrics"]["system_info"]["cpu_info"], "cpu"
                ) or "cpu-0"
                
                cpu_benchmark = LlamaRunResult(
                    prompt_speed=cpu_result["metrics"]["prompt_processing"]["avg_tokens_per_sec"],
                    generation_speed=cpu_result["metrics"]["generation"]["avg_tokens_per_sec"],
                    hw_id=cpu_hw_id
                )
        
        # Run GPU benchmarks if not CPU-only and Vulkan build was successful
        if not self.system_info.cpu_only and not getattr(args, 'no_gpu', False):
            if "vulkan_bench_binary" in build_result:
                vulkan_binary = build_result["vulkan_bench_binary"]
                for gpu_device in self.system_info.get_gpu_devices():
                    # Find the GPU device index for this hardware ID
                    gpu_index = int(gpu_device.hw_id.split('-')[1]) if '-' in gpu_device.hw_id else 0
                    benchmark.set_gpu_selection(gpu_device_index=gpu_index)
                    
                    gpu_result = benchmark._run_benchmark_command(
                        [vulkan_binary, "-m", benchmark.project_model_path, "-p", "2048", "-n", "64", "-sm", "none", "-mg", str(gpu_index)],
                        "gpu", 2048, 64, 99, benchmark._get_gpu_env_vars()
                    )
                    
                    if gpu_result.get("returncode") == 0 and not gpu_result.get("failed"):
                        gpu_benchmarks.append(LlamaRunResult(
                            prompt_speed=gpu_result["metrics"]["prompt_processing"]["avg_tokens_per_sec"],
                            generation_speed=gpu_result["metrics"]["generation"]["avg_tokens_per_sec"],
                            hw_id=gpu_device.hw_id
                        ))
        
        return LlamaBenchmarkResult(
            compile_time=build_result.get("total_time_seconds", 0.0),
            cpu_benchmark=cpu_benchmark,
            gpu_benchmarks=gpu_benchmarks
        )

    def _run_7zip_benchmark(self, args: Any = None) -> SevenZipBenchmarkResult:
        """Run 7zip benchmark with unified output."""
        benchmark = SevenZipBenchmark(self.output_dir, self.hardware_detector)
        benchmark.setup()
        return benchmark.benchmark()

    def _run_blender_benchmark(self, args: Any = None) -> BlenderBenchmarkResult:
        """Run Blender benchmark with unified output."""
        benchmark = BlenderBenchmark(self.output_dir)
        benchmark.setup()
        
        # Run the legacy benchmark
        legacy_result = benchmark.run()
        
        cpu_result = None
        gpu_results = []
        
        # Process device runs
        for device_run in legacy_result.get("device_runs", []):
            if device_run["device_framework"] == "CPU":
                # CPU result
                scenes = BlenderSceneResult()
                for scene_name, scene_data in device_run.get("scene_results", {}).items():
                    setattr(scenes, scene_name, scene_data.get("samples_per_minute"))
                cpu_result = scenes
                
            else:
                # GPU result
                gpu_hw_id = self.hardware_detector.find_matching_device(
                    device_run["device_name"], "gpu"
                ) or f"gpu-{len(gpu_results)}"
                
                scenes = BlenderSceneResult()
                for scene_name, scene_data in device_run.get("scene_results", {}).items():
                    setattr(scenes, scene_name, scene_data.get("samples_per_minute"))
                
                gpu_results.append(BlenderDeviceResult(
                    hw_id=gpu_hw_id,
                    scenes=scenes
                ))
        
        return BlenderBenchmarkResult(
            cpu=cpu_result,
            gpus=gpu_results
        )

    def run_all_benchmarks(self, args: Any = None) -> UnifiedBenchmarkResult:
        """Run all benchmarks and create unified result."""
        if not self.system_info:
            self.detect_hardware(cpu_only=getattr(args, 'no_gpu', False))
        
        print("🚀 Starting unified benchmark suite...")
        start_time = time.time()
        
        unified_result = UnifiedBenchmarkResult(meta=self.system_info)
        
        # List of benchmarks to run
        benchmarks_to_run = ["reversan", "llama", "sevenzip", "blender"]
        if hasattr(args, 'benchmark') and args.benchmark != "all":
            benchmark_map = {"7zip": "sevenzip"}  # Map 7zip to sevenzip
            mapped_benchmark = benchmark_map.get(args.benchmark, args.benchmark)
            benchmarks_to_run = [mapped_benchmark]
        
        for benchmark_name in benchmarks_to_run:
            try:
                print(f"\n{'='*50}")
                print(f"Running {benchmark_name.upper()} benchmark")
                print(f"{'='*50}")
                
                result = self.run_benchmark(benchmark_name, args)
                setattr(unified_result, benchmark_name, result)
                
                print(f"✅ {benchmark_name} benchmark completed successfully")
                
            except Exception as e:
                print(f"❌ {benchmark_name} benchmark failed: {e}")
                # Continue with other benchmarks
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All benchmarks completed in {elapsed_time:.1f} seconds")
        
        return unified_result

    def save_results(self, result: UnifiedBenchmarkResult, format: str = "json") -> str:
        """Save unified results to file."""
        timestamp = int(result.meta.timestamp)
        filename = f"unified_benchmark_results_{timestamp}.{format.lower()}"
        filepath = os.path.join(self.output_dir, filename)
        
        result.save_to_file(filepath, format)
        print(f"📁 Results saved to: {filepath}")
        
        return filepath

    def upload_results(self, result: UnifiedBenchmarkResult) -> bool:
        """Upload results to server (placeholder for future implementation)."""
        print("🌐 Upload functionality will be implemented after server-side changes")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for performance testing with single output file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--benchmark", "-b",
                        choices=["reversan", "llama", "7zip", "blender", "all"],
                        default="all",
                        help="Benchmark to run")

    parser.add_argument("--output-dir", "-o", default="results",
                        help="Output directory for results")

    parser.add_argument("--format", "-f", choices=["json", "yaml"], default="json",
                        help="Output format")

    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU benchmarks: CPU-only mode")

    parser.add_argument("--skip-build", action="store_true",
                        help="Skip building and reuse existing binaries (faster for debugging)")

    # Vulkan selection (for compatibility)
    parser.add_argument("--gpu-device", type=int,
                        help="Select specific GPU device for Vulkan benchmarking (0, 1, 2...)")
    
    parser.add_argument("--vk-driver",
                        help="Force specific Vulkan ICD driver file")
    
    parser.add_argument("--list-gpus", action="store_true",
                        help="List available Vulkan GPU devices and exit")

    parser.add_argument("--api-url", default="https://weirdbench.eu/api",
                        help="API URL for result uploads")

    args = parser.parse_args()

    runner = UnifiedBenchmarkRunner(args.output_dir, args.api_url)

    # Handle GPU listing
    if args.list_gpus:
        print("🔍 Detecting available GPUs...")
        runner.detect_hardware()
        gpu_devices = runner.system_info.get_gpu_devices()
        
        if gpu_devices:
            print(f"\nFound {len(gpu_devices)} GPU device(s):")
            for gpu in gpu_devices:
                print(f"  {gpu.hw_id}: {gpu.name} ({gpu.manufacturer}, {gpu.framework or 'Unknown framework'})")
        else:
            print("\nNo GPU devices found.")
        
        return 0

    try:
        # Run benchmarks
        result = runner.run_all_benchmarks(args)
        
        # Save results
        filepath = runner.save_results(result, args.format)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"System: {result.meta.host} ({result.meta.platform})")
        print(f"Hardware: {len(result.meta.hardware)} devices detected")
        
        for hw_id, device in result.meta.hardware.items():
            print(f"  {hw_id}: {device.name}")
        
        benchmarks_run = []
        if result.reversan:
            benchmarks_run.append("Reversan")
        if result.llama:
            benchmarks_run.append("Llama")
        if result.sevenzip:
            benchmarks_run.append("7-Zip")
        if result.blender:
            benchmarks_run.append("Blender")
        
        print(f"Benchmarks: {', '.join(benchmarks_run)}")
        print(f"Output: {filepath}")
        
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())