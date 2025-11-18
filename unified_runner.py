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
from benchmarks.console import red, green, grey


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
            print(grey("GNU time utility not found: detailed process metrics will be null for benchmarks that rely on it."))
        else:
            print(grey(f"GNU time detected at: {self.gnu_time}"))

    def _find_gnu_time(self) -> Optional[str]:
        """Find a usable 'time' utility. On many distros (incl. Arch), /usr/bin/time is GNU time by default."""
        import shutil
        candidates = ["/usr/bin/time", shutil.which("time"), shutil.which("gtime")]
        return next((c for c in candidates if c), None)

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
        
        print(grey(f"\nRunning {benchmark_name} benchmark..."))
        
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
                "time_seconds": run["metrics"]["user_seconds"],
                "memory_kb": run["metrics"]["max_rss_kb"]
            })
        
        thread_benchmarks = []
        for run in legacy_result.get("runs_threads", []):
            thread_benchmarks.append({
                "threads": run["threads"], 
                "time_seconds": run["metrics"]["user_seconds"],
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
        
        # Set NUMA options if specified
        if hasattr(args, 'numa_distribute') or hasattr(args, 'numa_isolate') or hasattr(args, 'numactl'):
            benchmark.set_numa_options(
                numa_distribute=getattr(args, 'numa_distribute', False),
                numa_isolate=getattr(args, 'numa_isolate', False),
                numactl_cmd=getattr(args, 'numactl', None)
            )
        
        # Pass args so the benchmark can honor flags like --skip-build and --no-gpu
        build_result = benchmark.build(args)
        
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
                # Prefer device indices from the LlamaBenchmark detection to ensure correct selection
                detected = getattr(benchmark, "available_gpus", []) or []
                # If none detected (shouldn't happen), fall back to system_info order assuming matching
                if detected:
                    # Map device index -> name
                    device_map = {d["index"]: d for d in detected if "index" in d}
                    indices = sorted(device_map.keys())
                    for gpu_index in indices:
                        benchmark.set_gpu_selection(gpu_device_index=gpu_index)
                        gpu_result = benchmark._run_benchmark_command(
                            [vulkan_binary, "-m", benchmark.project_model_path, "-p", "2048", "-n", "64", "-sm", "none", "-mg", "0"],
                            "gpu", 2048, 64, 99, benchmark._get_gpu_env_vars()
                        )
                        if gpu_result.get("returncode") == 0 and not gpu_result.get("failed"):
                            # Tag by hw_id resolved via detector matching on name
                            device_name = device_map[gpu_index]["name"]
                            match_hw = self.hardware_detector.find_matching_device(device_name, "gpu") or f"gpu-{gpu_index}"
                            gpu_benchmarks.append(LlamaRunResult(
                                prompt_speed=gpu_result["metrics"]["prompt_processing"]["avg_tokens_per_sec"],
                                generation_speed=gpu_result["metrics"]["generation"]["avg_tokens_per_sec"],
                                hw_id=match_hw
                            ))
                else:
                    # Fallback to iterating detected hardware GPUs by order
                    for gpu_device in self.system_info.get_gpu_devices():
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
        
        # Derive compile/build time similarly to Reversan: prefer CPU build timing total
        compile_time = 0.0
        cpu_timing = build_result.get("cpu_build_timing")
        if isinstance(cpu_timing, dict):
            compile_time = float(
                cpu_timing.get("total_time_seconds")
                or cpu_timing.get("build_time_seconds")
                or 0.0
            )
        else:
            # Fallback to any top-level timing keys if provided
            compile_time = float(
                build_result.get("total_time_seconds")
                or build_result.get("build_time_seconds")
                or 0.0
            )

        return LlamaBenchmarkResult(
            compile_time=compile_time,
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
        
        # Run the legacy benchmark, ensuring CLI args (e.g., --no-gpu) are respected
        legacy_result = benchmark.run(args)
        
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
                # Resolve a stable hw_id by device name
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
                
                print(green(f"{benchmark_name} benchmark completed successfully"))
                
            except Exception as e:
                print(red(f"{benchmark_name} benchmark failed: {e}"))
                # Continue with other benchmarks
                continue
        
        elapsed_time = time.time() - start_time
        print(grey(f"\nAll benchmarks completed in {elapsed_time:.1f} seconds"))
        
        return unified_result

    def save_results(self, result: UnifiedBenchmarkResult, format: str = "json") -> str:
        """Save unified results to file."""
        timestamp = int(result.meta.timestamp)
        filename = f"unified_benchmark_results_{timestamp}.{format.lower()}"
        filepath = os.path.join(self.output_dir, filename)
        
        result.save_to_file(filepath, format)
        print(f"📁 Results saved to: {filepath}")
        
        return filepath

    def upload_existing_file(self, file_path: str) -> bool:
        """Upload an existing unified results file to the server."""
        try:
            result = UnifiedBenchmarkResult.load_from_file(file_path)
        except Exception as e:
            print(f"❌ Failed to load results file '{file_path}': {e}")
            return False

        # Reuse the same upload path but ensure we pass required fields
        try:
            import requests
            print(f"📁 Results saved to: {file_path}")
            print(f"📁 Saved results to: {file_path}")
            print(f"🌐 Uploading results to {self.api_url}/upload...")

            run_id = f"{result.meta.host}-{int(result.meta.timestamp)}"
            form_data = {
                'run_id': run_id,
                'timestamp': str(int(result.meta.timestamp)),
            }

            with open(file_path, 'rb') as f:
                files = {'file': ('results.json', f, 'application/json')}
                response = requests.post(
                    f"{self.api_url}/upload",
                    data=form_data,
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('success'):
                    print(f"✅ Upload successful: {result_data.get('message', 'No message')}")
                    return True
                else:
                    print(f"❌ Upload failed: {result_data.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Upload failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return False

    def upload_results(self, result: UnifiedBenchmarkResult) -> bool:
        """Upload results to server."""
        try:
            import requests
            
            # Save to temporary file for upload
            temp_file = self.save_results(result, format="json")
            print(f"📁 Saved results to: {temp_file}")
            
            # Upload to server
            print(f"🌐 Uploading results to {self.api_url}/upload...")

            # Required form fields for backend
            run_id = f"{result.meta.host}-{int(result.meta.timestamp)}"
            form_data = {
                'run_id': run_id,
                'timestamp': str(int(result.meta.timestamp)),
            }

            with open(temp_file, 'rb') as f:
                files = {'file': ('results.json', f, 'application/json')}
                response = requests.post(
                    f"{self.api_url}/upload",
                    data=form_data,
                    files=files,
                    timeout=30
                )
                
            if response.status_code == 200:
                result_data = response.json()
                if result_data.get('success'):
                    print(f"✅ Upload successful: {result_data.get('message', 'No message')}")
                    return True
                else:
                    print(f"❌ Upload failed: {result_data.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Upload failed with status {response.status_code}: {response.text}")
                return False
                
        except ImportError:
            print("❌ Upload requires 'requests' library. Install with: pip install requests")
            return False
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return False


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

    # NUMA options for llama benchmark
    parser.add_argument("--numa-distribute", action="store_true",
                        help="Use NUMA distribute mode (numactl --interleave=all) for llama benchmark")
    
    parser.add_argument("--numa-isolate", action="store_true",
                        help="Use NUMA isolate mode for llama benchmark")
    
    parser.add_argument("--numactl",
                        help="Custom numactl command prefix for llama benchmark (e.g., 'numactl --interleave=all')")

    parser.add_argument("--api-url", default="https://weirdbench.eu/api",
                        help="API URL for result uploads")

    parser.add_argument("--upload", action="store_true",
                        help="Upload results to server after running benchmarks")

    parser.add_argument("--upload-existing", nargs="?", const="latest",
                        help="Upload an existing results file from output-dir. If no path is provided, uploads the newest unified results file in the output directory.")

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

    # If user requested to upload an existing results file, do so and exit
    if args.upload_existing is not None:
        # Determine which file to upload
        target_path = args.upload_existing
        if target_path == "latest":
            # Find newest unified results file in output dir (json or yaml)
            try:
                files = [
                    os.path.join(args.output_dir, f)
                    for f in os.listdir(args.output_dir)
                    if (f.startswith("unified_benchmark_results_") and (f.endswith(".json") or f.endswith(".yaml") or f.endswith(".yml")))
                ]
                if not files:
                    print("❌ No unified results files found to upload.")
                    return 1
                target_path = max(files, key=lambda p: os.path.getmtime(p))
            except Exception as e:
                print(f"❌ Failed to locate latest results file: {e}")
                return 1

        if not os.path.exists(target_path):
            print(f"❌ Results file not found: {target_path}")
            return 1

        print(f"🌐 Uploading existing results file: {target_path}")
        ok = runner.upload_existing_file(target_path)
        return 0 if ok else 1

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
        
        # Print detailed results
        print(f"\n{'='*60}")
        print("📈 DETAILED RESULTS")
        print(f"{'='*60}")
        
        # Reversan Results
        if result.reversan:
            print("\n🔄 REVERSAN")
            print(f"  Compile Time: {result.reversan.compile_time:.2f}s")
            
            if result.reversan.depth_benchmarks:
                print("\n  Depth Benchmarks:")
                for db in result.reversan.depth_benchmarks:
                    print(f"    Depth {db.depth}: {db.time_seconds:.2f}s, {db.memory_kb/1024:.1f}MB")
            
            if result.reversan.thread_benchmarks:
                print("\n  Thread Benchmarks:")
                for tb in result.reversan.thread_benchmarks:
                    print(f"    {tb.threads} threads: {tb.time_seconds:.2f}s, {tb.memory_kb/1024:.1f}MB")
        
        # Llama Results
        if result.llama:
            print("\n🦙 LLAMA")
            print(f"  Compile Time: {result.llama.compile_time:.2f}s")
            
            if result.llama.cpu_benchmark:
                cpu_hw = result.meta.hardware.get(result.llama.cpu_benchmark.hw_id)
                cpu_name = cpu_hw.name if cpu_hw else result.llama.cpu_benchmark.hw_id
                print(f"\n  CPU ({cpu_name}):")
                print(f"    Prompt: {result.llama.cpu_benchmark.prompt_speed:.1f} tokens/s")
                print(f"    Generation: {result.llama.cpu_benchmark.generation_speed:.1f} tokens/s")
            
            if result.llama.gpu_benchmarks:
                for gpu_bench in result.llama.gpu_benchmarks:
                    gpu_hw = result.meta.hardware.get(gpu_bench.hw_id)
                    gpu_name = gpu_hw.name if gpu_hw else gpu_bench.hw_id
                    print(f"\n  GPU ({gpu_name}):")
                    print(f"    Prompt: {gpu_bench.prompt_speed:.1f} tokens/s")
                    print(f"    Generation: {gpu_bench.generation_speed:.1f} tokens/s")
        
        # 7-Zip Results
        if result.sevenzip:
            print("\n🗜️  7-ZIP")
            print(f"  CPU Usage: {result.sevenzip.usage_percent:.0f}%")
            print(f"  R/U MIPS: {result.sevenzip.ru_mips:.0f}")
            print(f"  Total MIPS: {result.sevenzip.total_mips:.0f}")
        
        # Blender Results
        if result.blender:
            print("\n🎨 BLENDER")
            
            if result.blender.cpu:
                print("\n  CPU:")
                if result.blender.cpu.monster is not None:
                    print(f"    Monster: {result.blender.cpu.monster:.1f} samples/min")
                if result.blender.cpu.junkshop is not None:
                    print(f"    Junkshop: {result.blender.cpu.junkshop:.1f} samples/min")
                if result.blender.cpu.classroom is not None:
                    print(f"    Classroom: {result.blender.cpu.classroom:.1f} samples/min")
            
            if result.blender.gpus:
                for gpu_result in result.blender.gpus:
                    gpu_hw = result.meta.hardware.get(gpu_result.hw_id)
                    gpu_name = gpu_hw.name if gpu_hw else gpu_result.hw_id
                    print(f"\n  GPU ({gpu_name}):")
                    if gpu_result.scenes.monster is not None:
                        print(f"    Monster: {gpu_result.scenes.monster:.1f} samples/min")
                    if gpu_result.scenes.junkshop is not None:
                        print(f"    Junkshop: {gpu_result.scenes.junkshop:.1f} samples/min")
                    if gpu_result.scenes.classroom is not None:
                        print(f"    Classroom: {gpu_result.scenes.classroom:.1f} samples/min")
        
        print(f"\n{'='*60}")
        
        # Upload results if requested
        if args.upload:
            print("\n🌐 Uploading results to server...")
            upload_success = runner.upload_results(result)
            if not upload_success:
                print("⚠️  Benchmark completed but upload failed")
        
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())