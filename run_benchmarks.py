#!/usr/bin/env python3
"""
Main benchmark runner that orchestrates            try:
                print(f"\n{'='*60}")
                print(f"Starting {benchmark_name} benchmark...")
                print(f"{'='*60}")
                
                # Special handling for llama benchmark GPU selection
                if benchmark_name == "llama" and (args.gpu_device is not None or args.vk_driver):
                    benchmark = benchmark_class(output_dir)
                    benchmark.set_gpu_selection(args.gpu_device, args.vk_driver)
                    results = benchmark.run(args)
                else:
                    results = runner.run_benchmark(benchmark_name, args)enchmarks.
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
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Extract hardware information from the system."""
        try:
            # Get CPU info
            cpu_info = platform.processor()
            
            # Try to get more detailed CPU info on Linux
            if platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        cpu_lines = f.readlines()
                        for line in cpu_lines:
                            if line.startswith("model name"):
                                cpu_info = line.split(":")[1].strip()
                                break
                except:
                    pass
            
            # Get GPU info (basic detection)
            gpu_info = "Unknown"
            try:
                # Try nvidia-smi first
                result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
                else:
                    # Try lspci for AMD/Intel GPUs
                    result = subprocess.run(["lspci", "-k"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'VGA' in line or 'Display' in line:
                                if 'Radeon' in line or 'AMD' in line:
                                    # Extract GPU name
                                    parts = line.split('[')
                                    if len(parts) > 1:
                                        gpu_info = parts[-1].split(']')[0]
                                    break
            except:
                pass
            
            return {
                "cpu": cpu_info,
                "gpu": gpu_info,
                "platform": platform.platform(),
                "hostname": platform.node()
            }
        except Exception as e:
            print(f"⚠️  Could not extract hardware info: {e}")
            return {
                "cpu": "Unknown",
                "gpu": "Unknown", 
                "platform": platform.platform(),
                "hostname": platform.node()
            }
    
    def generate_run_id(self, hardware_info: Dict[str, Any]) -> str:
        """Generate a unique run ID based on timestamp and hardware."""
        timestamp = int(time.time())
        
        # Normalize CPU name
        cpu_name = hardware_info.get("cpu", "unknown-cpu").lower()
        cpu_name = cpu_name.replace(" ", "-").replace("(r)", "").replace("(tm)", "")
        cpu_name = "".join(c for c in cpu_name if c.isalnum() or c == "-")
        
        # Normalize GPU name
        gpu_name = hardware_info.get("gpu", "unknown-gpu").lower()
        if gpu_name != "unknown":
            gpu_name = gpu_name.replace(" ", "-").replace("(r)", "").replace("(tm)", "")
            gpu_name = "".join(c for c in gpu_name if c.isalnum() or c == "-")
            return f"{timestamp}_{cpu_name}_{gpu_name}"
        else:
            return f"{timestamp}_{cpu_name}"
    
    def upload_results(self, upload_all: bool = False) -> bool:
        """Upload benchmark results to the API."""
        print("📤 Uploading benchmark results...")
        
        try:
            # Get hardware info and generate run ID
            hardware_info = self.get_hardware_info()
            run_id = self.generate_run_id(hardware_info)
            
            print(f"🔧 Hardware detected: {hardware_info['cpu']}")
            if hardware_info['gpu'] != "Unknown":
                print(f"🎮 GPU detected: {hardware_info['gpu']}")
            print(f"🆔 Run ID: {run_id}")
            
            # Prepare files to upload
            files_to_upload = {}
            
            # Find benchmark result files
            for benchmark_name in self.benchmarks.keys():
                result_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
                if os.path.exists(result_file):
                    print(f"📁 Found {benchmark_name} results: {result_file}")
                    with open(result_file, 'rb') as f:
                        files_to_upload[f"{benchmark_name}_results"] = f.read()
            
            if not files_to_upload:
                print("❌ No benchmark result files found to upload")
                return False
            
            # Create multipart form data
            files = {}
            for key, content in files_to_upload.items():
                # Remove the "_results" suffix for file parameter names to match API expectation
                file_key = key.replace("_results", "")
                files[file_key] = (f"{key}.json", content, "application/json")
            
            # Add metadata - include action in form data, not URL parameter
            data = {
                "action": "upload",
                "run_id": run_id,
                "hardware_info": json.dumps(hardware_info),
                "timestamp": int(time.time())
            }
            
            # Upload to API - use /api/upload endpoint
            response = requests.post(f"{self.api_url}/upload", files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("✅ Upload successful!")
                    if "message" in result:
                        print(f"📝 Server response: {result['message']}")
                    
                    # Show detailed results if available
                    if "data" in result:
                        data = result["data"]
                        if "total_benchmarks_stored" in data:
                            print(f"📊 Benchmarks stored: {data['total_benchmarks_stored']}")
                        if "results" in data:
                            for hw_type, info in data["results"].items():
                                print(f"� {hw_type.upper()}: {info}")
                    
                    return True
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse server response as JSON: {e}")
                    print(f"📝 Raw response: {response.text[:500]}")
                    return False
            else:
                print(f"❌ Upload failed with status {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"📝 Error details: {error_info}")
                except:
                    print(f"📝 Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error during upload: {e}")
            return False
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def upload_existing_results(self) -> bool:
        """Upload existing results from the results folder."""
        print("📤 Uploading existing benchmark results...")
        
        # Check if results directory exists
        if not os.path.exists(self.output_dir):
            print(f"❌ Results directory '{self.output_dir}' does not exist")
            return False
        
        # List available result files
        available_files = []
        for benchmark_name in self.benchmarks.keys():
            result_file = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
            if os.path.exists(result_file):
                available_files.append((benchmark_name, result_file))
        
        if not available_files:
            print(f"❌ No benchmark result files found in '{self.output_dir}'")
            return False
        
        print(f"📁 Found {len(available_files)} result files:")
        for benchmark_name, file_path in available_files:
            print(f"   • {benchmark_name}: {file_path}")
        
        return self.upload_results(upload_all=True)


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
    
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to the backend API after benchmarking is complete"
    )
    
    parser.add_argument(
        "--upload-existing",
        action="store_true",
        help="Upload existing result files from the results folder without running new benchmarks"
    )
    
    parser.add_argument(
        "--api-url",
        default="https://weirdbench.eu/api",
        help="API endpoint for uploading results"
    )
    
    parser.add_argument(
        "--gpu-device",
        type=int,
        help="Select specific GPU device for Vulkan benchmarking (0, 1, 2...). Use --list-gpus to see available devices."
    )
    
    parser.add_argument(
        "--vk-driver",
        help="Force specific Vulkan ICD driver file (e.g., /usr/share/vulkan/icd.d/nvidia_icd.json)"
    )
    
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="List available Vulkan GPU devices and exit"
    )
    
    args = parser.parse_args()
    
    if args.runs < 1:
        print("Error: Number of runs must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    runner = BenchmarkRunner(args.output_dir, args.api_url)
    
    # Handle GPU listing
    if args.list_gpus:
        print("🎮 Detecting available Vulkan GPU devices...")
        try:
            # Create a temporary llama benchmark instance to detect GPUs
            from benchmarks.llama import LlamaBenchmark
            temp_benchmark = LlamaBenchmark(args.output_dir)
            temp_benchmark.setup()
            
            if temp_benchmark.available_gpus:
                print(f"\n✅ Found {len(temp_benchmark.available_gpus)} Vulkan GPU device(s):")
                print()
                for gpu in temp_benchmark.available_gpus:
                    print(f"  Device {gpu['index']}: {gpu['name']}")
                    print(f"    Driver: {gpu['driver']}")
                    if gpu['icd_path']:
                        print(f"    ICD File: {gpu['icd_path']}")
                    print()
                print("💡 Use --gpu-device N to select a specific GPU device")
                print("💡 Use --vk-driver PATH to force a specific Vulkan driver")
            else:
                print("\n❌ No Vulkan GPU devices detected")
                print("💡 Make sure Vulkan drivers are installed and vulkaninfo is available")
        except Exception as e:
            print(f"❌ Failed to detect GPUs: {e}")
        return 0
    
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
        print("    • Supports GPU device selection with --gpu-device and --vk-driver")
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
        print()
        print("GPU Selection Options (for llama benchmark):")
        print("  --gpu-device N        Select GPU device by index (0, 1, 2...)")
        print("  --vk-driver PATH      Force specific Vulkan driver ICD file")
        print("  --list-gpus           List available Vulkan GPU devices")
        print()
        print("Upload options:")
        print("  --upload                Upload results after benchmarking")
        print("  --upload-existing       Upload existing results from results/ folder")
        print("  --api-url URL           API endpoint (default: https://weirdbench.eu/api)")
        return 0
    
    try:
        if args.upload_existing:
            # Upload existing results without running benchmarks
            success = runner.upload_existing_results()
            return 0 if success else 1
        elif args.combine_results:
            runner.combine_results_from_files()
        elif args.benchmark == "all":
            # Special handling for llama benchmark in "all" mode
            if args.gpu_device is not None or args.vk_driver:
                print("🎯 GPU selection specified - configuring llama benchmark...")
                # We'll handle GPU selection in the individual benchmark loop
            runner.run_all_benchmarks(args)
        else:
            # Single benchmark
            if args.benchmark == "llama" and (args.gpu_device is not None or args.vk_driver):
                benchmark = runner.benchmarks[args.benchmark](args.output_dir)
                benchmark.set_gpu_selection(args.gpu_device, args.vk_driver)
                result = benchmark.run(args)
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
        
        # Upload results if requested
        if args.upload and not args.upload_existing:
            print("\n" + "="*50)
            success = runner.upload_results()
            if not success:
                print("⚠️  Upload failed, but benchmark results are still available locally")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())