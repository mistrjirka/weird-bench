#!/usr/bin/env python3
"""
Blender benchmark implementation.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import time
import tarfile
import urllib.request
from typing import Dict, Any, List, Optional

# pexpect no longer needed with the new CLI approach

from .base import BaseBenchmark


class BlenderBenchmark(BaseBenchmark):
    """Benchmark for Blender rendering performance."""
    
    def __init__(self, output_dir: str = "results"):
        super().__init__("blender", output_dir)
        self.benchmark_url = "https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-3.1.0-linux.tar.gz"
        self.benchmark_dir = os.path.abspath("blender-benchmark")
        self.launcher_path = os.path.join(self.benchmark_dir, "benchmark-launcher-cli")
        self.results["meta"]["benchmark_url"] = self.benchmark_url
    
    def setup(self) -> None:
        """Download and extract the Blender benchmark."""
        # Always delete and re-extract for clean state
        if os.path.isdir(self.benchmark_dir):
            print(f"Removing existing {self.benchmark_dir}")
            shutil.rmtree(self.benchmark_dir)
        
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        # Download benchmark
        tar_file = os.path.join(self.benchmark_dir, "blender-benchmark.tar.gz")
        print(f"Downloading Blender benchmark from {self.benchmark_url}")
        
        try:
            urllib.request.urlretrieve(self.benchmark_url, tar_file)
            print(f"Downloaded to {tar_file}")
        except Exception as e:
            print(f"❌ Failed to download Blender benchmark: {e}")
            print("💡 Please check your internet connection and URL accessibility")
            raise
        
        # Extract tar.gz
        print("Extracting Blender benchmark...")
        try:
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(self.benchmark_dir)
            
            # The tar contains a subdirectory, find the launcher
            for root, dirs, files in os.walk(self.benchmark_dir):
                if "benchmark-launcher-cli" in files:
                    self.launcher_path = os.path.join(root, "benchmark-launcher-cli")
                    break
            
            # Make launcher executable
            if os.path.exists(self.launcher_path):
                os.chmod(self.launcher_path, 0o755)
                print(f"Found launcher at: {self.launcher_path}")
            else:
                raise FileNotFoundError("benchmark-launcher-cli not found in extracted files")
                
        except Exception as e:
            print(f"❌ Failed to extract Blender benchmark: {e}")
            raise
        finally:
            # Clean up tar file
            if os.path.exists(tar_file):
                os.remove(tar_file)
    
    def build(self, args: Any = None) -> Dict[str, Any]:
        """No build step needed for Blender benchmark."""
        return {
            "launcher_path": self.launcher_path,
            "build_time_seconds": 0.0,
            "notes": "Blender benchmark uses pre-built binaries"
        }
    
    def _download_blender(self) -> bool:
        """Download Blender version if not already available."""
        print("🔽 Downloading Blender 4.5.0...")
        
        try:
            cmd = [self.launcher_path, "blender", "download", "4.5.0"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                  cwd=os.path.dirname(self.launcher_path))
            
            if result.returncode == 0:
                print("✅ Blender download completed")
                return True
            else:
                print(f"❌ Blender download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Blender download timed out")
            return False
        except Exception as e:
            print(f"❌ Blender download error: {e}")
            return False
    
    def _get_available_devices(self) -> List[Dict[str, str]]:
        """Get list of available devices for benchmarking."""
        print("🔍 Detecting available devices...")
        
        try:
            cmd = [self.launcher_path, "devices", "-b", "4.5.0"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                  cwd=os.path.dirname(self.launcher_path))
            
            if result.returncode != 0:
                print(f"❌ Failed to get device list: {result.stderr}")
                return []
            
            devices = []
            seen_devices = {}  # Track first framework seen for each device
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse format: "Device Name\tFramework"
                parts = line.rsplit('\t', 1)  # Split by tab from right
                if len(parts) == 2:
                    device_name, framework = parts[0].strip(), parts[1].strip()
                elif len(parts) == 1:
                    # Fallback to whitespace split if no tab
                    parts = line.rsplit(None, 1)
                    if len(parts) == 2:
                        device_name, framework = parts[0].strip(), parts[1].strip()
                    else:
                        continue
                else:
                    continue
                
                # Only use first framework seen for each device name
                if device_name not in seen_devices:
                    seen_devices[device_name] = framework
                    devices.append({
                        "name": device_name,
                        "framework": framework
                    })
                    print(f"📱 Found device: {device_name} ({framework})")
                else:
                    print(f"🔄 Skipping duplicate device: {device_name} ({framework}), already using {seen_devices[device_name]}")
            
            return devices
            
        except subprocess.TimeoutExpired:
            print("❌ Device detection timed out")
            return []
    def _download_scenes(self, scenes: List[str]) -> bool:
        """Download required scenes before running benchmark."""
        print("🔽 Downloading required scenes...")
        
        for scene in scenes:
            print(f"📥 Downloading scene: {scene}")
            try:
                cmd = [self.launcher_path, "scenes", "download", scene, "-b", "4.5.0"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                      cwd=os.path.dirname(self.launcher_path))
                
                if result.returncode == 0:
                    print(f"✅ Scene '{scene}' downloaded successfully")
                else:
                    print(f"❌ Failed to download scene '{scene}': {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"❌ Scene '{scene}' download timed out")
                return False
            except Exception as e:
                print(f"❌ Scene '{scene}' download error: {e}")
                return False
        
        print("✅ All scenes downloaded successfully")
        return True
    
    def _run_blender_benchmark(self, device_name: str, device_framework: str, scenes: List[str] = None, max_retries: int = 3) -> Dict[str, Any]:
        """Run Blender benchmark with specific device and scenes using JSON output, with retry mechanism."""
        if scenes is None:
            scenes = ["monster", "junkshop", "classroom"]
        
        scenes_str = " ".join(scenes)
        print(f"🎬 Running benchmark on {device_name} ({device_framework}) with scenes: {scenes_str}")
        
        # Download required scenes before benchmarking
        if not self._download_scenes(scenes):
            return {
                "device_name": device_name,
                "device_framework": device_framework,
                "scenes": scenes,
                "success": False,
                "error": "Failed to download required scenes"
            }
        
        # Retry mechanism
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt + 1}/{max_retries} for {device_name} ({device_framework})")
            
            start_time = time.perf_counter()
            result = self._run_single_blender_attempt(device_name, device_framework, scenes, start_time)
            
            if result["success"]:
                if attempt > 0:
                    print(f"✅ {device_name} ({device_framework}) succeeded on attempt {attempt + 1}")
                return result
            else:
                print(f"❌ Attempt {attempt + 1} failed: {result.get('error', 'Unknown error')}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
        
        print(f"❌ All {max_retries} attempts failed for {device_name} ({device_framework})")
        return result  # Return the last failed attempt
    
    def _run_single_blender_attempt(self, device_name: str, device_framework: str, scenes: List[str], start_time: float) -> Dict[str, Any]:
        """Run a single Blender benchmark attempt."""
        
        try:
            # Build command for JSON output
            cmd = [
                self.launcher_path, "benchmark",
                *scenes,  # Expand scenes list
                "--blender-version", "4.5.0",
                "--device-type", device_framework,
                "--device-name", device_name,
                "--json"
            ]
            
            print(f"� Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,  # 20 minute timeout
                                  cwd=os.path.dirname(self.launcher_path))
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return {
                    "device_name": device_name,
                    "device_framework": device_framework,
                    "scenes": scenes,
                    "success": False,
                    "error": f"Process failed with return code {result.returncode}",
                    "stderr": result.stderr,
                    "elapsed_seconds": elapsed_time
                }
            
            # Parse JSON output
            try:
                json_output = json.loads(result.stdout)
                print(f"✅ Benchmark completed in {elapsed_time:.1f}s")
                
                # Extract scene results
                scene_results = {}
                total_score = 0.0
                
                if "scenes" in json_output:
                    for scene_data in json_output["scenes"]:
                        scene_name = scene_data.get("label", "unknown")
                        samples_per_minute = scene_data.get("result", {}).get("samples_per_minute", 0.0)
                        
                        scene_results[scene_name.lower()] = {
                            "samples_per_minute": samples_per_minute,
                            "scene_name": scene_name
                        }
                        total_score += samples_per_minute
                        print(f"📊 {scene_name}: {samples_per_minute:.2f} samples per minute")
                
                return {
                    "device_name": device_name,
                    "device_framework": device_framework,
                    "scenes": scenes,
                    "success": True,
                    "elapsed_seconds": elapsed_time,
                    "scene_results": scene_results,
                    "total_score": total_score,
                    "raw_json": json_output,
                    "raw_output": result.stdout
                }
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON output: {e}")
                print(f"Raw output: {result.stdout[:500]}...")
                return {
                    "device_name": device_name,
                    "device_framework": device_framework,
                    "scenes": scenes,
                    "success": False,
                    "error": f"JSON parse error: {e}",
                    "elapsed_seconds": elapsed_time,
                    "raw_output": result.stdout
                }
            
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timed out after 20 minutes")
            return {
                "device_name": device_name,
                "device_framework": device_framework,
                "scenes": scenes,
                "success": False,
                "error": "Benchmark timed out after 20 minutes",
                "elapsed_seconds": 1200
            }
        except Exception as e:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"❌ Benchmark failed: {e}")
            return {
                "device_name": device_name,
                "device_framework": device_framework,
                "scenes": scenes,
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed_time
            }
    
    def _parse_blender_output(self, output: str) -> Dict[str, Any]:
        """Legacy method - now handled directly in _run_blender_benchmark with JSON."""
        # This method is kept for compatibility but is no longer used
        # The new CLI approach returns JSON directly
        return {
            "scenes": {},
            "total_score": 0.0,
            "note": "This method is deprecated - JSON parsing is now done in _run_blender_benchmark"
        }
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run Blender benchmarks on all available devices."""
        results = {
            "device_runs": [],
            "scenes_tested": ["monster", "junkshop", "classroom"]
        }
        
        # First, download Blender if needed
        if not self._download_blender():
            return {
                "device_runs": [],
                "error": "Failed to download Blender",
                "success": False
            }
        
        # Get available devices
        devices = self._get_available_devices()
        if not devices:
            print("❌ No devices found for benchmarking")
            return {
                "device_runs": [],
                "error": "No devices detected",
                "success": False
            }
        
        # Filter devices if --no-gpu is specified
        if args and getattr(args, 'no_gpu', False):
            original_count = len(devices)
            devices = [device for device in devices if device["framework"].upper() == "CPU"]
            print(f"🖥️  CPU-only mode: filtered {original_count} devices to {len(devices)} CPU device(s)")
            
            if not devices:
                print("❌ No CPU devices found for benchmarking")
                return {
                    "device_runs": [],
                    "error": "No CPU devices detected",
                    "success": False
                }
        
        print(f"🎯 Found {len(devices)} devices to benchmark")
        
        # Run benchmark on each device
        successful_runs = 0
        for i, device in enumerate(devices, 1):
            device_name = device["name"]
            framework = device["framework"]
            
            print(f"\n=== Running Blender benchmark {i}/{len(devices)}: {device_name} ({framework}) ===")
            
            # Try up to 3 times with retry logic
            result = None
            for attempt in range(1, 4):
                if attempt > 1:
                    print(f"🔄 Retry attempt {attempt}/3 for {device_name} ({framework})")
                    time.sleep(2 * attempt)  # Exponential backoff: 2s, 4s, 6s
                
                result = self._run_blender_benchmark(device_name, framework)
                
                if result["success"]:
                    if attempt > 1:
                        print(f"✅ Success on attempt {attempt}/3")
                    break
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"❌ Attempt {attempt}/3 failed: {error_msg}")
                    if attempt == 3:
                        print(f"💔 All 3 attempts failed for {device_name} ({framework})")
            
            result["device_name"] = device_name
            result["device_framework"] = framework
            result["attempts_made"] = attempt if result["success"] else 3
            
            results["device_runs"].append(result)
            
            if result["success"]:
                successful_runs += 1
                total_score = result.get("total_score", 0)
                print(f"✅ {device_name} ({framework}): {total_score:.2f} total samples/min")
            else:
                print(f"❌ {device_name} ({framework}) failed: {result.get('error', 'Unknown error')}")
        
        # Calculate device comparisons
        if successful_runs >= 2:
            print(f"\n📊 Comparing {successful_runs} successful device runs:")
            
            # Find CPU baseline for comparison
            cpu_result = None
            for result in results["device_runs"]:
                if result["success"] and result["device_framework"] == "CPU":
                    cpu_result = result
                    break
            
            if cpu_result:
                cpu_score = cpu_result.get("total_score", 0)
                if cpu_score > 0:
                    for result in results["device_runs"]:
                        if result["success"] and result["device_framework"] != "CPU":
                            device_score = result.get("total_score", 0)
                            speedup = device_score / cpu_score
                            result["cpu_speedup"] = speedup
                            device_name = result["device_name"]
                            framework = result["device_framework"]
                            print(f"📈 {device_name} ({framework}) vs CPU: {speedup:.2f}x speedup")
        
        results["successful_runs"] = successful_runs
        results["total_devices_tested"] = len(devices)
        
        if successful_runs > 0:
            print(f"\n🎉 Blender benchmark completed: {successful_runs}/{len(devices)} devices successful")
        else:
            print(f"\n❌ All device benchmarks failed")
        
        return results


def main():
    """Test the Blender benchmark directly."""
    benchmark = BlenderBenchmark("test_results")
    results = benchmark.run()
    print("Results:", results)


if __name__ == "__main__":
    main()