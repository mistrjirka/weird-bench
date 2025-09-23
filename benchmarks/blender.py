#!/usr/bin/env python3
"""
Blender benchmark implementation.
"""
import os
import re
import shutil
import subprocess
import sys
import time
import tarfile
import urllib.request
from typing import Dict, Any, List, Optional

try:
    import pexpect
except ImportError:
    print("❌ pexpect not found. Install with: pip install pexpect")
    sys.exit(1)

try:
    import pexpect
except ImportError:
    print("❌ pexpect not found. Install with: pip install pexpect")
    sys.exit(1)

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
    
    def build(self) -> Dict[str, Any]:
        """No build step needed for Blender benchmark."""
        return {
            "launcher_path": self.launcher_path,
            "build_time_seconds": 0.0,
            "notes": "Blender benchmark uses pre-built binaries"
        }
    
    def _run_blender_benchmark(self, device_type: str = "cpu") -> Dict[str, Any]:
        """Run Blender benchmark handling carriage returns and terminal control sequences."""
        print(f"Running Blender benchmark on {device_type.upper()}...")
        start_time = time.perf_counter()
        
        try:
            # Set environment variables to ensure clean terminal behavior
            env = os.environ.copy()
            env['TERM'] = 'dumb'  # Use dumb terminal to avoid control sequences
            env['COLUMNS'] = '80'
            env['LINES'] = '24'
            
            # Spawn with clean environment
            child = pexpect.spawn(
                self.launcher_path,
                cwd=os.path.dirname(self.launcher_path),
                timeout=30,
                encoding='utf-8',
                codec_errors='replace',
                env=env
            )
            
            # Disable terminal echo and special processing
            child.setecho(False)
            child.delaybeforesend = 0.1
            
            print("🚀 Starting Blender benchmark...")
            
            # Step 1: Handle version selection
            print("🔍 Step 1: Waiting for version selection...")
            try:
                # Look for the version selection prompt, handling carriage returns
                index = child.expect([
                    r'\? Choose a Blender version:',  # The actual prompt
                    r'Choose a Blender version:',     # Without the question mark
                    pexpect.TIMEOUT
                ], timeout=30)
                
                if index < 2:  # Found version prompt
                    print("✅ Found version selection prompt")
                    # Wait a moment for the prompt to stabilize
                    time.sleep(1)
                    child.sendline("")  # Send enter
                    print("📤 Sent: <ENTER> for default version")
                else:
                    print("⚠️ Version selection timeout")
                    
            except pexpect.EOF:
                print("❌ Process ended during version selection")
                raise Exception("Process ended unexpectedly during version selection")
            
            # Step 2: Handle download confirmation
            print("🔍 Step 2: Waiting for download confirmation...")
            try:
                # Look for download prompt, being flexible with the format
                index = child.expect([
                    r'No files need to be downloaded, continue\?',
                    r'download.*continue',
                    r'continue\?.*\(Y/n\)',
                    pexpect.TIMEOUT
                ], timeout=30)
                
                if index < 3:  # Found download prompt
                    print("✅ Found download confirmation prompt")
                    time.sleep(1)
                    child.sendline("Y")
                    print("📤 Sent: Y")
                else:
                    print("⚠️ Download confirmation timeout")
                    
            except pexpect.EOF:
                print("❌ Process ended during download confirmation")
                raise Exception("Process ended unexpectedly during download confirmation")
            
            # Step 3: Handle device selection (optional)
            print("🔍 Step 3: Looking for device selection...")
            try:
                # get output
                time.sleep(2)  # Wait a bit for the prompt to appear
                # print the output so far for debugging
                print("📄 Current output:\n", child.before)
                index = child.expect([
                    r'\?',
                    r'Select.*device',
                    r'monster:',  # Results already started
                    r'Rendering',
                    pexpect.TIMEOUT
                ], timeout=15)
                
                if index == 0 or index == 1:  # Device selection found
                    print("✅ Found device selection")
                    if device_type.lower() == "gpu":
                        print("🔽 Selecting GPU...")
                        child.send("\x1b[B")  # Arrow down
                        time.sleep(0.5)
                        child.sendline("")
                    else:
                        print("✅ Selecting default (CPU)")
                        child.sendline("")
                    print("📤 Device selection sent")

                    
                elif index == 2 or index == 3:  # Already started
                    print("ℹ️ Benchmark already started, no device selection needed")
                else:
                    print("ℹ️ No device selection prompt found")
                    
            except pexpect.EOF:
                print("ℹ️ Process ended during device selection check")
            
            # Step 4: Wait for benchmark completion
            print("🔍 Step 4: Waiting for benchmark to complete...")
            time.sleep(2)  # Give some time before starting to read output
            print("current output:\n", child.before)
            try:
                index = child.expect([
                    r'*Y*'
                ], timeout=10)
                print("current output:\n", child.before)

                if index == 0:
                    print("✅ Found final confirmation prompt")
                    child.sendline("Y")
                    print("📤 Sent: Y to finalize benchmark")
            except pexpect.TIMEOUT:
                print("ℹ️ No final confirmation prompt found, proceeding...")
            print("current output:\n", child.before)

            # Collect all output
            all_output = []
            benchmark_started = False
            
            try:
                while True:
                    try:
                        # Wait for output with generous timeout during benchmark
                        timeout = 600 if benchmark_started else 30
                        index = child.expect([
                            r'monster:.*samples per minute',     # Results line
                            r'junkshop:.*samples per minute',
                            r'classroom:.*samples per minute',
                            r'Rendering.*',                      # Benchmark started
                            r'.+',                               # Any other output
                            pexpect.EOF,
                            pexpect.TIMEOUT
                        ], timeout=timeout)
                        
                        if index <= 4:  # Got some output
                            output_line = child.after
                            if output_line:
                                # Clean up the output line - remove carriage returns and control sequences
                                clean_line = re.sub(r'\r+', '\n', output_line)
                                clean_line = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', clean_line)  # Remove ANSI sequences
                                all_output.append(clean_line)
                                
                                print(f"📊 {clean_line.strip()}")
                                
                                # Check if benchmark has started
                                if 'monster:' in clean_line or 'Rendering' in clean_line:
                                    benchmark_started = True
                                    print("🏁 Benchmark execution detected!")
                                    
                        elif index == 5:  # EOF
                            print("✅ Process completed (EOF)")
                            break
                        else:  # Timeout
                            if benchmark_started:
                                print("⏰ Benchmark timed out during execution")
                            else:
                                print("⏰ Timed out waiting for benchmark to start")
                            break
                            
                    except pexpect.EOF:
                        print("✅ Benchmark completed")
                        break
                        
            except KeyboardInterrupt:
                print("🛑 Benchmark interrupted by user")
                child.terminate()
                raise
            
            # Wait for process to finish
            child.close()
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            # Combine all output
            full_output = ''.join(all_output)
            
            # Also get any remaining output from child.before
            if hasattr(child, 'before') and child.before:
                full_output += child.before
            
            print(f"⏱️ Benchmark completed in {elapsed_time:.1f} seconds")
            print(f"📄 Output length: {len(full_output)} characters")
            print(f"🚪 Exit status: {child.exitstatus}")
            
            if child.exitstatus != 0:
                error_msg = f"Process failed with exit status {child.exitstatus}"
                print(f"❌ {error_msg}")
                return {
                    "device_type": device_type,
                    "success": False,
                    "error": error_msg,
                    "elapsed_seconds": elapsed_time,
                    "raw_output": full_output
                }
            
            # Parse results
            results = self._parse_blender_output(full_output)
            results.update({
                "device_type": device_type,
                "success": True,
                "elapsed_seconds": elapsed_time,
                "raw_output": full_output
            })
            
            print("✅ Benchmark completed successfully!")
            return results
            
        except Exception as e:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"❌ Blender benchmark failed: {e}")
            return {
                "device_type": device_type,
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed_time
            }
    
    def _parse_blender_output(self, output: str) -> Dict[str, Any]:
        """Parse Blender benchmark output to extract scene performance metrics."""
        results = {
            "scenes": {},
            "total_score": 0.0
        }
        
        # Look for benchmark results in the format:
        # "monster: 128.419147 samples per minute"
        # "junkshop: 82.744835 samples per minute"  
        # "classroom: 62.433857 samples per minute"
        
        scene_pattern = r'(\w+):\s*([\d.]+)\s*samples per minute'
        matches = re.findall(scene_pattern, output, re.IGNORECASE)
        
        total_score = 0.0
        for scene_name, score_str in matches:
            try:
                score = float(score_str)
                results["scenes"][scene_name.lower()] = {
                    "samples_per_minute": score,
                    "scene_name": scene_name
                }
                total_score += score
                print(f"📊 {scene_name}: {score:.2f} samples per minute")
            except ValueError:
                print(f"⚠️  Could not parse score for {scene_name}: {score_str}")
        
        results["total_score"] = total_score
        
        # Also try to extract device information
        device_match = re.search(r'Choose a device: (.+)', output)
        if device_match:
            results["detected_device"] = device_match.group(1).strip()
        
        # Extract Blender version
        version_match = re.search(r'Choose a Blender version: ([\d.]+)', output)
        if version_match:
            results["blender_version"] = version_match.group(1).strip()
        
        return results
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run Blender benchmarks for both CPU and GPU (if available)."""
        results = {
            "runs_cpu": [],
            "runs_gpu": []
        }
        
        # Always run CPU benchmark
        print("\n=== Running Blender CPU benchmark ===")
        cpu_result = self._run_blender_benchmark("cpu")
        results["runs_cpu"].append(cpu_result)
        
        if cpu_result["success"]:
            print(f"✅ CPU benchmark completed - Total score: {cpu_result.get('total_score', 0):.2f}")
        else:
            print(f"❌ CPU benchmark failed: {cpu_result.get('error', 'Unknown error')}")
        
        # Try GPU benchmark (might fail if no suitable GPU)
        print("\n=== Running Blender GPU benchmark ===")
        gpu_result = self._run_blender_benchmark("gpu")
        results["runs_gpu"].append(gpu_result)
        
        if gpu_result["success"]:
            print(f"✅ GPU benchmark completed - Total score: {gpu_result.get('total_score', 0):.2f}")
        else:
            print(f"❌ GPU benchmark failed: {gpu_result.get('error', 'Unknown error')}")
            print("💡 This is normal if no suitable GPU is available for Blender")
        
        # Calculate comparison if both succeeded
        if cpu_result["success"] and gpu_result["success"]:
            cpu_score = cpu_result.get("total_score", 0)
            gpu_score = gpu_result.get("total_score", 0)
            if cpu_score > 0:
                speedup = gpu_score / cpu_score
                results["gpu_vs_cpu_speedup"] = speedup
                print(f"📈 GPU vs CPU speedup: {speedup:.2f}x")
        
        return results


def main():
    """Test the Blender benchmark directly."""
    benchmark = BlenderBenchmark("test_results")
    results = benchmark.run()
    print("Results:", results)


if __name__ == "__main__":
    main()