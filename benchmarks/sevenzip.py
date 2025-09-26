#!/usr/bin/env python3
"""
7-Zip benchmark implementation.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, List, Optional

from .base import BaseBenchmark


class SevenZipBenchmark(BaseBenchmark):
    """Benchmark for 7-Zip compression performance."""
    
    def __init__(self, output_dir: str = "results"):
        super().__init__("7zip", output_dir)
        self.sevenzip_cmd = self._find_7zip_command()
        
    def _find_7zip_command(self) -> str:
        """Find the 7-Zip command available on the system."""
        # Try different common 7-Zip command names
        commands = ["7z", "7za", "7zz", "p7zip"]
        
        for cmd in commands:
            if shutil.which(cmd):
                print(f"Found 7-Zip command: {cmd}")
                return cmd
        
        # If none found, suggest installation
        print("❌ No 7-Zip command found. Please install 7-Zip:")
        print("   Ubuntu/Debian: sudo apt install p7zip-full")
        print("   Fedora/RHEL: sudo dnf install p7zip p7zip-plugins")
        print("   Arch: sudo pacman -S p7zip")
        raise FileNotFoundError("7-Zip not found")
    
    def setup(self) -> None:
        """Check if 7-Zip is available."""
        try:
            # Test that 7-Zip works
            result = subprocess.run([self.sevenzip_cmd], 
                                  capture_output=True, text=True, timeout=10)
            print(f"✅ 7-Zip is available: {self.sevenzip_cmd}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Failed to verify 7-Zip installation: {e}")
            raise
    
    def build(self, args: Any = None) -> Dict[str, Any]:
        """No build step needed for 7-Zip benchmark."""
        return {
            "sevenzip_command": self.sevenzip_cmd,
            "build_time_seconds": 0.0,
            "notes": "7-Zip benchmark uses system-installed binary"
        }
    
    def _create_test_data(self, size_mb: int = 100) -> str:
        """Create test data for compression benchmark."""
        test_dir = tempfile.mkdtemp(prefix="7zip_test_")
        
        # Create various types of files for realistic compression testing
        files_created = []
        
        # Text file (highly compressible)
        text_file = os.path.join(test_dir, "text_data.txt")
        with open(text_file, 'w') as f:
            # Write repetitive text data
            for i in range(size_mb * 1000):  # Approximate MB worth of text
                f.write(f"This is line {i} of test data for 7-Zip compression benchmark. " * 10 + "\n")
        files_created.append(text_file)
        
        # Binary file (less compressible)
        binary_file = os.path.join(test_dir, "binary_data.bin")
        with open(binary_file, 'wb') as f:
            # Write pseudo-random binary data
            import random
            random.seed(42)  # Reproducible "random" data
            for _ in range(size_mb * 1024):  # KB chunks
                chunk = bytes([random.randint(0, 255) for _ in range(1024)])
                f.write(chunk)
        files_created.append(binary_file)
        
        # Create some directory structure
        subdir = os.path.join(test_dir, "subdir")
        os.makedirs(subdir)
        
        small_file = os.path.join(subdir, "small.txt")
        with open(small_file, 'w') as f:
            f.write("Small file content\n" * 1000)
        files_created.append(small_file)
        
        print(f"📁 Created test data in {test_dir} with {len(files_created)} files")
        return test_dir
    
    def _run_compression_benchmark(self, test_dir: str, threads: int = 1) -> Dict[str, Any]:
        """Run compression benchmark with specified thread count."""
        archive_path = os.path.join(tempfile.gettempdir(), f"benchmark_archive_{threads}t.7z")
        
        # Remove existing archive if it exists
        if os.path.exists(archive_path):
            os.remove(archive_path)
        
        # Build 7-Zip command to compress the entire test directory
        cmd = [
            self.sevenzip_cmd, "a",  # Add to archive
            archive_path,
            test_dir,  # Compress entire directory
            f"-mmt{threads}",  # Set thread count
            "-mx=5",  # Compression level (0=none, 9=max)
            "-y"  # Yes to all prompts
        ]
        
        print(f"🔄 Running compression with {threads} threads...")
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "threads": threads,
                    "error": f"Compression failed: {result.stderr}",
                    "elapsed_seconds": elapsed_time
                }
            
            # Get archive size
            archive_size = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0
            
            # Parse 7-Zip output for compression ratio and speed
            compression_info = self._parse_7zip_output(result.stdout)
            
            return {
                "success": True,
                "threads": threads,
                "elapsed_seconds": elapsed_time,
                "archive_size_bytes": archive_size,
                "compression_ratio": compression_info.get("ratio", 0.0),
                "compression_speed_mb_s": compression_info.get("speed_mb_s", 0.0),
                "raw_output": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "threads": threads,
                "error": "Compression timed out after 5 minutes",
                "elapsed_seconds": 300
            }
        except Exception as e:
            return {
                "success": False,
                "threads": threads,
                "error": str(e),
                "elapsed_seconds": time.perf_counter() - start_time
            }
    
    def _parse_7zip_output(self, output: str) -> Dict[str, Any]:
        """Parse 7-Zip output to extract performance metrics."""
        info = {}
        
        # Look for compression ratio in the format "Archive size: X bytes (Y%)"
        ratio_patterns = [
            r'(\d+)%',  # Simple percentage
            r'Archive size:.*\((\d+)%\)',  # Archive size with percentage
            r'Ratio to input: (\d+)%'  # Explicit ratio
        ]
        
        for pattern in ratio_patterns:
            ratio_match = re.search(pattern, output)
            if ratio_match:
                info["ratio"] = float(ratio_match.group(1))
                break
        
        # Look for speed information 
        speed_patterns = [
            r'Speed:\s*(\d+(?:\.\d+)?)\s*MB/s',
            r'(\d+(?:\.\d+)?)\s*MB/s',
            r'(\d+(?:\.\d+)?)\s*MiB/s'
        ]
        
        for pattern in speed_patterns:
            speed_match = re.search(pattern, output)
            if speed_match:
                info["speed_mb_s"] = float(speed_match.group(1))
                break
        
        # Look for file sizes to calculate ratio manually if needed
        size_before_match = re.search(r'(\d+)\s+bytes\s+before', output)
        size_after_match = re.search(r'(\d+)\s+bytes\s+after', output)
        
        if size_before_match and size_after_match and "ratio" not in info:
            before = int(size_before_match.group(1))
            after = int(size_after_match.group(1))
            if before > 0:
                info["ratio"] = (after / before) * 100
        
        return info
    
    def _get_cpu_counts(self) -> tuple[int, int]:
        """Get the number of logical and physical CPU cores available."""
        logical_cores = 4  # Default fallback
        physical_cores = 2  # Default fallback
        
        try:
            import multiprocessing
            logical_cores = multiprocessing.cpu_count()
            print(f"Detected {logical_cores} logical CPU cores (threads)")
        except:
            print("Could not detect logical CPU count, defaulting to 4")
        
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores:
                print(f"Detected {physical_cores} physical CPU cores")
            else:
                # Fallback estimate
                physical_cores = max(1, logical_cores // 2)
                print(f"Estimated {physical_cores} physical cores (from {logical_cores} logical)")
        except ImportError:
            # Fallback to logical cores divided by 2 (assuming hyperthreading)
            physical_cores = max(1, logical_cores // 2)
            print(f"Estimated {physical_cores} physical cores (from {logical_cores} logical)")
        
        return logical_cores, physical_cores
    
    def _get_thread_test_counts(self, logical_cores: int, physical_cores: int) -> List[int]:
        """Generate thread counts for testing starting from logical cores, halving down, and including physical cores if needed."""
        if logical_cores <= 0:
            return [1]
        
        thread_counts = []
        current = logical_cores
        
        # Start from logical cores and halve until we reach 1
        while current >= 1:
            thread_counts.append(current)
            current //= 2
        
        # Add physical cores if they don't already exist in the halving pattern
        if physical_cores not in thread_counts and physical_cores > 1:
            thread_counts.append(physical_cores)
            print(f"Added physical core count {physical_cores} to test list")
        
        # Ensure we always test with 1 thread (if not already included)
        if thread_counts[-1] != 1:
            thread_counts.append(1)
        
        # Sort in descending order for cleaner output
        thread_counts.sort(reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_thread_counts = []
        for count in thread_counts:
            if count not in seen:
                seen.add(count)
                unique_thread_counts.append(count)
        
        return unique_thread_counts
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run 7-Zip benchmarks with different thread counts."""
        results = {
            "runs": [],
            "test_data_size_mb": 100
        }
        
        # Create test data
        test_dir = self._create_test_data(100)  # 100MB test data
        
        try:
            # Test different thread counts
            logical_cores, physical_cores = self._get_cpu_counts()
            thread_counts = self._get_thread_test_counts(logical_cores, physical_cores)
            
            print(f"🧮 Testing with thread counts: {thread_counts}")
            
            for threads in thread_counts:
                print(f"\n=== Running 7-Zip benchmark with {threads} threads ===")
                result = self._run_compression_benchmark(test_dir, threads)
                results["runs"].append(result)
                
                if result["success"]:
                    print(f"✅ {threads} threads: {result['elapsed_seconds']:.2f}s, "
                          f"ratio: {result.get('compression_ratio', 0):.1f}%, "
                          f"size: {result['archive_size_bytes'] / 1024 / 1024:.1f}MB")
                else:
                    print(f"❌ {threads} threads failed: {result.get('error', 'Unknown error')}")
            
            # Calculate thread scaling metrics
            if len([r for r in results["runs"] if r["success"]]) >= 2:
                single_thread_time = next((r["elapsed_seconds"] for r in results["runs"] 
                                         if r["success"] and r["threads"] == 1), None)
                if single_thread_time:
                    for result in results["runs"]:
                        if result["success"] and result["threads"] > 1:
                            speedup = single_thread_time / result["elapsed_seconds"]
                            efficiency = speedup / result["threads"] * 100
                            result["speedup"] = speedup
                            result["thread_efficiency_percent"] = efficiency
            
        finally:
            # Clean up test data
            try:
                shutil.rmtree(test_dir)
                print(f"🧹 Cleaned up test data from {test_dir}")
            except Exception as e:
                print(f"⚠️  Failed to clean up test data: {e}")
            
            # Clean up archive files
            try:
                import glob
                archive_pattern = os.path.join(tempfile.gettempdir(), "benchmark_archive_*.7z")
                for archive_file in glob.glob(archive_pattern):
                    os.remove(archive_file)
            except Exception as e:
                print(f"⚠️  Failed to clean up archive files: {e}")
        
        return results


def main():
    """Test the 7-Zip benchmark directly."""
    benchmark = SevenZipBenchmark("test_results")
    results = benchmark.run()
    print("Results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()