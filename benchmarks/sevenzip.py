#!/usr/bin/env python3
"""
7-Zip benchmark implementation using internal benchmark.
"""

import os
import subprocess
import time
import tempfile
import shutil
import re
from typing import Dict, Any, List, Optional

from .base import BaseBenchmark
from unified_models import SevenZipBenchmarkResult
from hardware_detector import GlobalHardwareDetector


class SevenZipBenchmark(BaseBenchmark):
    """Benchmark for 7-Zip compression performance using internal benchmark."""
    
    def __init__(self, output_dir: str = "results", hardware_detector: Optional[GlobalHardwareDetector] = None):
        super().__init__("7zip", output_dir)
        self.sevenzip_cmd = self._find_7zip_command()
        self.hardware_detector = hardware_detector
        
    def _find_7zip_command(self) -> str:
        """Find the 7-Zip command available on the system."""
        # Try different common 7-Zip command names
        commands = ["7z", "7za", "7zz", "p7zip"]
        
        for cmd in commands:
            if shutil.which(cmd):
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
            result = subprocess.run([self.sevenzip_cmd], capture_output=True, text=True, timeout=10)
            print(f"✅ 7-Zip found: {self.sevenzip_cmd}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"7-Zip setup failed: {e}")
    
    def build(self, args: Any = None) -> Dict[str, Any]:
        """No build step needed for 7-Zip benchmark."""
        return {
            "sevenzip_command": self.sevenzip_cmd,
            "build_time_seconds": 0.0,
            "notes": "7-Zip benchmark uses system-installed binary"
        }
    
    def _run_internal_benchmark(self) -> Dict[str, Any]:
        """Run 7-Zip internal benchmark using '7z b' command."""
        print("🚀 Running 7-Zip internal benchmark...")
        
        try:
            # Run the benchmark
            start_time = time.perf_counter()
            result = subprocess.run([self.sevenzip_cmd, 'b'], 
                                  capture_output=True, text=True, timeout=120)
            elapsed_seconds = time.perf_counter() - start_time
            
            if result.returncode != 0:
                raise RuntimeError(f"7z benchmark failed with return code {result.returncode}: {result.stderr}")
            
            # Parse the output
            parsed_results = self._parse_7zip_benchmark_output(result.stdout)
            
            return {
                "success": True,
                "elapsed_seconds": elapsed_seconds,
                "raw_output": result.stdout,
                "parsed_results": parsed_results
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("7-Zip benchmark timed out after 120 seconds")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": 0.0
            }
    
    def _parse_7zip_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse 7-Zip benchmark output to extract Tot: line metrics."""
        lines = output.strip().split('\n')
        
        # Look for the Tot: line which contains the final results
        # Format: "Tot:            1412   6735  95105"
        # Fields: Usage%, R/U Rating (MIPS), Total Rating (MIPS)
        
        tot_pattern = re.compile(r'^Tot:\s+(\d+)\s+(\d+)\s+(\d+)', re.MULTILINE)
        match = tot_pattern.search(output)
        
        if not match:
            raise ValueError("Could not find Tot: line in 7-Zip benchmark output")
        
        usage_percent = int(match.group(1))
        ru_mips = int(match.group(2))
        total_mips = int(match.group(3))
        
        # Also extract some additional system info if available
        system_info = {}
        
        # Look for CPU info line
        cpu_match = re.search(r'^([^:\n]+Processor.*?)$', output, re.MULTILINE)
        if cpu_match:
            system_info['cpu_detected'] = cpu_match.group(1).strip()
        
        # Look for threads info
        threads_match = re.search(r'Threads:(\d+)', output)
        if threads_match:
            system_info['threads_used'] = int(threads_match.group(1))
        
        # Look for RAM info
        ram_match = re.search(r'RAM size:\s+(\d+)\s+MB', output)
        if ram_match:
            system_info['ram_mb'] = int(ram_match.group(1))
        
        return {
            "usage_percent": usage_percent,
            "ru_mips": ru_mips,
            "total_mips": total_mips,
            "system_info": system_info
        }
    
    def benchmark(self, args: Any = None) -> SevenZipBenchmarkResult:
        """Run the 7-Zip benchmark and return structured results."""
        print("🔧 Running 7-Zip benchmark...")
        
        # Run the internal benchmark
        benchmark_result = self._run_internal_benchmark()
        
        if not benchmark_result["success"]:
            raise RuntimeError(f"7-Zip benchmark failed: {benchmark_result.get('error', 'Unknown error')}")
        
        parsed = benchmark_result["parsed_results"]
        
        # Create unified result
        result = SevenZipBenchmarkResult(
            usage_percent=float(parsed["usage_percent"]),
            ru_mips=float(parsed["ru_mips"]),
            total_mips=float(parsed["total_mips"])
        )
        
        print(f"✅ 7-Zip benchmark completed:")
        print(f"   CPU Usage: {result.usage_percent}%")
        print(f"   R/U MIPS: {result.ru_mips}")
        print(f"   Total MIPS: {result.total_mips}")
        
        return result
    
    def run(self, args: Any = None) -> Dict[str, Any]:
        """Legacy interface compatibility."""
        try:
            benchmark_result = self.benchmark(args)
            
            # Convert to legacy format for compatibility
            return {
                "meta": {
                    "benchmark_name": "7zip",
                    "host": os.uname().nodename,
                    "platform": f"{os.uname().sysname}-{os.uname().release}-{os.uname().machine}",
                    "timestamp": time.time()
                },
                "build": {
                    "sevenzip_command": self.sevenzip_cmd,
                    "build_time_seconds": 0.0,
                    "notes": "7-Zip benchmark uses internal benchmark (7z b)"
                },
                "results": {
                    "usage_percent": benchmark_result.usage_percent,
                    "ru_mips": benchmark_result.ru_mips,
                    "total_mips": benchmark_result.total_mips
                }
            }
        except Exception as e:
            return {
                "meta": {
                    "benchmark_name": "7zip",
                    "host": os.uname().nodename,
                    "platform": f"{os.uname().sysname}-{os.uname().release}-{os.uname().machine}",
                    "timestamp": time.time()
                },
                "error": str(e),
                "success": False
            }


def main():
    """Test the 7-Zip benchmark directly."""
    benchmark = SevenZipBenchmark("test_results")
    benchmark.setup()
    
    try:
        result = benchmark.benchmark()
        print("Unified Result:", result)
        
        legacy_result = benchmark.run()
        print("Legacy Result:", legacy_result)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()