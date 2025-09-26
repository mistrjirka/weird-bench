#!/usr/bin/env python3
"""
Base benchmark class defining the interface for all benchmarks.
"""
import json
import os
import subprocess
import time
import platform
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, name: str, output_dir: str = "results"):
        self.name = name
        self.output_dir = output_dir
        self.results = {
            "meta": {
                "benchmark_name": name,
                "host": platform.node(),
                "platform": platform.platform(),
                "timestamp": time.time(),
            }
        }
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, args: Any = None) -> Dict[str, Any]:
        """Run the complete benchmark and return results."""
        print(f"\n=== Running {self.name} benchmark ===")
        
        try:
            # Setup phase
            self.setup()
            
            # Build phase
            build_results = self.build(args)
            self.results["build"] = build_results
            
            # Benchmark phase
            benchmark_results = self.benchmark(args)
            self.results.update(benchmark_results)
            
            # Save results
            output_file = os.path.join(self.output_dir, f"{self.name}_results.json")
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            print(f"✅ {self.name} benchmark completed successfully!")
            print(f"📊 Results saved to: {output_file}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ {self.name} benchmark failed: {e}")
            raise
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the benchmark (clone repos, prepare environment, etc.)."""
        pass
    
    @abstractmethod
    def build(self, args: Any = None) -> Dict[str, Any]:
        """Build the benchmark (if needed). Override in subclasses."""
        return {}
    
    @abstractmethod
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run the actual benchmarks and return results."""
        pass
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None, 
                   check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True)
    
    def run_command_with_env(self, cmd: List[str], env: Dict[str, str], cwd: Optional[str] = None, 
                           check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command with custom environment variables and return the result."""
        return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True, env=env)
    
    def measure_time_and_run(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
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
        """Generate thread counts for testing starting from logical cores, halving down, and including physical cores if needed.
        
        If an odd value is encountered (except for the first value), it increases the number of cores to test by 1.
        Division by 2 rounds up.
        """
        if logical_cores <= 0:
            return [1]
        
        thread_counts = []
        current = logical_cores
        
        # Start from logical cores and halve until we reach 1
        # The first value (logical_cores) is kept as-is, even if odd
        first_value = True
        while current >= 1:
            if first_value:
                # Keep the first value as-is
                test_count = current
                first_value = False
            else:
                # For subsequent values, if odd, increase by 1
                test_count = current if current % 2 == 0 else current + 1
            
            if test_count not in thread_counts:
                thread_counts.append(test_count)
            
            # Round up division by 2, but stop if we would loop on 1
            next_current = (current + 1) // 2
            if next_current == current:
                # If we're not actually decreasing, break to avoid infinite loop
                break
            current = next_current
        
        # Add physical cores if they don't already exist in the halving pattern
        if physical_cores > 1:
            # If physical_cores is odd, increase by 1
            adjusted_physical = physical_cores if physical_cores % 2 == 0 else physical_cores + 1
            if adjusted_physical not in thread_counts:
                thread_counts.append(adjusted_physical)
                print(f"Added physical core count {adjusted_physical} to test list")
        
        # Ensure we always test with 1 thread (if not already included)
        if 1 not in thread_counts:
            thread_counts.append(1)
        
        # Sort in descending order for cleaner output
        thread_counts.sort(reverse=True)
        
        return thread_counts