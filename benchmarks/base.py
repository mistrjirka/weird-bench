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