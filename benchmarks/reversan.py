#!/usr/bin/env python3
"""
Reversan Engine benchmark implementation.
"""
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

from .base import BaseBenchmark


class ReversanBenchmark(BaseBenchmark):
    """Benchmark for Reversan Engine."""
    
    def __init__(self, output_dir: str = "results"):
        super().__init__("reversan", output_dir)
        self.repo_url = "https://github.com/Saniel0/Reversan-Engine.git"
        self.project_dir = os.path.abspath("Reversan-Engine")
        self.build_dir = os.path.join(self.project_dir, "build")
        self.gnu_time = self._find_gnu_time()
        self.results["meta"]["gnu_time"] = bool(self.gnu_time)
        self.results["meta"]["repo"] = self.repo_url
        
        # Generate unique build ID for cold builds
        import uuid
        self.build_id = str(uuid.uuid4())[:8]
    
    def _get_cold_build_env(self) -> Dict[str, str]:
        """Get environment variables that ensure cold builds (no caching)."""
        env = os.environ.copy()
        # Disable ccache globally
        env['CCACHE_DISABLE'] = '1'
        # Disable other potential caches
        env['CCACHE_DIR'] = '/dev/null'
        env['CCACHE_NOSTATS'] = '1'
        # Force empty launcher to override toolchain defaults
        env['CMAKE_C_COMPILER_LAUNCHER'] = ''
        env['CMAKE_CXX_COMPILER_LAUNCHER'] = ''
        return env
    
    def _find_gnu_time(self) -> Optional[str]:
        """Find GNU time command."""
        for cmd in ["/usr/bin/time", shutil.which("time"), shutil.which("gtime")]:
            if not cmd:
                continue
            try:
                p = subprocess.run([cmd, "-v", "true"], capture_output=True, text=True)
                if p.returncode == 0 and "Maximum resident set size" in (p.stderr or ""):
                    return cmd
            except Exception:
                pass
        return None
    
    def setup(self) -> None:
        """Clone or update the Reversan repository."""
        # Always delete and re-clone for clean state
        if os.path.isdir(self.project_dir):
            print(f"Removing existing {self.project_dir}")
            shutil.rmtree(self.project_dir)
        
        print(f"Cloning {self.repo_url} -> {self.project_dir}")
        self.run_command(["git", "clone", "--depth", "1", self.repo_url, self.project_dir])
    
    def build(self, args: Any = None) -> Dict[str, Any]:
        """Build the Reversan Engine and return build metrics."""
        os.makedirs(self.build_dir, exist_ok=True)
        
        print("Configuring CMake…")
        config_start = time.perf_counter()
        config_cmd = [
            "cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}",
            f"-DCMAKE_CXX_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}"
        ]
        self.run_command_with_env(config_cmd, self._get_cold_build_env(), cwd=self.project_dir)
        config_time = time.perf_counter() - config_start
        
        print("Building…")
        build_start = time.perf_counter()
        build_cmd = ["cmake", "--build", "build", "-j"]
        self.run_command_with_env(build_cmd, self._get_cold_build_env(), cwd=self.project_dir)
        build_time = time.perf_counter() - build_start
        
        total_compile_time = config_time + build_time
        print(f"Compile time: {total_compile_time:.2f}s (config: {config_time:.2f}s, build: {build_time:.2f}s)")
        
        binary_path = self._pick_binary()
        threads_supported = self._reversan_supports_threads(binary_path)
        
        # Capture help output for audit
        help_out = subprocess.run([binary_path, "--help"], capture_output=True, text=True)
        help_snippet = ((help_out.stdout or "") + (help_out.stderr or ""))[:2000]
        
        return {
            "binary": binary_path,
            "compile_time_seconds": total_compile_time,
            "config_time_seconds": config_time,
            "build_time_seconds": build_time,
            "time_measurer": "GNU time -v" if self.gnu_time else "fallback (wall-clock only)",
            "threads_supported": threads_supported,
            "help_snippet": help_snippet,
        }
    
    def _pick_binary(self) -> str:
        """Find the Reversan binary."""
        candidates = [
            os.path.join(self.project_dir, "reversan_avx2"),
            os.path.join(self.project_dir, "reversan_nosimd"),
        ]
        
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        
        # Last resort: any reversan* file
        for name in os.listdir(self.project_dir):
            if name.startswith("reversan") and os.access(os.path.join(self.project_dir, name), os.X_OK):
                return os.path.join(self.project_dir, name)
        
        raise RuntimeError("No 'reversan*' binary found in repo root.")
    
    def _get_cpu_count(self) -> int:
        """Get number of CPU cores/threads available."""
        try:
            # Try to get logical CPU count (includes hyperthreading)
            cpu_count = multiprocessing.cpu_count()
            print(f"Detected {cpu_count} CPU threads")
            return cpu_count
        except (ImportError, NotImplementedError):
            # Fallback to 8 if detection fails
            print("Could not detect CPU count, defaulting to 8 threads")
            return 8
    
    def _reversan_supports_threads(self, bin_path: str) -> bool:
        """Check if the binary supports threads."""
        try:
            p = subprocess.run([bin_path, "--help"], capture_output=True, text=True, timeout=10)
            txt = p.stdout + p.stderr
            return bool(re.search(r"(^|\s)-t(\s|=|,|/)|threads", txt, re.IGNORECASE))
        except Exception:
            return False
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run the Reversan benchmarks."""
        runs = getattr(args, 'runs', 1) if args else 1
        binary_path = self.results["build"]["binary"]
        threads_supported = self.results["build"]["threads_supported"]
        
        self.results["meta"]["test_runs_per_config"] = runs
        
        results = {
            "runs_depth": [],
            "runs_threads": [],
        }
        
        # Depth sweep tests (1-12)
        print(f"\n=== Running depth sweep tests (1-12) with {runs} run(s) each ===")
        for depth in range(1, 13):
            depth_runs = []
            for run_num in range(runs):
                run_label = f"run {run_num + 1}" if runs > 1 else ""
                print(f"Running depth {depth}{'...' if runs == 1 else f' {run_label}...'}")
                
                cmd = [binary_path, "--bot-vs-bot", "--depth", str(depth)]
                rc, metrics, out, err = self._measure_command(cmd)
                
                depth_runs.append({
                    "run": run_num + 1,
                    "returncode": rc,
                    "metrics": metrics,
                    "stderr_tail": (err or "")[-500:],
                })
            
            # Calculate average metrics if multiple runs
            if runs > 1:
                avg_metrics = self._calculate_average_metrics(depth_runs)
                results["runs_depth"].append({
                    "depth": depth,
                    "runs": depth_runs,
                    "average_metrics": avg_metrics,
                    "num_runs": runs,
                })
            else:
                # Single run - keep the original format for compatibility
                results["runs_depth"].append({
                    "depth": depth,
                    "returncode": depth_runs[0]["returncode"],
                    "metrics": depth_runs[0]["metrics"],
                    "stderr_tail": depth_runs[0]["stderr_tail"],
                })
        
        # Threads sweep tests @ depth 11
        max_threads = self._get_cpu_count()
        print(f"\n=== Running threads sweep tests (1-{max_threads}) at depth 11 with {runs} run(s) each ===")
        for t in range(1, max_threads + 1):
            if not threads_supported:
                print(f"Skipping threads test {t} (threads not supported)")
                results["runs_threads"].append({
                    "threads": t, 
                    "skipped": True, 
                    "reason": "threads_not_supported_by_binary"
                })
                continue
            
            thread_runs = []
            for run_num in range(runs):
                run_label = f"run {run_num + 1}" if runs > 1 else ""
                print(f"Running with {t} thread(s){'...' if runs == 1 else f' {run_label}...'}")
                
                cmd = [binary_path, "--bot-vs-bot", "--depth", "11", "-t", str(t)]
                rc, metrics, out, err = self._measure_command(cmd)
                
                thread_runs.append({
                    "run": run_num + 1,
                    "returncode": rc,
                    "metrics": metrics,
                    "stderr_tail": (err or "")[-500:],
                })
            
            # Calculate average metrics if multiple runs
            if runs > 1:
                avg_metrics = self._calculate_average_metrics(thread_runs)
                results["runs_threads"].append({
                    "threads": t,
                    "runs": thread_runs,
                    "average_metrics": avg_metrics,
                    "num_runs": runs,
                })
            else:
                # Single run - keep the original format for compatibility
                results["runs_threads"].append({
                    "threads": t,
                    "returncode": thread_runs[0]["returncode"],
                    "metrics": thread_runs[0]["metrics"],
                    "stderr_tail": thread_runs[0]["stderr_tail"],
                })
        
        return results
    
    def _measure_command(self, cmd: List[str], cwd: Optional[str] = None):
        """Measure command execution with GNU time if available."""
        if self.gnu_time:
            p = subprocess.run([self.gnu_time, "-v"] + cmd, cwd=cwd, capture_output=True, text=True)
            metrics = self._parse_gnu_time(p.stderr or "")
            return p.returncode, metrics, p.stdout, p.stderr
        
        start = time.perf_counter()
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        
        return p.returncode, {
            "max_rss_kb": None,
            "user_seconds": None,
            "sys_seconds": None,
            "elapsed_seconds": elapsed,
        }, p.stdout, p.stderr
    
    def _parse_gnu_time(self, stderr_text: str) -> Dict[str, Any]:
        """Parse GNU time output."""
        def get(rex, cast=float, default=None):
            m = re.search(rex, stderr_text)
            if not m:
                return default
            try:
                return cast(m.group(1).strip())
            except Exception:
                return m.group(1).strip()
        
        max_rss_kb = get(r"Maximum resident set size \(kbytes\):\s+(\d+)", int)
        user_sec = get(r"User time \(seconds\):\s+([0-9.]+)")
        sys_sec = get(r"System time \(seconds\):\s+([0-9.]+)")
        elapsed = get(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([0-9:]+)", str)
        
        def hms_to_s(s):
            if not isinstance(s, str):
                return None
            parts = s.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return None
        
        return {
            "max_rss_kb": max_rss_kb,
            "user_seconds": user_sec,
            "sys_seconds": sys_sec,
            "elapsed_seconds": hms_to_s(elapsed),
        }
    
    def _calculate_average_metrics(self, runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate average metrics from multiple runs."""
        if not runs:
            return None
        
        # Collect all metrics
        elapsed_times = [r["metrics"]["elapsed_seconds"] for r in runs if r["metrics"]["elapsed_seconds"] is not None]
        max_rss_values = [r["metrics"]["max_rss_kb"] for r in runs if r["metrics"]["max_rss_kb"] is not None]
        user_times = [r["metrics"]["user_seconds"] for r in runs if r["metrics"]["user_seconds"] is not None]
        sys_times = [r["metrics"]["sys_seconds"] for r in runs if r["metrics"]["sys_seconds"] is not None]
        
        return {
            "elapsed_seconds": sum(elapsed_times) / len(elapsed_times) if elapsed_times else None,
            "max_rss_kb": sum(max_rss_values) / len(max_rss_values) if max_rss_values else None,
            "user_seconds": sum(user_times) / len(user_times) if user_times else None,
            "sys_seconds": sum(sys_times) / len(sys_times) if sys_times else None,
            "elapsed_seconds_min": min(elapsed_times) if elapsed_times else None,
            "elapsed_seconds_max": max(elapsed_times) if elapsed_times else None,
            "max_rss_kb_min": min(max_rss_values) if max_rss_values else None,
            "max_rss_kb_max": max(max_rss_values) if max_rss_values else None,
        }