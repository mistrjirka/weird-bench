#!/usr/bin/env python3
"""
Llama.cpp benchmark implementation.
"""
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from typing import Dict, Any, List, Optional

from .base import BaseBenchmark


class CompilationToolbox:
    """Unified toolbox for managing compilation and build processes."""
    
    def __init__(self, project_dir: str, build_dir: str):
        self.project_dir = project_dir
        self.build_dir = build_dir
    
    def clean_build_environment(self) -> None:
        """Clean the entire project directory for a fresh build."""
        if os.path.exists(self.project_dir):
            print(f"🧹 Cleaning build environment: {self.project_dir}")
            shutil.rmtree(self.project_dir)
    
    def prepare_build_directory(self) -> None:
        """Ensure build directory exists."""
        os.makedirs(self.build_dir, exist_ok=True)
    
    def measure_build_phase(self, phase_name: str, cmd: List[str], cwd: str, env: Optional[Dict[str, str]] = None) -> tuple[float, subprocess.CompletedProcess]:
        """Measure the time for a build phase and return timing + result."""
        print(f"⚙️  {phase_name}...")
        print(f"🔧 Command: {' '.join(cmd)}")
        start_time = time.perf_counter()
        
        # For build commands, show real-time progress
        if "build" in phase_name.lower():
            process = subprocess.Popen(
                cmd, 
                cwd=cwd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True,
                env=env
            )
            
            stdout_lines = []
            line_count = 0
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    stdout_lines.append(line)
                    line_count += 1
                    
                    # Show progress every 20 lines or for important messages
                    if (line_count % 20 == 0 or 
                        any(keyword in line for keyword in ["Built target", "Linking", "error:", "Error", "llama-bench"])):
                        print(f"📈 {line}")
            
            process.wait()
            
            # Create result object
            class BuildResult:
                def __init__(self, returncode, stdout, stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = BuildResult(process.returncode, '\n'.join(stdout_lines))
        else:
            # For config commands, use regular capture
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        
        elapsed_time = time.perf_counter() - start_time
        print(f"✅ {phase_name} completed in {elapsed_time:.2f}s")
        
        return elapsed_time, result
    
    def build_variant_with_timing(self, variant_name: str, cmake_config_cmd: List[str], 
                                 error_handler=None, setup_callback=None, env: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Build a variant with detailed timing and error handling."""
        print(f"🔨 Building {variant_name} variant...")
        
        # Prepare clean environment
        self.clean_build_environment()
        
        # Call setup callback to prepare project (clone repo, copy model, etc.)
        if setup_callback:
            setup_callback()
        
        # Configuration phase
        config_time, config_result = self.measure_build_phase(
            f"Configuring {variant_name} build", 
            cmake_config_cmd, 
            self.project_dir,
            env
        )
        
        if config_result.returncode != 0:
            if error_handler:
                error_handler(config_result, "configuration", variant_name)
            raise subprocess.CalledProcessError(config_result.returncode, cmake_config_cmd)
        
        # Build phase with parallel jobs (full build)
        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)  # Limit to 20 jobs max
        build_cmd = ["cmake", "--build", "build", "--config", "Release", "--", f"-j{num_jobs}"]
        build_time, build_result = self.measure_build_phase(
            f"Building {variant_name} (using {num_jobs} jobs)", 
            build_cmd, 
            self.project_dir,
            env
        )
        
        if build_result.returncode != 0:
            if error_handler:
                error_handler(build_result, "build", variant_name)
            raise subprocess.CalledProcessError(build_result.returncode, build_cmd)
        
        total_time = config_time + build_time
        print(f"🎉 {variant_name.capitalize()} build completed! Total: {total_time:.2f}s (config: {config_time:.2f}s, build: {build_time:.2f}s)")
        
        return {
            "config_time_seconds": config_time,
            "build_time_seconds": build_time,
            "total_time_seconds": total_time
        }


class LlamaBenchmark(BaseBenchmark):
    """Benchmark for Llama.cpp."""
    
    def __init__(self, output_dir: str = "results"):
        super().__init__("llama", output_dir)
        self.repo_url = "https://github.com/ggml-org/llama.cpp"
        self.project_dir = os.path.abspath("llama.cpp")
        self.cpu_build_dir = os.path.join(self.project_dir, "build_cpu")
        self.vulkan_build_dir = os.path.join(self.project_dir, "build_vulkan")
        # Shared models directory above project directory
        self.shared_models_dir = os.path.abspath("models")
        self.project_models_dir = os.path.join(self.project_dir, "models")
        self.model_url = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-IQ4_NL.gguf?download=true"
        self.model_filename = "Qwen3-4B-Instruct-2507-IQ4_NL.gguf"
        self.shared_model_path = os.path.join(self.shared_models_dir, self.model_filename)
        self.project_model_path = os.path.join(self.project_models_dir, self.model_filename)
        
        self.results["meta"]["repo"] = self.repo_url
        self.results["meta"]["model_url"] = self.model_url
        
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
    
    def _check_dependencies(self) -> None:
        """Check for required system dependencies."""
        missing_deps = []
        
        # Check for essential build tools
        required_commands = {
            "git": "git (for cloning repositories)",
            "cmake": "cmake (for building projects)",
            "curl": "curl (required by llama.cpp build system)",
        }
        
        for cmd, description in required_commands.items():
            if not shutil.which(cmd):
                missing_deps.append(description)
        
        # Check for C++ compiler
        cpp_compilers = ["g++", "clang++", "c++"]
        has_compiler = any(shutil.which(compiler) for compiler in cpp_compilers)
        if not has_compiler:
            missing_deps.append("C++ compiler (g++, clang++, or c++)")
        
        if missing_deps:
            error_msg = "Missing required dependencies:\n"
            for dep in missing_deps:
                error_msg += f"  - {dep}\n"
            error_msg += "\nFor openSUSE, install with:\n"
            error_msg += "  sudo zypper install git cmake curl libcurl-devel gcc-c++\n"
            error_msg += "\nOr run the provided installation script:\n"
            error_msg += "  ./install_dependencies_opensuse.sh\n"
            error_msg += "\nPlease install the missing dependencies and try again."
            raise RuntimeError(error_msg)
    
    def setup(self) -> None:
        """Setup shared model directory and prepare for benchmarking."""
        # Check dependencies first
        print("🔍 Checking system dependencies...")
        self._check_dependencies()
        print("✅ All required dependencies found!")
        
        # Create shared models directory
        os.makedirs(self.shared_models_dir, exist_ok=True)
        
        # Download model to shared location if not exists
        if not os.path.exists(self.shared_model_path):
            print(f"📥 Downloading model {self.model_filename} to shared location...")
            self._download_model_to_shared_location()
        else:
            print(f"✅ Model {self.model_filename} already exists in shared location, skipping download")
            # Verify file size
            file_size = os.path.getsize(self.shared_model_path) // (1024 * 1024)
            print(f"📄 Model file size: {file_size} MB")
    
    def _download_model_to_shared_location(self) -> None:
        """Download the GGUF model file to shared models directory."""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r📥 Downloading: {percent}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(self.model_url, self.shared_model_path, progress_hook)
            print(f"\n✅ Model downloaded successfully to {self.shared_model_path}")
        except Exception as e:
            print(f"\n❌ Failed to download model: {e}")
            raise
    
    def _setup_project_and_copy_model(self) -> None:
        """Clone repository and copy model from shared location."""
        print(f"📂 Cloning {self.repo_url} -> {self.project_dir}")
        try:
            self.run_command(["git", "clone", "--depth", "1", self.repo_url, self.project_dir])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")
        
        # Create project models directory
        os.makedirs(self.project_models_dir, exist_ok=True)
        
        # Copy model from shared location
        print(f"📋 Copying model from shared location...")
        shutil.copy2(self.shared_model_path, self.project_model_path)
        print(f"✅ Model copied to {self.project_model_path}")
    
    def build(self, args: Any = None) -> Dict[str, Any]:
        """Build Llama.cpp with both CPU-only and Vulkan versions."""
        build_results = {}
        
        print(f"\n{'='*60}")
        print(f"🚀 Building llama.cpp with CPU and Vulkan versions")
        print(f"{'='*60}")
        
        # Check if --no-gpu is specified
        no_gpu = args and getattr(args, 'no_gpu', False)
        if no_gpu:
            print("🖥️  --no-gpu specified: skipping Vulkan build, CPU-only mode")
        
        # Step 1: Clean environment and setup project once
        print("🧹 Cleaning build environment...")
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)
        
        print("📦 Setting up project...")
        self._setup_project_and_copy_model()
        
        # Step 2: Build CPU version (with timing)
        print(f"\n🏗️  Building CPU version (measuring time)...")
        try:
            cpu_timing = self._build_cpu_version()
            build_results["cpu_build_timing"] = cpu_timing
            
            # Find CPU binary
            cpu_binary = self._find_bench_binary("cpu")
            build_results["cpu_bench_binary"] = cpu_binary
            print(f"✅ CPU build successful: {cpu_binary}")
            
        except Exception as e:
            print(f"❌ CPU build failed: {e}")
            build_results["cpu_build_error"] = str(e)
        
        # Step 3: Build Vulkan version (without timing, separate directory)
        print(f"\n🏗️  Building Vulkan version (no timing measurement)...")
        try:
            self._build_vulkan_version()
            
            # Find Vulkan binary
            vulkan_binary = self._find_bench_binary("vulkan")
            build_results["vulkan_bench_binary"] = vulkan_binary
            build_results["gpu_bench_binary"] = vulkan_binary
            print(f"✅ Vulkan build successful: {vulkan_binary}")
            
            # Check Vulkan device support
            vulkan_devices = self._check_vulkan_devices(vulkan_binary)
            build_results["vulkan_devices"] = vulkan_devices
            build_results["vulkan_supported"] = len(vulkan_devices) > 0 if vulkan_devices else False
            
        except Exception as e:
            print(f"❌ Vulkan build failed: {e}")
            build_results["vulkan_build_error"] = str(e)
        
        # Check if at least one build succeeded
        if not any(key.endswith("_bench_binary") for key in build_results.keys()):
            raise RuntimeError("All builds failed - cannot locate any benchmark binary")
        
        return build_results
    
    def _build_cpu_version(self) -> Dict[str, float]:
        """Build CPU-only version with timing."""
        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)
        
        # Configure CPU build
        print("⚙️  Configuring CPU build...")
        config_start = time.perf_counter()
        config_cmd = [
            "cmake", "-B", "build_cpu", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}",
            f"-DCMAKE_CXX_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}"
        ]
        print(f"🔧 Command: {' '.join(config_cmd)}")
        config_result = subprocess.run(config_cmd, cwd=self.project_dir, text=True, env=self._get_cold_build_env())
        config_time = time.perf_counter() - config_start
        
        if config_result.returncode != 0:
            raise subprocess.CalledProcessError(config_result.returncode, config_cmd)
        
        print(f"✅ CPU configuration completed in {config_time:.2f}s")
        
        # Build CPU version with output
        print(f"⚙️  Building CPU version (using {num_jobs} jobs)...")
        build_start = time.perf_counter()
        build_cmd = ["cmake", "--build", "build_cpu", "--config", "Release", "--", f"-j{num_jobs}"]
        print(f"🔧 Command: {' '.join(build_cmd)}")
        
        # Show build output in real-time
        process = subprocess.Popen(
            build_cmd, 
            cwd=self.project_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True,
            env=self._get_cold_build_env()
        )
        
        line_count = 0
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                line_count += 1
                # Show every 10th line or important messages
                if (line_count % 10 == 0 or 
                    any(keyword in line for keyword in ["Built target", "Linking", "error:", "Error", "llama-bench", "%"])):
                    print(f"📈 {line}")
        
        process.wait()
        build_time = time.perf_counter() - build_start
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, build_cmd)
        
        total_time = config_time + build_time
        print(f"✅ CPU build completed in {total_time:.2f}s (config: {config_time:.2f}s, build: {build_time:.2f}s)")
        
        return {
            "config_time_seconds": config_time,
            "build_time_seconds": build_time,
            "total_time_seconds": total_time
        }
    
    def _build_vulkan_version(self) -> None:
        """Build Vulkan version without timing (using separate directory)."""
        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)
        
        # Configure Vulkan build
        print("⚙️  Configuring Vulkan build...")
        config_cmd = [
            "cmake", "-B", "build_vulkan", "-DGGML_VULKAN=ON", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}",
            f"-DCMAKE_CXX_FLAGS=-DWEIRD_BENCH_BUILD_ID={self.build_id}"
        ]
        print(f"🔧 Command: {' '.join(config_cmd)}")
        config_result = subprocess.run(config_cmd, cwd=self.project_dir, text=True, env=self._get_cold_build_env())
        
        if config_result.returncode != 0:
            raise subprocess.CalledProcessError(config_result.returncode, config_cmd)
        
        print("✅ Vulkan configuration completed")
        
        # Build Vulkan version with output
        print(f"⚙️  Building Vulkan version (using {num_jobs} jobs)...")
        build_cmd = ["cmake", "--build", "build_vulkan", "--config", "Release", "--", f"-j{num_jobs}"]
        print(f"🔧 Command: {' '.join(build_cmd)}")
        
        # Show build output in real-time
        process = subprocess.Popen(
            build_cmd, 
            cwd=self.project_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True,
            env=self._get_cold_build_env()
        )
        
        line_count = 0
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                line_count += 1
                # Show every 10th line or important messages
                if (line_count % 10 == 0 or 
                    any(keyword in line for keyword in ["Built target", "Linking", "error:", "Error", "llama-bench", "%"])):
                    print(f"📈 {line}")
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, build_cmd)
        
        print("✅ Vulkan build completed")
    
    
    def _diagnose_cmake_error(self, result: subprocess.CompletedProcess, phase: str, variant_name: str) -> None:
        """Diagnose and report CMake build errors with helpful suggestions."""
        stderr = result.stderr or ""
        stdout = result.stdout or ""
        combined_output = stderr + stdout
        
        print(f"\n❌ CMake {phase} failed for {variant_name} build (exit code {result.returncode})")
        
        # Common error patterns and suggestions
        error_patterns = {
            "curl": {
                "message": "curl command not found",
                "suggestion": "Install curl: sudo zypper install curl libcurl-devel (openSUSE) or sudo apt-get install curl libcurl4-openssl-dev (Ubuntu/Debian) or brew install curl (macOS)"
            },
            "Could NOT find CURL": {
                "message": "CURL development headers not found",
                "suggestion": "Install CURL dev packages: sudo zypper install curl libcurl-devel (openSUSE) or sudo apt-get install curl libcurl4-openssl-dev (Ubuntu/Debian)"
            },
            "Could not find a package configuration file provided by \"Vulkan\"": {
                "message": "Vulkan SDK not found",
                "suggestion": "Install Vulkan SDK: sudo zypper install vulkan-devel (openSUSE) or from https://vulkan.lunarg.com/ or disable Vulkan with regular build"
            },
            "No CMAKE_CXX_COMPILER could be found": {
                "message": "C++ compiler not found",
                "suggestion": "Install build tools: sudo zypper install gcc-c++ cmake (openSUSE) or sudo apt-get install build-essential (Ubuntu/Debian) or xcode-select --install (macOS)"
            },
            "CMake Error": {
                "message": "CMake configuration error",
                "suggestion": "Check that all required dependencies are installed"
            }
        }
        
        # Look for known error patterns
        found_pattern = False
        for pattern, info in error_patterns.items():
            if pattern.lower() in combined_output.lower():
                print(f"🔍 Detected issue: {info['message']}")
                print(f"💡 Suggestion: {info['suggestion']}")
                found_pattern = True
                break
        
        if not found_pattern:
            print("🔍 Build error details:")
        
        # Show relevant error output (last 10 lines of stderr, then stdout)
        if stderr:
            stderr_lines = stderr.strip().split('\n')
            print("\nStderr (last 10 lines):")
            for line in stderr_lines[-10:]:
                print(f"  {line}")
        
        if stdout and not stderr:
            stdout_lines = stdout.strip().split('\n')
            print("\nStdout (last 10 lines):")
            for line in stdout_lines[-10:]:
                print(f"  {line}")
    
    def _find_bench_binary(self, build_type: str = "vulkan") -> str:
        """Find the llama-bench binary in the appropriate build directory."""
        if build_type == "cpu":
            build_dir = self.cpu_build_dir
        else:
            build_dir = self.vulkan_build_dir
            
        possible_paths = [
            os.path.join(build_dir, "bin", "llama-bench"),
            os.path.join(build_dir, "llama-bench"),
            os.path.join(self.project_dir, "llama-bench"),
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        raise RuntimeError(f"llama-bench binary not found in {build_type} build directory")
    
    def _check_vulkan_devices(self, bench_binary: str) -> Optional[List[str]]:
        """Check available Vulkan devices by running llama-bench."""
        try:
            # Run llama-bench without a model to see Vulkan device detection
            result = self.run_command([bench_binary], check=False)
            output = (result.stdout or "") + (result.stderr or "")
            
            # Parse Vulkan device detection output
            devices = []
            lines = output.split('\n')
            for line in lines:
                if "ggml_vulkan: Found" in line and "Vulkan devices:" in line:
                    # Extract number of devices
                    import re
                    match = re.search(r'Found (\d+) Vulkan devices:', line)
                    if match:
                        num_devices = int(match.group(1))
                        if num_devices > 0:
                            return [f"vulkan_device_{i}" for i in range(num_devices)]
                elif line.strip().startswith("ggml_vulkan:") and "=" in line:
                    # Parse individual device lines like "ggml_vulkan: 0 = AMD Radeon Graphics..."
                    devices.append(line.strip())
            
            return devices if devices else None
        except Exception:
            return None
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        """Run the Llama.cpp benchmarks."""
        # Note: We built both versions, but Vulkan build overwrote CPU build
        # So we need to rebuild CPU version for CPU benchmarks, then use Vulkan for GPU
        vulkan_supported = self.results["build"].get("vulkan_supported", False)
        
        results = {
            "runs_cpu": [],
            "runs_gpu": [],
        }
        
        # Limited test configurations as requested
        prompt_sizes = [512]
        generation_sizes = [64]
        
        # CPU benchmarks using CPU build
        if "cpu_bench_binary" in self.results["build"]:
            print("\n=== Running CPU benchmarks (using CPU build) ===")
            try:
                cpu_binary = self.results["build"]["cpu_bench_binary"]
                
                for p_size in prompt_sizes:
                    for g_size in generation_sizes:
                        print(f"Running CPU benchmark: prompt={p_size}, generation={g_size}")
                        
                        cmd = [
                            cpu_binary,
                            "-m", self.project_model_path,
                            "-p", str(p_size),
                            "-n", str(g_size)
                        ]
                        
                        result = self._run_benchmark_command(cmd, "cpu", p_size, g_size, 0)
                        results["runs_cpu"].append(result)
            except Exception as e:
                print(f"❌ Failed CPU benchmarking: {e}")
                results["cpu_skip_reason"] = "cpu_benchmark_failed"
        else:
            print("\n=== Skipping CPU benchmarks (CPU build failed) ===")
            results["cpu_skip_reason"] = "cpu_build_failed"
        
        # GPU benchmarks using Vulkan build
        no_gpu = args and getattr(args, 'no_gpu', False)
        if no_gpu:
            print(f"\n⏭️  Skipping GPU benchmarks (--no-gpu)")
            results["gpu_skip_reason"] = "no_gpu_flag_set"
        elif vulkan_supported and "vulkan_bench_binary" in self.results["build"]:
            print("\n=== Running GPU benchmarks (using Vulkan build) ===")
            try:
                gpu_binary = self.results["build"]["vulkan_bench_binary"]
                
                for p_size in prompt_sizes:
                    for g_size in generation_sizes:
                        print(f"Running GPU benchmark: prompt={p_size}, generation={g_size}")
                        
                        cmd = [
                            gpu_binary,
                            "-m", self.project_model_path,
                            "-p", str(p_size),
                            "-n", str(g_size)
                        ]
                        
                        result = self._run_benchmark_command(cmd, "gpu", p_size, g_size, 99)
                        results["runs_gpu"].append(result)
            except Exception as e:
                print(f"❌ Failed GPU benchmarking: {e}")
                results["gpu_skip_reason"] = "gpu_benchmark_failed"
        else:
            reason = "vulkan_not_supported" if not vulkan_supported else "vulkan_build_failed"
            print(f"\n=== Skipping GPU benchmarks ({reason.replace('_', ' ')}) ===")
            results["gpu_skip_reason"] = reason
        
        return results
    
    def _run_benchmark_command(self, cmd: List[str], run_type: str, prompt_size: int, 
                              generation_size: int, ngl: int) -> Dict[str, Any]:
        """Run a single benchmark command and parse results."""
        start_time = time.perf_counter()
        
        # Add JSON output flag at the start of the command
        json_cmd = [cmd[0]] + ["-o", "json"] + cmd[1:]
        
        try:
            print(f"🏃 Running {run_type} benchmark: prompt={prompt_size}, gen={generation_size}, ngl={ngl}")
            result = self.run_command(json_cmd, cwd=self.project_dir, check=False)
            elapsed_time = time.perf_counter() - start_time
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed: {result.stderr}")
                return {
                    "type": run_type,
                    "prompt_size": prompt_size,
                    "generation_size": generation_size,
                    "ngl": ngl,
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed_time,
                    "failed": True,
                    "error": result.stderr,
                    "metrics": {},
                }
            
            # Parse JSON output
            try:
                json_results = json.loads(result.stdout or "[]")
                metrics = self._parse_llama_bench_json(json_results, prompt_size, generation_size)
                
                return {
                    "type": run_type,
                    "prompt_size": prompt_size,
                    "generation_size": generation_size,
                    "ngl": ngl,
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed_time,
                    "metrics": metrics,
                    "raw_json": json_results,
                }
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse JSON output, falling back to text parsing: {e}")
                # Fallback to old text parsing
                metrics = self._parse_llama_bench_output(result.stdout or "")
                return {
                    "type": run_type,
                    "prompt_size": prompt_size,
                    "generation_size": generation_size,
                    "ngl": ngl,
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed_time,
                    "metrics": metrics,
                    "stdout_tail": (result.stdout or "")[-1000:],
                }
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"❌ Exception during benchmark: {e}")
            return {
                "type": run_type,
                "prompt_size": prompt_size,
                "generation_size": generation_size,
                "ngl": ngl,
                "returncode": -1,
                "elapsed_seconds": elapsed_time,
                "error": str(e),
                "metrics": {},
            }
    
    def _parse_llama_bench_json(self, json_results: List[Dict], prompt_size: int, 
                               generation_size: int) -> Dict[str, Any]:
        """Parse llama-bench JSON output to extract performance metrics."""
        metrics = {}
        
        if not json_results:
            return metrics
        
        # Find prompt and generation benchmark results
        prompt_result = None
        gen_result = None
        
        for result in json_results:
            if result.get("n_prompt", 0) == prompt_size and result.get("n_gen", 0) == 0:
                prompt_result = result
            elif result.get("n_prompt", 0) == 0 and result.get("n_gen", 0) == generation_size:
                gen_result = result
        
        # Extract system information from first result
        if json_results:
            first_result = json_results[0]
            metrics["system_info"] = {
                "cpu_info": first_result.get("cpu_info", "Unknown"),
                "gpu_info": first_result.get("gpu_info", "Unknown"),
                "backends": first_result.get("backends", "Unknown"),
                "model_type": first_result.get("model_type", "Unknown"),
                "model_size": first_result.get("model_size", 0),
                "model_n_params": first_result.get("model_n_params", 0),
                "n_threads": first_result.get("n_threads", 0),
                "n_gpu_layers": first_result.get("n_gpu_layers", 0),
            }
        
        # Extract prompt processing metrics
        if prompt_result:
            avg_ns = prompt_result.get("avg_ns", 0)
            avg_ts = prompt_result.get("avg_ts", 0)
            stddev_ns = prompt_result.get("stddev_ns", 0)
            stddev_ts = prompt_result.get("stddev_ts", 0)
            
            metrics["prompt_processing"] = {
                "avg_time_ns": avg_ns,
                "avg_tokens_per_sec": avg_ts,
                "stddev_time_ns": stddev_ns,
                "stddev_tokens_per_sec": stddev_ts,
                "avg_time_ms": avg_ns / 1_000_000 if avg_ns > 0 else 0,
                "samples_ns": prompt_result.get("samples_ns", []),
                "samples_ts": prompt_result.get("samples_ts", []),
            }
        
        # Extract generation metrics
        if gen_result:
            avg_ns = gen_result.get("avg_ns", 0)
            avg_ts = gen_result.get("avg_ts", 0)
            stddev_ns = gen_result.get("stddev_ns", 0)
            stddev_ts = gen_result.get("stddev_ts", 0)
            
            metrics["generation"] = {
                "avg_time_ns": avg_ns,
                "avg_tokens_per_sec": avg_ts,
                "stddev_time_ns": stddev_ns,
                "stddev_tokens_per_sec": stddev_ts,
                "avg_time_ms": avg_ns / 1_000_000 if avg_ns > 0 else 0,
                "samples_ns": gen_result.get("samples_ns", []),
                "samples_ts": gen_result.get("samples_ts", []),
            }
        
        # Calculate combined metrics for compatibility
        prompt_ts = metrics.get("prompt_processing", {}).get("avg_tokens_per_sec", 0)
        gen_ts = metrics.get("generation", {}).get("avg_tokens_per_sec", 0)
        
        # Use generation tokens/sec as primary metric (more relevant for inference)
        metrics["tokens_per_second"] = gen_ts if gen_ts > 0 else prompt_ts
        
        # Calculate total time
        prompt_time_ms = metrics.get("prompt_processing", {}).get("avg_time_ms", 0)
        gen_time_ms = metrics.get("generation", {}).get("avg_time_ms", 0)
        metrics["total_time_ms"] = prompt_time_ms + gen_time_ms
        
        return metrics
    
    def _parse_llama_bench_output(self, output: str) -> Dict[str, Any]:
        """Parse llama-bench output to extract performance metrics."""
        metrics = {}
        
        # Look for common llama-bench output patterns
        # These patterns depend on the actual output format of llama-bench
        
        # Example patterns (adjust based on actual llama-bench output):
        # tokens/s, ms/token, etc.
        
        # Tokens per second
        tokens_per_sec_match = re.search(r"(\d+\.?\d*)\s*tokens?/s", output, re.IGNORECASE)
        if tokens_per_sec_match:
            metrics["tokens_per_second"] = float(tokens_per_sec_match.group(1))
        
        # Milliseconds per token
        ms_per_token_match = re.search(r"(\d+\.?\d*)\s*ms/token", output, re.IGNORECASE)
        if ms_per_token_match:
            metrics["ms_per_token"] = float(ms_per_token_match.group(1))
        
        # Memory usage (if reported)
        memory_match = re.search(r"(\d+\.?\d*)\s*MB", output, re.IGNORECASE)
        if memory_match:
            metrics["memory_mb"] = float(memory_match.group(1))
        
        # Model load time
        load_time_match = re.search(r"load time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if load_time_match:
            metrics["load_time_ms"] = float(load_time_match.group(1))
        
        # Prompt processing time
        prompt_time_match = re.search(r"prompt eval time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if prompt_time_match:
            metrics["prompt_eval_time_ms"] = float(prompt_time_match.group(1))
        
        # Generation time
        gen_time_match = re.search(r"eval time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if gen_time_match:
            metrics["eval_time_ms"] = float(gen_time_match.group(1))
        
        return metrics