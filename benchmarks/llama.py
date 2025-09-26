#!/usr/bin/env python3
"""
Llama.cpp benchmark implementation with proper multi-GPU Vulkan detection
and optional build skipping for fast debugging.
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

                    if (line_count % 20 == 0 or
                        any(keyword in line for keyword in ["Built target", "Linking", "error:", "Error", "llama-bench"])):
                        print(f"📈 {line}")

            process.wait()

            class BuildResult:
                def __init__(self, returncode, stdout, stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            result = BuildResult(process.returncode, '\n'.join(stdout_lines))
        else:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)

        elapsed_time = time.perf_counter() - start_time
        print(f"✅ {phase_name} completed in {elapsed_time:.2f}s")

        return elapsed_time, result

    def build_variant_with_timing(self, variant_name: str, cmake_config_cmd: List[str],
                                  error_handler=None, setup_callback=None, env: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Build a variant with detailed timing and error handling."""
        print(f"🔨 Building {variant_name} variant...")

        self.clean_build_environment()

        if setup_callback:
            setup_callback()

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

        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)
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
        self.shared_models_dir = os.path.abspath("models")
        self.project_models_dir = os.path.join(self.project_dir, "models")
        self.model_url = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-IQ4_NL.gguf?download=true"
        self.model_filename = "Qwen3-4B-Instruct-2507-IQ4_NL.gguf"
        self.shared_model_path = os.path.join(self.shared_models_dir, self.model_filename)
        self.project_model_path = os.path.join(self.project_models_dir, self.model_filename)

        self.results["meta"]["repo"] = self.repo_url
        self.results["meta"]["model_url"] = self.model_url

        # GPU selection parameters
        self.gpu_device_index: Optional[int] = None
        self.vk_driver_files: Optional[str] = None
        self.available_gpus: List[Dict[str, Any]] = []

        import uuid
        self.build_id = str(uuid.uuid4())[:8]

    def _get_cold_build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env['CCACHE_DISABLE'] = '1'
        env['CCACHE_DIR'] = '/dev/null'
        env['CCACHE_NOSTATS'] = '1'
        env['CMAKE_C_COMPILER_LAUNCHER'] = ''
        env['CMAKE_CXX_COMPILER_LAUNCHER'] = ''
        return env

    def _check_dependencies(self) -> None:
        missing_deps = []
        required_commands = {
            "git": "git (for cloning repositories)",
            "cmake": "cmake (for building projects)",
            "curl": "curl (required by llama.cpp build system)",
        }
        for cmd, description in required_commands.items():
            if not shutil.which(cmd):
                missing_deps.append(description)

        cpp_compilers = ["g++", "clang++", "c++"]
        if not any(shutil.which(compiler) for compiler in cpp_compilers):
            missing_deps.append("C++ compiler (g++, clang++, or c++)")

        if missing_deps:
            error_msg = "Missing required dependencies:\n"
            for dep in missing_deps:
                error_msg += f"  - {dep}\n"
            error_msg += "\nFor openSUSE, install with:\n"
            error_msg += "  sudo zypper install git cmake curl libcurl-devel gcc-c++\n"
            error_msg += "\nOr run the provided installation script:\n"
            error_msg += "  ./install_dependencies_opensuse.sh\n"
            raise RuntimeError(error_msg)

    def setup(self, skip_build: bool = False) -> None:
        print("🔍 Checking system dependencies...")
        self._check_dependencies()
        print("✅ All required dependencies found!")

        print("🎮 Detecting available GPUs...")
        self.available_gpus = self._detect_available_gpus()
        if self.available_gpus:
            print(f"✅ Found {len(self.available_gpus)} Vulkan GPU(s):")
            for gpu in self.available_gpus:
                print(f"   • Device {gpu['index']}: {gpu['name']} (Driver: {gpu.get('driver','unknown')})")
        else:
            print("⚠️  No Vulkan GPUs detected")

        os.makedirs(self.shared_models_dir, exist_ok=True)

        if not os.path.exists(self.shared_model_path):
            print(f"📥 Downloading model {self.model_filename} to shared location...")
            self._download_model_to_shared_location()
        else:
            print(f"✅ Model {self.model_filename} already exists in shared location, skipping download")
            file_size = os.path.getsize(self.shared_model_path) // (1024 * 1024)
            print(f"📄 Model file size: {file_size} MB")

        if skip_build:
            # Ensure project dir exists to find binaries
            if not os.path.isdir(self.project_dir):
                print("⚠️  --skip-build requested but project directory does not exist. Cloning only.")
                self._setup_project_only()
            # Ensure model in project
            os.makedirs(self.project_models_dir, exist_ok=True)
            if not os.path.exists(self.project_model_path) and os.path.exists(self.shared_model_path):
                shutil.copy2(self.shared_model_path, self.project_model_path)

    def _download_model_to_shared_location(self) -> None:
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

    def _setup_project_only(self) -> None:
        print(f"📂 Cloning {self.repo_url} -> {self.project_dir}")
        try:
            self.run_command(["git", "clone", "--depth", "1", self.repo_url, self.project_dir])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")

    def _setup_project_and_copy_model(self) -> None:
        self._setup_project_only()
        os.makedirs(self.project_models_dir, exist_ok=True)
        print("📋 Copying model from shared location...")
        shutil.copy2(self.shared_model_path, self.project_model_path)
        print(f"✅ Model copied to {self.project_model_path}")

    def build(self, args: Any = None) -> Dict[str, Any]:
        build_results: Dict[str, Any] = {}

        print(f"\n{'='*60}")
        print(f"🚀 Building llama.cpp with CPU and Vulkan versions")
        print(f"{'='*60}")

        skip_build = bool(args and getattr(args, "skip_build", False))
        if skip_build:
            print("⏭️  --skip-build: skipping compilation and reusing existing binaries if present.")

        # If skipping build, only ensure repo+model exist and probe binaries
        if skip_build:
            self.setup(skip_build=True)
            cpu_binary = self._find_bench_binary("cpu", must_exist=False)
            vulkan_binary = self._find_bench_binary("vulkan", must_exist=False)
            if cpu_binary:
                print(f"✅ Found CPU bench binary: {cpu_binary}")
                build_results["cpu_bench_binary"] = cpu_binary
            else:
                print("⚠️  CPU bench binary not found (expected at build_cpu/bin/llama-bench).")

            if not getattr(args, "no_gpu", False) and vulkan_binary:
                print(f"✅ Found Vulkan bench binary: {vulkan_binary}")
                build_results["vulkan_bench_binary"] = vulkan_binary
                build_results["gpu_bench_binary"] = vulkan_binary
                vulkan_devices = self._check_vulkan_devices(vulkan_binary)
                build_results["vulkan_devices"] = vulkan_devices
                build_results["vulkan_supported"] = bool(vulkan_devices and len(vulkan_devices) > 0)
            else:
                print("ℹ️  Vulkan bench binary not found or --no-gpu set; skipping Vulkan probing.")
                build_results["vulkan_supported"] = False

            if not any(k.endswith("_bench_binary") for k in build_results.keys()):
                raise RuntimeError("No existing benchmark binaries found with --skip-build.")
            return build_results

        # Clean build
        print("🧹 Cleaning build environment...")
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

        print("📦 Setting up project...")
        self._setup_project_and_copy_model()

        # CPU build with timing
        print(f"\n🏗️  Building CPU version (measuring time)...")
        try:
            cpu_timing = self._build_cpu_version()
            build_results["cpu_build_timing"] = cpu_timing
            cpu_binary = self._find_bench_binary("cpu")
            build_results["cpu_bench_binary"] = cpu_binary
            print(f"✅ CPU build successful: {cpu_binary}")
        except Exception as e:
            print(f"❌ CPU build failed: {e}")
            build_results["cpu_build_error"] = str(e)

        # Vulkan build
        print(f"\n🏗️  Building Vulkan version (no timing measurement)...")
        try:
            self._build_vulkan_version()
            vulkan_binary = self._find_bench_binary("vulkan")
            build_results["vulkan_bench_binary"] = vulkan_binary
            build_results["gpu_bench_binary"] = vulkan_binary
            print(f"✅ Vulkan build successful: {vulkan_binary}")

            vulkan_devices = self._check_vulkan_devices(vulkan_binary)
            build_results["vulkan_devices"] = vulkan_devices
            build_results["vulkan_supported"] = bool(vulkan_devices and len(vulkan_devices) > 0)
        except Exception as e:
            print(f"❌ Vulkan build failed: {e}")
            build_results["vulkan_build_error"] = str(e)

        if not any(key.endswith("_bench_binary") for key in build_results.keys()):
            raise RuntimeError("All builds failed - cannot locate any benchmark binary")

        return build_results

    def _build_cpu_version(self) -> Dict[str, float]:
        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)

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

        print(f"⚙️  Building CPU version (using {num_jobs} jobs)...")
        build_start = time.perf_counter()
        build_cmd = ["cmake", "--build", "build_cpu", "--config", "Release", "--", f"-j{num_jobs}"]
        print(f"🔧 Command: {' '.join(build_cmd)}")

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
        import multiprocessing
        num_jobs = min(multiprocessing.cpu_count(), 20)

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

        print(f"⚙️  Building Vulkan version (using {num_jobs} jobs)...")
        build_cmd = ["cmake", "--build", "build_vulkan", "--config", "Release", "--", f"-j{num_jobs}"]
        print(f"🔧 Command: {' '.join(build_cmd)}")

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
                if (line_count % 10 == 0 or
                    any(keyword in line for keyword in ["Built target", "Linking", "error:", "Error", "llama-bench", "%"])):
                    print(f"📈 {line}")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, build_cmd)

        print("✅ Vulkan build completed")

    # ---------- Device detection & selection ----------

    def _vulkaninfo_json(self, extra_env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Run `vulkaninfo --json` and return parsed dict (or None)."""
        if not shutil.which("vulkaninfo"):
            return None
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        try:
            r = subprocess.run(
                ["vulkaninfo", "--json"],
                capture_output=True, text=True, env=env, timeout=30
            )
            if r.returncode != 0:
                return None
            data = r.stdout.strip()
            return json.loads(data)
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def _detect_available_gpus(self) -> List[Dict[str, Any]]:
        """
        Enumerate ALL Vulkan physical devices via vulkaninfo --json.
        Index corresponds directly to GGML_VULKAN_DEVICE.
        No ICD forcing, 'icd_path' is always None (kept for dict shape compatibility).
        """
        gpus: List[Dict[str, Any]] = []

        # Try JSON first
        info = self._vulkaninfo_json()
        print("Vulkaninfo JSON output:", info)
        if info and "physicalDevices" in info:
            for idx, dev in enumerate(info["physicalDevices"]):
                print("Physical Device:", dev)
                props = dev.get("properties", {})
                name = props.get("deviceName", f"Vulkan Device {idx}")
                driver_name = props.get("driverName") or props.get("driverID") or "unknown"
                gpus.append({
                    "index": idx,
                    "name": name,
                    "driver": str(driver_name),
                    "icd_path": None,  # never force an ICD
                })

        # Fallback: --summary
        if not gpus:
            try:
                r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
                if r.returncode == 0:
                    for line in r.stdout.splitlines():
                        m = re.match(r"\s*GPU\s*([0-9]+)\s*:\s*(.+)", line)
                        if m:
                            i = int(m.group(1))
                            name = m.group(2).strip()
                            print("Summary Device:", i, name)
                            gpus.append({
                                "index": i,
                                "name": name,
                                "driver": "unknown",
                                "icd_path": None,
                            })
            except Exception as e:
                print(f"Exception: {e}")

        # Last fallback: no devices
        return gpus


    def _list_vulkan_icd_files(self) -> List[str]:
        icd_paths = []
        for icd_dir in ['/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d', '/usr/local/share/vulkan/icd.d']:
            if os.path.exists(icd_dir):
                try:
                    for file in os.listdir(icd_dir):
                        if file.endswith('.json'):
                            icd_paths.append(os.path.join(icd_dir, file))
                except PermissionError:
                    continue
        return icd_paths

    def set_gpu_selection(self, gpu_device_index: Optional[int] = None, vk_driver_files: Optional[str] = None) -> None:
        """
        Configure which Vulkan device to use.
        Prefer setting only GGML_VULKAN_DEVICE. Avoid forcing VK_DRIVER_FILES unless truly necessary.
        """
        self.gpu_device_index = gpu_device_index
        self.vk_driver_files = vk_driver_files

        if gpu_device_index is not None:
            print(f"🎯 Selected GPU device index: {gpu_device_index}")
            if self.available_gpus:
                matching = [g for g in self.available_gpus if g['index'] == gpu_device_index]
                if matching:
                    g = matching[0]
                    print(f"   📋 GPU details: {g['name']} (Driver: {g.get('driver','unknown')})")
                else:
                    print(f"   ⚠️  Warning: No GPU found with index {gpu_device_index}")

        if vk_driver_files:
            print(f"🔧 Forcing Vulkan driver: {vk_driver_files}")
            if not os.path.exists(vk_driver_files):
                print(f"   ⚠️  Warning: ICD file not found: {vk_driver_files}")

    def _get_gpu_env_vars(self) -> Dict[str, str]:
        """
        Only set GGML_VULKAN_DEVICE by default.
        Do NOT set VK_DRIVER_FILES unless explicitly requested by user CLI.
        """
        env: Dict[str, str] = {}
        if self.gpu_device_index is not None:
            env['GGML_VULKAN_DEVICE'] = str(self.gpu_device_index)
        if self.vk_driver_files:
            # user explicitly asked to lock to an ICD (e.g., mixing vendors)
            env['VK_DRIVER_FILES'] = self.vk_driver_files
            env['VK_ICD_FILENAMES'] = self.vk_driver_files
        return env

    # ---------- Running ----------

    def run(self, args: Any = None) -> Dict[str, Any]:
        self.setup(skip_build=bool(args and getattr(args, "skip_build", False)))
        self.results["build"] = self.build(args)

        return self.benchmark(args)

    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        vulkan_supported = self.results["build"].get("vulkan_supported", False)

        results = {
            "runs_cpu": [],
            "runs_gpu": [],
        }

        if self.gpu_device_index is not None or self.vk_driver_files:
            results["gpu_selection"] = {
                "device_index": self.gpu_device_index,
                "vk_driver_files": self.vk_driver_files,
                "available_gpus": self.available_gpus
            }

        prompt_sizes = [512]
        generation_sizes = [64]

        # CPU
        if "cpu_bench_binary" in self.results["build"]:
            print("\n=== Running CPU benchmarks (using CPU build) ===")
            try:
                cpu_binary = self.results["build"]["cpu_bench_binary"]
                for p_size in prompt_sizes:
                    for g_size in generation_sizes:
                        print(f"Running CPU benchmark: prompt={p_size}, generation={g_size}")
                        cmd = [cpu_binary, "-m", self.project_model_path, "-p", str(p_size), "-n", str(g_size)]
                        result = self._run_benchmark_command(cmd, "cpu", p_size, g_size, 0)
                        results["runs_cpu"].append(result)
            except Exception as e:
                print(f"❌ Failed CPU benchmarking: {e}")
                results["cpu_skip_reason"] = "cpu_benchmark_failed"
        else:
            print("\n=== Skipping CPU benchmarks (CPU build failed or missing) ===")
            results["cpu_skip_reason"] = "cpu_build_failed"

        # GPU
        no_gpu = args and getattr(args, 'no_gpu', False)
        if no_gpu:
            print(f"\n⏭️  Skipping GPU benchmarks (--no-gpu)")
            results["gpu_skip_reason"] = "no_gpu_flag_set"
            return results

        if not vulkan_supported or "vulkan_bench_binary" not in self.results["build"]:
            reason = "vulkan_not_supported" if not vulkan_supported else "vulkan_build_failed"
            print(f"\n=== Skipping GPU benchmarks ({reason.replace('_', ' ')}) ===")
            results["gpu_skip_reason"] = reason
            return results

        print("\n=== Running GPU benchmarks (using Vulkan build) ===")
        gpu_binary = self.results["build"]["vulkan_bench_binary"]

        # Select GPUs to test
        if self.gpu_device_index is not None:
            gpus_to_test = [g for g in self.available_gpus if g['index'] == self.gpu_device_index]
            if not gpus_to_test and self.available_gpus:
                print(f"⚠️  Selected GPU index {self.gpu_device_index} not found. Falling back to device 0.")
                gpus_to_test = [self.available_gpus[0]]
        else:
            gpus_to_test = self.available_gpus

        if not gpus_to_test:
            print("⚠️  No GPUs available for testing")
            results["gpu_skip_reason"] = "no_gpus_available"
            return results

        print(f"🎯 Testing {len(gpus_to_test)} GPU(s): {[gpu['name'] for gpu in gpus_to_test]}")

        for gpu_device in gpus_to_test:
            print(f"\n🔄 Testing GPU: {gpu_device['name']} (Index: {gpu_device['index']})")
            # Only set GGML_VULKAN_DEVICE per device; do not force driver unless user asked
            self.set_gpu_selection(gpu_device['index'], self.vk_driver_files if self.vk_driver_files else None)

            for p_size in prompt_sizes:
                for g_size in generation_sizes:
                    print(f"Running GPU benchmark: prompt={p_size}, generation={g_size}")
                    cmd = [gpu_binary, "-m", self.project_model_path, "-p", str(p_size), "-n", str(g_size)]
                    gpu_env = self._get_gpu_env_vars()
                    if gpu_env:
                        print(f"🎯 GPU selection: {gpu_env}")
                    result = self._run_benchmark_command(cmd, "gpu", p_size, g_size, 99, gpu_env)
                    result["gpu_device"] = gpu_device
                    results["runs_gpu"].append(result)

            print(f"✅ Completed testing GPU: {gpu_device['name']}")

        # restore selection silently (no extra noisy print)
        self.gpu_device_index = None  # keep clean
        return results

    # ---------- Helpers ----------

    def _run_benchmark_command(self, cmd: List[str], run_type: str,
                               prompt_size: int, generation_size: int, ngl: int,
                               gpu_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        json_cmd = [cmd[0]] + ["-o", "json"] + cmd[1:]

        try:
            print(f"🏃 Running {run_type} benchmark: prompt={prompt_size}, gen={generation_size}, ngl={ngl}")
            env = os.environ.copy()
            if gpu_env:
                env.update(gpu_env)
                print(f"🎯 Using GPU environment: {gpu_env}")

            result = self.run_command_with_env(json_cmd, env, cwd=self.project_dir, check=False)
            elapsed_time = time.perf_counter() - start_time

            if result.returncode != 0:
                print(f"❌ Benchmark failed: {result.stderr}")
                return {
                    "type": run_type, "prompt_size": prompt_size, "generation_size": generation_size,
                    "ngl": ngl, "returncode": result.returncode, "elapsed_seconds": elapsed_time,
                    "failed": True, "error": result.stderr, "metrics": {},
                }

            try:
                json_results = json.loads(result.stdout or "[]")
                metrics = self._parse_llama_bench_json(json_results, prompt_size, generation_size)
                return {
                    "type": run_type, "prompt_size": prompt_size, "generation_size": generation_size,
                    "ngl": ngl, "returncode": result.returncode, "elapsed_seconds": elapsed_time,
                    "metrics": metrics, "raw_json": json_results,
                }
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse JSON output, falling back to text parsing: {e}")
                metrics = self._parse_llama_bench_output(result.stdout or "")
                return {
                    "type": run_type, "prompt_size": prompt_size, "generation_size": generation_size,
                    "ngl": ngl, "returncode": result.returncode, "elapsed_seconds": elapsed_time,
                    "metrics": metrics, "stdout_tail": (result.stdout or "")[-1000:],
                }

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"❌ Exception during benchmark: {e}")
            return {
                "type": run_type, "prompt_size": prompt_size, "generation_size": generation_size,
                "ngl": ngl, "returncode": -1, "elapsed_seconds": elapsed_time,
                "error": str(e), "metrics": {},
            }

    def _parse_llama_bench_json(self, json_results: List[Dict],
                                prompt_size: int, generation_size: int) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if not json_results:
            return metrics

        prompt_result = None
        gen_result = None
        for result in json_results:
            if result.get("n_prompt", 0) == prompt_size and result.get("n_gen", 0) == 0:
                prompt_result = result
            elif result.get("n_prompt", 0) == 0 and result.get("n_gen", 0) == generation_size:
                gen_result = result

        if json_results:
            first = json_results[0]
            metrics["system_info"] = {
                "cpu_info": first.get("cpu_info", "Unknown"),
                "gpu_info": first.get("gpu_info", "Unknown"),
                "backends": first.get("backends", "Unknown"),
                "model_type": first.get("model_type", "Unknown"),
                "model_size": first.get("model_size", 0),
                "model_n_params": first.get("model_n_params", 0),
                "n_threads": first.get("n_threads", 0),
                "n_gpu_layers": first.get("n_gpu_layers", 0),
            }

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

        prompt_ts = metrics.get("prompt_processing", {}).get("avg_tokens_per_sec", 0)
        gen_ts = metrics.get("generation", {}).get("avg_tokens_per_sec", 0)
        metrics["tokens_per_second"] = gen_ts if gen_ts > 0 else prompt_ts

        prompt_time_ms = metrics.get("prompt_processing", {}).get("avg_time_ms", 0)
        gen_time_ms = metrics.get("generation", {}).get("avg_time_ms", 0)
        metrics["total_time_ms"] = prompt_time_ms + gen_time_ms
        return metrics

    def _parse_llama_bench_output(self, output: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        m = re.search(r"(\d+\.?\d*)\s*tokens?/s", output, re.IGNORECASE)
        if m:
            metrics["tokens_per_second"] = float(m.group(1))
        m = re.search(r"(\d+\.?\d*)\s*ms/token", output, re.IGNORECASE)
        if m:
            metrics["ms_per_token"] = float(m.group(1))
        m = re.search(r"(\d+\.?\d*)\s*MB", output, re.IGNORECASE)
        if m:
            metrics["memory_mb"] = float(m.group(1))
        m = re.search(r"load time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if m:
            metrics["load_time_ms"] = float(m.group(1))
        m = re.search(r"prompt eval time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if m:
            metrics["prompt_eval_time_ms"] = float(m.group(1))
        m = re.search(r"eval time\s*[=:]\s*(\d+\.?\d*)\s*ms", output, re.IGNORECASE)
        if m:
            metrics["eval_time_ms"] = float(m.group(1))
        return metrics

    def _find_bench_binary(self, build_type: str = "vulkan", must_exist: bool = True) -> Optional[str]:
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
        if must_exist:
            raise RuntimeError(f"llama-bench binary not found in {build_type} build directory")
        return None

    def _check_vulkan_devices(self, bench_binary: str) -> Optional[List[str]]:
        try:
            res = self.run_command([bench_binary], check=False)
            output = (res.stdout or "") + (res.stderr or "")
            devices = []
            lines = output.split('\n')
            for line in lines:
                if "ggml_vulkan: Found" in line and "Vulkan devices:" in line:
                    m = re.search(r'Found (\d+) Vulkan devices:', line)
                    if m:
                        num = int(m.group(1))
                        if num > 0:
                            return [f"vulkan_device_{i}" for i in range(num)]
                elif line.strip().startswith("ggml_vulkan:") and "=" in line:
                    devices.append(line.strip())
            return devices if devices else None
        except Exception:
            return None
