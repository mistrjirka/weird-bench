# Weird Bench - Multi-Benchmark Performance Testing Suite

A comprehensive benchmarking suite that supports multiple performance testing scenarios with automated compilation, execution, and visualization.

## Architecture

The benchmark suite is built with a modular architecture featuring efficient resource management:

```
weird-bench/
├── benchmarks/             # Individual benchmark implementations
│   ├── __init__.py
│   ├── base.py            # Base benchmark class
│   ├── reversan.py        # Reversan Engine benchmark
│   ├── llama.py           # Llama.cpp benchmark
│   ├── blender.py         # Blender 3D rendering benchmark
│   └── sevenzip.py        # 7-Zip compression benchmark
├── results/               # Benchmark results (JSON files)
├── result_plots/          # Generated plots and visualizations
├── models/                # Cached ML models (auto-created)
├── install_deps_arch.sh   # Arch Linux dependency installer
├── install_deps_ubuntu.sh # Ubuntu/Debian dependency installer
└── install_deps_fedora.sh # Fedora/RHEL dependency installer
```

### Key Architecture Features

- **Shared Model Storage**: Large models are downloaded once to `models/` and copied as needed
- **Clean Build Environment**: Each compilation gets a fresh, isolated environment
- **Unified Build Toolbox**: `CompilationToolbox` provides consistent build timing and error handling
- **Detailed Timing**: Separate measurement of configuration and compilation phases
- **Resource Efficiency**: No redundant downloads or builds
- **Enhanced Error Reporting**: Specific diagnostics for common build failures (curl, Vulkan, etc.)

## Supported Benchmarks

### 1. Reversan Engine Benchmark
- **Repository**: https://github.com/Saniel0/Reversan-Engine.git
- **Tests**:
  - Depth sweep (1-12) performance analysis
  - Thread scaling performance (1 to max CPU threads - automatically detected)
- **Metrics**: Execution time, memory usage (with GNU time)
- **Compilation**: CMake with Release configuration

### 2. Llama.cpp Benchmark
- **Repository**: https://github.com/ggml-org/llama.cpp
- **Model**: Qwen3-4B-Instruct-2507-IQ4_NL.gguf (auto-downloaded to shared `models/` directory)
- **Tests**:
  - CPU performance (ngl=0): Various prompt/generation size combinations
  - GPU performance (ngl=99): Same configurations with Vulkan backend
  - Build time comparison: Regular vs Vulkan builds with detailed timing breakdown
- **Configurations**:
  - Prompt sizes: 512, 1024, 2048 tokens
  - Generation sizes: 64, 128, 256 tokens
- **Compilation**: 
  - Unified CompilationToolbox for clean, timed builds
  - Each build variant gets a fresh environment
  - Detailed timing: configuration phase + compilation phase
  - Model copied from shared location (no re-download between builds)

### 3. Blender 3D Rendering Benchmark
- **Benchmark Suite**: Official Blender Benchmark 2.0 (auto-downloaded)
- **Tests**:
  - Automatic device detection (CPU, GPU with various frameworks like HIP, CUDA)
  - Scene rendering: monster, junkshop, classroom
  - Performance comparison across all available devices
- **Output**: JSON results with samples per minute for each scene
- **Version**: Blender 4.5.0 with CLI automation

### 4. 7-Zip Compression Benchmark
- **Tests**:
  - Multi-threaded compression performance (1 to max CPU threads)
  - Realistic test data creation (text and binary files)
  - Thread scaling efficiency analysis
- **Metrics**: Compression time, archive size, speedup calculations
- **Requirements**: System-installed 7-Zip (p7zip-full package)

## Usage

### Available Benchmarks Summary

- **reversan**: Game engine performance with depth and thread scaling
- **llama**: LLM inference with CPU/GPU comparison and build timing
- **blender**: 3D rendering across all available devices (CPU, GPU)
- **7zip**: Compression performance with multi-threading analysis

### Installation

Install dependencies using the provided scripts:

```bash
# Arch Linux
./install_deps_arch.sh

# Ubuntu/Debian  
./install_deps_ubuntu.sh

# Fedora/RHEL
./install_deps_fedora.sh

# Test error handling improvements
python3 test_error_handling.py
```

### Run Specific Benchmarks
```bash
# Run only Reversan benchmark
python run_benchmarks.py --benchmark reversan --runs 5

# Run only Llama.cpp benchmark
python run_benchmarks.py --benchmark llama

# Run specific benchmark with complete pipeline
python run_complete_benchmark.py --benchmark llama
```

### Generate Plots Only
```bash
# Generate plots from existing results
python plot_all_results.py

# Custom results/output directories
python plot_all_results.py --results-dir custom_results --output-dir custom_plots
```

### List Available Benchmarks
```bash
python run_benchmarks.py --list
```

## Results Structure

### Individual Benchmark Results
Each benchmark generates its own JSON file in the `results/` directory:
- `reversan_results.json` - Reversan Engine results
- `llama_results.json` - Llama.cpp results

### Combined Results
When running all benchmarks, a combined result file is generated:
- `all_benchmarks_results.json` - All benchmark results with metadata

### Generated Plots
Plots are saved in both PNG and SVG formats in `result_plots/`:

**Reversan Engine**:
- `reversan_depth_performance.*` - Performance vs search depth
- `reversan_threads_performance.*` - Thread scaling analysis
- `reversan_benchmark_summary.*` - Combined summary

**Llama.cpp**:
- `llama_cpu_gpu_comparison.*` - CPU vs GPU performance
- `llama_performance_matrix.*` - Performance heatmap by configuration
- `llama_build_times.*` - Build time comparison

## Requirements

### System Dependencies
- Python 3.7+
- Git
- CMake
- C++ compiler (GCC/Clang)
- Optional: GNU time for detailed metrics
- Optional: Vulkan SDK for GPU acceleration

### Python Dependencies
```bash
pip install matplotlib numpy
```

### Hardware Requirements
- **Reversan**: CPU with multiple cores for thread testing
- **Llama.cpp**: 
  - ~4GB+ RAM for the 4B parameter model
  - GPU with Vulkan support for GPU benchmarks
  - ~10GB disk space for model download

## Configuration

### Benchmark Parameters

**Reversan**:
- Depth range: 1-12 (configurable in `benchmarks/reversan.py`)
- Thread range: 1-8 (configurable in `benchmarks/reversan.py`)
- Repetitions: Configurable via `--runs` parameter

**Llama.cpp**:
- Prompt sizes: 512, 1024, 2048 tokens (configurable in `benchmarks/llama.py`)
- Generation sizes: 64, 128, 256 tokens (configurable in `benchmarks/llama.py`)
- Model: Auto-downloaded Qwen3-4B model (URL configurable)

## Extending the Suite

### Adding New Benchmarks

1. Create a new file in `benchmarks/` directory (e.g., `benchmarks/mybench.py`)
2. Inherit from `BaseBenchmark` class:

```python
from .base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def __init__(self, output_dir: str = "results"):
        super().__init__("mybench", output_dir)
    
    def setup(self) -> None:
        # Setup code (clone repos, etc.)
        pass
    
    def build(self) -> Dict[str, Any]:
        # Build code and return metrics
        return {"build_time": 123.45}
    
    def benchmark(self, args: Any = None) -> Dict[str, Any]:
        # Run benchmarks and return results
        return {"test_results": [...]}
```

3. Register in `run_benchmarks.py`:
```python
from benchmarks.mybench import MyBenchmark

# Add to benchmarks dictionary
self.benchmarks = {
    "reversan": ReversanBenchmark,
    "llama": LlamaBenchmark,
    "mybench": MyBenchmark  # Add this line
}
```

4. Add plotting support in `plot_all_results.py`

## Performance Tips

1. **Use GNU time**: Install GNU time for detailed memory usage metrics
2. **SSD recommended**: Compilation and model loading benefit from fast storage
3. **Multiple runs**: Use `--runs` parameter for more stable Reversan results
4. **GPU memory**: Ensure sufficient VRAM for large Llama.cpp models
5. **Parallel builds**: CMake uses all available cores by default

## Troubleshooting

### Common Issues
- **CMake not found**: Install CMake development tools
- **Model download fails**: Check internet connection and disk space
- **Vulkan not available**: Install Vulkan SDK or run CPU-only tests
- **Memory errors**: Reduce model size or increase system RAM
- **Plot generation fails**: Install matplotlib and numpy

### Debug Mode
Set verbose output by modifying the benchmark classes or adding debug prints.

## Legacy Support

### Installation Scripts

Dependency installation scripts are provided for major Linux distributions:

```bash
# Arch Linux
./install_deps_arch.sh

# Ubuntu/Debian
./install_deps_ubuntu.sh

# Fedora/RHEL
./install_deps_fedora.sh
```

These scripts install all required dependencies including:
- Development tools (cmake, git, build-essential)
- Python packages (matplotlib, numpy, requests, pexpect)
- 7-Zip for compression benchmarks
- Vulkan development libraries for GPU acceleration

## License

This benchmarking suite is provided as-is. Individual benchmark targets (Reversan Engine, Llama.cpp) are subject to their respective licenses.