# NUMA Options for Llama Benchmark

The unified runner now supports NUMA (Non-Uniform Memory Access) options for the llama benchmark to optimize performance on multi-socket systems.

## Available Options

### 1. `--numa-distribute`
Uses `numactl --interleave=all` to distribute memory across all NUMA nodes.

**Example:**
```bash
python3 unified_runner.py --benchmark llama --numa-distribute
```

This will run llama-bench with:
```bash
numactl --interleave=all ./llama-bench -m model.gguf -p 512 -n 64 --numa distribute
```

### 2. `--numa-isolate`
Uses NUMA isolate mode to bind to specific nodes.

**Example:**
```bash
python3 unified_runner.py --benchmark llama --numa-isolate
```

### 3. `--numactl='custom command'`
Allows you to specify a custom numactl command prefix.

**Example:**
```bash
python3 unified_runner.py --benchmark llama --numactl='numactl --interleave=all'
```

This gives you full control over the numactl parameters.

## Advanced Example

For a 48-core system with specific NUMA requirements:

```bash
python3 unified_runner.py --benchmark llama \
  --numactl='numactl --interleave=all' \
  --skip-build
```

This will execute:
```bash
numactl --interleave=all ./llama-bench -m model.gguf -t 48 --numa distribute -o json
```

## Notes

- NUMA options only apply to the llama benchmark
- The `--numa` flag is automatically added to llama-bench when using NUMA options
- Custom `--numactl` command overrides `--numa-distribute` and `--numa-isolate`
