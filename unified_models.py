#!/usr/bin/env python3
"""
Unified benchmark data models for consistent hardware-aware benchmarking.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
import json
import yaml
from datetime import datetime


@dataclass
class HardwareDevice:
    """Represents a single hardware device (CPU or GPU)."""
    hw_id: str  # Global hardware ID (cpu-0, gpu-0, gpu-1, etc.)
    name: str   # Full device name
    type: str   # "cpu" or "gpu"
    manufacturer: str  # "Intel", "AMD", "NVIDIA", etc.
    # Additional optional fields
    cores: Optional[int] = None
    threads: Optional[int] = None
    framework: Optional[str] = None  # For GPUs: "CUDA", "OPENCL", "VULKAN", etc.
    driver_version: Optional[str] = None
    memory_mb: Optional[int] = None


@dataclass
class SystemInfo:
    """System information and hardware inventory."""
    platform: str
    host: str
    timestamp: float
    cpu_only: bool  # Whether this benchmark run was CPU-only
    hardware: Dict[str, HardwareDevice]  # hw_id -> HardwareDevice mapping
    
    def get_cpu_device(self) -> Optional[HardwareDevice]:
        """Get the primary CPU device."""
        for device in self.hardware.values():
            if device.type == "cpu":
                return device
        return None
    
    def get_gpu_devices(self) -> List[HardwareDevice]:
        """Get all GPU devices."""
        return [device for device in self.hardware.values() if device.type == "gpu"]


# Llama Benchmark Results
@dataclass
class LlamaRunResult:
    """Single Llama inference run result."""
    prompt_speed: float  # tokens per second for prompt processing
    generation_speed: float  # tokens per second for generation
    hw_id: str  # Hardware device used for this run


@dataclass
class LlamaBenchmarkResult:
    """Llama benchmark results with CPU/GPU separation."""
    compile_time: float  # Build/compilation time in seconds
    cpu_benchmark: Optional[LlamaRunResult] = None
    gpu_benchmarks: List[LlamaRunResult] = None
    
    def __post_init__(self):
        if self.gpu_benchmarks is None:
            self.gpu_benchmarks = []


# Reversan Benchmark Results
@dataclass
class ReversanDepthResult:
    """Single depth benchmark result."""
    depth: int
    time_seconds: float
    memory_kb: int


@dataclass
class ReversanThreadResult:
    """Single thread benchmark result."""
    threads: int
    time_seconds: float
    memory_kb: int


@dataclass
class ReversanBenchmarkResult:
    """Reversan benchmark results with depth and thread benchmarks."""
    compile_time: float
    depth_benchmarks: List[ReversanDepthResult]
    thread_benchmarks: List[ReversanThreadResult]


# 7zip Benchmark Results
@dataclass
class SevenZipBenchmarkResult:
    """7zip benchmark results using internal benchmark."""
    usage_percent: float  # CPU usage percentage
    ru_mips: float       # R/U MIPS (per core)
    total_mips: float    # Total MIPS


# Blender Benchmark Results
@dataclass
class BlenderSceneResult:
    """Result for a single Blender scene."""
    classroom: Optional[float] = None  # samples per minute
    junkshop: Optional[float] = None   # samples per minute
    monster: Optional[float] = None    # samples per minute


@dataclass
class BlenderDeviceResult:
    """Blender results for a specific device."""
    hw_id: str
    scenes: BlenderSceneResult


@dataclass
class BlenderBenchmarkResult:
    """Blender benchmark results with CPU/GPU separation."""
    cpu: Optional[BlenderSceneResult] = None
    gpus: List[BlenderDeviceResult] = None
    
    def __post_init__(self):
        if self.gpus is None:
            self.gpus = []


# Main unified benchmark result
@dataclass
class UnifiedBenchmarkResult:
    """Main unified benchmark result containing all benchmark data."""
    meta: SystemInfo
    llama: Optional[LlamaBenchmarkResult] = None
    reversan: Optional[ReversanBenchmarkResult] = None
    sevenzip: Optional[SevenZipBenchmarkResult] = None
    blender: Optional[BlenderBenchmarkResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedBenchmarkResult':
        """Create from dictionary."""
        # Convert nested dictionaries back to dataclass instances
        meta_data = data['meta']
        hardware = {}
        for hw_id, hw_data in meta_data['hardware'].items():
            hardware[hw_id] = HardwareDevice(**hw_data)
        
        meta = SystemInfo(
            platform=meta_data['platform'],
            host=meta_data['host'],
            timestamp=meta_data['timestamp'],
            cpu_only=meta_data['cpu_only'],
            hardware=hardware
        )
        
        result = cls(meta=meta)
        
        # Parse individual benchmarks
        if 'llama' in data and data['llama']:
            llama_data = data['llama']
            cpu_bench = None
            if llama_data.get('cpu_benchmark'):
                cpu_bench = LlamaRunResult(**llama_data['cpu_benchmark'])
            
            gpu_benchmarks = []
            if llama_data.get('gpu_benchmarks'):
                gpu_benchmarks = [LlamaRunResult(**gpu_data) for gpu_data in llama_data['gpu_benchmarks']]
            
            result.llama = LlamaBenchmarkResult(
                compile_time=llama_data['compile_time'],
                cpu_benchmark=cpu_bench,
                gpu_benchmarks=gpu_benchmarks
            )
        
        if 'reversan' in data and data['reversan']:
            rev_data = data['reversan']
            depth_benchmarks = [ReversanDepthResult(**depth) for depth in rev_data['depth_benchmarks']]
            thread_benchmarks = [ReversanThreadResult(**thread) for thread in rev_data['thread_benchmarks']]
            
            result.reversan = ReversanBenchmarkResult(
                compile_time=rev_data['compile_time'],
                depth_benchmarks=depth_benchmarks,
                thread_benchmarks=thread_benchmarks
            )
        
        if 'sevenzip' in data and data['sevenzip']:
            result.sevenzip = SevenZipBenchmarkResult(**data['sevenzip'])
        
        if 'blender' in data and data['blender']:
            blender_data = data['blender']
            cpu_result = None
            if blender_data.get('cpu'):
                cpu_result = BlenderSceneResult(**blender_data['cpu'])
            
            gpu_results = []
            if blender_data.get('gpus'):
                for gpu_data in blender_data['gpus']:
                    scenes = BlenderSceneResult(**gpu_data['scenes'])
                    gpu_results.append(BlenderDeviceResult(
                        hw_id=gpu_data['hw_id'],
                        scenes=scenes
                    ))
            
            result.blender = BlenderBenchmarkResult(
                cpu=cpu_result,
                gpus=gpu_results
            )
        
        return result
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UnifiedBenchmarkResult':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'UnifiedBenchmarkResult':
        """Create from YAML string."""
        return cls.from_dict(yaml.safe_load(yaml_str))
    
    def save_to_file(self, filepath: str, format: str = "json"):
        """Save to file in specified format."""
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        elif format.lower() == "yaml":
            with open(filepath, 'w') as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'UnifiedBenchmarkResult':
        """Load from file, detecting format from extension."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            return cls.from_yaml(content)
        else:
            return cls.from_json(content)


# Utility functions for hardware detection and ID generation
def generate_hardware_id(device_type: str, index: int) -> str:
    """Generate consistent hardware ID."""
    return f"{device_type}-{index}"


def normalize_hardware_name(name: str) -> str:
    """Normalize hardware names for consistent identification."""
    # Remove extra spaces and standardize format
    normalized = " ".join(name.split())
    
    # Common GPU name normalizations
    normalized = normalized.replace("(R)", "").replace("(TM)", "").replace("(C)", "")
    normalized = normalized.replace(" Graphics", "")
    
    return normalized.strip()


if __name__ == "__main__":
    # Example usage
    import time
    
    # Create example hardware setup
    cpu = HardwareDevice(
        hw_id="cpu-0",
        name="AMD Ryzen 7 5800X 8-Core Processor",
        type="cpu",
        manufacturer="AMD",
        cores=8,
        threads=16
    )
    
    gpu = HardwareDevice(
        hw_id="gpu-0", 
        name="NVIDIA GeForce RTX 3090",
        type="gpu",
        manufacturer="NVIDIA",
        framework="VULKAN"
    )
    
    system = SystemInfo(
        platform="Linux-6.12.47-1-lts-x86_64-with-glibc2.42",
        host="advantage",
        timestamp=time.time(),
        cpu_only=False,
        hardware={"cpu-0": cpu, "gpu-0": gpu}
    )
    
    # Create example benchmark results
    llama_result = LlamaBenchmarkResult(
        compile_time=81.0,
        cpu_benchmark=LlamaRunResult(
            prompt_speed=106.5,
            generation_speed=16.8,
            hw_id="cpu-0"
        ),
        gpu_benchmarks=[
            LlamaRunResult(
                prompt_speed=6221.2,
                generation_speed=194.5,
                hw_id="gpu-0"
            )
        ]
    )
    
    # Create unified result
    unified = UnifiedBenchmarkResult(
        meta=system,
        llama=llama_result
    )
    
    # Test serialization
    print("JSON output:")
    print(unified.to_json())
    
    print("\nYAML output:")
    print(unified.to_yaml())
    
    # Test round-trip
    json_str = unified.to_json()
    restored = UnifiedBenchmarkResult.from_json(json_str)
    print("\nRound-trip successful:", restored == unified)
