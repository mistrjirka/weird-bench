#!/usr/bin/env python3
"""
Global hardware detection system for consistent hardware identification.
"""

import subprocess
import json
import re
import shutil
from typing import Dict, List, Optional, Tuple
import os
from unified_models import HardwareDevice, generate_hardware_id, normalize_hardware_name
from benchmarks.vulkaninfo_utils import list_vulkan_gpus, get_vulkaninfo_text, parse_vulkaninfo_text


class GlobalHardwareDetector:
    """Detects and assigns consistent IDs to system hardware."""
    
    def __init__(self):
        self.detected_hardware: Dict[str, HardwareDevice] = {}
        self._cpu_count = 0
        self._gpu_count = 0
    
    def detect_all_hardware(self) -> Dict[str, HardwareDevice]:
        """Detect all system hardware and assign consistent IDs."""
        print("🔍 Detecting system hardware...")
        
        # Clear previous detection
        self.detected_hardware = {}
        self._cpu_count = 0
        self._gpu_count = 0
        
        # Detect CPU
        self._detect_cpu()
        
        # Detect GPUs
        self._detect_gpus()
        
        print(f"✅ Hardware detection complete: {self._cpu_count} CPU(s), {self._gpu_count} GPU(s)")
        return self.detected_hardware.copy()
    
    def _detect_cpu(self) -> None:
        """Detect CPU information."""
        try:
            # Try to get CPU info from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Extract CPU model name
            model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
            cpu_name = model_match.group(1).strip() if model_match else "Unknown CPU"
            
            # Count physical cores and threads
            cores = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))
            
            # Try to get physical core count (different from thread count)
            physical_cores = cores
            siblings_match = re.search(r'siblings\s*:\s*(\d+)', cpuinfo)
            cpu_cores_match = re.search(r'cpu cores\s*:\s*(\d+)', cpuinfo)
            
            if siblings_match and cpu_cores_match:
                siblings = int(siblings_match.group(1))
                physical_cores = int(cpu_cores_match.group(1))
                # If we have multiple physical packages, multiply
                packages = cores // siblings if siblings > 0 else 1
                physical_cores = physical_cores * packages
            
            # Detect manufacturer
            manufacturer = "Unknown"
            cpu_lower = cpu_name.lower()
            if "intel" in cpu_lower:
                manufacturer = "Intel"
            elif "amd" in cpu_lower:
                manufacturer = "AMD"
            elif "arm" in cpu_lower or "apple" in cpu_lower:
                manufacturer = "ARM"
            
            cpu_id = generate_hardware_id("cpu", self._cpu_count)
            cpu_device = HardwareDevice(
                hw_id=cpu_id,
                name=normalize_hardware_name(cpu_name),
                type="cpu",
                manufacturer=manufacturer,
                cores=physical_cores,
                threads=cores
            )
            
            self.detected_hardware[cpu_id] = cpu_device
            self._cpu_count += 1
            print(f"  🧠 CPU: {cpu_device.name} ({cpu_device.cores} cores, {cpu_device.threads} threads)")
            
        except Exception as e:
            print(f"⚠️  CPU detection failed: {e}")
            # Fallback CPU
            cpu_id = generate_hardware_id("cpu", self._cpu_count)
            cpu_device = HardwareDevice(
                hw_id=cpu_id,
                name="Unknown CPU",
                type="cpu",
                manufacturer="Unknown"
            )
            self.detected_hardware[cpu_id] = cpu_device
            self._cpu_count += 1
    
    def _detect_gpus(self) -> None:
        """Detect GPU information using multiple methods."""
        # Method 1: Try vulkaninfo (most reliable for gaming GPUs)
        gpus_vulkan = self._detect_gpus_vulkan()
        
        # Method 2: Try nvidia-ml-py for NVIDIA GPUs
        gpus_nvidia = self._detect_gpus_nvidia()
        
        # Method 3: Try lspci for fallback
        gpus_lspci = self._detect_gpus_lspci()
        
        # Combine and deduplicate
        all_gpus = []
        all_gpus.extend(gpus_vulkan)
        all_gpus.extend(gpus_nvidia) 
        all_gpus.extend(gpus_lspci)
        
        # Deduplicate by name similarity
        unique_gpus = self._deduplicate_gpus(all_gpus)
        
        for gpu_info in unique_gpus:
            gpu_id = generate_hardware_id("gpu", self._gpu_count)
            gpu_device = HardwareDevice(
                hw_id=gpu_id,
                name=normalize_hardware_name(gpu_info['name']),
                type="gpu",
                manufacturer=gpu_info.get('manufacturer', 'Unknown'),
                framework=gpu_info.get('framework'),
                driver_version=gpu_info.get('driver_version'),
                memory_mb=gpu_info.get('memory_mb')
            )
            
            self.detected_hardware[gpu_id] = gpu_device
            self._gpu_count += 1
            print(f"  🎮 GPU: {gpu_device.name} ({gpu_device.framework or 'Unknown framework'})")
    
    def _detect_gpus_vulkan(self) -> List[Dict[str, str]]:
        """Detect GPUs using vulkaninfo text output. Ignore software pipes."""
        try:
            text = get_vulkaninfo_text(timeout=30)
            if not text:
                return []
            devices = parse_vulkaninfo_text(text)
            gpus = []
            for dev in devices:
                name = dev.get('name', '')
                # Map driver to manufacturer naming used elsewhere
                driver = (dev.get('driver') or '').lower()
                if driver == 'nvidia':
                    manufacturer = 'NVIDIA'
                elif driver == 'amd':
                    manufacturer = 'AMD'
                elif driver == 'intel':
                    manufacturer = 'Intel'
                else:
                    manufacturer = self._detect_gpu_manufacturer(name)

                gpus.append({
                    'name': name,
                    'manufacturer': manufacturer,
                    'framework': 'VULKAN'
                })
            return gpus
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  ⚠️  Vulkan detection failed: {e}")
            return []
    
    # Remove JSON parsing path; rely on text method only per reliability concerns
    def _parse_vulkaninfo_json(self, json_output: str) -> List[Dict[str, str]]:
        return []
    
    def _parse_vulkaninfo_text(self, text_output: str) -> List[Dict[str, str]]:
        """Compat shim around shared parser; kept for backward use within this class."""
        devices = parse_vulkaninfo_text(text_output)
        gpus = []
        for dev in devices:
            name = dev.get('name', '')
            driver = (dev.get('driver') or '').lower()
            if driver == 'nvidia':
                manufacturer = 'NVIDIA'
            elif driver == 'amd':
                manufacturer = 'AMD'
            elif driver == 'intel':
                manufacturer = 'Intel'
            else:
                manufacturer = self._detect_gpu_manufacturer(name)
            gpus.append({'name': name, 'manufacturer': manufacturer, 'framework': 'VULKAN'})
        return gpus
    
    def _detect_gpus_nvidia(self) -> List[Dict[str, str]]:
        """Detect NVIDIA GPUs using nvidia-ml-py or nvidia-smi."""
        gpus = []
        
        try:
            # Try nvidia-smi first (more commonly available)
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            name, memory_str, driver = parts[0], parts[1], parts[2]
                            try:
                                memory_mb = int(memory_str) if memory_str.isdigit() else None
                            except:
                                memory_mb = None
                            
                            gpus.append({
                                'name': name,
                                'manufacturer': 'NVIDIA',
                                'framework': 'CUDA',
                                'driver_version': driver,
                                'memory_mb': memory_mb
                            })
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  ⚠️  NVIDIA detection failed: {e}")
        
        return gpus
    
    def _detect_gpus_lspci(self) -> List[Dict[str, str]]:
        """Detect GPUs using lspci as fallback."""
        gpus = []
        
        try:
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    original_line = line.strip()
                    line_lower = line.strip().lower()
                    if ('vga' in line_lower or 'display' in line_lower or '3d' in line_lower) and 'controller' in line_lower:
                        # Extract GPU name
                        # Format is usually: "01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)"
                        if ':' in original_line:
                            parts = original_line.split(':', 2)
                            if len(parts) >= 3:
                                gpu_info = parts[2].strip()
                                
                                # Extract bracketed name if available (preferred)
                                bracket_match = re.search(r'\[([^\]]+)\]', gpu_info)
                                if bracket_match:
                                    gpu_name = bracket_match.group(1)
                                else:
                                    # Fallback: clean up the manufacturer name
                                    gpu_name = gpu_info.split('(')[0].strip()
                                    # Remove common manufacturer suffixes
                                    gpu_name = re.sub(r'\s+(Corporation|Corp\.?|Inc\.?|Limited|Ltd\.?)\s*$', '', gpu_name, flags=re.IGNORECASE)
                                
                                manufacturer = self._detect_gpu_manufacturer(gpu_name)
                                
                                # Skip software renderers, virtual GPUs, and invalid entries
                                if (len(gpu_name) > 3 and 
                                    not any(skip in gpu_name.lower() for skip in ['software', 'virtual', 'vmware', 'qemu']) and
                                    manufacturer != 'Unknown'):
                                    gpus.append({
                                        'name': gpu_name,
                                        'manufacturer': manufacturer,
                                        'framework': 'UNKNOWN'
                                    })
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  ⚠️  lspci detection failed: {e}")
        
        return gpus
    
    def _detect_gpu_manufacturer(self, gpu_name: str) -> str:
        """Detect GPU manufacturer from device name."""
        name_lower = gpu_name.lower()
        
        if any(keyword in name_lower for keyword in ['nvidia', 'geforce', 'gtx', 'rtx', 'titan', 'quadro']):
            return 'NVIDIA'
        elif any(keyword in name_lower for keyword in ['amd', 'radeon', 'rx ', 'vega', 'fury']):
            return 'AMD'
        elif any(keyword in name_lower for keyword in ['intel', 'iris', 'uhd', 'hd graphics']):
            return 'Intel'
        elif any(keyword in name_lower for keyword in ['apple', 'mali', 'adreno']):
            return 'ARM'
        else:
            return 'Unknown'
    
    def _deduplicate_gpus(self, gpu_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate GPUs detected by different methods."""
        if not gpu_list:
            return []
        
        unique_gpus = []
        seen_names = set()
        
        for gpu in gpu_list:
            gpu_name = normalize_hardware_name(gpu['name'])
            
            # Check if we've seen a similar GPU name
            is_duplicate = False
            for seen_name in seen_names:
                if self._gpu_names_similar(gpu_name, seen_name):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_names.add(gpu_name)
                unique_gpus.append(gpu)
        
        return unique_gpus
    
    def _gpu_names_similar(self, name1: str, name2: str, threshold: float = 0.6) -> bool:
        """Check if two GPU names are similar enough to be the same device."""
        name1_clean = name1.lower().replace('-', ' ').replace('_', ' ')
        name2_clean = name2.lower().replace('-', ' ').replace('_', ' ')
        
        # Exact match
        if name1_clean == name2_clean:
            return True
        
        # Check if one name contains the other (allowing for abbreviations)
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Word-based similarity for more complex comparisons
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        
        if not words1 or not words2:
            return False
        
        # Remove common non-distinctive words
        common_words = {'graphics', 'card', 'controller', 'adapter', 'device'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def get_device_by_id(self, hw_id: str) -> Optional[HardwareDevice]:
        """Get device by hardware ID."""
        return self.detected_hardware.get(hw_id)
    
    def get_cpu_device(self) -> Optional[HardwareDevice]:
        """Get the primary CPU device."""
        for device in self.detected_hardware.values():
            if device.type == "cpu":
                return device
        return None
    
    def get_gpu_devices(self) -> List[HardwareDevice]:
        """Get all GPU devices."""
        return [device for device in self.detected_hardware.values() if device.type == "gpu"]
    
    def find_matching_device(self, device_name: str, device_type: str = None) -> Optional[str]:
        """Find hardware ID for a device by name matching."""
        if not device_name:
            return None
            
        device_name_clean = normalize_hardware_name(device_name).lower()
        
        best_match = None
        best_similarity = 0.0
        
        for hw_id, device in self.detected_hardware.items():
            if device_type and device.type != device_type:
                continue
            
            detected_name_clean = device.name.lower()
            
            # Exact match gets priority
            if device_name_clean == detected_name_clean:
                return hw_id
            
            # Substring match
            if device_name_clean in detected_name_clean or detected_name_clean in device_name_clean:
                similarity = min(len(device_name_clean), len(detected_name_clean)) / max(len(device_name_clean), len(detected_name_clean))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = hw_id
            
            # Word-based similarity as fallback
            words_query = set(device_name_clean.replace('-', ' ').replace('_', ' ').split())
            words_detected = set(detected_name_clean.replace('-', ' ').replace('_', ' ').split())
            
            if words_query and words_detected:
                intersection = words_query.intersection(words_detected)
                union = words_query.union(words_detected)
                word_similarity = len(intersection) / len(union)
                
                if word_similarity > best_similarity and word_similarity > 0.3:
                    best_similarity = word_similarity
                    best_match = hw_id
        
        return best_match if best_similarity > 0.3 else None


if __name__ == "__main__":
    # Test the hardware detector
    detector = GlobalHardwareDetector()
    hardware = detector.detect_all_hardware()
    
    print("\nDetected Hardware:")
    for hw_id, device in hardware.items():
        print(f"  {hw_id}: {device.name} ({device.type}, {device.manufacturer})")
        if device.cores:
            print(f"    Cores: {device.cores}, Threads: {device.threads}")
        if device.framework:
            print(f"    Framework: {device.framework}")
    
    # Test device lookup
    print("\nTesting device matching:")
    cpu_id = detector.find_matching_device("AMD Ryzen 7 5800X", "cpu")
    gpu_id = detector.find_matching_device("NVIDIA GeForce RTX 3090", "gpu")
    
    print(f"CPU match: {cpu_id}")
    print(f"GPU match: {gpu_id}")
