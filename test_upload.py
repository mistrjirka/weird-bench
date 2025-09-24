#!/usr/bin/env python3
"""
Test script to upload benchmark results to our API
"""
import json
import os
import requests
import time
import platform
import subprocess

def get_hardware_info():
    """Extract hardware information from the system."""
    try:
        # Get CPU info
        cpu_info = platform.processor()
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpu_lines = f.readlines()
                    for line in cpu_lines:
                        if line.startswith("model name"):
                            cpu_info = line.split(":")[1].strip()
                            break
            except:
                pass
        
        # Get GPU info (basic detection)
        gpu_info = "Unknown"
        try:
            # Try nvidia-smi first
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
            else:
                # Try lspci for AMD/Intel GPUs
                result = subprocess.run(["lspci", "-k"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or 'Display' in line:
                            if 'Radeon' in line or 'AMD' in line:
                                # Extract GPU name
                                parts = line.split('[')
                                if len(parts) > 1:
                                    gpu_info = parts[-1].split(']')[0]
                                break
        except:
            pass
        
        return {
            "cpu": cpu_info,
            "gpu": gpu_info,
            "platform": platform.platform(),
            "hostname": platform.node()
        }
    except Exception as e:
        print(f"⚠️  Could not extract hardware info: {e}")
        return {
            "cpu": "Unknown",
            "gpu": "Unknown", 
            "platform": platform.platform(),
            "hostname": platform.node()
        }

def upload_results():
    """Upload benchmark results to the API."""
    api_url = "http://localhost:8090/api"
    results_dir = "/home/jirka/programovani/weird-bench/results"
    
    print("📤 Uploading benchmark results...")
    
    try:
        # Get hardware info
        hardware_info = get_hardware_info()
        print(f"🔧 Hardware detected: {hardware_info['cpu']}")
        if hardware_info['gpu'] != "Unknown":
            print(f"🎮 GPU detected: {hardware_info['gpu']}")
        
        # Find benchmark result files
        benchmarks = ["reversan", "llama", "7zip", "blender"]
        files_to_upload = {}
        
        for benchmark_name in benchmarks:
            result_file = os.path.join(results_dir, f"{benchmark_name}_results.json")
            if os.path.exists(result_file):
                print(f"📁 Found {benchmark_name} results: {result_file}")
                with open(result_file, 'rb') as f:
                    files_to_upload[benchmark_name] = f.read()
        
        if not files_to_upload:
            print("❌ No benchmark result files found to upload")
            return False
        
        # Create multipart form data in the format our API expects
        files = {}
        for key, content in files_to_upload.items():
            files[key] = (f"{key}_results.json", content, "application/json")
        
        # Add action to form data (not URL parameter)
        data = {
            "action": "upload",
            "hardware_info": json.dumps(hardware_info),
            "timestamp": int(time.time())
        }
        
        print(f"🌐 Uploading to: {api_url}/api.php")
        print(f"📦 Uploading files: {list(files_to_upload.keys())}")
        
        # Upload to API
        response = requests.post(f"{api_url}/api.php", files=files, data=data, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📝 Response content: {response.text[:500]}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Upload successful!")
                if "message" in result:
                    print(f"📝 Server response: {result['message']}")
                if "data" in result:
                    print(f"📊 Results summary:")
                    data = result["data"]
                    if "total_benchmarks_stored" in data:
                        print(f"   • Benchmarks stored: {data['total_benchmarks_stored']}")
                    if "results" in data:
                        for hw_type, info in data["results"].items():
                            print(f"   • {hw_type.upper()}: {info}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                return False
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            try:
                error_info = response.json()
                print(f"📝 Error details: {error_info}")
            except:
                print(f"📝 Raw response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error during upload: {e}")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_results()