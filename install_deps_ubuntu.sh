#!/bin/bash
"""
Ubuntu/Debian dependency installer for weird-bench
Installs all required packages for running the benchmark suite.
"""

set -e  # Exit on any error

echo "🏗️  Installing dependencies for weird-bench on Ubuntu/Debian..."
echo ""

# Update package databases
echo "📦 Updating package databases..."
sudo apt update

# Core development tools
echo ""
echo "🔧 Installing core development tools..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    pkg-config

# 7-Zip for compression benchmark
echo ""
echo "🗜️  Installing 7-Zip..."
sudo apt install -y p7zip-full

# Vulkan support for Llama.cpp GPU acceleration
echo ""
echo "🎮 Installing Vulkan support..."
sudo apt install -y \
    libvulkan-dev \
    vulkan-tools \
    vulkan-validationlayers-dev

# Additional Python packages via pip
echo ""
echo "🐍 Installing Python packages..."
pip3 install --user \
    matplotlib \
    numpy \
    requests \
    pexpect

# Optional: Install GPU drivers (user can uncomment what they need)
echo ""
echo "💡 GPU Driver Installation (optional):"
echo "   For NVIDIA: sudo apt install nvidia-vulkan-dev"
echo "   For AMD: sudo apt install mesa-vulkan-drivers"
echo "   For Intel: sudo apt install intel-media-va-driver"
echo ""

# Check installations
echo "✅ Verifying installations..."
echo ""

# Check if commands are available
commands=("cmake" "git" "python3" "7z" "pip3")
missing=()

for cmd in "${commands[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "✓ $cmd is available"
    else
        echo "❌ $cmd is NOT available"
        missing+=("$cmd")
    fi
done

# Check Python packages
python_packages=("matplotlib" "numpy" "requests" "pexpect")
echo ""
echo "🐍 Checking Python packages..."

for pkg in "${python_packages[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "✓ Python package '$pkg' is available"
    else
        echo "❌ Python package '$pkg' is NOT available"
        missing+=("python3-$pkg")
    fi
done

echo ""
if [ ${#missing[@]} -eq 0 ]; then
    echo "🎉 All dependencies installed successfully!"
    echo ""
    echo "You can now run benchmarks:"
    echo "  python3 run_benchmarks.py --list"
    echo "  python3 run_benchmarks.py --benchmark reversan"
    echo "  python3 run_benchmarks.py --benchmark llama"
    echo "  python3 run_benchmarks.py --benchmark 7zip"
    echo "  python3 run_benchmarks.py --benchmark all"
else
    echo "❌ Some dependencies are missing: ${missing[*]}"
    echo "Please check the installation and try again."
    exit 1
fi