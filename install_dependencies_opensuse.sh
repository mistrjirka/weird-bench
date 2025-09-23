#!/bin/bash
# Install dependencies for weird-bench on openSUSE
# This script installs all required dependencies for running both Reversan and Llama.cpp benchmarks

set -e

echo "🔧 Installing dependencies for weird-bench on openSUSE..."

# Core build tools and utilities
echo "📦 Installing core build tools..."
sudo zypper install -y \
    git \
    cmake \
    make \
    gcc-c++ \
    curl \
    libcurl-devel \
    python3 \
    python3-pip \
    pkg-config

# Time utility for detailed process metrics (used by Reversan benchmark)
echo "📦 Installing GNU time..."
sudo zypper install -y time

# 7-Zip for compression benchmark
echo "📦 Installing 7-Zip..."
sudo zypper install -y p7zip

# Python dependencies
echo "🐍 Installing Python dependencies..."
sudo zypper install -y python3-matplotlib python3-numpy || {
    echo "⚠️  System Python packages not available, trying pip with --break-system-packages..."
    pip3 install --user --break-system-packages matplotlib numpy
}

# Additional Python packages needed by weird-bench
echo "🐍 Installing additional Python packages..."
sudo zypper install -y python3-requests python3-pexpect python3-psutil || {
    echo "⚠️  Some Python packages not available, trying pip with --break-system-packages..."
    pip3 install --user --break-system-packages requests pexpect psutil
}

# Optional: Vulkan SDK for GPU acceleration (Llama.cpp)
echo "🎮 Installing Vulkan development packages (optional for GPU support)..."
sudo zypper install -y vulkan-devel vulkan-loader || {
    echo "⚠️  Vulkan packages not available - GPU acceleration will be disabled"
    echo "   You can still run CPU benchmarks"
}

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "📋 Installed packages:"
echo "   - git, cmake, make, gcc-c++ (build tools)"
echo "   - curl, libcurl-devel (HTTP client and development headers)"
echo "   - python3, python3-pip (Python runtime)"
echo "   - matplotlib, numpy (Python plotting libraries)"
echo "   - time (GNU time utility for process metrics)"
echo "   - vulkan-devel, vulkan-loader (GPU acceleration, optional)"
echo ""
echo "🚀 You can now run benchmarks with:"
echo "   python3 run_benchmarks.py --benchmark all"
echo ""