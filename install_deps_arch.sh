#!/bin/bash
"""
Arch Linux dependency installer for weird-bench
Installs all required packages for running the benchmark suite.
"""

set -e  # Exit on any error

echo "🏗️  Installing dependencies for weird-bench on Arch Linux..."
echo ""

# Update package databases
echo "📦 Updating package databases..."
sudo pacman -Sy

# Core development tools
echo ""
echo "🔧 Installing core development tools..."
sudo pacman -S --needed \
    base-devel \
    cmake \
    git \
    python \
    python-pip \
    pkgconf \
    vulkan-tools


# 7-Zip for compression benchmark
echo ""
echo "🗜️  Installing 7-Zip..."
sudo pacman -S --needed p7zip

# Vulkan support for Llama.cpp GPU acceleration
echo ""
echo "🎮 Installing Vulkan support..."
sudo pacman -S --needed \
    vulkan-headers \
    vulkan-validation-layers \
    shaderc
sudo pacman -S --needed libxi

# GNU time (provides /usr/bin/time)
sudo pacman -S --needed time


# Additional Python packages via pip
echo ""
echo "🐍 Installing Python packages..."
sudo pacman -S python-matplotlib python-numpy python-requests python-pexpect
# Optional: Install GPU drivers (user can uncomment what they need)
echo ""
echo "💡 GPU Driver Installation (you may need these for Vulkan):"
echo "   For NVIDIA: sudo pacman -S nvidia nvidia-vulkan-dev"
echo "   For AMD: sudo pacman -S vulkan-radeon"
echo "   For Intel: sudo pacman -S vulkan-intel"
echo ""
echo "ℹ️  Note: You already have vulkan-intel installed, but you may need:"
echo "   sudo pacman -S vulkan-headers shaderc"
echo ""

# Check installations
echo "✅ Verifying installations..."
echo ""

# Check if commands are available
commands=("cmake" "git" "python3" "7z" "pip" "time")
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
python_packages=("matplotlib" "numpy" "requests" "pexpect" "psutil")
echo ""
echo "🐍 Checking Python packages..."

for pkg in "${python_packages[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "✓ Python package '$pkg' is available"
    else
        echo "❌ Python package '$pkg' is NOT available"
        missing+=("python-$pkg")
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