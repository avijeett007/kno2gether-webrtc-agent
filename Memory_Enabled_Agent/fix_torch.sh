#!/bin/bash

# Fix PyTorch installation for memoryagent Conda environment
# This script fixes the "Symbol not found" error with torch on macOS

echo "===== PyTorch Installation Fix for OYOS Agent ====="
echo "This script will fix the PyTorch installation in your memoryagent Conda environment."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is required but not installed. Please install Conda and try again."
    exit 1
fi

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate memoryagent

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate 'memoryagent' environment. Make sure it exists."
    exit 1
fi

echo "Conda environment 'memoryagent' activated successfully."
echo "Python version: $(python --version)"
echo ""

# Remove existing PyTorch installation completely
echo "Step 1: Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio
conda remove -y --force-remove pytorch torchvision torchaudio cudatoolkit

# Clean pip cache
pip cache purge

echo "Step 2: Reinstalling PyTorch using conda..."

# Check operating system
OS="$(uname)"
if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS system, installing PyTorch for CPU..."
    
    # For macOS, use conda to install PyTorch (using CPU version)
    conda install -y -c pytorch pytorch=2.0.0 torchvision torchaudio
else
    # For Linux/other systems
    echo "Detected non-macOS system, installing PyTorch for CPU..."
    conda install -y -c pytorch pytorch=2.0.0 torchvision torchaudio cpuonly
fi

# Verify installation
echo ""
echo "Step 3: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed successfully!"
else
    echo "❌ PyTorch installation verification failed."
    echo "Please try reinstalling manually with:"
    echo "  conda activate memoryagent"
    echo "  conda install -c pytorch pytorch=2.0.0 torchvision torchaudio"
    exit 1
fi

# Check if other dependencies are correctly installed
echo ""
echo "Step 4: Checking other dependencies..."

# Check transformers
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || pip install transformers>=4.0.0

# Reinstall turn detector plugin (which uses PyTorch)
echo ""
echo "Step 5: Reinstalling turn detector plugin..."
pip install --force-reinstall livekit-plugins-turn-detector

echo ""
echo "===== Installation Fix Complete ====="
echo "You can now try running the OYOS agent again with:"
echo "  python oyos_coach_disc_agent.py start"
echo "" 