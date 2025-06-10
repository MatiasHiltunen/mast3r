#!/bin/bash

echo "Setting up MASt3R on macOS with uv..."

# Create and activate virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install ASMK separately (may need special handling)
echo "Installing ASMK..."
git clone https://github.com/jenicek/asmk.git temp_asmk || true
cd temp_asmk
pip install cython
cythonize cython/*.pyx
pip install -e .
cd ..
rm -rf temp_asmk

# Create necessary directories
mkdir -p checkpoints
mkdir -p data

echo "Setup complete!"
echo "To activate the environment in the future, run: source .venv/bin/activate"
echo ""
echo "Note: On macOS, CUDA is not available. All operations will run on CPU."
echo "The RoPE CUDA kernels will not be compiled, but the model will still work." 