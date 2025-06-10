#!/bin/bash

echo "Installing MASt3R with uv on macOS..."

# Install base dependencies with uv
echo "Installing dependencies with uv..."
uv pip sync

# Install ASMK manually since it requires compilation
echo "Installing ASMK..."
uv pip install cython numpy
git clone https://github.com/jenicek/asmk.git temp_asmk 2>/dev/null || true
cd temp_asmk
cd cython
python -c "from Cython.Build import cythonize; import glob; cythonize(glob.glob('*.pyx'))"
cd ..
uv pip install -e .
cd ..
rm -rf temp_asmk

# Install the package in editable mode
echo "Installing MASt3R in editable mode..."
uv pip install -e .

echo "Installation complete!" 