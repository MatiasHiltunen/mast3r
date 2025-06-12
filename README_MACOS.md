# MASt3R on macOS with uv

This guide explains how to set up and run MASt3R on macOS using `uv` for dependency management.

## Prerequisites

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Ensure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/naver/mast3r
cd mast3r
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
# Install PyTorch CPU version (CUDA is not available on macOS)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
uv pip install -r requirements.txt -r dust3r/requirements.txt -r dust3r/requirements_optional.txt

# Install cython for ASMK
uv pip install cython numpy
```

4. Install ASMK (required for retrieval features):
```bash
# Clone to a permanent location outside the project
git clone https://github.com/jenicek/asmk.git ../asmk_install
cd ../asmk_install/cython
python -c "from Cython.Build import cythonize; import glob; cythonize(glob.glob('*.pyx'))"
cd ..
uv pip install -e .
cd ../mast3r
```

## Download Model Checkpoints

```bash
# Create checkpoints directory
mkdir -p checkpoints/

# Download MASt3R model
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

# Optional: Download retrieval model for advanced pairing
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## Running the Demo

The demo will automatically detect that you're on macOS and use the appropriate device (MPS if available, otherwise CPU):

```bash
python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```

### Demo Options

- `--weights`: Load a checkpoint from a local file
- `--retrieval_model`: Enable retrieval-based pairing (requires retrieval checkpoint)
- `--device`: Override device selection (defaults to best available: mps > cpu)
- `--local_network`: Make the demo accessible on your local network
- `--server_port`: Specify a custom port (default: 7860)

### Example with retrieval:
```bash
python demo.py \
    --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --retrieval_model checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
```

## macOS-Specific Notes

1. **No CUDA Support**: macOS doesn't support CUDA. The code has been modified to:
   - Use MPS (Metal Performance Shaders) when available for GPU acceleration on Apple Silicon
   - Fall back to CPU if MPS is not available
   - Skip CUDA-specific optimizations like RoPE kernels

2. **Performance**: 
   - On Apple Silicon (M1/M2/M3), MPS provides GPU acceleration
   - Processing will be slower than on CUDA-enabled systems
   - Consider using smaller image sizes for faster processing

3. **Memory Usage**:
   - Monitor memory usage, especially with large images
   - The `--image_size` parameter can be set to 224 for lower memory usage

## Troubleshooting

1. **Import errors**: Make sure you're in the activated virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. **ASMK installation issues**: 
   - Ensure Xcode Command Line Tools are installed
   - Try installing with system Python if issues persist

3. **Out of memory errors**:
   - Reduce image size: `--image_size 224`
   - Process fewer images at once
   - Close other applications to free up memory

4. **MPS errors**: If you encounter MPS-related errors, force CPU usage:
   ```bash
   python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --device cpu
   ```

## Alternative Installation Script

For convenience, you can use the provided setup script:
```bash
chmod +x setup_macos.sh
./setup_macos.sh
```

This script automates the environment setup and dependency installation process. 