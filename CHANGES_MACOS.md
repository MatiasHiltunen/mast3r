# Changes Made for macOS Compatibility with uv

This document summarizes all the changes made to make MASt3R work on macOS with uv package manager.

## New Files Created

1. **`pyproject.toml`** - Created a proper Python package configuration with:
   - uv-specific configuration
   - PyTorch CPU-only index for macOS
   - All dependencies consolidated
   - ASMK as a git dependency

2. **`.python-version`** - Specifies Python 3.11 for uv

3. **`README_MACOS.md`** - Comprehensive guide for macOS users with:
   - Installation instructions using uv
   - macOS-specific notes about performance and limitations
   - Troubleshooting guide

4. **`setup_macos.sh`** - Automated setup script for macOS

5. **`install_uv.sh`** - Simplified installation script using uv commands

6. **`test_macos.py`** - Test script to verify the installation

## Code Changes

### Device Detection
Updated default device selection in multiple files to auto-detect the best available device (cuda > mps > cpu):
- `demo.py` - Added conditional CUDA settings
- `dust3r/dust3r/demo.py` - Added automatic device detection
- `make_pairs.py` - Added automatic device detection
- `visloc.py` - Added automatic device detection  
- `kapture_mast3r_mapping.py` - Added automatic device detection and torch import

### CUDA-specific Code
Made CUDA-specific calls conditional:
- `demo.py` - Wrapped `torch.backends.cuda.matmul.allow_tf32` in CUDA availability check
- `make_pairs.py` - Made `torch.cuda.empty_cache()` conditional
- `mast3r/demo.py` - Made `torch.cuda.empty_cache()` conditional
- `mast3r/cloud_opt/sparse_ga.py` - Made `torch.cuda.empty_cache()` conditional
- `mast3r/demo_glomap.py` - Made `torch.cuda.empty_cache()` conditional

### Autocast Fix
Fixed deprecated `torch.cuda.amp.autocast` syntax to `torch.amp.autocast('cuda', ...)` in:
- `dust3r/dust3r/cloud_opt/base_opt.py` - Also made it device-aware
- `dust3r/dust3r/inference.py`
- `dust3r/dust3r/model.py`
- `mast3r/cloud_opt/sparse_ga.py`

### Retrieval Support
- Added `faiss-cpu` to dependencies (CPU-only version for macOS)
- Modified `mast3r/retrieval/processor.py` to handle CPU-only faiss and missing ASMK gracefully
- ASMK must be installed separately as shown in README_MACOS.md

### Other Changes
- Updated `.gitignore` to include uv-related files (`.venv/`, `uv.lock`)

## Key Differences on macOS

1. **No CUDA Support**: All GPU operations use Metal Performance Shaders (MPS) on Apple Silicon or CPU
2. **No RoPE CUDA Kernels**: Falls back to PyTorch implementation (slower but functional)
3. **Performance**: Expect slower processing compared to CUDA-enabled systems
4. **Memory**: Consider using smaller image sizes (`--image_size 224`) for better performance

## Usage

After setting up the environment as described in `README_MACOS.md`, the demo can be run with:

```bash
python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```

The system will automatically use MPS (if available on Apple Silicon) or CPU for computation. 