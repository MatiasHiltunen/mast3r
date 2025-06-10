#!/usr/bin/env python3
"""Simple test script to verify MASt3R works on macOS"""

import torch
import numpy as np
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

def test_device_detection():
    """Test device detection on macOS"""
    print("Testing device detection...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Selected device: {device}")
    
    # Test tensor creation
    test_tensor = torch.randn(2, 3, 4).to(device)
    print(f"Successfully created tensor on {device}: shape={test_tensor.shape}")
    
    return device

def test_model_loading(device):
    """Test loading MASt3R model"""
    print("\nTesting model loading...")
    try:
        # This will download from HuggingFace if not already cached
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        print(f"Successfully loaded model: {model_name}")
        print(f"Model device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_inference(model, device):
    """Test basic inference with dummy data"""
    print("\nTesting inference with dummy data...")
    try:
        # Create dummy images
        dummy_imgs = [
            {'img': torch.randn(1, 3, 224, 224).to(device), 'true_shape': torch.tensor([224, 224]).to(device), 'idx': 0},
            {'img': torch.randn(1, 3, 224, 224).to(device), 'true_shape': torch.tensor([224, 224]).to(device), 'idx': 1}
        ]
        
        # Run inference
        output = inference([tuple(dummy_imgs)], model, device, batch_size=1, verbose=False)
        
        print("Inference successful!")
        print(f"Output keys: {output.keys()}")
        print(f"View1 shape: {output['view1']['pts3d'].shape}")
        print(f"View2 shape: {output['view2']['pts3d'].shape}")
        
        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

if __name__ == '__main__':
    print("=== MASt3R macOS Test ===\n")
    
    # Test device detection
    device = test_device_detection()
    
    # Test model loading
    model = test_model_loading(device)
    
    if model is not None:
        # Test inference
        success = test_inference(model, device)
        
        if success:
            print("\n✅ All tests passed! MASt3R is working correctly on macOS.")
        else:
            print("\n❌ Inference test failed.")
    else:
        print("\n❌ Model loading failed.")
    
    print("\nNote: For actual usage, download model checkpoints as described in README_MACOS.md") 