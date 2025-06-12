# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""
Metal-accelerated Fast Nearest Neighbor Matching for MASt3R

This module provides Metal-accelerated implementations of nearest neighbor
search operations used in MASt3R, with automatic fallback to MPS and CPU.
"""

import platform

# Only import Metal NN on macOS
if platform.system() == 'Darwin':
    try:
        from .metal_nn import (
            bruteforce_reciprocal_nns,
            cdistMatcher,
            MetalNearestNeighbor,
            MPSNearestNeighbor,
            metal_bruteforce_reciprocal_nns
        )
        
        # Check availability
        if MetalNearestNeighbor.is_available():
            print("✓ Metal-accelerated NN available")
        else:
            print("ⓘ Metal NN not available, using MPS/CPU fallback")
            
    except ImportError as e:
        print(f"Warning: Failed to import Metal NN: {e}")
        # Fall back to original implementation
        from mast3r.fast_nn import bruteforce_reciprocal_nns, cdistMatcher
else:
    # Not on macOS, use original implementation
    from mast3r.fast_nn import bruteforce_reciprocal_nns, cdistMatcher

__all__ = ['bruteforce_reciprocal_nns', 'cdistMatcher'] 