"""
Metal-accelerated Fast Nearest Neighbor Matching for MASt3R
Copyright (C) 2024-present Naver Corporation. All rights reserved.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""

import torch
import numpy as np
import math
import os
from pathlib import Path
from typing import Tuple, Optional, Union

# Try to import Metal compute libraries
try:
    import metalcompute as mc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

class MetalNearestNeighbor:
    """Metal-accelerated nearest neighbor search"""
    
    _device = None
    _library = None
    _kernels = {}
    
    @classmethod
    def _init_metal(cls):
        """Initialize Metal device and compile kernels"""
        if cls._device is None and METAL_AVAILABLE:
            try:
                cls._device = mc.Device()
                
                # Load and compile Metal shaders
                shader_path = Path(__file__).parent / "metal_nn_kernels.metal"
                if shader_path.exists():
                    with open(shader_path, 'r') as f:
                        shader_source = f.read()
                    
                    cls._library = cls._device.compile(shader_source)
                    
                    # Cache all kernel functions
                    kernel_names = [
                        'compute_l2_distances',
                        'compute_l2_distances_optimized', 
                        'compute_dot_similarities',
                        'compute_dot_similarities_optimized',
                        'find_argmin_rows',
                        'find_argmin_cols',
                        'find_argmax_rows', 
                        'find_argmax_cols',
                        'compute_block_distances'
                    ]
                    
                    for name in kernel_names:
                        cls._kernels[name] = cls._library.get_function(name)
                        
                    print("✓ Metal NN kernels loaded successfully")
                    return True
                else:
                    print(f"Warning: Metal shader file not found at {shader_path}")
                    return False
            except Exception as e:
                print(f"Warning: Failed to initialize Metal NN: {e}")
                cls._device = None
                return False
        return cls._device is not None
    
    @classmethod
    def is_available(cls):
        """Check if Metal NN is available"""
        return METAL_AVAILABLE and cls._init_metal()


class MPSNearestNeighbor:
    """MPS-accelerated nearest neighbor search using PyTorch's MPS backend"""
    
    @staticmethod
    def bruteforce_reciprocal_nns(A: torch.Tensor, B: torch.Tensor, 
                                device: str = 'mps', block_size: Optional[int] = None, 
                                dist: str = 'l2') -> Tuple[np.ndarray, np.ndarray]:
        """
        MPS-accelerated brute force reciprocal nearest neighbors
        
        Args:
            A: Query descriptors [NA, D]
            B: Database descriptors [NB, D] 
            device: Device to use ('mps' or fallback)
            block_size: Block size for memory efficiency
            dist: Distance type ('l2' or 'dot')
            
        Returns:
            Tuple of (nn_A, nn_B) as numpy arrays
        """
        # Ensure we're on MPS if available
        if torch.backends.mps.is_available() and device == 'mps':
            A = A.to('mps')
            B = B.to('mps')
        
        if dist == 'l2':
            dist_func = torch.cdist
            argmin_func = torch.min
        elif dist == 'dot':
            def dist_func(A, B):
                return A @ B.T
            def argmin_func(X, dim):
                sim, nn = torch.max(X, dim=dim)
                return sim.neg_(), nn
        else:
            raise ValueError(f'Unknown distance type: {dist}')

        NA, NB = len(A), len(B)
        
        if block_size is None or NA * NB <= block_size**2:
            # Full matrix computation
            with torch.amp.autocast('mps', enabled=True):
                dists = dist_func(A, B)
                _, nn_A = argmin_func(dists, dim=1)
                _, nn_B = argmin_func(dists, dim=0)
        else:
            # Block-wise computation for memory efficiency
            dis_A = torch.full((NA,), float('inf'), device=A.device, dtype=A.dtype)
            dis_B = torch.full((NB,), float('inf'), device=A.device, dtype=A.dtype)
            nn_A = torch.full((NA,), -1, device=A.device, dtype=torch.int64)
            nn_B = torch.full((NB,), -1, device=A.device, dtype=torch.int64)
            
            num_blocks_A = math.ceil(NA / block_size)
            num_blocks_B = math.ceil(NB / block_size)
            
            for i in range(num_blocks_A):
                A_start, A_end = i * block_size, min((i + 1) * block_size, NA)
                A_block = A[A_start:A_end]
                
                for j in range(num_blocks_B):
                    B_start, B_end = j * block_size, min((j + 1) * block_size, NB)
                    B_block = B[B_start:B_end]
                    
                    with torch.amp.autocast('mps', enabled=True):
                        dists_block = dist_func(A_block, B_block)
                        min_A_block, argmin_A_block = argmin_func(dists_block, dim=1)
                        min_B_block, argmin_B_block = argmin_func(dists_block, dim=0)
                    
                    # Update global minimums for A
                    A_mask = min_A_block < dis_A[A_start:A_end]
                    dis_A[A_start:A_end][A_mask] = min_A_block[A_mask]
                    nn_A[A_start:A_end][A_mask] = argmin_A_block[A_mask] + B_start
                    
                    # Update global minimums for B  
                    B_mask = min_B_block < dis_B[B_start:B_end]
                    dis_B[B_start:B_end][B_mask] = min_B_block[B_mask]
                    nn_B[B_start:B_end][B_mask] = argmin_B_block[B_mask] + A_start
        
        return nn_A.cpu().numpy(), nn_B.cpu().numpy()


def metal_bruteforce_reciprocal_nns(A: torch.Tensor, B: torch.Tensor,
                                   device: str = 'auto', block_size: Optional[int] = None,
                                   dist: str = 'l2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Metal-accelerated brute force reciprocal nearest neighbors with automatic fallback
    
    Args:
        A: Query descriptors [NA, D]
        B: Database descriptors [NB, D]
        device: Device to use ('auto', 'metal', 'mps', 'cpu')
        block_size: Block size for memory efficiency  
        dist: Distance type ('l2' or 'dot')
        
    Returns:
        Tuple of (nn_A, nn_B) as numpy arrays
    """
    # Convert to tensors if needed
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B)
    
    # Auto device selection
    if device == 'auto':
        if MetalNearestNeighbor.is_available():
            device = 'metal'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Try Metal acceleration first
    if device == 'metal' and MetalNearestNeighbor.is_available():
        try:
            return _metal_bruteforce_reciprocal_nns_impl(A, B, block_size, dist)
        except Exception as e:
            print(f"Warning: Metal NN failed ({e}), falling back to MPS")
            device = 'mps'
    
    # Fallback to MPS acceleration
    if device in ['mps', 'auto'] and torch.backends.mps.is_available():
        try:
            return MPSNearestNeighbor.bruteforce_reciprocal_nns(A, B, 'mps', block_size, dist)
        except Exception as e:
            print(f"Warning: MPS NN failed ({e}), falling back to CPU")
            device = 'cpu'
    
    # Final fallback to CPU
    from mast3r.fast_nn import bruteforce_reciprocal_nns
    return bruteforce_reciprocal_nns(A, B, device='cpu', block_size=block_size, dist=dist)


def _metal_bruteforce_reciprocal_nns_impl(A: torch.Tensor, B: torch.Tensor,
                                        block_size: Optional[int], dist: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal implementation using Metal kernels
    """
    if not MetalNearestNeighbor._init_metal():
        raise RuntimeError("Metal not initialized")
    
    device = MetalNearestNeighbor._device
    kernels = MetalNearestNeighbor._kernels
    
    # Ensure contiguous and float32
    A = A.contiguous().float().cpu()
    B = B.contiguous().float().cpu()
    NA, D = A.shape
    NB = B.shape[0]
    
    # Choose kernel based on distance type
    use_l2 = (dist == 'l2')
    
    if block_size is None or NA * NB <= block_size**2:
        # Full matrix computation
        if use_l2:
            kernel = kernels['compute_l2_distances_optimized'] if D % 4 == 0 else kernels['compute_l2_distances']
        else:
            kernel = kernels['compute_dot_similarities_optimized'] if D % 4 == 0 else kernels['compute_dot_similarities']
        
        # Create Metal buffers
        if D % 4 == 0 and use_l2:
            # Use vectorized version
            A_buffer = device.buffer(A.view(-1, D//4, 4).numpy().astype(np.float32))
            B_buffer = device.buffer(B.view(-1, D//4, 4).numpy().astype(np.float32))
            distances = np.zeros((NA, NB), dtype=np.float32)
            distances_buffer = device.buffer(distances)
            
            # Set parameters
            kernel.set_buffer(A_buffer, index=0)
            kernel.set_buffer(B_buffer, index=1) 
            kernel.set_buffer(distances_buffer, index=2)
            kernel.set_int(NA, index=3)
            kernel.set_int(NB, index=4)
            kernel.set_int(D//4, index=5)
            
            # Execute with appropriate threadgroup size
            threads_per_group = (16, 16)  # Optimize for Metal
            groups = ((NB + 15) // 16, (NA + 15) // 16)
            if use_l2:
                shared_mem_size = (D//4) * 16  # For shared memory optimization
                kernel.dispatch(groups, threads_per_group, shared_mem_size)
            else:
                kernel.dispatch(groups, threads_per_group)
        else:
            # Use regular version
            A_buffer = device.buffer(A.numpy())
            B_buffer = device.buffer(B.numpy())
            distances = np.zeros((NA, NB), dtype=np.float32)
            distances_buffer = device.buffer(distances)
            
            kernel.set_buffer(A_buffer, index=0)
            kernel.set_buffer(B_buffer, index=1)
            kernel.set_buffer(distances_buffer, index=2)
            kernel.set_int(NA, index=3)
            kernel.set_int(NB, index=4) 
            kernel.set_int(D, index=5)
            
            threads_per_group = (16, 16)
            groups = ((NB + 15) // 16, (NA + 15) // 16)
            kernel.dispatch(groups, threads_per_group)
        
        # Wait for completion
        device.wait()
        
        # Get results back from Metal
        distances = distances_buffer.contents().as_array(distances.shape, distances.dtype)
        distances = torch.from_numpy(distances)
        
        # Find argmin/argmax
        if use_l2:
            _, nn_A = torch.min(distances, dim=1)
            _, nn_B = torch.min(distances, dim=0)
        else:
            _, nn_A = torch.max(distances, dim=1)
            _, nn_B = torch.max(distances, dim=0)
        
        return nn_A.numpy(), nn_B.numpy()
    
    else:
        # Block-wise computation - fall back to MPS for now
        # TODO: Implement block-wise Metal kernels
        return MPSNearestNeighbor.bruteforce_reciprocal_nns(
            torch.from_numpy(A.numpy()), torch.from_numpy(B.numpy()), 
            'mps' if torch.backends.mps.is_available() else 'cpu', 
            block_size, dist
        )


class MetalcdistMatcher:
    """Metal-accelerated version of cdistMatcher"""
    
    def __init__(self, db_pts: torch.Tensor, device: str = 'auto'):
        self.db_pts = db_pts
        self.device = device
        
        # Auto-select best device
        if device == 'auto':
            if MetalNearestNeighbor.is_available():
                self.device = 'metal'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
                self.db_pts = self.db_pts.to('mps')
            else:
                self.device = 'cpu'
    
    def query(self, queries: torch.Tensor, k: int = 1, **kw):
        """Query the database for nearest neighbors"""
        assert k == 1, "Only k=1 supported currently"
        
        if queries.numel() == 0:
            return None, []
        
        # Use Metal-accelerated nearest neighbor search
        nnA, nnB = metal_bruteforce_reciprocal_nns(
            queries, self.db_pts, device=self.device, **kw
        )
        
        return None, nnA  # dis=None to match original API


# Export the main functions with the same API as the original
def bruteforce_reciprocal_nns(A: Union[torch.Tensor, np.ndarray], 
                            B: Union[torch.Tensor, np.ndarray],
                            device: str = 'auto', 
                            block_size: Optional[int] = None, 
                            dist: str = 'l2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for the original bruteforce_reciprocal_nns with Metal acceleration
    
    Args:
        A: Query descriptors
        B: Database descriptors
        device: Device to use ('auto', 'metal', 'mps', 'cuda', 'cpu')
        block_size: Block size for memory efficiency
        dist: Distance type ('l2' or 'dot')
        
    Returns:
        Tuple of (nn_A, nn_B) as numpy arrays
    """
    # Handle CUDA device requests by falling back to original implementation
    if isinstance(device, str) and device.startswith('cuda'):
        from mast3r.fast_nn import bruteforce_reciprocal_nns as original_bruteforce
        return original_bruteforce(A, B, device=device, block_size=block_size, dist=dist)
    
    return metal_bruteforce_reciprocal_nns(A, B, device=device, block_size=block_size, dist=dist)


def cdistMatcher(db_pts: torch.Tensor, device: str = 'auto'):
    """
    Drop-in replacement for the original cdistMatcher with Metal acceleration
    """
    # Handle CUDA device requests by falling back to original implementation  
    if isinstance(device, str) and device.startswith('cuda'):
        from mast3r.fast_nn import cdistMatcher as original_cdistMatcher
        return original_cdistMatcher(db_pts, device=device)
    
    return MetalcdistMatcher(db_pts, device=device) 