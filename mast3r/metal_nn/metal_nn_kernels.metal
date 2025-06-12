/*
  Metal implementation of Fast Nearest Neighbor Matching for MASt3R
  Copyright (C) 2024-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <metal_stdlib>
using namespace metal;

// L2 distance computation kernel - optimized for memory bandwidth
kernel void compute_l2_distances(
    constant float* A [[buffer(0)]],           // [NA, D] - Query descriptors
    constant float* B [[buffer(1)]],           // [NB, D] - Database descriptors  
    device float* distances [[buffer(2)]],     // [NA, NB] - Output distances
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    const uint i = gid.y;  // Query index (A)
    const uint j = gid.x;  // Database index (B)
    
    if (i >= NA || j >= NB) return;
    
    // Compute L2 distance between A[i] and B[j]
    float dist = 0.0;
    for (uint d = 0; d < D; d++) {
        float diff = A[i * D + d] - B[j * D + d];
        dist += diff * diff;
    }
    
    distances[i * NB + j] = sqrt(dist);
}

// Optimized L2 distance with shared memory and vectorization
kernel void compute_l2_distances_optimized(
    constant float4* A [[buffer(0)]],          // [NA, D/4] - Query descriptors (vectorized)
    constant float4* B [[buffer(1)]],          // [NB, D/4] - Database descriptors (vectorized)
    device float* distances [[buffer(2)]],     // [NA, NB] - Output distances
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    constant uint& D4 [[buffer(5)]],           // D/4 (vectorized dimension)
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float4* shared_A [[threadgroup(0)]])
{
    const uint i = gid.y;  // Query index (A)
    const uint j_base = gid.x * tg_size.x;  // Base database index (B)
    const uint local_j = tid.x;
    
    if (i >= NA) return;
    
    // Load query descriptor into shared memory
    if (local_j == 0) {
        for (uint d = 0; d < D4; d++) {
            shared_A[d] = A[i * D4 + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute distances for multiple B vectors in parallel
    for (uint j_offset = 0; j_offset < tg_size.x; j_offset++) {
        uint j = j_base + j_offset;
        if (j >= NB) break;
        
        if (local_j == j_offset) {
            float4 dist_vec = float4(0.0);
            for (uint d = 0; d < D4; d++) {
                float4 diff = shared_A[d] - B[j * D4 + d];
                dist_vec += diff * diff;
            }
            
            // Sum components of the distance vector
            float dist = dist_vec.x + dist_vec.y + dist_vec.z + dist_vec.w;
            distances[i * NB + j] = sqrt(dist);
        }
    }
}

// Dot product similarity computation
kernel void compute_dot_similarities(
    constant float* A [[buffer(0)]],           // [NA, D] - Query descriptors
    constant float* B [[buffer(1)]],           // [NB, D] - Database descriptors
    device float* similarities [[buffer(2)]],  // [NA, NB] - Output similarities
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.y;  // Query index (A)
    const uint j = gid.x;  // Database index (B)
    
    if (i >= NA || j >= NB) return;
    
    // Compute dot product between A[i] and B[j]
    float sim = 0.0;
    for (uint d = 0; d < D; d++) {
        sim += A[i * D + d] * B[j * D + d];
    }
    
    similarities[i * NB + j] = sim;
}

// Optimized dot product with vectorization
kernel void compute_dot_similarities_optimized(
    constant float4* A [[buffer(0)]],          // [NA, D/4] - Query descriptors (vectorized)
    constant float4* B [[buffer(1)]],          // [NB, D/4] - Database descriptors (vectorized)
    device float* similarities [[buffer(2)]],  // [NA, NB] - Output similarities
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    constant uint& D4 [[buffer(5)]],           // D/4 (vectorized dimension)
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.y;  // Query index (A)
    const uint j = gid.x;  // Database index (B)
    
    if (i >= NA || j >= NB) return;
    
    // Compute dot product using float4 vectorization
    float4 sim_vec = float4(0.0);
    for (uint d = 0; d < D4; d++) {
        sim_vec += A[i * D4 + d] * B[j * D4 + d];
    }
    
    // Sum components of the similarity vector
    similarities[i * NB + j] = sim_vec.x + sim_vec.y + sim_vec.z + sim_vec.w;
}

// Find minimum distances and indices (argmin)
kernel void find_argmin_rows(
    constant float* distances [[buffer(0)]],   // [NA, NB] - Input distances
    device float* min_distances [[buffer(1)]], // [NA] - Output minimum distances
    device uint* min_indices [[buffer(2)]],    // [NA] - Output minimum indices
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NA) return;
    
    float min_dist = INFINITY;
    uint min_idx = 0;
    
    // Find minimum in row gid
    for (uint j = 0; j < NB; j++) {
        float dist = distances[gid * NB + j];
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = j;
        }
    }
    
    min_distances[gid] = min_dist;
    min_indices[gid] = min_idx;
}

// Find minimum distances and indices (argmin) along columns
kernel void find_argmin_cols(
    constant float* distances [[buffer(0)]],   // [NA, NB] - Input distances  
    device float* min_distances [[buffer(1)]], // [NB] - Output minimum distances
    device uint* min_indices [[buffer(2)]],    // [NB] - Output minimum indices
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NB) return;
    
    float min_dist = INFINITY;
    uint min_idx = 0;
    
    // Find minimum in column gid
    for (uint i = 0; i < NA; i++) {
        float dist = distances[i * NB + gid];
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }
    
    min_distances[gid] = min_dist;
    min_indices[gid] = min_idx;
}

// Find maximum similarities and indices (argmax) for dot product
kernel void find_argmax_rows(
    constant float* similarities [[buffer(0)]], // [NA, NB] - Input similarities
    device float* max_similarities [[buffer(1)]], // [NA] - Output maximum similarities  
    device uint* max_indices [[buffer(2)]],     // [NA] - Output maximum indices
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NA) return;
    
    float max_sim = -INFINITY;
    uint max_idx = 0;
    
    // Find maximum in row gid
    for (uint j = 0; j < NB; j++) {
        float sim = similarities[gid * NB + j];
        if (sim > max_sim) {
            max_sim = sim;
            max_idx = j;
        }
    }
    
    max_similarities[gid] = max_sim;
    max_indices[gid] = max_idx;
}

// Find maximum similarities and indices (argmax) along columns
kernel void find_argmax_cols(
    constant float* similarities [[buffer(0)]], // [NA, NB] - Input similarities
    device float* max_similarities [[buffer(1)]], // [NB] - Output maximum similarities
    device uint* max_indices [[buffer(2)]],     // [NB] - Output maximum indices  
    constant uint& NA [[buffer(3)]],
    constant uint& NB [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NB) return;
    
    float max_sim = -INFINITY;
    uint max_idx = 0;
    
    // Find maximum in column gid
    for (uint i = 0; i < NA; i++) {
        float sim = similarities[i * NB + gid];
        if (sim > max_sim) {
            max_sim = sim;
            max_idx = i;
        }
    }
    
    max_similarities[gid] = max_sim;
    max_indices[gid] = max_idx;
}

// Block-wise distance computation for memory efficiency
kernel void compute_block_distances(
    constant float* A [[buffer(0)]],           // [NA, D] - Query descriptors
    constant float* B [[buffer(1)]],           // [NB, D] - Database descriptors
    device float* distances [[buffer(2]]],     // [block_A_size, block_B_size] - Output block
    constant uint& block_A_start [[buffer(3)]],
    constant uint& block_B_start [[buffer(4)]],
    constant uint& block_A_size [[buffer(5)]],
    constant uint& block_B_size [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& use_l2 [[buffer(8)]],       // 1 for L2, 0 for dot product
    uint2 gid [[thread_position_in_grid]])
{
    const uint i_local = gid.y;  // Local query index in block
    const uint j_local = gid.x;  // Local database index in block
    
    if (i_local >= block_A_size || j_local >= block_B_size) return;
    
    const uint i_global = block_A_start + i_local;
    const uint j_global = block_B_start + j_local;
    
    float result = 0.0;
    
    if (use_l2) {
        // L2 distance computation
        for (uint d = 0; d < D; d++) {
            float diff = A[i_global * D + d] - B[j_global * D + d];
            result += diff * diff;
        }
        result = sqrt(result);
    } else {
        // Dot product computation
        for (uint d = 0; d < D; d++) {
            result += A[i_global * D + d] * B[j_global * D + d];
        }
    }
    
    distances[i_local * block_B_size + j_local] = result;
} 