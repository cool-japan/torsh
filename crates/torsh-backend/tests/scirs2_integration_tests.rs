//! SciRS2 Integration Tests
//!
//! These tests validate the integration of SciRS2 performance features:
//! - Phase 1: Parallel operations (2-4x speedup)
//! - Phase 3: Memory-aligned SIMD (2-4x speedup)
//! - Phase 4: Intelligent chunking (15-30% improvement)

use torsh_backend::cpu::scirs2_chunking_prelude::*;
use torsh_backend::cpu::scirs2_parallel::current_num_threads;
use torsh_backend::cpu::scirs2_parallel::prelude::*;
use torsh_backend::cpu::scirs2_simd_prelude::*;

#[test]
fn test_scirs2_parallel_thread_count() {
    let num_threads = current_num_threads();
    assert!(num_threads > 0, "Should have at least one thread available");
    println!("SciRS2 Parallel: {} threads available", num_threads);
}

#[test]
fn test_scirs2_parallel_for_range() {
    use std::sync::{Arc, Mutex};

    let data = Arc::new(Mutex::new(vec![0i32; 1000]));
    let data_clone = Arc::clone(&data);

    parallel_for_range(0, 1000, move |i| {
        let mut d = data_clone.lock().expect("lock should not be poisoned");
        d[i] = (i * i) as i32;
    });

    let result = data.lock().expect("lock should not be poisoned");
    for i in 0..1000 {
        assert_eq!(result[i], (i * i) as i32);
    }
}

#[test]
fn test_scirs2_parallel_map_range() {
    let result = parallel_map_range(0, 100, |i| i * 2);
    assert_eq!(result.len(), 100);
    for i in 0..100 {
        assert_eq!(result[i], i * 2);
    }
}

#[test]
fn test_scirs2_simd_available() {
    let available = scirs2_simd_available();
    println!("SciRS2 SIMD available: {}", available);

    // Should match feature flag
    #[cfg(feature = "simd")]
    assert!(available);

    #[cfg(not(feature = "simd"))]
    assert!(!available);
}

#[test]
fn test_scirs2_simd_alignment() {
    let alignment = scirs2_simd_alignment();
    assert_eq!(alignment, 64, "Should use 64-byte cache line alignment");
    assert!(alignment.is_power_of_two());
}

#[test]
fn test_scirs2_simd_aligned_vec() {
    let mut vec = AlignedVec::with_capacity(100);
    for i in 0..100 {
        vec.push(i as f32);
    }

    assert_eq!(vec.len(), 100);
    assert_eq!(vec.alignment(), 64);

    // Test aligned operations
    let mut result = AlignedVec::with_capacity(100);
    for _ in 0..100 {
        result.push(0.0);
    }

    aligned_add_f32(vec.as_slice(), vec.as_slice(), result.as_mut_slice());

    for i in 0..100 {
        assert_eq!(result.as_slice()[i], (i * 2) as f32);
    }
}

#[test]
fn test_scirs2_simd_operations() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];
    let mut result = vec![0.0f32; 5];

    // Test aligned add
    aligned_add_f32(&a, &b, &mut result);
    assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0]);

    // Test aligned mul
    aligned_mul_f32(&a, &b, &mut result);
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0]);

    // Test aligned dot
    let dot = aligned_dot_f32(&a, &b);
    assert_eq!(dot, 2.0 + 6.0 + 12.0 + 20.0 + 30.0);

    // Test aligned sum
    let sum = aligned_sum_f32(&a);
    assert_eq!(sum, 15.0);
}

#[test]
fn test_scirs2_simd_adaptive_operations() {
    let a = vec![1.0f32; 1000];
    let b = vec![2.0f32; 1000];
    let mut result = vec![0.0f32; 1000];

    // Adaptive operations should automatically choose SIMD or scalar
    adaptive_add_f32(&a, &b, &mut result);
    for &val in &result {
        assert_eq!(val, 3.0);
    }

    adaptive_mul_f32(&a, &b, &mut result);
    for &val in &result {
        assert_eq!(val, 2.0);
    }

    let dot = adaptive_dot_f32(&a, &b);
    assert_eq!(dot, 2000.0);
}

#[test]
fn test_scirs2_simd_feature_detection() {
    println!("AVX2 available: {}", has_avx2());
    println!("AVX-512 available: {}", has_avx512());
    println!("NEON available: {}", has_neon());
    println!("F32 vector width: {}", f32_vector_width());
    println!("F64 vector width: {}", f64_vector_width());

    // Feature detection should not panic
    assert!(f32_vector_width() > 0);
    assert!(f64_vector_width() > 0);

    // SIMD should be recommended for large arrays
    assert!(should_use_simd(10000) || !scirs2_simd_available());
    assert!(!should_use_simd(8) || scirs2_simd_available());
}

#[test]
fn test_scirs2_chunking_config() {
    let config = ChunkingConfig::default();

    assert!(config.l1_cache_size > 0);
    assert!(config.l2_cache_size > config.l1_cache_size);
    assert!(config.l3_cache_size > config.l2_cache_size);
    assert!(config.num_cores > 0);
    assert_eq!(config.cache_line_size, 64);

    println!("Chunking config:");
    println!("  L1 cache: {} KB", config.l1_cache_size / 1024);
    println!("  L2 cache: {} KB", config.l2_cache_size / 1024);
    println!("  L3 cache: {} MB", config.l3_cache_size / 1024 / 1024);
    println!("  CPU cores: {}", config.num_cores);
}

#[test]
fn test_scirs2_chunking_strategies() {
    // Test different workload types
    let element_size = 4; // f32
    let total_size = 10000;

    let chunk_elementwise =
        ChunkingUtils::optimal_chunk_size(WorkloadType::Elementwise, element_size, total_size);
    let chunk_matrix =
        ChunkingUtils::optimal_chunk_size(WorkloadType::Matrix, element_size, total_size);
    let chunk_reduction =
        ChunkingUtils::optimal_chunk_size(WorkloadType::Reduction, element_size, total_size);

    println!("Optimal chunk sizes for 10000 f32 elements:");
    println!("  Elementwise: {}", chunk_elementwise);
    println!("  Matrix: {}", chunk_matrix);
    println!("  Reduction: {}", chunk_reduction);

    assert!(chunk_elementwise > 0 && chunk_elementwise <= total_size);
    assert!(chunk_matrix > 0 && chunk_matrix <= total_size);
    assert!(chunk_reduction > 0 && chunk_reduction <= total_size);
}

#[test]
fn test_scirs2_chunking_range_splitting() {
    let chunks = ChunkingUtils::chunk_range(0, 10000, WorkloadType::Elementwise, 4);

    assert!(!chunks.is_empty());
    assert_eq!(chunks.first().unwrap().0, 0);
    assert_eq!(chunks.last().unwrap().1, 10000);

    // Verify chunks are contiguous
    for window in chunks.windows(2) {
        assert_eq!(window[0].1, window[1].0, "Chunks should be contiguous");
    }

    // Verify total coverage
    let total_elements: usize = chunks.iter().map(|(start, end)| end - start).sum();
    assert_eq!(total_elements, 10000);

    println!("Split 10000 elements into {} chunks", chunks.len());
}

#[test]
fn test_scirs2_chunking_matrix_blocks() {
    let (block_m, block_n, block_k) = ChunkingUtils::matrix_blocks(1024, 1024, 1024, 4);

    assert!(block_m > 0 && block_m <= 1024);
    assert!(block_n > 0 && block_n <= 1024);
    assert!(block_k > 0 && block_k <= 1024);

    println!("Matrix blocking for 1024x1024x1024 f32:");
    println!("  Block M: {}", block_m);
    println!("  Block N: {}", block_n);
    println!("  Block K: {}", block_k);

    // Blocks should fit in cache
    let config = ChunkingConfig::default();
    let block_bytes = block_m * block_n * 4;
    assert!(
        block_bytes <= config.l2_cache_size,
        "Block should fit in L2 cache"
    );
}

#[test]
fn test_scirs2_chunking_cache_alignment() {
    let element_size = 4; // f32
    let config = ChunkingConfig::default();

    // Test cache line alignment
    let aligned_size = config.cache_line_size / element_size;
    assert!(ChunkingUtils::is_cache_aligned(aligned_size, element_size));

    // Test alignment correction
    let unaligned = 100;
    let aligned = ChunkingUtils::align_to_cache_line(unaligned, element_size);
    assert!(ChunkingUtils::is_cache_aligned(aligned, element_size));
    assert!(aligned >= unaligned);

    println!(
        "Aligned {} elements to {} for cache alignment",
        unaligned, aligned
    );
}

#[test]
fn test_scirs2_chunking_strategy_builder() {
    let strategy = ChunkingStrategy::new(WorkloadType::Elementwise, 4).with_alignment(true);

    let chunk_size = strategy.chunk_size(10000);
    assert!(ChunkingUtils::is_cache_aligned(chunk_size, 4));

    let chunks = strategy.split_range(0, 10000);
    assert!(!chunks.is_empty());
    assert_eq!(chunks.first().unwrap().0, 0);
    assert_eq!(chunks.last().unwrap().1, 10000);
}

#[test]
fn test_scirs2_integrated_parallel_simd() {
    use std::sync::{Arc, Mutex};

    // Test integration of parallel and SIMD operations
    let size = 10000;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let result = Arc::new(Mutex::new(vec![0.0f32; size]));

    // Use parallel processing with SIMD operations
    let chunk_size = ChunkingUtils::optimal_chunk_size(WorkloadType::Elementwise, 4, size);

    let result_clone = Arc::clone(&result);
    let chunks: Vec<_> = (0..size).step_by(chunk_size).collect();

    chunks.into_par_iter().for_each(|start| {
        let end = (start + chunk_size).min(size);
        let mut temp_result = vec![0.0f32; end - start];

        aligned_add_f32(&a[start..end], &b[start..end], &mut temp_result);

        let mut res = result_clone.lock().expect("lock should not be poisoned");
        res[start..end].copy_from_slice(&temp_result);
    });

    // Verify results
    let final_result = result.lock().expect("lock should not be poisoned");
    for &val in final_result.iter() {
        assert_eq!(val, 3.0);
    }
}

#[test]
fn test_scirs2_workload_specific_chunking() {
    let configs = [
        ChunkingConfig::compute_intensive(),
        ChunkingConfig::memory_intensive(),
        ChunkingConfig::cache_friendly(),
    ];

    for (i, config) in configs.iter().enumerate() {
        let chunk = config.optimal_elementwise_chunk(4);
        println!("Config {}: elementwise chunk = {}", i, chunk);
        assert!(chunk > 0);
    }
}

#[test]
fn test_scirs2_performance_monitoring() {
    // This test verifies that all SciRS2 integration points are working
    println!("\n=== SciRS2 Integration Status ===");
    println!("Phase 1 (Parallel): {} threads", current_num_threads());
    println!(
        "Phase 3 (SIMD): {}",
        if scirs2_simd_available() {
            "Available"
        } else {
            "Unavailable"
        }
    );
    println!("Phase 4 (Chunking): Configured");

    let config = ChunkingConfig::default();
    println!("  CPU cores: {}", config.num_cores);
    println!("  L1 cache: {} KB", config.l1_cache_size / 1024);
    println!("  L2 cache: {} KB", config.l2_cache_size / 1024);
    println!("  SIMD width (f32): {}", f32_vector_width());

    // All integration points should be functional
    assert!(current_num_threads() > 0);
    assert!(config.num_cores > 0);
    assert!(f32_vector_width() > 0);
}
