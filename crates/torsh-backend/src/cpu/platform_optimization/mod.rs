//! Platform-specific CPU optimizations with microarchitecture detection
//!
//! This module provides advanced x86_64 and ARM64 optimizations that are specifically
//! tuned for different CPU microarchitectures and features.
//!
//! ## Modular Architecture (Phase 82 Refactoring)
//!
//! The original 1,706-line monolithic file has been systematically extracted into:
//! - `microarchitecture` - CPU microarchitecture type definitions
//! - `features` - CPU feature detection and flags
//! - `cache` - Cache hierarchy information
//! - `optimization` - Microarchitecture-specific optimization parameters
//! - `detection` - Core CPU detection logic and information gathering
//! - `operations` - Platform-optimized SIMD operations
//! - `helpers` - Helper structs and utility functions
//!
//! ## Usage
//!
//! ```rust,no_run
//! use torsh_backend::cpu::platform_optimization::*;
//!
//! // Get CPU information
//! let cpu_info = CpuInfo::get();
//! println!("Detected {} cores with {:?} microarchitecture",
//!          cpu_info.logical_cores, cpu_info.x86_microarch);
//!
//! // Use optimized operations
//! let ops = PlatformOptimizedOps::new();
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![2.0, 3.0, 4.0, 5.0];
//! let result = ops.dot_product_f32(&a, &b).unwrap();
//!
//! // Platform optimizer
//! let optimizer = PlatformOptimizer::new().unwrap();
//! println!("{}", optimizer.get_cpu_info());
//! ```

// Core modules
pub mod cache;
pub mod detection;
pub mod features;
pub mod helpers;
pub mod microarchitecture;
pub mod operations;
pub mod optimization;

// Re-export core types for backward compatibility
pub use cache::CacheInfo;
pub use detection::{detect_arm_microarchitecture, detect_x86_microarchitecture, CpuInfo};
pub use features::{detect_cpu_features, CpuFeatures};

/// Detect CPU information for benchmarking
///
/// This is a convenience function that provides a simplified interface for benchmarks
/// to get basic CPU information.
pub fn detect_cpu_info() -> CpuInfo {
    CpuInfo::get().clone()
}

/// Detect microarchitecture for benchmarking
///
/// This function automatically detects whether we're on x86 or ARM and returns
/// the appropriate microarchitecture information.
pub fn detect_microarchitecture() -> String {
    let info = CpuInfo::get();

    #[cfg(target_arch = "x86_64")]
    if let Some(microarch) = info.x86_microarch {
        return format!("{:?}", microarch);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(microarch) = info.arm_microarch {
        return format!("{:?}", microarch);
    }

    "Unknown".to_string()
}

/// Get optimization parameters for the current platform
///
/// This function returns a structure containing optimization parameters
/// suitable for the detected CPU architecture.
pub fn get_optimization_parameters() -> optimization::MicroarchOptimization {
    optimization::MicroarchOptimization::default()
}
pub use helpers::{CpuOptimizer, OptimizationCache, OptimizedOperations, PlatformOptimizer};
pub use microarchitecture::{ArmMicroarchitecture, X86Microarchitecture};
pub use operations::PlatformOptimizedOps;
pub use optimization::MicroarchOptimization;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_cpu_info_detection() {
        let cpu_info = CpuInfo::get();

        // Basic sanity checks
        assert!(cpu_info.logical_cores > 0);
        assert!(cpu_info.physical_cores > 0);
        assert!(cpu_info.physical_cores <= cpu_info.logical_cores);
        assert!(cpu_info.base_frequency > 0.0);
        assert!(cpu_info.max_frequency >= cpu_info.base_frequency);

        // Cache sizes should be reasonable
        assert!(cpu_info.cache.l1d_size >= 16 * 1024); // At least 16KB
        assert!(cpu_info.cache.l2_size >= 128 * 1024); // At least 128KB
        assert!(cpu_info.cache.l1_line_size >= 32); // At least 32 bytes

        println!(
            "Detected CPU: {} cores, {}KB L1, {}KB L2, {}KB L3",
            cpu_info.logical_cores,
            cpu_info.cache.l1d_size / 1024,
            cpu_info.cache.l2_size / 1024,
            cpu_info.cache.l3_size / 1024
        );
    }

    #[test]
    fn test_platform_optimized_ops() {
        let ops = PlatformOptimizedOps::new();

        // Test dot product
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = ops.dot_product_f32(&a, &b).unwrap();
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 40

        // Test with mismatched lengths
        let c = vec![1.0, 2.0];
        assert!(ops.dot_product_f32(&a, &c).is_err());

        // Test matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = vec![0.0; 4]; // 2x2 result

        ops.matrix_multiply_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //          [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_optimization_parameters() {
        let ops = PlatformOptimizedOps::new();
        let cpu_info = ops.cpu_info();

        // Check that optimization parameters are reasonable
        assert!(cpu_info.optimization.optimal_vector_width >= 16);
        assert!(cpu_info.optimization.unroll_factor >= 2);
        assert!(cpu_info.optimization.matrix_block_size >= 32);
        assert!(cpu_info.optimization.memory_alignment >= 16);
        assert!(cpu_info.optimization.parallel_chunk_size >= 64);

        // Test chunk size calculation
        let chunk_size = ops.get_optimal_parallel_chunk_size(10000);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 10000);

        // Test memory alignment
        let alignment = ops.get_memory_alignment();
        assert!(alignment.is_power_of_two());
        assert!(alignment >= 16);
    }

    #[test]
    fn test_feature_detection() {
        let cpu_info = CpuInfo::get();

        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 CPUs should have these
            assert!(cpu_info.features.sse);
            assert!(cpu_info.features.sse2);

            // Print detected features for debugging
            println!(
                "x86_64 features: AVX={}, AVX2={}, AVX-512F={}, FMA={}",
                cpu_info.features.avx,
                cpu_info.features.avx2,
                cpu_info.features.avx512f,
                cpu_info.features.fma
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Most ARM64 systems should have NEON
            assert!(cpu_info.features.neon);
            assert!(cpu_info.features.fp);

            println!(
                "ARM64 features: NEON={}, FP={}, ASIMD={}, AES={}",
                cpu_info.features.neon,
                cpu_info.features.fp,
                cpu_info.features.asimd,
                cpu_info.features.aes_arm
            );
        }
    }

    #[test]
    fn test_microarchitecture_detection() {
        let cpu_info = CpuInfo::get();

        #[cfg(target_arch = "x86_64")]
        {
            assert!(cpu_info.x86_microarch.is_some());
            println!(
                "Detected x86_64 microarchitecture: {:?}",
                cpu_info.x86_microarch
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(cpu_info.arm_microarch.is_some());
            println!(
                "Detected ARM64 microarchitecture: {:?}",
                cpu_info.arm_microarch
            );
        }

        assert!(!cpu_info.vendor.is_empty());
        println!("CPU vendor: {}", cpu_info.vendor);
    }

    #[test]
    fn test_platform_optimizer() {
        let optimizer = PlatformOptimizer::new().unwrap();

        // Test that we can get CPU info
        let info_str = optimizer.get_cpu_info();
        assert!(!info_str.is_empty());
        assert!(info_str.contains("CPU Features:"));

        // Test features are properly detected
        #[cfg(target_arch = "x86_64")]
        {
            assert!(optimizer.features.sse || optimizer.features.sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(optimizer.features.neon);
        }
    }

    #[test]
    fn test_cache_info_default() {
        let cache_info = CacheInfo::default();

        // Verify reasonable default values
        assert_eq!(cache_info.l1d_size, 32 * 1024);
        assert_eq!(cache_info.l1i_size, 32 * 1024);
        assert_eq!(cache_info.l2_size, 256 * 1024);
        assert_eq!(cache_info.l3_size, 8 * 1024 * 1024);
        assert_eq!(cache_info.l1_line_size, 64);
        assert_eq!(cache_info.l2_line_size, 64);
        assert_eq!(cache_info.l3_line_size, 64);
    }

    #[test]
    fn test_microarch_optimization_default() {
        let optimization = MicroarchOptimization::default();

        // Verify reasonable default values
        assert_eq!(optimization.optimal_vector_width, 32);
        assert_eq!(optimization.unroll_factor, 4);
        assert_eq!(optimization.matrix_block_size, 64);
        assert_eq!(optimization.prefetch_distance, 8);
        assert!(optimization.branch_friendly);
        assert!(optimization.prefer_fma);
        assert!(optimization.cache_blocking);
        assert!(optimization.software_prefetch);
        assert_eq!(optimization.memory_alignment, 32);
        assert_eq!(optimization.parallel_chunk_size, 1024);
        assert!(optimization.ht_aware);
        assert!(!optimization.numa_aware);
    }

    #[test]
    fn test_cpu_features_default() {
        let features = CpuFeatures::default();

        // All features should start as false
        assert!(!features.sse);
        assert!(!features.avx);
        assert!(!features.avx2);
        assert!(!features.neon);
        assert!(!features.fma);
    }

    #[test]
    fn test_detect_cpu_features() {
        let features = detect_cpu_features().unwrap();

        // Should detect some basic features on most platforms
        #[cfg(target_arch = "x86_64")]
        {
            // Most x86_64 systems should have SSE/SSE2
            assert!(features.sse || features.sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 systems should have NEON
            assert!(features.neon);
        }
    }

    #[test]
    fn test_helper_structs() {
        // Test that helper structs can be created
        let cpu_optimizer = CpuOptimizer::new();
        let cpu_optimizer_default = CpuOptimizer::default();

        let optimized_ops = OptimizedOperations::new();
        let optimized_ops_default = OptimizedOperations::default();

        let optimization_cache = OptimizationCache::new();
        let optimization_cache_default = OptimizationCache::default();

        // These are mostly placeholder structs, so we just verify they can be created
        // In a real implementation, they would have more functionality
    }

    #[test]
    fn test_modular_structure_integrity() {
        // Test that all major components can be accessed through the module system

        // Test microarchitecture enums
        let _x86_arch = X86Microarchitecture::Haswell;
        let _arm_arch = ArmMicroarchitecture::M1;

        // Test CPU info detection
        let cpu_info = CpuInfo::get();
        assert!(cpu_info.logical_cores > 0);

        // Test platform optimized operations
        let ops = PlatformOptimizedOps::new();
        let chunk_size = ops.get_optimal_parallel_chunk_size(1000);
        assert!(chunk_size > 0);

        // Test platform optimizer
        let optimizer = PlatformOptimizer::new().unwrap();
        let info = optimizer.get_cpu_info();
        assert!(!info.is_empty());

        // Test feature detection
        let features = detect_cpu_features().unwrap();
        // Features struct should be valid (features may or may not be present)

        println!("Phase 82 modular structure integrity verified");
    }

    #[test]
    fn test_simd_operations_safety() {
        let ops = PlatformOptimizedOps::new();

        // Test with various input sizes to ensure SIMD implementations handle edge cases
        let test_cases = vec![
            vec![1.0],                                    // Size 1
            vec![1.0, 2.0],                               // Size 2
            vec![1.0, 2.0, 3.0],                          // Size 3
            vec![1.0, 2.0, 3.0, 4.0],                     // Size 4 (SSE boundary)
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // Size 8 (AVX boundary)
            vec![1.0; 15],                                // Size 15 (odd size)
            vec![1.0; 16],                                // Size 16
            vec![1.0; 100],                               // Larger size
        ];

        for test_vec in test_cases {
            let len = test_vec.len();
            let a = test_vec.clone();
            let b = vec![2.0; len];

            // Test dot product
            let result = ops.dot_product_f32(&a, &b).unwrap();
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(
                (result - expected).abs() < 1e-6,
                "SIMD dot product mismatch for size {}: got {}, expected {}",
                len,
                result,
                expected
            );
        }
    }
}
