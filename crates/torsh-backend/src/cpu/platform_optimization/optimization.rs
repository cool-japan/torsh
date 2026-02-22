//! Microarchitecture-specific optimization parameters
//!
//! This module defines optimization parameters that are tuned for specific
//! CPU microarchitectures to achieve optimal performance.

/// Microarchitecture-specific optimization parameters
#[derive(Debug, Clone)]
pub struct MicroarchOptimization {
    /// Optimal vector width for SIMD operations
    pub optimal_vector_width: usize,
    /// Preferred unroll factor for loops
    pub unroll_factor: usize,
    /// Optimal block size for matrix operations
    pub matrix_block_size: usize,
    /// Memory prefetch distance (cache lines ahead)
    pub prefetch_distance: usize,
    /// Branch predictor friendly loop structure
    pub branch_friendly: bool,
    /// Instruction scheduling preferences
    pub prefer_fma: bool,
    /// Cache blocking strategy
    pub cache_blocking: bool,
    /// Software prefetching enabled
    pub software_prefetch: bool,
    /// Memory alignment requirements
    pub memory_alignment: usize,
    /// Optimal chunk size for parallel operations
    pub parallel_chunk_size: usize,
    /// Hyper-threading awareness
    pub ht_aware: bool,
    /// NUMA awareness
    pub numa_aware: bool,
}

impl Default for MicroarchOptimization {
    fn default() -> Self {
        Self {
            optimal_vector_width: 32, // 256-bit vectors (AVX2)
            unroll_factor: 4,
            matrix_block_size: 64,
            prefetch_distance: 8,
            branch_friendly: true,
            prefer_fma: true,
            cache_blocking: true,
            software_prefetch: true,
            memory_alignment: 32,
            parallel_chunk_size: 1024,
            ht_aware: true,
            numa_aware: false,
        }
    }
}
