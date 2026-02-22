//! Advanced memory access pattern optimization
//!
//! This module provides sophisticated memory access pattern optimizations
//! including cache-friendly data layouts, prefetching strategies, and
//! NUMA-aware memory allocation patterns.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::cpu::error::CpuResult;

/// Cache line size (typically 64 bytes on modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// Memory access pattern types
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,     // Linear access through memory
    Strided(usize), // Access with fixed stride
    Random,         // Random access pattern
    Blocked(usize), // Block-based access with given block size
}

/// Memory layout optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum LayoutStrategy {
    RowMajor,       // Standard row-major layout
    ColumnMajor,    // Column-major layout
    Blocked(usize), // Blocked layout with given block size
    ZMorton,        // Z-order (Morton) curve layout
    Hilbert,        // Hilbert curve layout
    CacheOblivious, // Cache-oblivious layout
}

/// NUMA memory policy
#[derive(Debug, Clone, Copy)]
pub enum NumaPolicy {
    Local,            // Allocate on local NUMA node
    Interleave,       // Interleave across NUMA nodes
    Preferred(usize), // Prefer specific NUMA node
    Bind(usize),      // Bind to specific NUMA node
}

/// Memory prefetch strategy
#[derive(Debug, Clone, Copy)]
pub enum PrefetchStrategy {
    None,
    Software(usize), // Software prefetch with distance
    Hardware,        // Hardware prefetching
    Adaptive,        // Adaptive prefetching based on pattern
}

/// Cache-aligned memory allocator
pub struct CacheAlignedAllocator {
    alignment: usize,
    numa_policy: NumaPolicy,
}

impl Default for CacheAlignedAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAlignedAllocator {
    pub fn new() -> Self {
        Self {
            alignment: CACHE_LINE_SIZE,
            numa_policy: NumaPolicy::Local,
        }
    }

    pub fn with_alignment(alignment: usize) -> Self {
        Self {
            alignment,
            numa_policy: NumaPolicy::Local,
        }
    }

    pub fn with_numa_policy(numa_policy: NumaPolicy) -> Self {
        Self {
            alignment: CACHE_LINE_SIZE,
            numa_policy,
        }
    }

    /// Allocate cache-aligned memory
    pub fn allocate<T>(&self, count: usize) -> CpuResult<CacheAlignedBuffer<T>> {
        let layout = Layout::from_size_align(
            count * std::mem::size_of::<T>(),
            self.alignment.max(std::mem::align_of::<T>()),
        )
        .map_err(|_| crate::cpu::error::cpu_errors::memory_allocation_error("Invalid layout"))?;

        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(crate::cpu::error::cpu_errors::memory_allocation_error(
                "Allocation failed",
            ));
        }

        Ok(CacheAlignedBuffer {
            ptr: NonNull::new(ptr as *mut T).expect("ptr should not be null after null check"),
            len: count,
            layout,
            numa_policy: self.numa_policy,
        })
    }

    /// Allocate zero-initialized memory
    pub fn allocate_zeroed<T>(&self, count: usize) -> CpuResult<CacheAlignedBuffer<T>> {
        let layout = Layout::from_size_align(
            count * std::mem::size_of::<T>(),
            self.alignment.max(std::mem::align_of::<T>()),
        )
        .map_err(|_| crate::cpu::error::cpu_errors::memory_allocation_error("Invalid layout"))?;

        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(crate::cpu::error::cpu_errors::memory_allocation_error(
                "Allocation failed",
            ));
        }

        Ok(CacheAlignedBuffer {
            ptr: NonNull::new(ptr as *mut T).expect("ptr should not be null after null check"),
            len: count,
            layout,
            numa_policy: self.numa_policy,
        })
    }
}

/// Cache-aligned memory buffer
pub struct CacheAlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    #[allow(dead_code)]
    numa_policy: NumaPolicy,
}

impl<T> CacheAlignedBuffer<T> {
    /// Get raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get as slice
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is valid and points to `len` elements of type `T`
    /// - The memory is properly aligned for type `T`
    /// - The memory is not deallocated during the lifetime of the returned slice
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
    }

    /// Get as mutable slice
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is valid and points to `len` elements of type `T`
    /// - The memory is properly aligned for type `T`
    /// - The memory is not deallocated during the lifetime of the returned slice
    /// - No other references to the same memory exist during the lifetime of the returned slice
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
    }

    /// Check if pointer is cache-aligned
    pub fn is_cache_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(CACHE_LINE_SIZE)
    }
}

impl<T> Drop for CacheAlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for CacheAlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for CacheAlignedBuffer<T> {}

/// Memory access pattern optimizer
pub struct AccessPatternOptimizer {
    prefetch_strategy: PrefetchStrategy,
    layout_strategy: LayoutStrategy,
    cache_line_size: usize,
    l1_cache_size: usize,
    l2_cache_size: usize,
    l3_cache_size: usize,
}

impl Default for AccessPatternOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessPatternOptimizer {
    pub fn new() -> Self {
        Self {
            prefetch_strategy: PrefetchStrategy::Adaptive,
            layout_strategy: LayoutStrategy::RowMajor,
            cache_line_size: CACHE_LINE_SIZE,
            l1_cache_size: 32 * 1024,       // 32KB
            l2_cache_size: 256 * 1024,      // 256KB
            l3_cache_size: 8 * 1024 * 1024, // 8MB
        }
    }

    pub fn with_cache_sizes(l1: usize, l2: usize, l3: usize) -> Self {
        Self {
            prefetch_strategy: PrefetchStrategy::Adaptive,
            layout_strategy: LayoutStrategy::RowMajor,
            cache_line_size: CACHE_LINE_SIZE,
            l1_cache_size: l1,
            l2_cache_size: l2,
            l3_cache_size: l3,
        }
    }

    /// Set prefetch strategy
    pub fn set_prefetch_strategy(&mut self, strategy: PrefetchStrategy) {
        self.prefetch_strategy = strategy;
    }

    /// Set layout strategy
    pub fn set_layout_strategy(&mut self, strategy: LayoutStrategy) {
        self.layout_strategy = strategy;
    }

    /// Calculate optimal block size for cache efficiency
    pub fn optimal_block_size(&self, element_size: usize, cache_level: CacheLevel) -> usize {
        let cache_size = match cache_level {
            CacheLevel::L1 => self.l1_cache_size,
            CacheLevel::L2 => self.l2_cache_size,
            CacheLevel::L3 => self.l3_cache_size,
        };

        // Use fraction of cache to leave room for other data
        let usable_cache = cache_size / 2;
        let _elements_in_cache = usable_cache / element_size;

        // Find largest power of 2 that fits
        let mut block_size = 1;
        while block_size * block_size * element_size <= usable_cache {
            block_size *= 2;
        }
        block_size / 2
    }

    /// Optimize matrix access pattern for cache efficiency
    pub fn optimize_matrix_access<T: Copy>(
        &self,
        matrix: &[T],
        rows: usize,
        cols: usize,
        access_pattern: AccessPattern,
    ) -> Vec<usize> {
        match access_pattern {
            AccessPattern::Sequential => (0..matrix.len()).collect(),
            AccessPattern::Strided(stride) => (0..matrix.len()).step_by(stride).collect(),
            AccessPattern::Random => {
                // Generate cache-friendly random access pattern
                let mut indices: Vec<usize> = (0..matrix.len()).collect();
                // Sort by cache line to improve locality
                indices.sort_by_key(|&i| i / (self.cache_line_size / std::mem::size_of::<T>()));
                indices
            }
            AccessPattern::Blocked(block_size) => {
                self.generate_blocked_access_pattern(rows, cols, block_size)
            }
        }
    }

    /// Generate blocked access pattern for matrices
    fn generate_blocked_access_pattern(
        &self,
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<usize> {
        let mut indices = Vec::new();

        let row_blocks = (rows + block_size - 1) / block_size;
        let col_blocks = (cols + block_size - 1) / block_size;

        for block_row in 0..row_blocks {
            for block_col in 0..col_blocks {
                let start_row = block_row * block_size;
                let end_row = ((block_row + 1) * block_size).min(rows);
                let start_col = block_col * block_size;
                let end_col = ((block_col + 1) * block_size).min(cols);

                for i in start_row..end_row {
                    for j in start_col..end_col {
                        indices.push(i * cols + j);
                    }
                }
            }
        }

        indices
    }

    /// Prefetch memory for improved cache performance
    pub fn prefetch_memory<T>(&self, ptr: *const T, len: usize, access_pattern: AccessPattern) {
        match self.prefetch_strategy {
            PrefetchStrategy::None => {}
            PrefetchStrategy::Software(distance) => {
                self.software_prefetch(ptr, len, distance, access_pattern);
            }
            PrefetchStrategy::Hardware => {
                // Enable hardware prefetching (platform-specific)
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    // Intel hardware prefetch hints
                    std::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
            PrefetchStrategy::Adaptive => {
                // Choose strategy based on access pattern
                match access_pattern {
                    AccessPattern::Sequential => {
                        self.software_prefetch(ptr, len, 4, access_pattern);
                    }
                    AccessPattern::Strided(stride) if stride <= 16 => {
                        self.software_prefetch(ptr, len, 2, access_pattern);
                    }
                    _ => {
                        // Use hardware prefetching for complex patterns
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(
                                ptr as *const i8,
                                std::arch::x86_64::_MM_HINT_T1,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Software prefetching implementation
    fn software_prefetch<T>(
        &self,
        _ptr: *const T,
        len: usize,
        distance: usize,
        pattern: AccessPattern,
    ) {
        match pattern {
            AccessPattern::Sequential => {
                for _i in (0..len).step_by(self.cache_line_size / std::mem::size_of::<T>()) {
                    if _i + distance < len {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            let prefetch_ptr = _ptr.add(_i + distance);
                            std::arch::x86_64::_mm_prefetch(
                                prefetch_ptr as *const i8,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }
                }
            }
            AccessPattern::Strided(stride) => {
                for _i in (0..len).step_by(stride) {
                    if _i + distance * stride < len {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            let prefetch_ptr = _ptr.add(_i + distance * stride);
                            std::arch::x86_64::_mm_prefetch(
                                prefetch_ptr as *const i8,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }
                }
            }
            _ => {
                // Default to cache line granularity
                for _i in (0..len).step_by(self.cache_line_size / std::mem::size_of::<T>()) {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        let prefetch_ptr = _ptr.add(_i);
                        std::arch::x86_64::_mm_prefetch(
                            prefetch_ptr as *const i8,
                            std::arch::x86_64::_MM_HINT_T1,
                        );
                    }
                }
            }
        }
    }

    /// Calculate memory bandwidth utilization
    pub fn calculate_bandwidth_utilization(
        &self,
        bytes_transferred: usize,
        time_elapsed: std::time::Duration,
    ) -> f64 {
        if time_elapsed.as_secs_f64() > 0.0 {
            bytes_transferred as f64 / time_elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
        // GB/s
        } else {
            0.0
        }
    }

    /// Get optimal chunk size for streaming operations
    pub fn optimal_streaming_chunk_size(&self, element_size: usize) -> usize {
        // Target 75% of L2 cache for streaming
        let target_bytes = (self.l2_cache_size * 3) / 4;
        let chunk_elements = target_bytes / element_size;

        // Round down to nearest cache line boundary
        let elements_per_line = self.cache_line_size / element_size;
        (chunk_elements / elements_per_line) * elements_per_line
    }
}

/// Cache level enumeration
#[derive(Debug, Clone, Copy)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// Memory layout transformer for different access patterns
pub struct LayoutTransformer {
    optimizer: AccessPatternOptimizer,
}

impl Default for LayoutTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutTransformer {
    pub fn new() -> Self {
        Self {
            optimizer: AccessPatternOptimizer::new(),
        }
    }

    /// Transform matrix from row-major to blocked layout
    pub fn to_blocked_layout<T: Copy>(
        &self,
        input: &[T],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<T> {
        let mut output = vec![input[0]; input.len()]; // Use first element as default

        let row_blocks = (rows + block_size - 1) / block_size;
        let col_blocks = (cols + block_size - 1) / block_size;

        let mut out_idx = 0;

        for block_row in 0..row_blocks {
            for block_col in 0..col_blocks {
                let start_row = block_row * block_size;
                let end_row = ((block_row + 1) * block_size).min(rows);
                let start_col = block_col * block_size;
                let end_col = ((block_col + 1) * block_size).min(cols);

                for i in start_row..end_row {
                    for j in start_col..end_col {
                        if out_idx < output.len() {
                            output[out_idx] = input[i * cols + j];
                            out_idx += 1;
                        }
                    }
                }
            }
        }

        output
    }

    /// Transform matrix from blocked back to row-major layout
    pub fn from_blocked_layout<T: Copy>(
        &self,
        input: &[T],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<T> {
        let mut output = vec![input[0]; input.len()];

        let row_blocks = (rows + block_size - 1) / block_size;
        let col_blocks = (cols + block_size - 1) / block_size;

        let mut in_idx = 0;

        for block_row in 0..row_blocks {
            for block_col in 0..col_blocks {
                let start_row = block_row * block_size;
                let end_row = ((block_row + 1) * block_size).min(rows);
                let start_col = block_col * block_size;
                let end_col = ((block_col + 1) * block_size).min(cols);

                for i in start_row..end_row {
                    for j in start_col..end_col {
                        if in_idx < input.len() {
                            output[i * cols + j] = input[in_idx];
                            in_idx += 1;
                        }
                    }
                }
            }
        }

        output
    }

    /// Calculate cache miss rate estimate for given access pattern
    pub fn estimate_cache_miss_rate(
        &self,
        data_size: usize,
        access_pattern: AccessPattern,
        cache_level: CacheLevel,
    ) -> f64 {
        let cache_size = match cache_level {
            CacheLevel::L1 => self.optimizer.l1_cache_size,
            CacheLevel::L2 => self.optimizer.l2_cache_size,
            CacheLevel::L3 => self.optimizer.l3_cache_size,
        };

        match access_pattern {
            AccessPattern::Sequential => {
                if data_size <= cache_size {
                    0.01 // Very low miss rate for sequential access within cache
                } else {
                    1.0 - (cache_size as f64 / data_size as f64)
                }
            }
            AccessPattern::Strided(stride) => {
                let effective_size = data_size * stride;
                if effective_size <= cache_size {
                    0.05
                } else {
                    0.8
                }
            }
            AccessPattern::Random => {
                if data_size <= cache_size {
                    0.1
                } else {
                    0.95 // Very high miss rate for random access
                }
            }
            AccessPattern::Blocked(block_size) => {
                let block_data_size = block_size * block_size * 4; // Assume 4 bytes per element
                if block_data_size <= cache_size {
                    0.02
                } else {
                    0.3
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_allocator() {
        let allocator = CacheAlignedAllocator::new();
        let buffer = allocator.allocate::<f32>(1000).unwrap();

        assert_eq!(buffer.len(), 1000);
        assert!(buffer.is_cache_aligned());
    }

    #[test]
    fn test_access_pattern_optimizer() {
        let optimizer = AccessPatternOptimizer::new();

        let block_size = optimizer.optimal_block_size(4, CacheLevel::L1);
        assert!(block_size > 0);
        assert!(block_size.is_power_of_two());

        let streaming_chunk = optimizer.optimal_streaming_chunk_size(4);
        assert!(streaming_chunk > 0);
        assert!(streaming_chunk % (CACHE_LINE_SIZE / 4) == 0);
    }

    #[test]
    fn test_matrix_access_optimization() {
        let optimizer = AccessPatternOptimizer::new();
        let matrix = vec![1.0f32; 100];

        let indices = optimizer.optimize_matrix_access(&matrix, 10, 10, AccessPattern::Sequential);

        assert_eq!(indices.len(), matrix.len());
        assert_eq!(indices, (0..100).collect::<Vec<_>>());

        let blocked_indices =
            optimizer.optimize_matrix_access(&matrix, 10, 10, AccessPattern::Blocked(4));

        assert_eq!(blocked_indices.len(), matrix.len());
    }

    #[test]
    fn test_layout_transformer() {
        let transformer = LayoutTransformer::new();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let blocked = transformer.to_blocked_layout(&input, 4, 4, 2);
        assert_eq!(blocked.len(), input.len());

        let restored = transformer.from_blocked_layout(&blocked, 4, 4, 2);
        assert_eq!(restored, input);
    }

    #[test]
    fn test_cache_miss_estimation() {
        let transformer = LayoutTransformer::new();

        let miss_rate_seq =
            transformer.estimate_cache_miss_rate(1000, AccessPattern::Sequential, CacheLevel::L1);

        let miss_rate_random =
            transformer.estimate_cache_miss_rate(1000, AccessPattern::Random, CacheLevel::L1);

        assert!(miss_rate_seq < miss_rate_random);
    }

    #[test]
    fn test_bandwidth_calculation() {
        let optimizer = AccessPatternOptimizer::new();
        let bandwidth = optimizer.calculate_bandwidth_utilization(
            1024 * 1024 * 1024, // 1GB
            std::time::Duration::from_secs(1),
        );

        assert!((bandwidth - 1.0).abs() < 0.01); // Should be ~1 GB/s
    }
}

/// Calculate memory bandwidth in GB/s given bytes transferred and time elapsed
///
/// This is a standalone utility function for benchmarking purposes
pub fn calculate_memory_bandwidth(bytes: usize, elapsed: std::time::Duration) -> f64 {
    if elapsed.as_secs_f64() > 0.0 {
        bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0) // GB/s
    } else {
        0.0
    }
}
