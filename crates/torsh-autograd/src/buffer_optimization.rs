//! Temporary buffer allocation optimization for autograd operations
//!
//! This module provides efficient management of temporary buffers used during
//! gradient computations, including buffer pools, allocation patterns optimization,
//! memory alignment, and cache-friendly allocation strategies.

use crate::error_handling::{AutogradError, AutogradResult};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Buffer size categories for efficient pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BufferSizeCategory {
    Tiny = 0,   // < 1KB
    Small = 1,  // 1KB - 16KB
    Medium = 2, // 16KB - 256KB
    Large = 3,  // 256KB - 4MB
    Huge = 4,   // > 4MB
}

impl BufferSizeCategory {
    /// Get the size category for a given buffer size
    pub fn from_size(size: usize) -> Self {
        if size < 1024 {
            BufferSizeCategory::Tiny
        } else if size < 16 * 1024 {
            BufferSizeCategory::Small
        } else if size < 256 * 1024 {
            BufferSizeCategory::Medium
        } else if size < 4 * 1024 * 1024 {
            BufferSizeCategory::Large
        } else {
            BufferSizeCategory::Huge
        }
    }

    /// Get the maximum size for this category
    pub fn max_size(&self) -> usize {
        match self {
            BufferSizeCategory::Tiny => 1024,
            BufferSizeCategory::Small => 16 * 1024,
            BufferSizeCategory::Medium => 256 * 1024,
            BufferSizeCategory::Large => 4 * 1024 * 1024,
            BufferSizeCategory::Huge => usize::MAX,
        }
    }

    /// Get the minimum size for this category
    pub fn min_size(&self) -> usize {
        match self {
            BufferSizeCategory::Tiny => 0,
            BufferSizeCategory::Small => 1024,
            BufferSizeCategory::Medium => 16 * 1024,
            BufferSizeCategory::Large => 256 * 1024,
            BufferSizeCategory::Huge => 4 * 1024 * 1024,
        }
    }
}

/// Allocation strategy for buffer management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Pool-based allocation with size categories
    Pooled,
    /// Direct allocation for each request
    Direct,
    /// Hybrid approach: pooled for small buffers, direct for large
    Hybrid,
    /// Memory-mapped allocation for very large buffers
    MemoryMapped,
    /// Stack-based allocation for very small buffers
    StackBased,
}

/// Buffer allocation alignment requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAlignment {
    /// No specific alignment requirements
    None,
    /// 8-byte alignment for basic types
    Align8 = 8,
    /// 16-byte alignment for SIMD operations
    Align16 = 16,
    /// 32-byte alignment for AVX operations
    Align32 = 32,
    /// 64-byte alignment for cache line optimization
    Align64 = 64,
    /// Page-aligned for memory-mapped operations
    PageAligned = 4096,
}

/// Temporary buffer metadata
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    pub size: usize,
    pub alignment: BufferAlignment,
    pub category: BufferSizeCategory,
    pub allocation_time: Instant,
    pub last_access_time: Instant,
    pub access_count: usize,
    pub allocation_location: String,
    pub is_active: bool,
}

/// Managed temporary buffer
#[derive(Debug)]
pub struct TempBuffer {
    ptr: NonNull<u8>,
    metadata: BufferMetadata,
    layout: Layout,
}

impl TempBuffer {
    /// Create a new temporary buffer
    pub fn new(size: usize, alignment: BufferAlignment, location: &str) -> AutogradResult<Self> {
        let align = match alignment {
            BufferAlignment::None => 1,
            BufferAlignment::Align8 => 8,
            BufferAlignment::Align16 => 16,
            BufferAlignment::Align32 => 32,
            BufferAlignment::Align64 => 64,
            BufferAlignment::PageAligned => 4096,
        };

        let layout = Layout::from_size_align(size, align).map_err(|e| {
            AutogradError::gradient_computation(
                "buffer_allocation",
                format!("Invalid buffer layout: {}", e),
            )
        })?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(AutogradError::gradient_computation(
                "buffer_allocation",
                "Failed to allocate buffer memory",
            ));
        }

        let metadata = BufferMetadata {
            size,
            alignment,
            category: BufferSizeCategory::from_size(size),
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            allocation_location: location.to_string(),
            is_active: true,
        };

        Ok(Self {
            ptr: NonNull::new(ptr).expect("memory allocation returned null pointer"),
            metadata,
            layout,
        })
    }

    /// Get a raw pointer to the buffer data
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.metadata.last_access_time = Instant::now();
        self.metadata.access_count += 1;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.metadata.size) }
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.metadata.size
    }

    /// Get buffer metadata
    pub fn metadata(&self) -> &BufferMetadata {
        &self.metadata
    }

    /// Check if buffer is suitable for reuse
    pub fn is_reusable(&self, required_size: usize, required_alignment: BufferAlignment) -> bool {
        self.metadata.size >= required_size
            && self.metadata.alignment as usize >= required_alignment as usize
    }

    /// Mark buffer as accessed
    pub fn mark_accessed(&mut self) {
        self.metadata.last_access_time = Instant::now();
        self.metadata.access_count += 1;
    }
}

unsafe impl Send for TempBuffer {}
unsafe impl Sync for TempBuffer {}

impl Drop for TempBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Buffer pool for efficient reuse of temporary buffers
#[derive(Debug)]
pub struct BufferPool {
    pools: HashMap<BufferSizeCategory, VecDeque<TempBuffer>>,
    max_pool_size: usize,
    max_idle_time: Duration,
    allocation_stats: AllocationStats,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(max_pool_size: usize, max_idle_time: Duration) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
            max_idle_time,
            allocation_stats: AllocationStats::default(),
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(
        &mut self,
        size: usize,
        alignment: BufferAlignment,
        location: &str,
    ) -> AutogradResult<TempBuffer> {
        let category = BufferSizeCategory::from_size(size);

        // Try to reuse a buffer from the pool
        if let Some(pool) = self.pools.get_mut(&category) {
            // Find a suitable buffer
            for i in 0..pool.len() {
                if pool[i].is_reusable(size, alignment) {
                    let mut buffer = pool.remove(i).expect("index i is within pool bounds");
                    buffer.mark_accessed();
                    self.allocation_stats.pool_hits += 1;
                    return Ok(buffer);
                }
            }
        }

        // No suitable buffer found, allocate a new one
        let buffer = TempBuffer::new(size, alignment, location)?;
        self.allocation_stats.new_allocations += 1;
        self.allocation_stats.total_allocated += size;
        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, mut buffer: TempBuffer) {
        buffer.metadata.is_active = false;
        let category = buffer.metadata.category;

        // Clean up old buffers before adding new one
        self.cleanup_idle_buffers();

        // Add to pool if there's space
        let pool = self.pools.entry(category).or_insert_with(VecDeque::new);
        if pool.len() < self.max_pool_size {
            pool.push_back(buffer);
        } else {
            // Pool is full, drop the buffer
            self.allocation_stats.buffers_dropped += 1;
        }
    }

    /// Clean up buffers that have been idle too long
    pub fn cleanup_idle_buffers(&mut self) {
        let now = Instant::now();
        let max_idle = self.max_idle_time;

        for pool in self.pools.values_mut() {
            pool.retain(|buffer| {
                let idle_time = now.duration_since(buffer.metadata.last_access_time);
                if idle_time > max_idle {
                    self.allocation_stats.buffers_cleaned += 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Get allocation statistics
    pub fn get_stats(&self) -> AllocationStats {
        self.allocation_stats.clone()
    }

    /// Get pool utilization statistics
    pub fn get_pool_utilization(&self) -> HashMap<BufferSizeCategory, (usize, usize)> {
        let mut utilization = HashMap::new();
        for (&category, pool) in &self.pools {
            utilization.insert(category, (pool.len(), self.max_pool_size));
        }
        utilization
    }
}

/// Statistics about buffer allocation patterns
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub new_allocations: usize,
    pub pool_hits: usize,
    pub buffers_dropped: usize,
    pub buffers_cleaned: usize,
    pub total_allocated: usize,
    pub peak_pool_size: usize,
}

impl AllocationStats {
    /// Calculate pool hit rate
    pub fn hit_rate(&self) -> f64 {
        let total_requests = self.new_allocations + self.pool_hits;
        if total_requests == 0 {
            0.0
        } else {
            self.pool_hits as f64 / total_requests as f64
        }
    }

    /// Get average allocation size
    pub fn average_allocation_size(&self) -> f64 {
        if self.new_allocations == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.new_allocations as f64
        }
    }
}

/// Optimized buffer allocator with multiple strategies
#[derive(Debug)]
pub struct OptimizedBufferAllocator {
    strategy: AllocationStrategy,
    pool: Option<Arc<Mutex<BufferPool>>>,
    allocation_patterns: AllocationPatternAnalyzer,
    cache_optimization: CacheOptimizer,
}

impl OptimizedBufferAllocator {
    /// Create a new optimized buffer allocator
    pub fn new(strategy: AllocationStrategy) -> Self {
        let pool = match strategy {
            AllocationStrategy::Pooled | AllocationStrategy::Hybrid => {
                Some(Arc::new(Mutex::new(BufferPool::new(
                    1000,                     // max pool size
                    Duration::from_secs(300), // 5 minutes idle time
                ))))
            }
            _ => None,
        };

        Self {
            strategy,
            pool,
            allocation_patterns: AllocationPatternAnalyzer::new(),
            cache_optimization: CacheOptimizer::new(),
        }
    }

    /// Allocate a temporary buffer with optimization
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: BufferAlignment,
        location: &str,
    ) -> AutogradResult<TempBuffer> {
        // Record allocation pattern
        self.allocation_patterns
            .record_allocation(size, alignment, location);

        // Apply cache optimization suggestions
        let optimized_size = self.cache_optimization.optimize_size(size);
        let optimized_alignment = self.cache_optimization.optimize_alignment(alignment);

        match self.strategy {
            AllocationStrategy::Pooled => {
                if let Some(ref pool) = self.pool {
                    pool.lock()
                        .expect("lock should not be poisoned")
                        .get_buffer(optimized_size, optimized_alignment, location)
                } else {
                    TempBuffer::new(optimized_size, optimized_alignment, location)
                }
            }
            AllocationStrategy::Direct => {
                TempBuffer::new(optimized_size, optimized_alignment, location)
            }
            AllocationStrategy::Hybrid => {
                let category = BufferSizeCategory::from_size(optimized_size);
                if matches!(
                    category,
                    BufferSizeCategory::Tiny
                        | BufferSizeCategory::Small
                        | BufferSizeCategory::Medium
                ) {
                    // Use pool for smaller buffers
                    if let Some(ref pool) = self.pool {
                        pool.lock()
                            .expect("lock should not be poisoned")
                            .get_buffer(optimized_size, optimized_alignment, location)
                    } else {
                        TempBuffer::new(optimized_size, optimized_alignment, location)
                    }
                } else {
                    // Direct allocation for larger buffers
                    TempBuffer::new(optimized_size, optimized_alignment, location)
                }
            }
            AllocationStrategy::MemoryMapped => {
                // For very large buffers, consider memory mapping
                if optimized_size > 16 * 1024 * 1024 {
                    // Would implement memory mapping here
                    TempBuffer::new(optimized_size, optimized_alignment, location)
                } else {
                    TempBuffer::new(optimized_size, optimized_alignment, location)
                }
            }
            AllocationStrategy::StackBased => {
                // For very small buffers, could use stack allocation
                TempBuffer::new(optimized_size, optimized_alignment, location)
            }
        }
    }

    /// Deallocate a temporary buffer
    pub fn deallocate(&mut self, buffer: TempBuffer) {
        match self.strategy {
            AllocationStrategy::Pooled | AllocationStrategy::Hybrid => {
                if let Some(ref pool) = self.pool {
                    pool.lock()
                        .expect("lock should not be poisoned")
                        .return_buffer(buffer);
                }
                // If no pool, buffer will be dropped automatically
            }
            _ => {
                // Direct deallocation - buffer will be dropped automatically
            }
        }
    }

    /// Get allocation statistics
    pub fn get_allocation_stats(&self) -> Option<AllocationStats> {
        self.pool.as_ref().map(|pool| {
            pool.lock()
                .expect("lock should not be poisoned")
                .get_stats()
        })
    }

    /// Get allocation pattern analysis
    pub fn get_pattern_analysis(&self) -> &AllocationPatternAnalyzer {
        &self.allocation_patterns
    }

    /// Perform maintenance (cleanup, optimization)
    pub fn perform_maintenance(&mut self) {
        if let Some(ref pool) = self.pool {
            pool.lock()
                .expect("lock should not be poisoned")
                .cleanup_idle_buffers();
        }
        self.allocation_patterns.analyze_patterns();
        self.cache_optimization
            .update_optimizations(&self.allocation_patterns);
    }
}

/// Analyzes allocation patterns to suggest optimizations
#[derive(Debug)]
pub struct AllocationPatternAnalyzer {
    allocation_history: VecDeque<AllocationRecord>,
    size_frequency: HashMap<usize, usize>,
    alignment_frequency: HashMap<BufferAlignment, usize>,
    location_frequency: HashMap<String, usize>,
    max_history_size: usize,
}

#[derive(Debug, Clone)]
struct AllocationRecord {
    size: usize,
    alignment: BufferAlignment,
    location: String,
    timestamp: Instant,
}

impl AllocationPatternAnalyzer {
    /// Create a new allocation pattern analyzer
    pub fn new() -> Self {
        Self {
            allocation_history: VecDeque::new(),
            size_frequency: HashMap::new(),
            alignment_frequency: HashMap::new(),
            location_frequency: HashMap::new(),
            max_history_size: 10000,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize, alignment: BufferAlignment, location: &str) {
        let record = AllocationRecord {
            size,
            alignment,
            location: location.to_string(),
            timestamp: Instant::now(),
        };

        self.allocation_history.push_back(record);

        // Update frequency counters
        *self.size_frequency.entry(size).or_insert(0) += 1;
        *self.alignment_frequency.entry(alignment).or_insert(0) += 1;
        *self
            .location_frequency
            .entry(location.to_string())
            .or_insert(0) += 1;

        // Limit history size
        if self.allocation_history.len() > self.max_history_size {
            if let Some(old_record) = self.allocation_history.pop_front() {
                // Decrement frequency counters for removed record
                if let Some(count) = self.size_frequency.get_mut(&old_record.size) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.size_frequency.remove(&old_record.size);
                    }
                }
                if let Some(count) = self.alignment_frequency.get_mut(&old_record.alignment) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.alignment_frequency.remove(&old_record.alignment);
                    }
                }
                if let Some(count) = self.location_frequency.get_mut(&old_record.location) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.location_frequency.remove(&old_record.location);
                    }
                }
            }
        }
    }

    /// Analyze patterns and suggest optimizations
    pub fn analyze_patterns(&self) -> PatternAnalysis {
        let most_common_sizes: Vec<_> = {
            let mut sizes: Vec<_> = self.size_frequency.iter().collect();
            sizes.sort_by(|a, b| b.1.cmp(a.1));
            sizes
                .into_iter()
                .take(10)
                .map(|(&size, &count)| (size, count))
                .collect()
        };

        let most_common_alignments: Vec<_> = {
            let mut alignments: Vec<_> = self.alignment_frequency.iter().collect();
            alignments.sort_by(|a, b| b.1.cmp(a.1));
            alignments
                .into_iter()
                .take(5)
                .map(|(&alignment, &count)| (alignment, count))
                .collect()
        };

        let hottest_locations: Vec<_> = {
            let mut locations: Vec<_> = self.location_frequency.iter().collect();
            locations.sort_by(|a, b| b.1.cmp(a.1));
            locations
                .into_iter()
                .take(10)
                .map(|(location, &count)| (location.clone(), count))
                .collect()
        };

        PatternAnalysis {
            total_allocations: self.allocation_history.len(),
            most_common_sizes,
            most_common_alignments,
            hottest_locations,
        }
    }

    /// Get current allocation rate (allocations per second)
    pub fn get_allocation_rate(&self) -> f64 {
        if self.allocation_history.is_empty() {
            return 0.0;
        }

        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60); // Last minute

        let recent_allocations = self
            .allocation_history
            .iter()
            .rev()
            .take_while(|record| record.timestamp > cutoff)
            .count();

        recent_allocations as f64 / 60.0
    }
}

/// Results of allocation pattern analysis
#[derive(Debug)]
pub struct PatternAnalysis {
    pub total_allocations: usize,
    pub most_common_sizes: Vec<(usize, usize)>,
    pub most_common_alignments: Vec<(BufferAlignment, usize)>,
    pub hottest_locations: Vec<(String, usize)>,
}

/// Cache-aware optimization for buffer allocation
#[derive(Debug)]
pub struct CacheOptimizer {
    cache_line_size: usize,
    preferred_alignments: HashMap<usize, BufferAlignment>,
    size_adjustments: HashMap<usize, usize>,
}

impl CacheOptimizer {
    /// Create a new cache optimizer
    pub fn new() -> Self {
        Self {
            cache_line_size: 64, // Typical cache line size
            preferred_alignments: HashMap::new(),
            size_adjustments: HashMap::new(),
        }
    }

    /// Optimize buffer size for cache efficiency
    pub fn optimize_size(&self, size: usize) -> usize {
        // Check if we have a specific optimization for this size
        if let Some(&adjusted_size) = self.size_adjustments.get(&size) {
            return adjusted_size;
        }

        // Round up to cache line boundary for larger buffers
        if size > self.cache_line_size {
            let cache_lines = (size + self.cache_line_size - 1) / self.cache_line_size;
            cache_lines * self.cache_line_size
        } else {
            // For smaller buffers, round to power of 2
            size.next_power_of_two()
        }
    }

    /// Optimize alignment for cache efficiency
    pub fn optimize_alignment(&self, alignment: BufferAlignment) -> BufferAlignment {
        match alignment {
            BufferAlignment::None => BufferAlignment::Align8,
            BufferAlignment::Align8 => BufferAlignment::Align16,
            other => other,
        }
    }

    /// Update optimizations based on allocation patterns
    pub fn update_optimizations(&mut self, analyzer: &AllocationPatternAnalyzer) {
        let analysis = analyzer.analyze_patterns();

        // Update preferred alignments based on common patterns
        for (alignment, count) in analysis.most_common_alignments {
            if count > 100 {
                // Threshold for frequent usage
                for (size, _) in &analysis.most_common_sizes {
                    self.preferred_alignments.insert(*size, alignment);
                }
            }
        }
    }
}

/// Global optimized buffer allocator
static GLOBAL_ALLOCATOR: std::sync::OnceLock<Arc<RwLock<OptimizedBufferAllocator>>> =
    std::sync::OnceLock::new();

/// Get the global buffer allocator
pub fn get_global_allocator() -> &'static Arc<RwLock<OptimizedBufferAllocator>> {
    GLOBAL_ALLOCATOR.get_or_init(|| {
        Arc::new(RwLock::new(OptimizedBufferAllocator::new(
            AllocationStrategy::Hybrid,
        )))
    })
}

/// Convenience functions for global buffer allocation
pub fn allocate_temp_buffer(
    size: usize,
    alignment: BufferAlignment,
    location: &str,
) -> AutogradResult<TempBuffer> {
    get_global_allocator()
        .write()
        .expect("lock should not be poisoned")
        .allocate(size, alignment, location)
}

pub fn deallocate_temp_buffer(buffer: TempBuffer) {
    get_global_allocator()
        .write()
        .expect("lock should not be poisoned")
        .deallocate(buffer);
}

pub fn get_global_allocation_stats() -> Option<AllocationStats> {
    get_global_allocator()
        .read()
        .expect("lock should not be poisoned")
        .get_allocation_stats()
}

pub fn perform_global_maintenance() {
    get_global_allocator()
        .write()
        .expect("lock should not be poisoned")
        .perform_maintenance();
}

/// RAII wrapper for automatic buffer cleanup
#[derive(Debug)]
pub struct AutoBuffer {
    buffer: Option<TempBuffer>,
}

impl AutoBuffer {
    /// Create a new auto-managed buffer
    pub fn new(size: usize, alignment: BufferAlignment, location: &str) -> AutogradResult<Self> {
        let buffer = allocate_temp_buffer(size, alignment, location)?;
        Ok(Self {
            buffer: Some(buffer),
        })
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.buffer.as_mut().map(|b| b.as_mut_slice())
    }

    /// Get buffer size
    pub fn size(&self) -> Option<usize> {
        self.buffer.as_ref().map(|b| b.size())
    }

    /// Take ownership of the inner buffer
    pub fn take(mut self) -> Option<TempBuffer> {
        self.buffer.take()
    }
}

impl Drop for AutoBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            deallocate_temp_buffer(buffer);
        }
    }
}

/// Macro for convenient temporary buffer allocation
#[macro_export]
macro_rules! temp_buffer {
    ($size:expr) => {
        $crate::buffer_optimization::AutoBuffer::new(
            $size,
            $crate::buffer_optimization::BufferAlignment::Align16,
            concat!(file!(), ":", line!()),
        )
    };
    ($size:expr, $align:expr) => {
        $crate::buffer_optimization::AutoBuffer::new($size, $align, concat!(file!(), ":", line!()))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_category() {
        assert_eq!(BufferSizeCategory::from_size(512), BufferSizeCategory::Tiny);
        assert_eq!(
            BufferSizeCategory::from_size(8192),
            BufferSizeCategory::Small
        );
        assert_eq!(
            BufferSizeCategory::from_size(128 * 1024),
            BufferSizeCategory::Medium
        );
        assert_eq!(
            BufferSizeCategory::from_size(2 * 1024 * 1024),
            BufferSizeCategory::Large
        );
        assert_eq!(
            BufferSizeCategory::from_size(8 * 1024 * 1024),
            BufferSizeCategory::Huge
        );
    }

    #[test]
    fn test_temp_buffer_creation() {
        let buffer = TempBuffer::new(1024, BufferAlignment::Align16, "test").unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.metadata().alignment, BufferAlignment::Align16);
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(10, Duration::from_secs(60));

        // Allocate a buffer
        let buffer1 = pool
            .get_buffer(1024, BufferAlignment::Align16, "test1")
            .unwrap();
        assert_eq!(buffer1.size(), 1024);

        // Return it to the pool
        pool.return_buffer(buffer1);

        // Get another buffer of the same size (should reuse)
        let buffer2 = pool
            .get_buffer(1024, BufferAlignment::Align16, "test2")
            .unwrap();
        assert_eq!(buffer2.size(), 1024);

        let stats = pool.get_stats();
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.new_allocations, 1);
    }

    #[test]
    fn test_optimized_allocator() {
        let mut allocator = OptimizedBufferAllocator::new(AllocationStrategy::Pooled);

        let buffer1 = allocator
            .allocate(1024, BufferAlignment::Align16, "test1")
            .unwrap();
        assert_eq!(buffer1.size(), 1024);

        allocator.deallocate(buffer1);

        let buffer2 = allocator
            .allocate(1024, BufferAlignment::Align16, "test2")
            .unwrap();
        assert_eq!(buffer2.size(), 1024);

        if let Some(stats) = allocator.get_allocation_stats() {
            assert!(stats.total_allocated > 0);
        }
    }

    #[test]
    fn test_allocation_pattern_analyzer() {
        let mut analyzer = AllocationPatternAnalyzer::new();

        // Record some allocations
        analyzer.record_allocation(1024, BufferAlignment::Align16, "test1");
        analyzer.record_allocation(1024, BufferAlignment::Align16, "test2");
        analyzer.record_allocation(2048, BufferAlignment::Align32, "test3");

        let analysis = analyzer.analyze_patterns();
        assert_eq!(analysis.total_allocations, 3);
        assert!(!analysis.most_common_sizes.is_empty());
        assert!(!analysis.most_common_alignments.is_empty());

        let rate = analyzer.get_allocation_rate();
        assert!(rate >= 0.0);
    }

    #[test]
    fn test_cache_optimizer() {
        let optimizer = CacheOptimizer::new();

        // Test size optimization
        let optimized_size = optimizer.optimize_size(100);
        assert!(optimized_size >= 100);

        // Test alignment optimization
        let optimized_alignment = optimizer.optimize_alignment(BufferAlignment::None);
        assert_ne!(optimized_alignment, BufferAlignment::None);
    }

    #[test]
    fn test_auto_buffer() {
        let mut auto_buffer = AutoBuffer::new(1024, BufferAlignment::Align16, "test").unwrap();
        assert_eq!(auto_buffer.size(), Some(1024));

        if let Some(slice) = auto_buffer.as_mut_slice() {
            assert_eq!(slice.len(), 1024);
            slice[0] = 42;
            assert_eq!(slice[0], 42);
        }
    }

    #[test]
    fn test_global_allocator() {
        let buffer = allocate_temp_buffer(1024, BufferAlignment::Align16, "test").unwrap();
        assert_eq!(buffer.size(), 1024);

        deallocate_temp_buffer(buffer);

        if let Some(stats) = get_global_allocation_stats() {
            assert!(stats.total_allocated > 0);
        }

        perform_global_maintenance();
    }
}
