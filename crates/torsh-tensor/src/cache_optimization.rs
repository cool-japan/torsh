// Cache optimization module for improving memory layout and access patterns

use crate::{Tensor, TensorStorage};
use scirs2_core::simd_aligned::AlignedVec;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
    shape::Shape,
};

/// Cache analysis report providing detailed performance metrics
#[derive(Debug, Clone)]
pub struct CacheAnalysisReport {
    /// Overall cache efficiency score (0.0 to 1.0)
    pub cache_efficiency: f64,
    /// Estimated number of cache misses for typical access patterns
    pub estimated_cache_misses: usize,
    /// Spatial locality score (0.0 to 1.0)
    pub spatial_locality_score: f64,
    /// Temporal locality score (0.0 to 1.0)
    pub temporal_locality_score: f64,
    /// Whether current memory layout is optimal
    pub memory_layout_optimal: bool,
    /// List of recommended optimizations
    pub recommended_optimizations: Vec<String>,
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Memory layout optimization for cache efficiency
    /// Analyzes and optimizes the tensor's memory layout to improve cache performance
    pub fn optimize_cache_layout(&mut self) -> Result<()> {
        // Check if tensor is large enough to benefit from optimization
        if self.numel() < 1024 {
            return Ok(()); // Skip small tensors
        }

        // Analyze current access pattern and stride layout
        let current_strides = self.compute_strides();
        let optimal_order = self.determine_optimal_dimension_order(&current_strides);

        // If current layout is already optimal, return early
        if optimal_order.iter().enumerate().all(|(i, &dim)| dim == i) {
            return Ok(());
        }

        // Reorganize data for better cache locality
        self.reorder_dimensions(&optimal_order)?;

        // Add padding for cache line alignment if beneficial
        self.add_cache_padding()?;

        Ok(())
    }

    /// Determine optimal dimension order for cache efficiency
    /// Prioritizes dimensions that are accessed more frequently together
    fn determine_optimal_dimension_order(&self, strides: &[usize]) -> Vec<usize> {
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        let mut dim_priorities: Vec<(usize, f64)> = (0..dims.len())
            .map(|i| {
                // Calculate priority based on dimension size and stride
                let size_factor = dims[i] as f64;
                let stride_factor = 1.0 / (strides[i] as f64 + 1.0);
                let cache_friendliness = size_factor * stride_factor;
                (i, cache_friendliness)
            })
            .collect();

        // Sort by cache friendliness (higher is better)
        dim_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        dim_priorities.into_iter().map(|(dim, _)| dim).collect()
    }

    /// Reorder tensor dimensions for optimal cache access
    fn reorder_dimensions(&mut self, optimal_order: &[usize]) -> Result<()> {
        if optimal_order.len() != self.ndim() {
            return Err(TorshError::InvalidOperation(
                "Dimension order length mismatch".to_string(),
            ));
        }

        // Create permutation for transpose operation
        let data = self.to_vec()?;
        let old_dims = self.shape().dims().to_vec();
        let old_strides = self.compute_strides();

        // Calculate new dimensions and create reordered data
        let new_dims: Vec<usize> = optimal_order.iter().map(|&i| old_dims[i]).collect();
        let new_numel = new_dims.iter().product::<usize>();
        let mut new_data = vec![data[0]; new_numel]; // Initialize with first element

        // Reorder data according to optimal dimension order
        #[allow(clippy::needless_range_loop)]
        for i in 0..new_numel {
            let mut old_indices = vec![0; self.ndim()];
            let mut remaining = i;

            // Convert flat index to multi-dimensional indices in new layout
            for (j, &dim_size) in new_dims.iter().enumerate().rev() {
                old_indices[optimal_order[j]] = remaining % dim_size;
                remaining /= dim_size;
            }

            // Calculate flat index in original layout
            let old_flat_index: usize = old_indices
                .iter()
                .zip(old_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum();

            new_data[i] = data[old_flat_index];
        }

        // Update tensor with optimized layout
        self.storage = TensorStorage::create_optimal(new_data)?;
        self.shape = Shape::new(new_dims);

        Ok(())
    }

    /// Add cache-line aligned padding for better memory access patterns
    fn add_cache_padding(&mut self) -> Result<()> {
        const CACHE_LINE_SIZE: usize = 64; // bytes
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = CACHE_LINE_SIZE / element_size;

        // Only add padding if it would be beneficial
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        if dims.is_empty() || dims[dims.len() - 1] % elements_per_cache_line == 0 {
            return Ok(()); // Already aligned or no benefit
        }

        // Calculate padding needed for last dimension
        let last_dim = dims[dims.len() - 1];
        let padded_last_dim = last_dim.div_ceil(elements_per_cache_line) * elements_per_cache_line;
        let padding_needed = padded_last_dim - last_dim;

        // Only add padding if overhead is reasonable (< 25%)
        if (padding_needed as f64 / last_dim as f64) > 0.25 {
            return Ok(());
        }

        let data = self.to_vec()?;
        let mut new_dims = dims.to_vec();
        let last_idx = new_dims.len() - 1;
        new_dims[last_idx] = padded_last_dim;

        // Create padded data
        let new_numel = new_dims.iter().product::<usize>();
        let mut padded_data = Vec::with_capacity(new_numel);

        let outer_size = new_numel / padded_last_dim;
        for i in 0..outer_size {
            let start_idx = i * last_dim;
            let end_idx = (i + 1) * last_dim;

            // Copy original data
            padded_data.extend_from_slice(&data[start_idx..end_idx]);

            // Add padding (zeros)
            for _ in 0..padding_needed {
                padded_data.push(data[0]); // Use first element as padding value
            }
        }

        // Update tensor with padded layout
        self.storage = TensorStorage::create_optimal(padded_data)?;
        self.shape = Shape::new(new_dims);

        Ok(())
    }

    /// Analyze memory access patterns and provide optimization recommendations
    pub fn analyze_cache_performance(&self) -> CacheAnalysisReport {
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        let strides = self.compute_strides();
        let numel = self.numel();

        // Calculate cache efficiency metrics
        let mut cache_misses_estimate = 0f64;

        // Estimate cache misses based on stride patterns
        for (i, &stride) in strides.iter().enumerate() {
            let dimension_accesses = dims[i] as f64;
            let stride_penalty = if stride > 64 {
                stride as f64 / 64.0
            } else {
                1.0
            };
            cache_misses_estimate += dimension_accesses * stride_penalty;
        }

        // Calculate spatial locality (how well adjacent elements are accessed together)
        let spatial_locality_score = if strides.last().copied().unwrap_or(1) == 1usize {
            1.0
        } else {
            1.0 / strides.last().copied().unwrap_or(1) as f64
        };

        // Calculate temporal locality (reuse of recently accessed data)
        let temporal_locality_score = 1.0 / (numel as f64).log2().max(1.0);

        CacheAnalysisReport {
            cache_efficiency: (spatial_locality_score + temporal_locality_score) / 2.0,
            estimated_cache_misses: cache_misses_estimate as usize,
            spatial_locality_score,
            temporal_locality_score,
            memory_layout_optimal: strides.last().copied().unwrap_or(1) == 1usize,
            recommended_optimizations: self.generate_optimization_recommendations(&strides),
        }
    }

    /// Generate specific optimization recommendations based on current layout
    fn generate_optimization_recommendations(&self, strides: &[usize]) -> Vec<String> {
        let mut recommendations = Vec::new();
        let shape_binding = self.shape();
        let dims = shape_binding.dims();

        // Check for non-contiguous memory layout
        if strides.last().copied().unwrap_or(1) != 1 {
            recommendations
                .push("Consider using .contiguous() to ensure row-major layout".to_string());
        }

        // Check for small tensors that don't benefit from optimization
        if self.numel() < 1024 {
            recommendations.push("Tensor too small to benefit from cache optimization".to_string());
        }

        // Check for dimensions that could benefit from reordering
        if dims.len() > 2 {
            let largest_dim = dims.iter().enumerate().max_by_key(|(_, &size)| size);
            if let Some((largest_idx, _)) = largest_dim {
                if largest_idx != dims.len() - 1 {
                    recommendations.push(format!(
                        "Consider moving dimension {largest_idx} to the end for better cache locality"
                    ));
                }
            }
        }

        // Check for padding opportunities
        const CACHE_LINE_SIZE: usize = 64;
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = CACHE_LINE_SIZE / element_size;

        if !dims.is_empty() {
            let last_dim = dims[dims.len() - 1];
            if last_dim % elements_per_cache_line != 0 {
                recommendations
                    .push("Consider adding cache-line padding for better alignment".to_string());
            }
        }

        recommendations
    }

    /// Create a cache-optimized copy of the tensor
    pub fn to_cache_optimized(&self) -> Result<Self> {
        let mut optimized = self.clone();
        optimized.optimize_cache_layout()?;
        Ok(optimized)
    }

    /// Get memory usage statistics for the tensor
    pub fn memory_stats(&self) -> MemoryStats {
        let element_size = std::mem::size_of::<T>();
        let total_elements = self.numel();
        let total_bytes = total_elements * element_size;

        // Estimate memory overhead based on storage type
        let overhead_bytes = match &self.storage {
            TensorStorage::InMemory(_) => {
                // Arc + RwLock overhead
                std::mem::size_of::<std::sync::Arc<std::sync::RwLock<Vec<T>>>>()
            }
            TensorStorage::MemoryMapped(_) => {
                // Memory mapped storage overhead
                1024 // Approximate overhead for file handles, cache, etc.
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(_) => {
                // Arc + RwLock + AlignedVec overhead
                std::mem::size_of::<std::sync::Arc<std::sync::RwLock<AlignedVec<T>>>>()
            }
        };

        MemoryStats {
            total_bytes,
            element_size,
            total_elements,
            overhead_bytes,
            is_memory_mapped: matches!(&self.storage, TensorStorage::MemoryMapped(_)),
        }
    }
}

/// Memory usage statistics for a tensor
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory used by tensor data in bytes
    pub total_bytes: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Memory overhead from storage structures
    pub overhead_bytes: usize,
    /// Whether tensor uses memory-mapped storage
    pub is_memory_mapped: bool,
}

impl MemoryStats {
    /// Get effective memory usage (data + overhead)
    pub fn effective_bytes(&self) -> usize {
        self.total_bytes + self.overhead_bytes
    }

    /// Get memory efficiency (data bytes / total bytes)
    pub fn efficiency(&self) -> f64 {
        self.total_bytes as f64 / self.effective_bytes() as f64
    }
}

/// Global memory pool for temporary tensor allocations
pub struct TensorMemoryPool {
    /// Pooled memory blocks organized by size
    pool: Arc<Mutex<HashMap<usize, Vec<Vec<u8>>>>>,
    /// Memory allocation statistics
    stats: Arc<Mutex<PoolStatistics>>,
    /// Maximum memory pool size in bytes
    max_pool_size: usize,
    /// Current pool size in bytes
    current_pool_size: Arc<Mutex<usize>>,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    pub allocations: usize,
    pub deallocations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub peak_memory_usage: usize,
    pub total_memory_saved: usize,
}

impl TensorMemoryPool {
    /// Create a new memory pool with specified maximum size
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PoolStatistics::default())),
            max_pool_size: max_size_mb * 1024 * 1024,
            current_pool_size: Arc::new(Mutex::new(0)),
        }
    }

    /// Allocate memory from pool or create new
    pub fn allocate(&self, size_bytes: usize) -> Vec<u8> {
        let mut pool = self.pool.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.allocations += 1;

        // Round up to next power of 2 for better pooling
        let rounded_size = size_bytes.next_power_of_two();

        if let Some(pool_vec) = pool.get_mut(&rounded_size) {
            if let Some(memory) = pool_vec.pop() {
                stats.cache_hits += 1;
                let mut current_size = self.current_pool_size.lock().unwrap();
                *current_size -= rounded_size;
                return memory;
            }
        }

        stats.cache_misses += 1;
        vec![0u8; rounded_size]
    }

    /// Return memory to pool
    pub fn deallocate(&self, mut memory: Vec<u8>) {
        let size = memory.len();
        let mut pool = self.pool.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let mut current_size = self.current_pool_size.lock().unwrap();

        stats.deallocations += 1;

        // Only pool if under size limit
        if *current_size + size <= self.max_pool_size {
            // Clear the memory before pooling for security
            memory.fill(0);

            pool.entry(size).or_default().push(memory);
            *current_size += size;
            stats.total_memory_saved += size;
        }

        stats.peak_memory_usage = stats.peak_memory_usage.max(*current_size);
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the entire pool
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        let mut current_size = self.current_pool_size.lock().unwrap();

        pool.clear();
        *current_size = 0;
    }
}

/// Memory pressure detection and adaptive allocation
pub struct MemoryPressureMonitor {
    /// Memory usage samples
    samples: Arc<Mutex<Vec<(Instant, usize)>>>,
    /// Current pressure level (0.0 to 1.0)
    pressure_level: Arc<Mutex<f64>>,
    /// System memory threshold for high pressure
    high_pressure_threshold: usize,
}

impl MemoryPressureMonitor {
    pub fn new(memory_limit_mb: usize) -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            pressure_level: Arc::new(Mutex::new(0.0)),
            high_pressure_threshold: memory_limit_mb * 1024 * 1024,
        }
    }

    /// Record memory usage sample
    pub fn record_usage(&self, bytes_used: usize) {
        let mut samples = self.samples.lock().unwrap();
        let mut pressure = self.pressure_level.lock().unwrap();

        let now = Instant::now();
        samples.push((now, bytes_used));

        // Keep only recent samples (last 60 seconds)
        samples.retain(|(time, _)| now.duration_since(*time) < Duration::from_secs(60));

        // Calculate pressure based on recent usage
        let avg_usage = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|(_, usage)| *usage as f64).sum::<f64>() / samples.len() as f64
        };

        *pressure = (avg_usage / self.high_pressure_threshold as f64).min(1.0);
    }

    /// Get current memory pressure level
    pub fn get_pressure_level(&self) -> f64 {
        *self.pressure_level.lock().unwrap()
    }

    /// Check if system is under high memory pressure
    pub fn is_high_pressure(&self) -> bool {
        self.get_pressure_level() > 0.8
    }
}

/// NUMA-aware memory allocation hints
#[derive(Debug, Clone, Copy)]
pub enum NumaNode {
    Local,
    Node(u32),
    Interleaved,
}

#[derive(Debug, Clone)]
pub struct NumaAllocationHint {
    pub preferred_node: NumaNode,
    pub allow_fallback: bool,
    pub bind_threads: bool,
}

impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Advanced memory optimization with NUMA awareness
    pub fn optimize_memory_layout(&mut self, numa_hint: Option<NumaAllocationHint>) -> Result<()> {
        // Basic cache optimization
        self.optimize_cache_layout()?;

        // Apply NUMA optimization if hint provided
        if let Some(hint) = numa_hint {
            self.apply_numa_optimization(hint)?;
        }

        // Memory access pattern prediction
        self.optimize_access_patterns()?;

        Ok(())
    }

    /// Apply NUMA-specific optimizations
    fn apply_numa_optimization(&mut self, _hint: NumaAllocationHint) -> Result<()> {
        // NUMA optimization would require platform-specific implementation
        // For now, we'll implement basic interleaving for large tensors
        if self.numel() > 1_000_000 {
            // Large tensors benefit from interleaved allocation
            // This would require platform-specific NUMA API calls
            // For now, just ensure contiguous layout
            if !self.is_contiguous() {
                let contiguous_tensor = self.contiguous()?;
                *self = contiguous_tensor;
            }
        }
        Ok(())
    }

    /// Optimize memory access patterns based on predicted usage
    fn optimize_access_patterns(&mut self) -> Result<()> {
        let shape_binding = self.shape();
        let dims = shape_binding.dims();

        // For matrices, optimize for row-major access
        if dims.len() == 2 && dims[0] > 64 && dims[1] > 64 {
            // Check if we should transpose for better cache behavior
            let row_size = dims[1] * std::mem::size_of::<T>();
            let cache_line_size = 64;

            // If rows don't align well with cache lines, consider optimization
            if row_size % cache_line_size != 0 && row_size < cache_line_size * 4 {
                self.add_cache_padding()?;
            }
        }

        // For 3D+ tensors, ensure innermost dimension is cache-friendly
        if dims.len() >= 3 {
            let innermost_size = dims[dims.len() - 1] * std::mem::size_of::<T>();
            if !(32..=256).contains(&innermost_size) {
                // Consider reshaping for better cache utilization
                self.add_cache_padding()?;
            }
        }

        Ok(())
    }

    /// Memory-mapped tensor creation with optimization hints
    pub fn create_memory_mapped_optimized(
        data: Vec<T>,
        shape: Vec<usize>,
        numa_hint: Option<NumaAllocationHint>,
    ) -> Result<Self> {
        let mut tensor = Self::from_data(data, shape, torsh_core::device::DeviceType::Cpu)?;
        tensor.optimize_memory_layout(numa_hint)?;
        Ok(tensor)
    }

    /// Prefetch memory pages for better performance
    pub fn prefetch_data(&self) -> Result<()> {
        // This would use madvise/PrefetchVirtualMemory on supported platforms
        // For now, we'll implement a simple memory access pattern
        if self.numel() > 10_000 {
            let data = self.to_vec()?;
            let stride = data.len() / 100; // Sample every 1% of data

            // Touch memory at regular intervals to trigger prefetch
            let mut _sum = T::default();
            for i in (0..data.len()).step_by(stride.max(1)) {
                _sum = data[i]; // Simple memory access to trigger prefetch
            }
        }
        Ok(())
    }
}

// Global memory pool instance
static GLOBAL_MEMORY_POOL: std::sync::OnceLock<TensorMemoryPool> = std::sync::OnceLock::new();
static MEMORY_PRESSURE_MONITOR: std::sync::OnceLock<MemoryPressureMonitor> =
    std::sync::OnceLock::new();

/// Get global memory pool
pub fn get_memory_pool() -> &'static TensorMemoryPool {
    GLOBAL_MEMORY_POOL.get_or_init(|| TensorMemoryPool::new(1024)) // 1GB default
}

/// Get memory pressure monitor
pub fn get_memory_pressure_monitor() -> &'static MemoryPressureMonitor {
    MEMORY_PRESSURE_MONITOR.get_or_init(|| MemoryPressureMonitor::new(8192)) // 8GB default
}

#[cfg(test)]
mod tests {
    use crate::creation::*;

    #[test]
    fn test_cache_optimization() {
        let mut tensor = ones::<f32>(&[100, 100]).unwrap();
        assert!(tensor.optimize_cache_layout().is_ok());
    }

    #[test]
    fn test_cache_analysis() {
        let tensor = ones::<f32>(&[64, 64]).unwrap();
        let report = tensor.analyze_cache_performance();
        assert!(report.cache_efficiency >= 0.0 && report.cache_efficiency <= 1.0);
    }

    #[test]
    fn test_contiguous_layout() {
        let tensor = ones::<f32>(&[10, 10]).unwrap();
        assert!(tensor.is_contiguous());

        let contiguous = tensor.contiguous().unwrap();
        assert!(contiguous.is_contiguous());
    }

    #[test]
    fn test_memory_stats() {
        let tensor = ones::<f32>(&[100, 100]).unwrap();
        let stats = tensor.memory_stats();
        assert_eq!(stats.total_elements, 10000);
        assert_eq!(stats.element_size, 4); // f32 is 4 bytes
        assert_eq!(stats.total_bytes, 40000);
    }

    #[test]
    fn test_memory_pool() {
        use super::*;

        let pool = TensorMemoryPool::new(10); // 10 MB

        // Test allocation
        let memory1 = pool.allocate(1024);
        assert_eq!(memory1.len(), 1024);

        let memory2 = pool.allocate(2048);
        assert_eq!(memory2.len(), 2048);

        // Test deallocation and reuse
        pool.deallocate(memory1);
        let memory3 = pool.allocate(1024);
        assert_eq!(memory3.len(), 1024);

        // Check statistics
        let stats = pool.get_statistics();
        assert!(stats.allocations > 0);
        assert!(stats.deallocations > 0);

        pool.deallocate(memory2);
        pool.deallocate(memory3);
    }

    #[test]
    fn test_memory_pressure_monitor() {
        use super::*;

        let monitor = MemoryPressureMonitor::new(100); // 100 MB limit

        // Test pressure calculation - monitor uses average of samples
        monitor.record_usage(50 * 1024 * 1024); // 50 MB
        assert!(monitor.get_pressure_level() < 0.6);

        monitor.record_usage(90 * 1024 * 1024); // 90 MB
                                                // Average of 50MB and 90MB = 70MB = 0.7 pressure
        assert!(monitor.get_pressure_level() > 0.6);
        assert!(monitor.get_pressure_level() < 0.8);
        assert!(!monitor.is_high_pressure()); // 0.7 < 0.8, so not high pressure

        // Add a higher pressure reading to trigger high pressure
        monitor.record_usage(95 * 1024 * 1024); // 95 MB
                                                // Average of 50MB, 90MB, and 95MB = ~78MB = 0.78 pressure (still < 0.8)
        monitor.record_usage(100 * 1024 * 1024); // 100 MB
                                                 // This should push the average above 0.8
        assert!(monitor.is_high_pressure());
    }

    #[test]
    fn test_advanced_memory_optimization() {
        let mut tensor = ones::<f32>(&[64, 64]).unwrap();

        // Test with NUMA hint
        let numa_hint = super::NumaAllocationHint {
            preferred_node: super::NumaNode::Local,
            allow_fallback: true,
            bind_threads: false,
        };

        assert!(tensor.optimize_memory_layout(Some(numa_hint)).is_ok());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_cache_optimized_creation() {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let shape = vec![100, 100];

        let numa_hint = super::NumaAllocationHint {
            preferred_node: super::NumaNode::Interleaved,
            allow_fallback: true,
            bind_threads: false,
        };

        let tensor = super::Tensor::create_memory_mapped_optimized(data, shape, Some(numa_hint));
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        // Shape may be optimized with padding for cache efficiency
        let shape = tensor.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 100); // First dimension should be preserved
        assert!(dims[1] >= 100); // Second dimension may have padding
    }

    #[test]
    fn test_memory_prefetch() {
        let tensor = ones::<f32>(&[200, 200]).unwrap();
        assert!(tensor.prefetch_data().is_ok());
    }

    #[test]
    fn test_global_memory_pool_access() {
        use super::*;

        let pool = get_memory_pool();
        let memory = pool.allocate(1024);
        assert_eq!(memory.len(), 1024);
        pool.deallocate(memory);

        let monitor = get_memory_pressure_monitor();
        monitor.record_usage(1024 * 1024); // 1 MB
        assert!(monitor.get_pressure_level() >= 0.0);
    }

    #[test]
    fn test_pool_statistics() {
        use super::*;

        let pool = TensorMemoryPool::new(5); // 5 MB

        // Perform multiple allocations and deallocations
        let mut memories = Vec::new();
        for i in 0..10 {
            let size = (i + 1) * 512;
            memories.push(pool.allocate(size));
        }

        for memory in memories {
            pool.deallocate(memory);
        }

        let stats = pool.get_statistics();
        assert_eq!(stats.allocations, 10);
        assert_eq!(stats.deallocations, 10);
        assert!(stats.cache_hits + stats.cache_misses == 10);

        pool.clear();
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        let tensor = ones::<f32>(&[50, 50]).unwrap();
        let stats = tensor.memory_stats();

        let efficiency = stats.efficiency();
        assert!(efficiency > 0.0 && efficiency <= 1.0);

        let effective = stats.effective_bytes();
        assert!(effective >= stats.total_bytes);
    }
}
