//! Advanced CPU backend optimizations

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::cpu::error::CpuResult;
use rayon::prelude::*;

/// Kernel fusion optimizer for combining multiple operations
pub struct KernelFusionOptimizer {
    /// Cache of fused kernels
    fused_kernels: Arc<Mutex<HashMap<String, Box<dyn FusedKernel + Send + Sync>>>>,
    /// Enable/disable fusion
    enabled: bool,
}

impl Default for KernelFusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelFusionOptimizer {
    /// Create new kernel fusion optimizer
    pub fn new() -> Self {
        Self {
            fused_kernels: Arc::new(Mutex::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Enable or disable kernel fusion
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if fusion is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register a fused kernel
    pub fn register_fused_kernel(&self, name: String, kernel: Box<dyn FusedKernel + Send + Sync>) {
        let mut kernels = self.fused_kernels.lock().unwrap();
        kernels.insert(name, kernel);
    }

    /// Execute fused operation if available
    pub fn try_execute_fused(
        &self,
        operation_sequence: &[&str],
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> CpuResult<bool> {
        if !self.enabled {
            return Ok(false);
        }

        let fusion_key = operation_sequence.join("->");
        let kernels = self.fused_kernels.lock().unwrap();

        if let Some(kernel) = kernels.get(&fusion_key) {
            kernel.execute(inputs, outputs)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Common fused operations
    #[allow(clippy::too_many_arguments)]
    pub fn conv_relu_fusion(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        input_shape: (usize, usize, usize, usize), // (N, C, H, W)
        kernel_shape: (usize, usize, usize, usize), // (K, C, H, W)
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> CpuResult<()> {
        // Fused convolution + ReLU operation
        let (n, c, h, w) = input_shape;
        let (k, _, kh, kw) = kernel_shape;
        let (sh, sw) = stride;
        let (ph, pw) = padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        // Parallel processing over output channels and batch
        output
            .par_chunks_mut(out_h * out_w)
            .enumerate()
            .for_each(|(out_idx, out_slice)| {
                let batch_idx = out_idx / k;
                let channel_idx = out_idx % k;

                if batch_idx >= n {
                    return;
                }

                // Perform convolution for this output channel
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        // Convolve with kernel
                        for ic in 0..c {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    let ih = oh * sh + kh_idx;
                                    let iw = ow * sw + kw_idx;

                                    if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                                        let input_h = ih - ph;
                                        let input_w = iw - pw;

                                        if input_h < h && input_w < w {
                                            let input_idx = batch_idx * c * h * w
                                                + ic * h * w
                                                + input_h * w
                                                + input_w;

                                            let weight_idx = channel_idx * c * kh * kw
                                                + ic * kh * kw
                                                + kh_idx * kw
                                                + kw_idx;

                                            sum += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if let Some(bias_data) = bias {
                            sum += bias_data[channel_idx];
                        }

                        // Apply ReLU activation (fusion)
                        let out_idx = oh * out_w + ow;
                        out_slice[out_idx] = sum.max(0.0);
                    }
                }
            });

        Ok(())
    }

    /// Linear + activation fusion
    #[allow(clippy::too_many_arguments)]
    pub fn linear_activation_fusion(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        input_shape: (usize, usize),  // (batch_size, input_dim)
        weight_shape: (usize, usize), // (output_dim, input_dim)
        activation: ActivationType,
    ) -> CpuResult<()> {
        let (batch_size, input_dim) = input_shape;
        let (output_dim, _) = weight_shape;

        // Parallel processing over batch
        output
            .par_chunks_mut(output_dim)
            .enumerate()
            .for_each(|(batch_idx, batch_output)| {
                if batch_idx >= batch_size {
                    return;
                }

                // Compute linear transformation
                for out_idx in 0..output_dim {
                    let mut sum = 0.0f32;

                    // Matrix multiplication
                    for in_idx in 0..input_dim {
                        let input_val = input[batch_idx * input_dim + in_idx];
                        let weight_val = weight[out_idx * input_dim + in_idx];
                        sum += input_val * weight_val;
                    }

                    // Add bias if provided
                    if let Some(bias_data) = bias {
                        sum += bias_data[out_idx];
                    }

                    // Apply activation
                    batch_output[out_idx] = match activation {
                        ActivationType::ReLU => sum.max(0.0),
                        ActivationType::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                        ActivationType::Tanh => sum.tanh(),
                        ActivationType::GELU => {
                            sum * 0.5
                                * (1.0 + (sum * std::f32::consts::FRAC_2_SQRT_PI * 0.5).tanh())
                        }
                        ActivationType::None => sum,
                    };
                }
            });

        Ok(())
    }
}

/// Trait for fused kernels
pub trait FusedKernel {
    fn execute(&self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> CpuResult<()>;
}

/// Activation types for fusion
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
}

/// Memory optimization manager
pub struct MemoryOptimizer {
    /// Memory pool for reusing buffers
    memory_pool: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>,
    /// Enable memory pooling
    pooling_enabled: bool,
    /// Memory usage statistics
    stats: Arc<Mutex<MemoryStats>>,
}

#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub peak_usage: usize,
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryOptimizer {
    /// Create new memory optimizer
    pub fn new() -> Self {
        Self {
            memory_pool: Arc::new(Mutex::new(HashMap::new())),
            pooling_enabled: true,
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    /// Enable or disable memory pooling
    pub fn set_pooling_enabled(&mut self, enabled: bool) {
        self.pooling_enabled = enabled;
    }

    /// Get buffer from pool or allocate new one
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        if !self.pooling_enabled {
            return vec![0.0; size];
        }

        let mut pool = self.memory_pool.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(buffers) = pool.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                stats.pool_hits += 1;
                return buffer;
            }
        }

        stats.pool_misses += 1;
        stats.total_allocated += size * std::mem::size_of::<f32>();
        stats.peak_usage = stats.peak_usage.max(stats.total_allocated);

        vec![0.0; size]
    }

    /// Return buffer to pool
    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        if !self.pooling_enabled {
            return;
        }

        let size = buffer.len();
        buffer.fill(0.0); // Clear for reuse

        let mut pool = self.memory_pool.lock().unwrap();
        pool.entry(size).or_default().push(buffer);
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let stats = self.stats.lock().unwrap();
        MemoryStats {
            total_allocated: stats.total_allocated,
            pool_hits: stats.pool_hits,
            pool_misses: stats.pool_misses,
            peak_usage: stats.peak_usage,
        }
    }

    /// Clear memory pool
    pub fn clear_pool(&self) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.clear();

        let mut stats = self.stats.lock().unwrap();
        stats.total_allocated = 0;
    }
}

/// Cache line size for memory optimization (typically 64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Work stealing statistics for thread pool optimization
#[derive(Debug, Default)]
pub struct WorkStealingStats {
    pub tasks_executed: AtomicUsize,
    pub tasks_stolen: AtomicUsize,
    pub idle_time_ns: AtomicUsize,
    pub work_time_ns: AtomicUsize,
}

impl WorkStealingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_task_executed(&self) {
        self.tasks_executed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_task_stolen(&self) {
        self.tasks_stolen.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_work_time(&self, duration: Duration) {
        self.work_time_ns
            .fetch_add(duration.as_nanos() as usize, Ordering::Relaxed);
    }

    pub fn record_idle_time(&self, duration: Duration) {
        self.idle_time_ns
            .fetch_add(duration.as_nanos() as usize, Ordering::Relaxed);
    }

    pub fn get_steal_ratio(&self) -> f64 {
        let executed = self.tasks_executed.load(Ordering::Relaxed);
        let stolen = self.tasks_stolen.load(Ordering::Relaxed);
        if executed > 0 {
            stolen as f64 / executed as f64
        } else {
            0.0
        }
    }

    pub fn get_efficiency(&self) -> f64 {
        let work_time = self.work_time_ns.load(Ordering::Relaxed);
        let idle_time = self.idle_time_ns.load(Ordering::Relaxed);
        let total_time = work_time + idle_time;
        if total_time > 0 {
            work_time as f64 / total_time as f64
        } else {
            0.0
        }
    }
}

/// Thread pool optimizer for better parallelization
pub struct ThreadPoolOptimizer {
    /// Number of threads to use
    num_threads: usize,
    /// Thread affinity settings
    affinity_enabled: bool,
    /// Cache size per core (in bytes)
    l1_cache_size: usize,
    l2_cache_size: usize,
    l3_cache_size: usize,
    /// Work stealing metrics
    work_stealing_stats: Arc<WorkStealingStats>,
    /// Custom thread pool for advanced scheduling
    custom_pool: Option<rayon::ThreadPool>,
}

impl Default for ThreadPoolOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadPoolOptimizer {
    /// Create new thread pool optimizer
    pub fn new() -> Self {
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Detect cache sizes (simplified detection, could be enhanced with CPUID)
        let (l1_cache, l2_cache, l3_cache) = Self::detect_cache_sizes();

        Self {
            num_threads,
            affinity_enabled: false,
            l1_cache_size: l1_cache,
            l2_cache_size: l2_cache,
            l3_cache_size: l3_cache,
            work_stealing_stats: Arc::new(WorkStealingStats::new()),
            custom_pool: None,
        }
    }

    /// Detect CPU cache sizes (simplified implementation)
    fn detect_cache_sizes() -> (usize, usize, usize) {
        // Default values for common x86_64 architectures
        // In a real implementation, we would use CPUID or system files
        #[cfg(target_arch = "x86_64")]
        {
            (32 * 1024, 256 * 1024, 8 * 1024 * 1024) // 32KB L1, 256KB L2, 8MB L3
        }
        #[cfg(target_arch = "aarch64")]
        {
            (64 * 1024, 512 * 1024, 4 * 1024 * 1024) // Apple Silicon typical values
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            (32 * 1024, 256 * 1024, 2 * 1024 * 1024) // Conservative defaults
        }
    }

    /// Set number of threads
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;

        // Create custom thread pool with enhanced settings
        let builder = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(2 * 1024 * 1024) // 2MB stack for deep recursions
            .thread_name(|index| format!("torsh-cpu-{}", index))
            .panic_handler(|_| {
                eprintln!("Thread panic in torsh CPU backend");
            });

        match builder.build() {
            Ok(pool) => {
                self.custom_pool = Some(pool);
            }
            Err(_) => {
                // Fallback to global pool update
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build_global()
                    .ok();
            }
        }
    }

    /// Get number of threads
    pub fn get_num_threads(&self) -> usize {
        self.num_threads
    }

    /// Enable thread affinity
    pub fn set_affinity_enabled(&mut self, enabled: bool) {
        self.affinity_enabled = enabled;
    }

    /// Get cache-aware chunk size for optimal memory access patterns
    pub fn get_cache_aware_chunk_size(&self, total_size: usize, element_size: usize) -> usize {
        // Calculate optimal chunk size based on L1 cache
        let l1_elements = self.l1_cache_size / element_size;
        let cache_friendly_chunk = l1_elements / 2; // Leave room for other data

        // Balance between cache efficiency and parallelism
        let parallel_chunk = total_size / self.num_threads;
        let min_chunk = 64; // Minimum to avoid overhead

        // Choose the size that provides good cache locality while maintaining parallelism
        if cache_friendly_chunk < min_chunk {
            min_chunk
        } else if parallel_chunk < cache_friendly_chunk {
            parallel_chunk.max(min_chunk)
        } else {
            cache_friendly_chunk
        }
    }

    /// Get optimal chunk size for parallel operations (legacy method)
    pub fn get_optimal_chunk_size(&self, total_size: usize, min_chunk_size: usize) -> usize {
        self.get_cache_aware_chunk_size(total_size, 4)
            .max(min_chunk_size)
    }

    /// Calculate optimal block size for matrix operations
    pub fn get_matrix_block_size(&self, _matrix_elements: usize, element_size: usize) -> usize {
        // For square matrices, find block size that fits in L2 cache
        let l2_elements = self.l2_cache_size / element_size;
        let sqrt_elements = (l2_elements as f64).sqrt() as usize;

        // Ensure block size is a power of 2 for better alignment
        let mut block_size = 1;
        while block_size < sqrt_elements {
            block_size *= 2;
        }
        block_size / 2 // Back off one level for safety
    }

    /// Get cache line aligned size
    pub fn get_cache_aligned_size(&self, size: usize) -> usize {
        ((size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE
    }

    /// Execute parallel operation with optimal chunking
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, op: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        let chunk_size = self.get_optimal_chunk_size(range.len(), 1);

        if let Some(pool) = &self.custom_pool {
            pool.install(|| {
                range.into_par_iter().with_min_len(chunk_size).for_each(op);
            });
        } else {
            range.into_par_iter().with_min_len(chunk_size).for_each(op);
        }
    }

    /// Execute cache-aware parallel operation with memory locality optimization
    pub fn parallel_for_cache_aware<F>(
        &self,
        range: std::ops::Range<usize>,
        element_size: usize,
        op: F,
    ) where
        F: Fn(usize) + Send + Sync,
    {
        let chunk_size = self.get_cache_aware_chunk_size(range.len(), element_size);
        let stats = Arc::clone(&self.work_stealing_stats);

        let timed_op = move |idx: usize| {
            let start = Instant::now();
            stats.record_task_executed();
            op(idx);
            stats.record_work_time(start.elapsed());
        };

        if let Some(pool) = &self.custom_pool {
            pool.install(|| {
                range
                    .into_par_iter()
                    .with_min_len(chunk_size)
                    .for_each(timed_op);
            });
        } else {
            range
                .into_par_iter()
                .with_min_len(chunk_size)
                .for_each(timed_op);
        }
    }

    /// Execute parallel map operation with cache-aware chunking
    pub fn parallel_map<T, R, F>(&self, data: &[T], op: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Send + Sync,
    {
        let chunk_size = self.get_optimal_chunk_size(data.len(), 1);

        if let Some(pool) = &self.custom_pool {
            pool.install(|| data.par_iter().with_min_len(chunk_size).map(op).collect())
        } else {
            data.par_iter().with_min_len(chunk_size).map(op).collect()
        }
    }

    /// Cache-aware parallel matrix multiplication block processing
    pub fn parallel_matrix_blocks<F>(&self, rows: usize, cols: usize, element_size: usize, op: F)
    where
        F: Fn(usize, usize, usize, usize) + Send + Sync, // (row_start, row_end, col_start, col_end)
    {
        let block_size = self.get_matrix_block_size(rows * cols, element_size);

        let row_blocks = (rows + block_size - 1) / block_size;
        let col_blocks = (cols + block_size - 1) / block_size;

        (0..row_blocks).into_par_iter().for_each(|row_block| {
            (0..col_blocks).into_par_iter().for_each(|col_block| {
                let row_start = row_block * block_size;
                let row_end = ((row_block + 1) * block_size).min(rows);
                let col_start = col_block * block_size;
                let col_end = ((col_block + 1) * block_size).min(cols);

                op(row_start, row_end, col_start, col_end);
            });
        });
    }

    /// Parallel reduction with work stealing optimization
    pub fn parallel_reduce<T, F, R, C>(&self, data: &[T], identity: R, map_op: F, reduce_op: C) -> R
    where
        T: Sync,
        R: Send + Clone + Sync,
        F: Fn(R, &T) -> R + Send + Sync,
        C: Fn(R, R) -> R + Send + Sync,
    {
        let chunk_size = self.get_cache_aware_chunk_size(data.len(), std::mem::size_of::<T>());

        if let Some(pool) = &self.custom_pool {
            pool.install(|| {
                data.par_chunks(chunk_size)
                    .map(|chunk| chunk.iter().fold(identity.clone(), &map_op))
                    .reduce(|| identity.clone(), &reduce_op)
            })
        } else {
            data.par_chunks(chunk_size)
                .map(|chunk| chunk.iter().fold(identity.clone(), &map_op))
                .reduce(|| identity.clone(), &reduce_op)
        }
    }

    /// Get work stealing statistics
    pub fn get_work_stealing_stats(&self) -> (f64, f64) {
        (
            self.work_stealing_stats.get_steal_ratio(),
            self.work_stealing_stats.get_efficiency(),
        )
    }

    /// Reset work stealing statistics
    pub fn reset_stats(&self) {
        let stats = &self.work_stealing_stats;
        stats.tasks_executed.store(0, Ordering::Relaxed);
        stats.tasks_stolen.store(0, Ordering::Relaxed);
        stats.work_time_ns.store(0, Ordering::Relaxed);
        stats.idle_time_ns.store(0, Ordering::Relaxed);
    }

    /// Get cache information
    pub fn get_cache_info(&self) -> (usize, usize, usize) {
        (self.l1_cache_size, self.l2_cache_size, self.l3_cache_size)
    }
}

/// Combined optimization manager
pub struct OptimizationManager {
    /// Kernel fusion optimizer
    pub kernel_fusion: KernelFusionOptimizer,
    /// Memory optimizer
    pub memory: MemoryOptimizer,
    /// Thread pool optimizer
    pub thread_pool: ThreadPoolOptimizer,
    /// Global optimization settings
    optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    None,
    Basic,
    Balanced,
    Aggressive,
}

impl OptimizationManager {
    /// Create new optimization manager
    pub fn new(level: OptimizationLevel) -> Self {
        let mut manager = Self {
            kernel_fusion: KernelFusionOptimizer::new(),
            memory: MemoryOptimizer::new(),
            thread_pool: ThreadPoolOptimizer::new(),
            optimization_level: level,
        };

        // Configure based on optimization level
        match level {
            OptimizationLevel::None => {
                manager.kernel_fusion.set_enabled(false);
                manager.memory.set_pooling_enabled(false);
            }
            OptimizationLevel::Basic => {
                manager.kernel_fusion.set_enabled(true);
                manager.memory.set_pooling_enabled(true);
            }
            OptimizationLevel::Balanced => {
                manager.kernel_fusion.set_enabled(true);
                manager.memory.set_pooling_enabled(true);
                // Moderate thread pool optimizations
            }
            OptimizationLevel::Aggressive => {
                manager.kernel_fusion.set_enabled(true);
                manager.memory.set_pooling_enabled(true);
                manager.thread_pool.set_affinity_enabled(true);

                // Use more threads for aggressive optimization
                let max_threads = thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                manager.thread_pool.set_num_threads(max_threads);
            }
        }

        manager
    }

    /// Get optimization level
    pub fn get_optimization_level(&self) -> OptimizationLevel {
        self.optimization_level
    }

    /// Set optimization level
    pub fn set_optimization_level(&mut self, level: OptimizationLevel) {
        self.optimization_level = level;

        // Reconfigure optimizers
        match level {
            OptimizationLevel::None => {
                self.kernel_fusion.set_enabled(false);
                self.memory.set_pooling_enabled(false);
            }
            OptimizationLevel::Basic => {
                self.kernel_fusion.set_enabled(true);
                self.memory.set_pooling_enabled(true);
            }
            OptimizationLevel::Balanced => {
                self.kernel_fusion.set_enabled(true);
                self.memory.set_pooling_enabled(true);
                // Moderate thread pool optimizations
            }
            OptimizationLevel::Aggressive => {
                self.kernel_fusion.set_enabled(true);
                self.memory.set_pooling_enabled(true);
                self.thread_pool.set_affinity_enabled(true);
            }
        }
    }

    /// Print optimization statistics
    pub fn print_stats(&self) {
        let mem_stats = self.memory.get_stats();
        let (steal_ratio, efficiency) = self.thread_pool.get_work_stealing_stats();
        let (l1, l2, l3) = self.thread_pool.get_cache_info();

        println!("Optimization Statistics:");
        println!("  Memory Pool Hits: {}", mem_stats.pool_hits);
        println!("  Memory Pool Misses: {}", mem_stats.pool_misses);
        println!("  Total Allocated: {} bytes", mem_stats.total_allocated);
        println!("  Peak Usage: {} bytes", mem_stats.peak_usage);
        println!("  Threads: {}", self.thread_pool.get_num_threads());
        println!("  Kernel Fusion: {}", self.kernel_fusion.is_enabled());
        println!("  Work Steal Ratio: {:.2}%", steal_ratio * 100.0);
        println!("  Thread Efficiency: {:.2}%", efficiency * 100.0);
        println!(
            "  Cache Sizes: L1={} KB, L2={} KB, L3={} KB",
            l1 / 1024,
            l2 / 1024,
            l3 / 1024
        );
    }
}

impl Default for OptimizationManager {
    fn default() -> Self {
        Self::new(OptimizationLevel::Basic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_fusion_optimizer() {
        let optimizer = KernelFusionOptimizer::new();
        assert!(optimizer.is_enabled());
    }

    #[test]
    fn test_memory_optimizer() {
        let optimizer = MemoryOptimizer::new();

        // Test buffer allocation and return
        let buffer = optimizer.get_buffer(1024);
        assert_eq!(buffer.len(), 1024);

        optimizer.return_buffer(buffer);

        // Test pool hit
        let buffer2 = optimizer.get_buffer(1024);
        assert_eq!(buffer2.len(), 1024);

        let stats = optimizer.get_stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_thread_pool_optimizer() {
        let optimizer = ThreadPoolOptimizer::new();
        assert!(optimizer.get_num_threads() > 0);

        let chunk_size = optimizer.get_optimal_chunk_size(1000, 10);
        assert!(chunk_size >= 10);

        // Test cache-aware chunk size
        let cache_chunk = optimizer.get_cache_aware_chunk_size(1000, 4);
        assert!(cache_chunk > 0);

        // Test matrix block size
        let block_size = optimizer.get_matrix_block_size(1000, 4);
        assert!(block_size > 0);
        assert!(block_size.is_power_of_two());

        // Test cache alignment
        let aligned_size = optimizer.get_cache_aligned_size(100);
        assert_eq!(aligned_size % CACHE_LINE_SIZE, 0);

        // Test cache info
        let (l1, l2, l3) = optimizer.get_cache_info();
        assert!(l1 > 0);
        assert!(l2 > l1);
        assert!(l3 > l2);
    }

    #[test]
    fn test_optimization_manager() {
        let manager = OptimizationManager::new(OptimizationLevel::Basic);
        assert!(manager.kernel_fusion.is_enabled());
        assert!(manager.memory.pooling_enabled);
    }

    #[test]
    fn test_conv_relu_fusion() {
        let optimizer = KernelFusionOptimizer::new();

        // Simple test case
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
        let weight = vec![0.5, 0.5, 0.5, 0.5]; // 1x1x2x2
        let mut output = vec![0.0; 1]; // 1x1x1x1

        let result = optimizer.conv_relu_fusion(
            &input,
            &weight,
            None,
            &mut output,
            (1, 1, 2, 2), // input shape
            (1, 1, 2, 2), // kernel shape
            (1, 1),       // stride
            (0, 0),       // padding
        );

        assert!(result.is_ok());
        assert!(output[0] > 0.0); // ReLU should have activated
    }

    #[test]
    fn test_linear_activation_fusion() {
        let optimizer = KernelFusionOptimizer::new();

        let input = vec![1.0, 2.0]; // batch_size=1, input_dim=2
        let weight = vec![0.5, 0.5]; // output_dim=1, input_dim=2
        let bias = vec![0.1];
        let mut output = vec![0.0]; // batch_size=1, output_dim=1

        let result = optimizer.linear_activation_fusion(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            (1, 2), // input shape
            (1, 2), // weight shape
            ActivationType::ReLU,
        );

        assert!(result.is_ok());
        assert!(output[0] > 0.0); // Should be 1.0*0.5 + 2.0*0.5 + 0.1 = 1.6
    }

    #[test]
    fn test_work_stealing_stats() {
        let stats = WorkStealingStats::new();

        // Record some metrics
        stats.record_task_executed();
        stats.record_task_stolen();
        stats.record_work_time(Duration::from_millis(10));
        stats.record_idle_time(Duration::from_millis(2));

        assert!(stats.get_steal_ratio() > 0.0);
        assert!(stats.get_efficiency() > 0.0);
    }

    #[test]
    fn test_cache_aware_parallel_operations() {
        let optimizer = ThreadPoolOptimizer::new();

        // Test parallel_for_cache_aware
        let data = vec![0; 1000];
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        optimizer.parallel_for_cache_aware(0..data.len(), 4, move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), data.len());

        // Test parallel_reduce
        let data: Vec<i32> = (0..1000).collect();
        let sum = optimizer.parallel_reduce(&data, 0i32, |acc, &val| acc + val, |a, b| a + b);

        let expected_sum: i32 = (0..1000).sum();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn test_parallel_matrix_blocks() {
        let optimizer = ThreadPoolOptimizer::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        optimizer.parallel_matrix_blocks(100, 100, 4, move |_r1, _r2, _c1, _c2| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        assert!(counter.load(Ordering::Relaxed) > 0);
    }
}
