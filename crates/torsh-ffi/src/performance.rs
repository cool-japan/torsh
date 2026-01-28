//! Performance optimizations and batched operations for ToRSh FFI
//!
//! This module provides optimized operations for better performance in FFI scenarios,
//! including batched operations, memory pooling, and async processing.

#![allow(dead_code)]

use crate::c_api::*;
use crate::error::{FfiError, FfiResult};
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Performance statistics for monitoring FFI operations
/// Uses atomic operations for lock-free counter updates
#[derive(Debug)]
pub struct PerformanceStats {
    pub total_operations: AtomicU64,
    pub total_time_ms: AtomicU64,
    pub min_time_ms: AtomicU64,
    pub max_time_ms: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub memory_pool_allocations: AtomicU64,
    pub memory_pool_deallocations: AtomicU64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            total_time_ms: AtomicU64::new(0),
            min_time_ms: AtomicU64::new(u64::MAX),
            max_time_ms: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            memory_pool_allocations: AtomicU64::new(0),
            memory_pool_deallocations: AtomicU64::new(0),
        }
    }

    /// Record an operation with lock-free atomic updates
    pub fn record_operation(&self, duration_ms: u64) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.total_time_ms.fetch_add(duration_ms, Ordering::Relaxed);

        // Update min using compare-and-swap loop
        let mut current_min = self.min_time_ms.load(Ordering::Relaxed);
        while duration_ms < current_min {
            match self.min_time_ms.compare_exchange_weak(
                current_min,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max using compare-and-swap loop
        let mut current_max = self.max_time_ms.load(Ordering::Relaxed);
        while duration_ms > current_max {
            match self.max_time_ms.compare_exchange_weak(
                current_max,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_allocation(&self) {
        self.memory_pool_allocations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_deallocation(&self) {
        self.memory_pool_deallocations
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn avg_time_ms(&self) -> f64 {
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        if total_ops == 0 {
            0.0
        } else {
            self.total_time_ms.load(Ordering::Relaxed) as f64 / total_ops as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }

    /// Get a snapshot of stats for display/reporting
    pub fn snapshot(&self) -> PerformanceStatsSnapshot {
        PerformanceStatsSnapshot {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            total_time_ms: self.total_time_ms.load(Ordering::Relaxed),
            avg_time_ms: self.avg_time_ms(),
            min_time_ms: self.min_time_ms.load(Ordering::Relaxed),
            max_time_ms: self.max_time_ms.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            memory_pool_allocations: self.memory_pool_allocations.load(Ordering::Relaxed),
            memory_pool_deallocations: self.memory_pool_deallocations.load(Ordering::Relaxed),
        }
    }

    /// Reset all stats to zero
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.total_time_ms.store(0, Ordering::Relaxed);
        self.min_time_ms.store(u64::MAX, Ordering::Relaxed);
        self.max_time_ms.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.memory_pool_allocations.store(0, Ordering::Relaxed);
        self.memory_pool_deallocations.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of performance statistics (for display/reporting)
#[derive(Debug, Clone, Default)]
pub struct PerformanceStatsSnapshot {
    pub total_operations: u64,
    pub total_time_ms: u64,
    pub avg_time_ms: f64,
    pub min_time_ms: u64,
    pub max_time_ms: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_pool_allocations: u64,
    pub memory_pool_deallocations: u64,
}

impl PerformanceStatsSnapshot {
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }
}

/// Global performance statistics (lock-free with atomics)
static PERF_STATS: std::sync::LazyLock<Arc<PerformanceStats>> =
    std::sync::LazyLock::new(|| Arc::new(PerformanceStats::new()));

/// Operation cache for frequently used operations
#[derive(Debug)]
pub struct OperationCache {
    cache: RwLock<std::collections::HashMap<String, CachedOperation>>,
    max_size: usize,
    ttl_ms: u64,
}

#[derive(Debug, Clone)]
struct CachedOperation {
    result: Vec<u8>, // Serialized result
    created_at: Instant,
    access_count: u64,
}

impl OperationCache {
    pub fn new(max_size: usize, ttl_ms: u64) -> Self {
        Self {
            cache: RwLock::new(std::collections::HashMap::new()),
            max_size,
            ttl_ms,
        }
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        let cache = self.cache.read();
        if let Some(cached) = cache.get(key) {
            if cached.created_at.elapsed().as_millis() as u64 <= self.ttl_ms {
                PERF_STATS.record_cache_hit();
                return Some(cached.result.clone());
            }
        }

        PERF_STATS.record_cache_miss();
        None
    }

    fn put(&self, key: String, value: Vec<u8>) {
        let mut cache = self.cache.write();

        // Remove expired entries and maintain size limit
        if cache.len() >= self.max_size {
            let now = Instant::now();
            cache.retain(|_, v| now.duration_since(v.created_at).as_millis() as u64 <= self.ttl_ms);

            // If still at capacity, remove oldest entries
            if cache.len() >= self.max_size {
                let mut entries: Vec<_> = cache.iter().collect();
                entries.sort_by_key(|(_, v)| v.created_at);

                // Collect keys to remove first
                let keys_to_remove: Vec<_> = entries
                    .iter()
                    .take(cache.len() - self.max_size + 1)
                    .map(|(key, _)| (*key).clone())
                    .collect();

                // Now remove them
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        cache.insert(
            key,
            CachedOperation {
                result: value,
                created_at: Instant::now(),
                access_count: 1,
            },
        );
    }
}

/// Global operation cache
static OP_CACHE: std::sync::LazyLock<OperationCache> =
    std::sync::LazyLock::new(|| OperationCache::new(1000, 300000)); // 1000 items, 5 min TTL

/// Memory pool for efficient tensor memory management
#[derive(Debug)]
pub struct TensorMemoryPool {
    pools: RwLock<std::collections::HashMap<usize, Vec<Vec<f32>>>>,
    max_pool_size: usize,
    max_buffer_size: usize,
}

impl TensorMemoryPool {
    fn new(max_pool_size: usize, max_buffer_size: usize) -> Self {
        Self {
            pools: RwLock::new(std::collections::HashMap::new()),
            max_pool_size,
            max_buffer_size,
        }
    }

    /// Get a pre-allocated buffer from the pool or create a new one
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        if size > self.max_buffer_size {
            // For very large buffers, don't use pooling
            return Vec::with_capacity(size);
        }

        let mut pools = self.pools.write();
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.clear();
                buffer.reserve(size);

                PERF_STATS.record_allocation();

                return buffer;
            }
        }

        // No available buffer in pool, create new one
        Vec::with_capacity(size)
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        let size = buffer.capacity();
        if size > self.max_buffer_size {
            return; // Don't pool very large buffers
        }

        buffer.clear();

        let mut pools = self.pools.write();
        let pool = pools.entry(size).or_insert_with(Vec::new);

        if pool.len() < self.max_pool_size {
            pool.push(buffer);

            PERF_STATS.record_deallocation();
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let pools = self.pools.read();
        let mut total_pools = 0;
        let mut total_buffers = 0;
        let mut memory_usage = 0;

        for (size, pool) in pools.iter() {
            total_pools += 1;
            total_buffers += pool.len();
            memory_usage += size * pool.len() * std::mem::size_of::<f32>();
        }

        PoolStats {
            total_pools,
            total_buffers,
            memory_usage,
        }
    }

    /// Clean up expired or unused buffers
    pub fn cleanup(&self) {
        let mut pools = self.pools.write();
        // Keep only smaller pools and limit buffer count per pool
        pools.retain(|size, pool| {
            if *size > self.max_buffer_size / 4 {
                pool.truncate(self.max_pool_size / 4);
            }
            !pool.is_empty()
        });
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_pools: usize,
    pub total_buffers: usize,
    pub memory_usage: usize,
}

/// Global memory pool
static MEMORY_POOL: std::sync::LazyLock<TensorMemoryPool> =
    std::sync::LazyLock::new(|| TensorMemoryPool::new(50, 1_000_000)); // 50 buffers per size, max 1M elements

/// Batched tensor operations for better performance
#[derive(Debug)]
pub struct BatchedOperations {
    tensors: Vec<*mut TorshTensor>,
    operations: Vec<BatchOperation>,
}

#[derive(Debug, Clone)]
pub enum BatchOperation {
    Add { a_idx: usize, b_idx: usize },
    Mul { a_idx: usize, b_idx: usize },
    MatMul { a_idx: usize, b_idx: usize },
    ReLU { tensor_idx: usize },
    ScalarAdd { tensor_idx: usize, scalar: f32 },
    ScalarMul { tensor_idx: usize, scalar: f32 },
}

impl BatchedOperations {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_tensor(&mut self, tensor: *mut TorshTensor) -> usize {
        let idx = self.tensors.len();
        self.tensors.push(tensor);
        idx
    }

    pub fn add_operation(&mut self, op: BatchOperation) {
        self.operations.push(op);
    }

    pub fn execute_batch(&self) -> FfiResult<Vec<*mut TorshTensor>> {
        let start = Instant::now();
        let mut results = Vec::new();

        for operation in &self.operations {
            let result = match operation {
                BatchOperation::Add { a_idx, b_idx } => unsafe {
                    // Create output tensor with same shape as first input
                    let ndim = torsh_tensor_ndim(self.tensors[*a_idx]);
                    if ndim == 0 {
                        return Err(FfiError::Tensor {
                            message: "Invalid tensor dimensions".to_string(),
                        });
                    }
                    let mut shape_vec = vec![0usize; ndim];
                    let mut ndim_out = ndim;
                    torsh_tensor_shape(self.tensors[*a_idx], shape_vec.as_mut_ptr(), &mut ndim_out);
                    let output_tensor = torsh_tensor_zeros(shape_vec.as_ptr(), ndim);
                    if output_tensor.is_null() {
                        ptr::null_mut()
                    } else {
                        let error = torsh_tensor_add(
                            self.tensors[*a_idx],
                            self.tensors[*b_idx],
                            output_tensor,
                        );
                        if error != TorshError::Success {
                            ptr::null_mut()
                        } else {
                            output_tensor
                        }
                    }
                },
                BatchOperation::Mul { a_idx, b_idx } => unsafe {
                    // Create output tensor with same shape as first input
                    let ndim = torsh_tensor_ndim(self.tensors[*a_idx]);
                    if ndim == 0 {
                        return Err(FfiError::Tensor {
                            message: "Invalid tensor dimensions".to_string(),
                        });
                    }
                    let mut shape_vec = vec![0usize; ndim];
                    let mut ndim_out = ndim;
                    torsh_tensor_shape(self.tensors[*a_idx], shape_vec.as_mut_ptr(), &mut ndim_out);
                    let output_tensor = torsh_tensor_zeros(shape_vec.as_ptr(), ndim);
                    if output_tensor.is_null() {
                        ptr::null_mut()
                    } else {
                        let error = torsh_tensor_mul(
                            self.tensors[*a_idx],
                            self.tensors[*b_idx],
                            output_tensor,
                        );
                        if error != TorshError::Success {
                            ptr::null_mut()
                        } else {
                            output_tensor
                        }
                    }
                },
                BatchOperation::MatMul { a_idx, b_idx } => unsafe {
                    // Create output tensor with same shape as first input
                    let ndim = torsh_tensor_ndim(self.tensors[*a_idx]);
                    if ndim == 0 {
                        return Err(FfiError::Tensor {
                            message: "Invalid tensor dimensions".to_string(),
                        });
                    }
                    let mut shape_vec = vec![0usize; ndim];
                    let mut ndim_out = ndim;
                    torsh_tensor_shape(self.tensors[*a_idx], shape_vec.as_mut_ptr(), &mut ndim_out);
                    let output_tensor = torsh_tensor_zeros(shape_vec.as_ptr(), ndim);
                    if output_tensor.is_null() {
                        ptr::null_mut()
                    } else {
                        let error = torsh_tensor_matmul(
                            self.tensors[*a_idx],
                            self.tensors[*b_idx],
                            output_tensor,
                        );
                        if error != TorshError::Success {
                            ptr::null_mut()
                        } else {
                            output_tensor
                        }
                    }
                },
                BatchOperation::ReLU { tensor_idx } => unsafe {
                    // Create output tensor with same shape as first input
                    let ndim = torsh_tensor_ndim(self.tensors[*tensor_idx]);
                    if ndim == 0 {
                        return Err(FfiError::Tensor {
                            message: "Invalid tensor dimensions".to_string(),
                        });
                    }
                    let mut shape_vec = vec![0usize; ndim];
                    let mut ndim_out = ndim;
                    torsh_tensor_shape(
                        self.tensors[*tensor_idx],
                        shape_vec.as_mut_ptr(),
                        &mut ndim_out,
                    );
                    let output_tensor = torsh_tensor_zeros(shape_vec.as_ptr(), ndim);
                    if output_tensor.is_null() {
                        ptr::null_mut()
                    } else {
                        let error = torsh_tensor_relu(self.tensors[*tensor_idx], output_tensor);
                        if error != TorshError::Success {
                            ptr::null_mut()
                        } else {
                            output_tensor
                        }
                    }
                },
                BatchOperation::ScalarAdd { tensor_idx, scalar } => {
                    // Note: Would need to implement scalar operations in C API
                    unsafe { torsh_tensor_add_scalar(self.tensors[*tensor_idx], *scalar) }
                }
                BatchOperation::ScalarMul { tensor_idx, scalar } => {
                    // Note: Would need to implement scalar operations in C API
                    unsafe { torsh_tensor_mul_scalar(self.tensors[*tensor_idx], *scalar) }
                }
            };

            if result.is_null() {
                return Err(FfiError::Tensor {
                    message: "Batch operation failed".to_string(),
                });
            }

            results.push(result);
        }

        let duration = start.elapsed().as_millis() as u64;
        PERF_STATS.record_operation(duration);

        Ok(results)
    }
}

/// Asynchronous operation queue for non-blocking operations
pub struct AsyncOperationQueue {
    queue: Arc<Mutex<VecDeque<AsyncOperation>>>,
    max_queue_size: usize,
}

struct AsyncOperation {
    operation: Box<dyn Fn() -> *mut TorshTensor + Send + Sync>,
    callback: Option<Box<dyn Fn(*mut TorshTensor) + Send + Sync>>,
    created_at: Instant,
}

impl AsyncOperationQueue {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            max_queue_size,
        }
    }

    pub fn enqueue<F, C>(&self, operation: F, callback: Option<C>) -> FfiResult<()>
    where
        F: Fn() -> *mut TorshTensor + Send + Sync + 'static,
        C: Fn(*mut TorshTensor) + Send + Sync + 'static,
    {
        let mut queue = self.queue.lock();

        if queue.len() >= self.max_queue_size {
            return Err(FfiError::AllocationFailed {
                message: "Async operation queue is full".to_string(),
            });
        }

        queue.push_back(AsyncOperation {
            operation: Box::new(operation),
            callback: callback.map(|c| Box::new(c) as Box<dyn Fn(*mut TorshTensor) + Send + Sync>),
            created_at: Instant::now(),
        });

        Ok(())
    }

    pub fn process_next(&self) -> FfiResult<bool> {
        let operation = {
            let mut queue = self.queue.lock();
            queue.pop_front()
        };

        if let Some(op) = operation {
            let start = Instant::now();
            let result = (op.operation)();
            let duration = start.elapsed().as_millis() as u64;

            PERF_STATS.record_operation(duration);

            if let Some(callback) = op.callback {
                callback(result);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn queue_size(&self) -> usize {
        self.queue.lock().len()
    }
}

/// Global async operation queue
static ASYNC_QUEUE: std::sync::LazyLock<AsyncOperationQueue> =
    std::sync::LazyLock::new(|| AsyncOperationQueue::new(10000));

/// C API wrappers for performance optimizations

/// Create a new batched operations context
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_new() -> *mut BatchedOperations {
    Box::into_raw(Box::new(BatchedOperations::new()))
}

/// Add a tensor to the batch context
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_add_tensor(
    batch: *mut BatchedOperations,
    tensor: *mut TorshTensor,
) -> c_int {
    if batch.is_null() || tensor.is_null() {
        return -1;
    }

    let batch_ref = &mut *batch;
    batch_ref.add_tensor(tensor) as c_int
}

/// Add an addition operation to the batch
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_add_add_op(
    batch: *mut BatchedOperations,
    a_idx: c_int,
    b_idx: c_int,
) -> TorshError {
    if batch.is_null() {
        return TorshError::InvalidArgument;
    }

    let batch_ref = &mut *batch;
    batch_ref.add_operation(BatchOperation::Add {
        a_idx: a_idx as usize,
        b_idx: b_idx as usize,
    });

    TorshError::Success
}

/// Add a multiplication operation to the batch
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_add_mul_op(
    batch: *mut BatchedOperations,
    a_idx: c_int,
    b_idx: c_int,
) -> TorshError {
    if batch.is_null() {
        return TorshError::InvalidArgument;
    }

    let batch_ref = &mut *batch;
    batch_ref.add_operation(BatchOperation::Mul {
        a_idx: a_idx as usize,
        b_idx: b_idx as usize,
    });

    TorshError::Success
}

/// Add a ReLU operation to the batch
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_add_relu_op(
    batch: *mut BatchedOperations,
    tensor_idx: c_int,
) -> TorshError {
    if batch.is_null() {
        return TorshError::InvalidArgument;
    }

    let batch_ref = &mut *batch;
    batch_ref.add_operation(BatchOperation::ReLU {
        tensor_idx: tensor_idx as usize,
    });

    TorshError::Success
}

/// Execute all operations in the batch
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_execute(
    batch: *mut BatchedOperations,
    results: *mut *mut TorshTensor,
    max_results: c_int,
    actual_results: *mut c_int,
) -> TorshError {
    if batch.is_null() || results.is_null() || actual_results.is_null() {
        return TorshError::InvalidArgument;
    }

    let batch_ref = &*batch;
    match batch_ref.execute_batch() {
        Ok(result_tensors) => {
            let copy_count = std::cmp::min(result_tensors.len(), max_results as usize);

            for (i, tensor) in result_tensors.iter().take(copy_count).enumerate() {
                *results.add(i) = *tensor;
            }

            *actual_results = copy_count as c_int;
            TorshError::Success
        }
        Err(_) => TorshError::RuntimeError,
    }
}

/// Free a batched operations context
#[no_mangle]
pub unsafe extern "C" fn torsh_batch_free(batch: *mut BatchedOperations) {
    if !batch.is_null() {
        let _ = Box::from_raw(batch);
    }
}

/// Get current performance statistics
#[no_mangle]
pub unsafe extern "C" fn torsh_get_performance_stats(
    total_ops: *mut u64,
    avg_time_ms: *mut c_float,
    cache_hit_rate: *mut c_float,
) -> TorshError {
    if total_ops.is_null() || avg_time_ms.is_null() || cache_hit_rate.is_null() {
        return TorshError::InvalidArgument;
    }

    let snapshot = PERF_STATS.snapshot();
    *total_ops = snapshot.total_operations;
    *avg_time_ms = snapshot.avg_time_ms as c_float;
    *cache_hit_rate = snapshot.cache_hit_rate() as c_float;
    TorshError::Success
}

/// Reset performance statistics
#[no_mangle]
pub unsafe extern "C" fn torsh_reset_performance_stats() -> TorshError {
    PERF_STATS.reset();
    TorshError::Success
}

/// Process one item from the async operation queue
#[no_mangle]
pub unsafe extern "C" fn torsh_process_async_queue() -> c_int {
    match ASYNC_QUEUE.process_next() {
        Ok(true) => 1,  // Processed an operation
        Ok(false) => 0, // Queue was empty
        Err(_) => -1,   // Error
    }
}

/// Get the current async queue size
#[no_mangle]
pub unsafe extern "C" fn torsh_async_queue_size() -> c_int {
    ASYNC_QUEUE.queue_size() as c_int
}

/// Clear the operation cache
#[no_mangle]
pub unsafe extern "C" fn torsh_clear_operation_cache() -> TorshError {
    {
        let mut cache = OP_CACHE.cache.write();
        cache.clear();
    }
    TorshError::Success
}

// =============================================================================
// Memory Pool API Functions
// =============================================================================

/// Get memory pool statistics
#[no_mangle]
pub unsafe extern "C" fn torsh_get_memory_pool_stats(
    total_pools: *mut usize,
    total_buffers: *mut usize,
    memory_usage: *mut usize,
) -> TorshError {
    if total_pools.is_null() || total_buffers.is_null() || memory_usage.is_null() {
        return TorshError::InvalidArgument;
    }

    let stats = MEMORY_POOL.stats();
    *total_pools = stats.total_pools;
    *total_buffers = stats.total_buffers;
    *memory_usage = stats.memory_usage;

    TorshError::Success
}

/// Clean up unused memory pool buffers
#[no_mangle]
pub unsafe extern "C" fn torsh_cleanup_memory_pool() -> TorshError {
    MEMORY_POOL.cleanup();
    TorshError::Success
}

/// Get a buffer from the memory pool (for internal use by other modules)
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    MEMORY_POOL.get_buffer(size)
}

/// Return a buffer to the memory pool (for internal use by other modules)
pub fn return_pooled_buffer(buffer: Vec<f32>) {
    MEMORY_POOL.return_buffer(buffer);
}

/// Get current performance statistics snapshot (Rust API)
pub fn get_performance_stats() -> PerformanceStatsSnapshot {
    PERF_STATS.snapshot()
}

/// Advanced performance profiler for fine-grained analysis
#[derive(Debug, Default)]
pub struct AdvancedProfiler {
    operation_timings: RwLock<std::collections::HashMap<String, Vec<u64>>>,
    memory_snapshots: RwLock<Vec<(String, usize, Instant)>>,
}

impl AdvancedProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record timing for a specific operation type
    pub fn record_timing(&self, operation: &str, duration_ms: u64) {
        let mut timings = self.operation_timings.write();
        timings
            .entry(operation.to_string())
            .or_default()
            .push(duration_ms);
    }

    /// Record memory usage snapshot
    pub fn record_memory_snapshot(&self, label: &str, memory_bytes: usize) {
        let mut snapshots = self.memory_snapshots.write();
        snapshots.push((label.to_string(), memory_bytes, Instant::now()));

        // Keep only recent snapshots
        if snapshots.len() > 1000 {
            snapshots.drain(0..500);
        }
    }

    /// Get timing statistics for an operation
    pub fn get_timing_stats(&self, operation: &str) -> Option<TimingStats> {
        let timings = self.operation_timings.read();
        if let Some(times) = timings.get(operation) {
            if times.is_empty() {
                return None;
            }

            let mut sorted_times = times.clone();
            sorted_times.sort_unstable();

            let sum: u64 = times.iter().sum();
            let count = times.len();
            let min = *sorted_times
                .first()
                .expect("sorted_times should not be empty after is_empty check");
            let max = *sorted_times
                .last()
                .expect("sorted_times should not be empty after is_empty check");
            let avg = sum as f64 / count as f64;

            let median = if count % 2 == 0 {
                (sorted_times[count / 2 - 1] + sorted_times[count / 2]) as f64 / 2.0
            } else {
                sorted_times[count / 2] as f64
            };

            let p95_idx = (count as f64 * 0.95).ceil() as usize - 1;
            let p95 = sorted_times[p95_idx.min(count - 1)];

            Some(TimingStats {
                operation: operation.to_string(),
                count,
                min,
                max,
                avg,
                median,
                p95,
            })
        } else {
            None
        }
    }

    /// Get memory usage over time
    pub fn get_memory_usage_trend(&self) -> Vec<(String, usize, std::time::Duration)> {
        let snapshots = self.memory_snapshots.read();
        let start_time = snapshots
            .first()
            .map(|(_, _, t)| *t)
            .unwrap_or_else(Instant::now);

        snapshots
            .iter()
            .map(|(label, memory, time)| (label.clone(), *memory, time.duration_since(start_time)))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct TimingStats {
    pub operation: String,
    pub count: usize,
    pub min: u64,
    pub max: u64,
    pub avg: f64,
    pub median: f64,
    pub p95: u64,
}

/// Global advanced profiler
static ADVANCED_PROFILER: std::sync::LazyLock<AdvancedProfiler> =
    std::sync::LazyLock::new(AdvancedProfiler::new);

/// Record a profiling measurement
pub fn profile_operation<F, R>(operation: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_millis() as u64;

    ADVANCED_PROFILER.record_timing(operation, duration);

    PERF_STATS.record_operation(duration);

    result
}

/// Get profiling statistics for an operation (C API)
#[no_mangle]
pub unsafe extern "C" fn torsh_get_operation_timing_stats(
    operation: *const c_char,
    count: *mut usize,
    min_ms: *mut u64,
    max_ms: *mut u64,
    avg_ms: *mut c_float,
) -> TorshError {
    if operation.is_null()
        || count.is_null()
        || min_ms.is_null()
        || max_ms.is_null()
        || avg_ms.is_null()
    {
        return TorshError::InvalidArgument;
    }

    let operation_str = match std::ffi::CStr::from_ptr(operation).to_str() {
        Ok(s) => s,
        Err(_) => return TorshError::InvalidArgument,
    };

    if let Some(stats) = ADVANCED_PROFILER.get_timing_stats(operation_str) {
        *count = stats.count;
        *min_ms = stats.min;
        *max_ms = stats.max;
        *avg_ms = stats.avg as c_float;
        TorshError::Success
    } else {
        TorshError::NotImplemented
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats::new();

        stats.record_operation(100);
        stats.record_operation(200);
        stats.record_operation(150);

        assert_eq!(stats.total_operations.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_time_ms(), 150.0);
        assert_eq!(stats.min_time_ms.load(Ordering::Relaxed), 100);
        assert_eq!(stats.max_time_ms.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn test_cache_hit_rate() {
        let stats = PerformanceStats::new();

        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_miss();

        assert_eq!(stats.cache_hit_rate(), 2.0 / 3.0);
    }

    #[test]
    fn test_operation_cache() {
        let cache = OperationCache::new(10, 1000);

        let key = "test_op".to_string();
        let value = vec![1, 2, 3, 4];

        cache.put(key.clone(), value.clone());

        let retrieved = cache.get(&key);
        assert_eq!(retrieved, Some(value));

        let non_existent = cache.get("non_existent");
        assert_eq!(non_existent, None);
    }

    #[test]
    fn test_batched_operations() {
        let mut batch = BatchedOperations::new();

        // Note: These would be real tensor pointers in practice
        let tensor1 = 0x12345 as *mut TorshTensor;
        let tensor2 = 0x67890 as *mut TorshTensor;

        let idx1 = batch.add_tensor(tensor1);
        let idx2 = batch.add_tensor(tensor2);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);

        batch.add_operation(BatchOperation::Add {
            a_idx: idx1,
            b_idx: idx2,
        });
        batch.add_operation(BatchOperation::ReLU { tensor_idx: idx1 });

        assert_eq!(batch.operations.len(), 2);
    }

    #[test]
    fn test_async_operation_queue() {
        let queue = AsyncOperationQueue::new(10);

        let operation = || 0x12345 as *mut TorshTensor;
        let callback = |_tensor: *mut TorshTensor| {
            // Callback logic would go here
        };

        let result = queue.enqueue(operation, Some(callback));
        assert!(result.is_ok());

        assert_eq!(queue.queue_size(), 1);

        let processed = queue.process_next();
        assert!(processed.is_ok());
        assert_eq!(processed.unwrap(), true);

        assert_eq!(queue.queue_size(), 0);
    }
}
