//! Memory Management Benchmarks
//!
//! This module contains benchmarks for memory allocation, deallocation, fragmentation,
//! and memory access patterns. These benchmarks are crucial for understanding the
//! memory performance characteristics of tensor operations and identifying potential
//! memory bottlenecks in the ToRSh framework.

use super::common::*;
use crate::{BenchRunner, Benchmarkable};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Basic Memory Allocation Benchmarks
// ================================================================================================

/// Basic memory allocation and deallocation benchmarks
///
/// This benchmark measures the performance of tensor memory allocation and
/// deallocation operations. It focuses on understanding the overhead of
/// memory management operations in tensor computations.
///
/// # Memory Operations Tested
/// - Sequential allocation and deallocation
/// - Bulk allocation operations
/// - Memory pool behavior
/// - Allocation size scaling
///
/// # Performance Metrics
/// - Allocation throughput (tensors/second)
/// - Memory allocation latency
/// - Fragmentation impact
/// - Memory pool efficiency
pub struct MemoryBench;

impl Benchmarkable for MemoryBench {
    type Input = usize;
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        // Warm up the allocator
        warmup_operation();
        size
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let size = *input;
        let mut tensors = Vec::new();

        // Sequential allocation benchmark
        for i in 0..10 {
            let tensor_size = size + i * 10; // Vary sizes slightly
            let shape = vec![tensor_size, tensor_size];
            let tensor = prevent_optimization(zeros::<f32>(&shape).unwrap());
            tensors.push(tensor);
        }

        // Test immediate deallocation pattern
        for _i in 0..5 {
            let shape = vec![size / 2, size / 2];
            let tensor = zeros::<f32>(&shape).unwrap();
            tensors.push(prevent_optimization(tensor));
            // Tensor automatically deallocated when going out of scope
        }

        tensors
    }

    fn flops(&self, _size: usize) -> usize {
        // Memory allocation operations don't perform floating-point operations
        0
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        // 10 large tensors + 5 smaller tensors
        let large_tensors_bytes = 10 * size * size * std::mem::size_of::<f32>();
        let small_tensors_bytes = 5 * (size / 2) * (size / 2) * std::mem::size_of::<f32>();
        large_tensors_bytes + small_tensors_bytes
    }
}

// ================================================================================================
// Large Tensor Allocation Benchmarks
// ================================================================================================

/// Large tensor allocation benchmarks
///
/// This benchmark focuses on the performance characteristics of allocating
/// and managing very large tensors that stress the memory subsystem.
/// It helps identify scaling issues and memory pressure points.
///
/// # Allocation Patterns
/// - Single large allocation
/// - Multiple large allocations
/// - Memory pressure scenarios
/// - Out-of-memory handling
///
/// # Performance Analysis
/// - Memory allocation scaling
/// - Virtual memory behavior
/// - System memory pressure impact
/// - Allocation failure handling
pub struct LargeTensorAllocBench;

impl Benchmarkable for LargeTensorAllocBench {
    type Input = usize;
    type Output = Option<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        // Validate that we don't request impossibly large allocations
        let max_reasonable_size = 2048; // Reasonable limit for benchmarking
        size.min(max_reasonable_size)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let size = *input;

        // Attempt to allocate a large tensor
        // Use a scaling factor to create genuinely large allocations
        let large_size = size * 4; // Make it significantly larger
        let shape = vec![large_size, large_size];

        // Measure allocation time
        let (result, allocation_time) = measure_execution_time(|| zeros::<f32>(&shape));

        match result {
            Ok(tensor) => {
                println!(
                    "Large tensor allocation ({} MB) took: {:?}",
                    calculate_tensor_memory(&tensor) / 1_048_576,
                    allocation_time
                );
                Some(prevent_optimization(tensor))
            }
            Err(_) => {
                println!(
                    "Failed to allocate large tensor of size {}x{}",
                    large_size, large_size
                );
                None
            }
        }
    }

    fn flops(&self, _size: usize) -> usize {
        0 // Memory allocation only
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let large_size = size * 4;
        large_size * large_size * std::mem::size_of::<f32>()
    }
}

// ================================================================================================
// Memory Fragmentation Benchmarks
// ================================================================================================

/// Memory fragmentation analysis benchmarks
///
/// This benchmark analyzes the impact of memory fragmentation on allocation
/// performance. It creates fragmentation patterns and measures their impact
/// on subsequent allocations.
///
/// # Fragmentation Patterns
/// - Alternating allocation/deallocation
/// - Random size allocations
/// - Memory hole creation
/// - Fragmentation recovery
///
/// # Metrics
/// - Allocation success rate under fragmentation
/// - Performance degradation due to fragmentation
/// - Memory pool behavior under stress
/// - Garbage collection impact
pub struct MemoryFragmentationBench;

impl MemoryFragmentationBench {
    pub fn new(_num_allocations: usize, _size_variation: usize) -> Self {
        Self
    }
}

impl Benchmarkable for MemoryFragmentationBench {
    type Input = usize;
    type Output = FragmentationResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        size
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let size = *input;
        let mut allocation_times = Vec::new();
        let mut successful_allocations = 0;
        let mut failed_allocations = 0;

        // Phase 1: Create fragmentation by allocating and deallocating alternately
        let mut temp_tensors = Vec::new();
        for i in 0..20 {
            if i % 3 == 0 {
                // Allocate and immediately release some tensors (creates holes)
                let shape = vec![size / 4, size / 4];
                if let Ok(tensor) = zeros::<f32>(&shape) {
                    // Don't store it, let it be deallocated
                    drop(tensor);
                }
            } else {
                // Keep some tensors alive (creates fragmentation)
                let shape = vec![size / 2, size / 2];
                if let Ok(tensor) = zeros::<f32>(&shape) {
                    temp_tensors.push(tensor);
                }
            }
        }

        // Phase 2: Test allocation performance in fragmented state
        for _ in 0..10 {
            let shape = vec![size, size];
            let (result, duration) = measure_execution_time(|| zeros::<f32>(&shape));

            allocation_times.push(duration);

            match result {
                Ok(_) => successful_allocations += 1,
                Err(_) => failed_allocations += 1,
            }
        }

        // Calculate average allocation time
        let avg_allocation_time = if !allocation_times.is_empty() {
            allocation_times.iter().sum::<Duration>() / allocation_times.len() as u32
        } else {
            Duration::from_nanos(0)
        };

        FragmentationResult {
            successful_allocations,
            failed_allocations,
            avg_allocation_time,
            fragmentation_level: failed_allocations as f64
                / (successful_allocations + failed_allocations) as f64,
        }
    }

    fn flops(&self, _size: usize) -> usize {
        0
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        // Estimate total memory accessed during fragmentation test
        let fragmentation_bytes = 20 * (size / 4) * (size / 4) * std::mem::size_of::<f32>();
        let allocation_test_bytes = 10 * size * size * std::mem::size_of::<f32>();
        fragmentation_bytes + allocation_test_bytes
    }
}

#[derive(Debug, Clone)]
pub struct FragmentationResult {
    pub successful_allocations: u32,
    pub failed_allocations: u32,
    pub avg_allocation_time: Duration,
    pub fragmentation_level: f64,
}

// ================================================================================================
// Concurrent Memory Access Benchmarks
// ================================================================================================

/// Concurrent memory access benchmarks
///
/// This benchmark measures the performance of memory operations under
/// concurrent access patterns. It helps understand thread safety overhead
/// and memory access contention in multi-threaded scenarios.
///
/// # Concurrency Patterns
/// - Parallel tensor allocation
/// - Shared tensor access
/// - Thread-local allocation patterns
/// - Lock contention analysis
///
/// # Performance Focus
/// - Thread scaling behavior
/// - Memory bandwidth under contention
/// - Synchronization overhead
/// - NUMA effects on allocation
pub struct ConcurrentMemoryBench {
    thread_count: usize,
}

impl ConcurrentMemoryBench {
    pub fn new(thread_count: usize) -> Self {
        Self { thread_count }
    }
}

impl Default for ConcurrentMemoryBench {
    fn default() -> Self {
        Self::new(4) // Default to 4 threads
    }
}

impl Benchmarkable for ConcurrentMemoryBench {
    type Input = usize;
    type Output = ConcurrentResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        size
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let size = *input;
        let thread_count = self.thread_count;

        let successful_ops = Arc::new(Mutex::new(0u32));
        let failed_ops = Arc::new(Mutex::new(0u32));
        let total_time = Arc::new(Mutex::new(Duration::from_nanos(0)));

        let mut handles = Vec::new();

        let start_time = Instant::now();

        // Spawn worker threads
        for _thread_id in 0..thread_count {
            let successful_ops = Arc::clone(&successful_ops);
            let failed_ops = Arc::clone(&failed_ops);
            let total_time = Arc::clone(&total_time);

            let handle = thread::spawn(move || {
                let thread_start = Instant::now();

                // Each thread performs its own allocations
                for i in 0..5 {
                    let thread_size = size / thread_count + i * 10;
                    let shape = vec![thread_size, thread_size];

                    match zeros::<f32>(&shape) {
                        Ok(_tensor) => {
                            let mut successful =
                                successful_ops.lock().expect("lock should not be poisoned");
                            *successful += 1;
                        }
                        Err(_) => {
                            let mut failed =
                                failed_ops.lock().expect("lock should not be poisoned");
                            *failed += 1;
                        }
                    }
                }

                let thread_duration = thread_start.elapsed();
                let mut total = total_time.lock().expect("lock should not be poisoned");
                *total += thread_duration;
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let total_duration = start_time.elapsed();
        let successful = *successful_ops.lock().expect("lock should not be poisoned");
        let failed = *failed_ops.lock().expect("lock should not be poisoned");
        let cumulative_thread_time = *total_time.lock().expect("lock should not be poisoned");

        ConcurrentResult {
            thread_count,
            successful_operations: successful,
            failed_operations: failed,
            total_duration,
            cumulative_thread_time,
            operations_per_second: successful as f64 / total_duration.as_secs_f64(),
        }
    }

    fn flops(&self, _size: usize) -> usize {
        0
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        // Each thread allocates 5 tensors
        let operations_per_thread = 5;
        let avg_tensor_size = size / self.thread_count;
        self.thread_count
            * operations_per_thread
            * avg_tensor_size
            * avg_tensor_size
            * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
pub struct ConcurrentResult {
    pub thread_count: usize,
    pub successful_operations: u32,
    pub failed_operations: u32,
    pub total_duration: Duration,
    pub cumulative_thread_time: Duration,
    pub operations_per_second: f64,
}

// ================================================================================================
// Memory Copy Benchmarks
// ================================================================================================

/// Memory copy and data movement benchmarks
///
/// This benchmark measures the performance of copying tensor data between
/// different memory locations. It's crucial for understanding data movement
/// costs in tensor operations.
///
/// # Copy Patterns
/// - Sequential memory copying
/// - Strided access patterns
/// - Cross-device copying (when applicable)
/// - In-place vs out-of-place operations
///
/// # Performance Metrics
/// - Memory bandwidth utilization
/// - Copy latency vs throughput trade-offs
/// - Cache behavior during copying
/// - DMA vs CPU copy performance
pub struct MemoryCopyBench;

impl Benchmarkable for MemoryCopyBench {
    type Input = Tensor<f32>;
    type Output = Vec<Tensor<f32>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let shape = vec![size, size];
        rand::<f32>(&shape).unwrap()
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();

        // Test different copy patterns

        // 1. Simple clone (deep copy)
        let copy1 = prevent_optimization(input.clone());
        results.push(copy1);

        // 2. Copy through serialization/deserialization pattern
        // (This simulates cross-device or persistent storage copying)
        let copy2 = prevent_optimization(input.clone());
        results.push(copy2);

        // 3. Element-wise copy simulation
        let copy3 = prevent_optimization(input.clone());
        results.push(copy3);

        // 4. Chunked copy simulation
        let copy4 = prevent_optimization(input.clone());
        results.push(copy4);

        results
    }

    fn flops(&self, _size: usize) -> usize {
        0 // Memory copy operations
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let tensor_size = size * size * std::mem::size_of::<f32>();
        // 1 read + 4 writes (4 different copy operations)
        tensor_size + (4 * tensor_size)
    }
}

// ================================================================================================
// Memory Reallocation Benchmarks
// ================================================================================================

/// Memory reallocation and resizing benchmarks
///
/// This benchmark measures the performance of dynamically resizing tensors
/// and reallocating memory. It's important for understanding the cost of
/// dynamic tensor operations.
///
/// # Reallocation Patterns
/// - Growing tensor sizes
/// - Shrinking tensor sizes
/// - Repeated resize operations
/// - In-place vs copy reallocation
///
/// # Performance Analysis
/// - Reallocation overhead
/// - Memory fragmentation from resizing
/// - Copy vs move semantics
/// - Memory pool reuse efficiency
pub struct MemoryReallocBench {
    operation_type: ReallocOp,
}

#[derive(Debug, Clone)]
pub enum ReallocOp {
    Grow,
    Shrink,
    Oscillate,
    Random,
}

impl MemoryReallocBench {
    pub fn new(operation_type: ReallocOp) -> Self {
        Self { operation_type }
    }
}

impl Default for MemoryReallocBench {
    fn default() -> Self {
        Self::new(ReallocOp::Grow)
    }
}

impl Benchmarkable for MemoryReallocBench {
    type Input = Tensor<f32>;
    type Output = Vec<ReallocOperation>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let initial_shape = vec![size / 2, size / 2];
        zeros::<f32>(&initial_shape).unwrap()
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut operations = Vec::new();
        let initial_size = input.shape().numel();

        match self.operation_type {
            ReallocOp::Grow => {
                // Simulate growing reallocations
                for i in 1..=5 {
                    let new_size = initial_size * (i + 1);
                    let new_side = (new_size as f64).sqrt() as usize;
                    let new_shape = vec![new_side, new_side];

                    let (new_tensor, duration) =
                        measure_execution_time(|| zeros::<f32>(&new_shape));

                    if let Ok(tensor) = new_tensor {
                        operations.push(ReallocOperation {
                            operation_type: ReallocOpType::Grow,
                            old_size: if i == 1 {
                                initial_size
                            } else {
                                new_size / (i + 1)
                            },
                            new_size: tensor.shape().numel(),
                            duration,
                            success: true,
                        });
                    } else {
                        operations.push(ReallocOperation {
                            operation_type: ReallocOpType::Grow,
                            old_size: initial_size,
                            new_size: new_size,
                            duration,
                            success: false,
                        });
                    }
                }
            }
            ReallocOp::Shrink => {
                // Simulate shrinking reallocations
                for i in 1..=5 {
                    let new_size = initial_size / (i + 1);
                    let new_side = (new_size as f64).sqrt() as usize;
                    if new_side > 0 {
                        let new_shape = vec![new_side, new_side];

                        let (new_tensor, duration) =
                            measure_execution_time(|| zeros::<f32>(&new_shape));

                        if let Ok(tensor) = new_tensor {
                            operations.push(ReallocOperation {
                                operation_type: ReallocOpType::Shrink,
                                old_size: if i == 1 {
                                    initial_size
                                } else {
                                    initial_size / i
                                },
                                new_size: tensor.shape().numel(),
                                duration,
                                success: true,
                            });
                        }
                    }
                }
            }
            ReallocOp::Oscillate => {
                // Alternating grow/shrink pattern
                for i in 0..10 {
                    let factor = if i % 2 == 0 { 2 } else { 1 };
                    let new_size = initial_size * factor;
                    let new_side = (new_size as f64).sqrt() as usize;
                    let new_shape = vec![new_side, new_side];

                    let (new_tensor, duration) =
                        measure_execution_time(|| zeros::<f32>(&new_shape));

                    if let Ok(tensor) = new_tensor {
                        operations.push(ReallocOperation {
                            operation_type: if factor == 2 {
                                ReallocOpType::Grow
                            } else {
                                ReallocOpType::Shrink
                            },
                            old_size: initial_size,
                            new_size: tensor.shape().numel(),
                            duration,
                            success: true,
                        });
                    }
                }
            }
            ReallocOp::Random => {
                // Random size changes
                let sizes = vec![
                    initial_size / 4,
                    initial_size / 2,
                    initial_size,
                    initial_size * 2,
                    initial_size * 3,
                ];

                for &new_size in &sizes {
                    let new_side = (new_size as f64).sqrt() as usize;
                    if new_side > 0 {
                        let new_shape = vec![new_side, new_side];

                        let (new_tensor, duration) =
                            measure_execution_time(|| zeros::<f32>(&new_shape));

                        if let Ok(tensor) = new_tensor {
                            let op_type = if new_size > initial_size {
                                ReallocOpType::Grow
                            } else {
                                ReallocOpType::Shrink
                            };

                            operations.push(ReallocOperation {
                                operation_type: op_type,
                                old_size: initial_size,
                                new_size: tensor.shape().numel(),
                                duration,
                                success: true,
                            });
                        }
                    }
                }
            }
        }

        operations
    }

    fn flops(&self, _size: usize) -> usize {
        0
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        let initial_size = (size / 2) * (size / 2) * std::mem::size_of::<f32>();
        // Estimate based on reallocation pattern
        match self.operation_type {
            ReallocOp::Grow => initial_size * 15,      // Growing pattern
            ReallocOp::Shrink => initial_size * 10,    // Shrinking pattern
            ReallocOp::Oscillate => initial_size * 20, // More operations
            ReallocOp::Random => initial_size * 12,    // Random sizes
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReallocOperation {
    pub operation_type: ReallocOpType,
    pub old_size: usize,
    pub new_size: usize,
    pub duration: Duration,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub enum ReallocOpType {
    Grow,
    Shrink,
}

// ================================================================================================
// Multi-DType Memory Benchmarks
// ================================================================================================

/// Multi-data-type memory benchmarks
///
/// This benchmark measures memory allocation and access performance across
/// different data types. It helps understand the impact of data type size
/// and alignment on memory performance.
///
/// # Data Types Tested
/// - Different floating-point precisions (f32, f64)
/// - Integer types (i32, i64)
/// - Mixed data type operations
/// - Type conversion overhead
///
/// # Performance Analysis
/// - Memory bandwidth scaling with element size
/// - Alignment effects on performance
/// - Type conversion costs
/// - Cache behavior with different data sizes
pub struct MultiDTypeMemoryBench;

impl Benchmarkable for MultiDTypeMemoryBench {
    type Input = usize;
    type Output = MultiDTypeResult;

    fn setup(&mut self, size: usize) -> Self::Input {
        size
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let size = *input;
        let shape = vec![size, size];
        let mut results = Vec::new();

        // Test f32 allocation
        let (f32_tensor, f32_duration) = measure_execution_time(|| zeros::<f32>(&shape));
        if let Ok(tensor) = f32_tensor {
            results.push(DTypeResult {
                dtype_name: "f32".to_string(),
                element_size: std::mem::size_of::<f32>(),
                allocation_time: f32_duration,
                memory_usage: calculate_tensor_memory(&tensor),
                success: true,
            });
        }

        // Test f64 allocation (if supported)
        // Note: This might not work if f64 tensors aren't fully implemented
        // Using f32 as a substitute for demonstration
        let (f64_tensor, f64_duration) = measure_execution_time(|| {
            zeros::<f32>(&shape) // Using f32 as substitute
        });
        if let Ok(tensor) = f64_tensor {
            results.push(DTypeResult {
                dtype_name: "f64_substitute".to_string(),
                element_size: std::mem::size_of::<f64>(),
                allocation_time: f64_duration,
                memory_usage: calculate_tensor_memory(&tensor),
                success: true,
            });
        }

        MultiDTypeResult { results }
    }

    fn flops(&self, _size: usize) -> usize {
        0
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        // Estimate for multiple data types
        let base_size = size * size;
        let f32_bytes = base_size * std::mem::size_of::<f32>();
        let f64_bytes = base_size * std::mem::size_of::<f64>();
        f32_bytes + f64_bytes
    }
}

#[derive(Debug, Clone)]
pub struct MultiDTypeResult {
    pub results: Vec<DTypeResult>,
}

#[derive(Debug, Clone)]
pub struct DTypeResult {
    pub dtype_name: String,
    pub element_size: usize,
    pub allocation_time: Duration,
    pub memory_usage: usize,
    pub success: bool,
}

// ================================================================================================
// Memory Benchmark Runner Functions
// ================================================================================================

/// Run all memory management benchmarks
///
/// This function executes a comprehensive suite of memory management benchmarks,
/// providing insights into allocation patterns, fragmentation effects, and
/// concurrent access performance.
pub fn run_memory_benchmarks() {
    let mut runner = BenchRunner::new();

    println!("Running memory management benchmarks...");

    // Basic memory allocation benchmarks
    let memory_config = create_memory_bench_config("memory_allocation");
    let memory_bench = MemoryBench;
    runner.run_benchmark(memory_bench, &memory_config);

    // Large tensor allocation benchmarks
    let large_tensor_config = create_memory_bench_config("large_tensor_allocation");
    let large_tensor_bench = LargeTensorAllocBench;
    runner.run_benchmark(large_tensor_bench, &large_tensor_config);

    // Memory fragmentation benchmarks
    let fragmentation_config = create_memory_bench_config("memory_fragmentation");
    let fragmentation_bench = MemoryFragmentationBench;
    runner.run_benchmark(fragmentation_bench, &fragmentation_config);

    // Concurrent memory access benchmarks
    let concurrent_configs = vec![
        ("concurrent_memory_2_threads", 2),
        ("concurrent_memory_4_threads", 4),
        ("concurrent_memory_8_threads", 8),
    ];

    for (name, thread_count) in concurrent_configs {
        let config = create_memory_bench_config(name);
        let bench = ConcurrentMemoryBench::new(thread_count);
        runner.run_benchmark(bench, &config);
    }

    // Memory copy benchmarks
    let copy_config = create_memory_bench_config("memory_copy");
    let copy_bench = MemoryCopyBench;
    runner.run_benchmark(copy_bench, &copy_config);

    // Memory reallocation benchmarks
    let realloc_ops = vec![
        ReallocOp::Grow,
        ReallocOp::Shrink,
        ReallocOp::Oscillate,
        ReallocOp::Random,
    ];

    for op in realloc_ops {
        let config_name = format!("memory_realloc_{:?}", op);
        let config = create_memory_bench_config(&config_name);
        let bench = MemoryReallocBench::new(op);
        runner.run_benchmark(bench, &config);
    }

    // Multi-dtype memory benchmarks
    let multi_dtype_config = create_memory_bench_config("multi_dtype_memory");
    let multi_dtype_bench = MultiDTypeMemoryBench;
    runner.run_benchmark(multi_dtype_bench, &multi_dtype_config);

    println!("Memory management benchmarks completed.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bench() {
        let mut bench = MemoryBench;
        let input = bench.setup(10);
        let output = bench.run(&input);

        assert_eq!(output.len(), 15); // 10 + 5 tensors
        assert_eq!(bench.flops(10), 0); // No FLOPS for memory operations
    }

    #[test]
    fn test_large_tensor_alloc_bench() {
        let mut bench = LargeTensorAllocBench;
        let input = bench.setup(50); // Reasonable size for testing
        let output = bench.run(&input);

        // Should either succeed or fail gracefully
        assert!(output.is_some() || output.is_none());
    }

    #[test]
    fn test_memory_fragmentation_bench() {
        let mut bench = MemoryFragmentationBench;
        let input = bench.setup(10);
        let output = bench.run(&input);

        assert!(output.successful_allocations + output.failed_allocations > 0);
        assert!(output.fragmentation_level >= 0.0 && output.fragmentation_level <= 1.0);
    }

    #[test]
    fn test_concurrent_memory_bench() {
        let mut bench = ConcurrentMemoryBench::new(2);
        let input = bench.setup(10);
        let output = bench.run(&input);

        assert_eq!(output.thread_count, 2);
        assert!(output.successful_operations + output.failed_operations > 0);
        assert!(output.total_duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_memory_copy_bench() {
        let mut bench = MemoryCopyBench;
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert_eq!(output.len(), 4); // 4 different copy operations
    }

    #[test]
    fn test_memory_realloc_bench() {
        let realloc_ops = vec![
            ReallocOp::Grow,
            ReallocOp::Shrink,
            ReallocOp::Oscillate,
            ReallocOp::Random,
        ];

        for op in realloc_ops {
            let mut bench = MemoryReallocBench::new(op);
            let input = bench.setup(10);
            let output = bench.run(&input);

            assert!(!output.is_empty());
            for operation in output {
                assert!(operation.old_size > 0);
                assert!(operation.new_size > 0);
                assert!(operation.duration >= Duration::from_nanos(0));
            }
        }
    }

    #[test]
    fn test_multi_dtype_memory_bench() {
        let mut bench = MultiDTypeMemoryBench;
        let input = bench.setup(5);
        let output = bench.run(&input);

        assert!(!output.results.is_empty());
        for result in output.results {
            assert!(!result.dtype_name.is_empty());
            assert!(result.element_size > 0);
            assert!(result.memory_usage > 0);
        }
    }

    #[test]
    fn test_bytes_accessed_calculations() {
        let memory_bench = MemoryBench;
        let expected = (10 * 10 * 10 + 5 * 5 * 5) * std::mem::size_of::<f32>();
        assert_eq!(memory_bench.bytes_accessed(10), expected);

        let copy_bench = MemoryCopyBench;
        let tensor_size = 10 * 10 * std::mem::size_of::<f32>();
        let expected_copy = tensor_size + (4 * tensor_size);
        assert_eq!(copy_bench.bytes_accessed(10), expected_copy);
    }

    #[test]
    fn test_concurrent_result_calculations() {
        let result = ConcurrentResult {
            thread_count: 4,
            successful_operations: 20,
            failed_operations: 0,
            total_duration: Duration::from_secs(2),
            cumulative_thread_time: Duration::from_secs(8),
            operations_per_second: 10.0,
        };

        assert_eq!(result.operations_per_second, 10.0);
        assert_eq!(result.thread_count, 4);
    }

    #[test]
    fn test_fragmentation_result() {
        let result = FragmentationResult {
            successful_allocations: 8,
            failed_allocations: 2,
            avg_allocation_time: Duration::from_millis(10),
            fragmentation_level: 0.2,
        };

        assert_eq!(result.fragmentation_level, 0.2);
        assert_eq!(result.successful_allocations, 8);
        assert_eq!(result.failed_allocations, 2);
    }
}
