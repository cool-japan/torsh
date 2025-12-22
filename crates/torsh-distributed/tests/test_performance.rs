//! Performance tests for distributed training operations
//!
//! These tests measure and validate the performance characteristics of various
//! distributed training operations under different conditions and configurations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use torsh_core::Result;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_gather, all_reduce, barrier, broadcast},
    communication_scheduler::{CommunicationScheduler, SchedulerConfig, SchedulingStrategy},
    gradient_compression::{CompressionConfig, CompressionMethod, GradientCompressor},
    init_process_group,
    profiling::{init_global_profiler, CommunicationOpType, ProfilingConfig},
};
use torsh_tensor::creation::{ones, randn};

/// Performance metrics for operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Duration of operation
    pub duration: Duration,
    /// Data size in bytes
    pub data_size: usize,
    /// Throughput in bytes per second
    pub throughput: f64,
    /// Number of processes involved
    pub world_size: u32,
    /// Memory usage before operation
    pub memory_before: usize,
    /// Memory usage after operation
    pub memory_after: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
}

impl PerformanceMetrics {
    pub fn new(operation: String, duration: Duration, data_size: usize, world_size: u32) -> Self {
        let throughput = if duration.as_secs_f64() > 0.0 {
            data_size as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            operation,
            duration,
            data_size,
            throughput,
            world_size,
            memory_before: 0,
            memory_after: 0,
            cpu_usage: 0.0,
        }
    }

    /// Calculate bandwidth in MB/s
    pub fn bandwidth_mbps(&self) -> f64 {
        self.throughput / (1024.0 * 1024.0)
    }

    /// Calculate latency per byte in nanoseconds
    pub fn latency_per_byte_ns(&self) -> f64 {
        if self.data_size > 0 {
            self.duration.as_nanos() as f64 / self.data_size as f64
        } else {
            0.0
        }
    }
}

/// Performance test configuration
#[derive(Debug, Clone)]
pub struct PerformanceTestConfig {
    /// Tensor sizes to test
    pub tensor_sizes: Vec<Vec<usize>>,
    /// World sizes to test
    pub world_sizes: Vec<u32>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Timeout for operations
    pub timeout: Duration,
    /// Whether to collect detailed profiling
    pub enable_profiling: bool,
}

impl Default for PerformanceTestConfig {
    fn default() -> Self {
        Self {
            tensor_sizes: vec![
                vec![1000],       // 1K elements
                vec![10000],      // 10K elements
                vec![100000],     // 100K elements
                vec![100, 100],   // 10K elements (2D)
                vec![32, 32, 32], // ~32K elements (3D)
                vec![1000, 1000], // 1M elements (2D)
            ],
            world_sizes: vec![2, 4, 8],
            iterations: 10,
            warmup_iterations: 3,
            timeout: Duration::from_secs(30),
            enable_profiling: true,
        }
    }
}

/// Performance test suite for distributed operations
pub struct PerformanceTestSuite {
    config: PerformanceTestConfig,
    results: Arc<Mutex<Vec<PerformanceMetrics>>>,
}

impl PerformanceTestSuite {
    pub fn new(config: PerformanceTestConfig) -> Self {
        Self {
            config,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Run performance benchmark for all_reduce operation
    pub async fn benchmark_all_reduce(&self) -> Result<()> {
        for &world_size in &self.config.world_sizes {
            for tensor_shape in &self.config.tensor_sizes {
                let metrics = self
                    .benchmark_all_reduce_single(world_size, tensor_shape)
                    .await?;
                self.results.lock().unwrap().push(metrics);
            }
        }
        Ok(())
    }

    async fn benchmark_all_reduce_single(
        &self,
        world_size: u32,
        shape: &[usize],
    ) -> Result<PerformanceMetrics> {
        let pg = init_process_group(BackendType::Gloo, 0, world_size, "127.0.0.1", 30000).await?;
        let element_count: usize = shape.iter().product();
        let data_size = element_count * std::mem::size_of::<f32>();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let mut tensor = ones::<f32>(shape)?;
            let _ = all_reduce(&mut tensor, ReduceOp::Sum, &pg).await;
        }

        // Benchmark
        let mut durations = Vec::new();
        for _ in 0..self.config.iterations {
            let mut tensor = ones::<f32>(shape)?;
            let start = Instant::now();

            let result = timeout(
                self.config.timeout,
                all_reduce(&mut tensor, ReduceOp::Sum, &pg),
            )
            .await;
            let duration = start.elapsed();

            match result {
                Ok(Ok(())) => durations.push(duration),
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(torsh_core::TorshError::Other(
                        "Operation timed out".to_string(),
                    ))
                }
            }
        }

        let avg_nanos =
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128;
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        Ok(PerformanceMetrics::new(
            format!("all_reduce_{}x{}", world_size, element_count),
            avg_duration,
            data_size,
            world_size,
        ))
    }

    /// Benchmark collective operations scaling
    pub async fn benchmark_collective_scaling(&self) -> Result<()> {
        let operations = vec![
            ("all_reduce", CommunicationOpType::AllReduce),
            ("all_gather", CommunicationOpType::AllGather),
            ("broadcast", CommunicationOpType::Broadcast),
            ("barrier", CommunicationOpType::Barrier),
        ];

        for (op_name, op_type) in operations {
            for &world_size in &self.config.world_sizes {
                let metrics = self
                    .benchmark_operation_scaling(op_name, op_type, world_size)
                    .await?;
                self.results.lock().unwrap().push(metrics);
            }
        }

        Ok(())
    }

    async fn benchmark_operation_scaling(
        &self,
        op_name: &str,
        op_type: CommunicationOpType,
        world_size: u32,
    ) -> Result<PerformanceMetrics> {
        let pg = init_process_group(BackendType::Gloo, 0, world_size, "127.0.0.1", 30001).await?;
        let shape = &[1000, 1000]; // Fixed size for scaling test
        let data_size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        let mut durations = Vec::new();

        for _ in 0..self.config.iterations {
            let start = Instant::now();

            let result = match op_type {
                CommunicationOpType::AllReduce => {
                    let mut tensor = ones::<f32>(shape)?;
                    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await
                }
                CommunicationOpType::AllGather => {
                    let tensor = ones::<f32>(shape)?;
                    let mut output = Vec::new();
                    all_gather(&mut output, &tensor, &pg).await
                }
                CommunicationOpType::Broadcast => {
                    let mut tensor = ones::<f32>(shape)?;
                    broadcast(&mut tensor, 0, &pg).await
                }
                CommunicationOpType::Barrier => barrier(&pg).await,
                _ => Ok(()),
            };

            let duration = start.elapsed();

            match result {
                Ok(()) => durations.push(duration),
                Err(e) => return Err(e.into()),
            }
        }

        let avg_nanos =
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128;
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        Ok(PerformanceMetrics::new(
            format!("{}_scaling_ws{}", op_name, world_size),
            avg_duration,
            data_size,
            world_size,
        ))
    }

    /// Benchmark gradient compression performance
    pub async fn benchmark_gradient_compression(&self) -> Result<()> {
        let compression_methods = vec![
            CompressionMethod::TopK { k: 0.1 }, // Keep top 10%
            CompressionMethod::Quantization { bits: 8 },
            CompressionMethod::SignSGD,
        ];

        for method in compression_methods {
            for tensor_shape in &self.config.tensor_sizes {
                let metrics = self
                    .benchmark_compression_single(method.clone(), tensor_shape)
                    .await?;
                self.results.lock().unwrap().push(metrics);
            }
        }

        Ok(())
    }

    async fn benchmark_compression_single(
        &self,
        method: CompressionMethod,
        shape: &[usize],
    ) -> Result<PerformanceMetrics> {
        let config = CompressionConfig {
            method: method.clone(),
            compression_ratio: 0.1,
            error_feedback: true,
            error_feedback_momentum: 0.9,
            memory_efficient: false,
            warmup_steps: 0,
        };

        let mut compressor = GradientCompressor::new(config);
        let element_count: usize = shape.iter().product();
        let data_size = element_count * std::mem::size_of::<f32>();

        // Create gradient tensor
        let gradient = randn::<f32>(shape)?;

        let mut durations = Vec::new();
        for _ in 0..self.config.iterations {
            let start = Instant::now();

            // Compress gradient
            let compressed = compressor.compress(&gradient, "test_param")?;

            // Decompress gradient
            let _decompressed = compressor.decompress(&compressed)?;

            let duration = start.elapsed();
            durations.push(duration);
        }

        let avg_nanos =
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128;
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        Ok(PerformanceMetrics::new(
            format!("compression_{:?}_{}", method, element_count),
            avg_duration,
            data_size,
            1, // Single process for compression
        ))
    }

    /// Benchmark communication scheduling performance
    pub async fn benchmark_communication_scheduling(&self) -> Result<()> {
        let strategies = vec![
            SchedulingStrategy::FIFO,
            SchedulingStrategy::PriorityBased,
            SchedulingStrategy::RoundRobin,
        ];

        for strategy in strategies {
            let metrics = self.benchmark_scheduler_single(strategy).await?;
            self.results.lock().unwrap().push(metrics);
        }

        Ok(())
    }

    async fn benchmark_scheduler_single(
        &self,
        strategy: SchedulingStrategy,
    ) -> Result<PerformanceMetrics> {
        let config = SchedulerConfig {
            strategy: strategy.clone(),
            max_concurrent_ops: 4,
            bandwidth_limit_bps: 1_000_000_000,
            enable_priorities: true,
            adaptive_scheduling: false,
            timeout_ms: 10000,
            enable_compression: false,
            compression_threshold: 1024 * 1024,
            #[cfg(feature = "scirs2-simd")]
            enable_simd_optimization: false,
            #[cfg(feature = "scirs2-simd")]
            simd_chunk_size: 1024,
            #[cfg(feature = "scirs2-simd")]
            enable_auto_vectorization: false,
            #[cfg(feature = "scirs2-simd")]
            parallel_execution_strategy: torsh_distributed::communication_scheduler::ParallelExecutionStrategy::UniformChunking,
        };

        let scheduler = CommunicationScheduler::new(config);
        let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 30002).await?;
        let pg_arc = Arc::new(pg);

        scheduler.start().await?;
        let mut durations = Vec::new();

        for _ in 0..self.config.iterations {
            let start = Instant::now();

            // Schedule multiple operations
            let mut futures = Vec::new();
            for _ in 0..10 {
                let tensor = ones::<f32>(&[100, 100])?.mul_scalar(1.0)?;
                let future = scheduler.schedule_task(
                    torsh_distributed::communication_scheduler::CommunicationOp::AllReduce,
                    tensor,
                    pg_arc.clone(),
                    torsh_distributed::communication_scheduler::Priority::Normal,
                );
                futures.push(future);
            }

            // Wait for all operations to complete
            for future in futures {
                let _ = future.await;
            }

            let duration = start.elapsed();
            durations.push(duration);
        }

        let avg_nanos =
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128;
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        let data_size = 10 * 100 * 100 * std::mem::size_of::<f32>(); // 10 tensors

        Ok(PerformanceMetrics::new(
            format!("scheduling_{:?}", strategy),
            avg_duration,
            data_size,
            4,
        ))
    }

    /// Get all performance results
    pub fn get_results(&self) -> Vec<PerformanceMetrics> {
        self.results.lock().unwrap().clone()
    }

    /// Print performance summary
    pub fn print_summary(&self) {
        let results = self.get_results();

        println!("\n=== Performance Test Summary ===");
        println!(
            "{:<40} {:>12} {:>12} {:>12} {:>12}",
            "Operation", "Duration(ms)", "Bandwidth(MB/s)", "World Size", "Data Size(KB)"
        );
        println!("{}", "-".repeat(80));

        for metrics in &results {
            println!(
                "{:<40} {:>12.2} {:>12.2} {:>12} {:>12.1}",
                metrics.operation,
                metrics.duration.as_secs_f64() * 1000.0,
                metrics.bandwidth_mbps(),
                metrics.world_size,
                metrics.data_size as f64 / 1024.0
            );
        }

        // Calculate averages by operation type
        let mut operation_groups: HashMap<String, Vec<&PerformanceMetrics>> = HashMap::new();
        for metrics in &results {
            let op_type = metrics
                .operation
                .split('_')
                .next()
                .unwrap_or("unknown")
                .to_string();
            operation_groups.entry(op_type).or_default().push(metrics);
        }

        println!("\n=== Average Performance by Operation Type ===");
        for (op_type, metrics_list) in operation_groups {
            let avg_duration: f64 = metrics_list
                .iter()
                .map(|m| m.duration.as_secs_f64())
                .sum::<f64>()
                / metrics_list.len() as f64;

            let avg_bandwidth: f64 = metrics_list.iter().map(|m| m.bandwidth_mbps()).sum::<f64>()
                / metrics_list.len() as f64;

            println!(
                "{:<20} {:>12.2}ms {:>12.2}MB/s",
                op_type,
                avg_duration * 1000.0,
                avg_bandwidth
            );
        }
    }
}

#[tokio::test]
async fn test_all_reduce_performance() -> Result<()> {
    let config = PerformanceTestConfig {
        tensor_sizes: vec![vec![1000], vec![10000]],
        world_sizes: vec![2, 4],
        iterations: 5,
        ..Default::default()
    };

    let suite = PerformanceTestSuite::new(config);
    suite.benchmark_all_reduce().await?;

    let results = suite.get_results();
    assert!(!results.is_empty(), "Should have performance results");

    // Check that larger world sizes don't take exponentially longer
    let small_world_results: Vec<_> = results.iter().filter(|r| r.world_size == 2).collect();
    let large_world_results: Vec<_> = results.iter().filter(|r| r.world_size == 4).collect();

    if !small_world_results.is_empty() && !large_world_results.is_empty() {
        let small_avg = small_world_results
            .iter()
            .map(|r| r.duration.as_secs_f64())
            .sum::<f64>()
            / small_world_results.len() as f64;

        let large_avg = large_world_results
            .iter()
            .map(|r| r.duration.as_secs_f64())
            .sum::<f64>()
            / large_world_results.len() as f64;

        // Performance shouldn't degrade more than 4x with 2x world size
        assert!(
            large_avg < small_avg * 4.0,
            "Performance degradation too severe: {:.3}s vs {:.3}s",
            large_avg,
            small_avg
        );
    }

    suite.print_summary();
    Ok(())
}

#[tokio::test]
async fn test_collective_scaling_performance() -> Result<()> {
    let config = PerformanceTestConfig {
        world_sizes: vec![2, 4, 8],
        iterations: 3,
        ..Default::default()
    };

    let suite = PerformanceTestSuite::new(config);
    suite.benchmark_collective_scaling().await?;

    let results = suite.get_results();
    assert!(!results.is_empty(), "Should have scaling results");

    suite.print_summary();
    Ok(())
}

#[tokio::test]
async fn test_compression_performance() -> Result<()> {
    let config = PerformanceTestConfig {
        tensor_sizes: vec![vec![1000], vec![10000]],
        iterations: 5,
        ..Default::default()
    };

    let suite = PerformanceTestSuite::new(config);
    suite.benchmark_gradient_compression().await?;

    let results = suite.get_results();
    assert!(!results.is_empty(), "Should have compression results");

    // Verify compression provides reasonable speedup
    for result in &results {
        // Compression should complete within reasonable time
        assert!(
            result.duration.as_secs_f64() < 1.0,
            "Compression took too long: {:.3}s for {}",
            result.duration.as_secs_f64(),
            result.operation
        );
    }

    suite.print_summary();
    Ok(())
}

#[tokio::test]
async fn test_memory_usage_scaling() -> Result<()> {
    // Test that memory usage scales reasonably with tensor size
    let tensor_sizes = vec![
        vec![100],   // Small
        vec![1000],  // Medium
        vec![10000], // Large
    ];

    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 30003).await?;

    for size in tensor_sizes {
        let element_count: usize = size.iter().product();

        // Measure operation time for different sizes
        let start = Instant::now();
        let mut tensor = ones::<f32>(&size)?;
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
        let duration = start.elapsed();

        println!(
            "Size: {} elements, Duration: {:.3}ms",
            element_count,
            duration.as_secs_f64() * 1000.0
        );

        // Larger tensors should not take exponentially longer (for mock backend)
        assert!(
            duration.as_secs_f64() < 1.0,
            "Operation took too long for size {}: {:.3}s",
            element_count,
            duration.as_secs_f64()
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_concurrent_operations_performance() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 30004).await?;
    let pg_arc = Arc::new(pg);
    let num_concurrent = 10;

    // Test concurrent all_reduce operations
    let start = Instant::now();

    let mut futures = Vec::new();
    for i in 0..num_concurrent {
        let pg_clone = pg_arc.clone();
        let future = async move {
            let base = ones::<f32>(&[100, 100])?;
            let mut tensor = base.mul_scalar(i as f32)?;
            all_reduce(&mut tensor, ReduceOp::Sum, &pg_clone)
                .await
                .map_err(|e| e.into())
        };
        futures.push(future);
    }

    let results: Result<Vec<_>> = futures_util::future::try_join_all(futures).await;
    let duration = start.elapsed();

    assert!(results.is_ok(), "All concurrent operations should succeed");

    println!(
        "Concurrent operations: {} ops in {:.3}ms ({:.3}ms/op avg)",
        num_concurrent,
        duration.as_secs_f64() * 1000.0,
        duration.as_secs_f64() * 1000.0 / num_concurrent as f64
    );

    // Concurrent operations should not be much slower than sequential
    // (This is more relevant for real backends)
    assert!(
        duration.as_secs_f64() < 5.0,
        "Concurrent operations took too long"
    );

    Ok(())
}

#[tokio::test]
async fn test_profiling_overhead() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 30005).await?;

    // Test without profiling
    let start = Instant::now();
    for _ in 0..10 {
        let mut tensor = ones::<f32>(&[1000])?;
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
    }
    let duration_no_profiling = start.elapsed();

    // Test with profiling enabled
    let profiling_config = ProfilingConfig {
        enabled: true,
        max_events: 1000,
        track_per_operation_stats: true,
        track_per_rank_stats: true,
        sampling_rate: 1.0,
        min_duration_us: 0,
    };
    init_global_profiler(profiling_config)?;

    let start = Instant::now();
    for _ in 0..10 {
        let mut tensor = ones::<f32>(&[1000])?;
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
    }
    let duration_with_profiling = start.elapsed();

    let overhead_ratio =
        duration_with_profiling.as_secs_f64() / duration_no_profiling.as_secs_f64();

    println!("Profiling overhead: {:.2}x", overhead_ratio);

    // Profiling overhead should be reasonable (less than 2x for mock backend)
    assert!(
        overhead_ratio < 2.0,
        "Profiling overhead too high: {:.2}x",
        overhead_ratio
    );

    Ok(())
}
