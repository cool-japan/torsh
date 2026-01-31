//! Stress tests for distributed training system reliability
//!
//! These tests push the distributed training system to its limits to verify
//! reliability under extreme conditions and high load scenarios.

use futures_util::future::join_all;
use scirs2_core::random::quick::random_usize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use torsh_core::Result;
#[cfg(feature = "scirs2-simd")]
use torsh_distributed::communication_scheduler::ParallelExecutionStrategy;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_gather, all_reduce, barrier, broadcast, reduce, scatter},
    communication_scheduler::{
        CommunicationOp, CommunicationScheduler, Priority, SchedulerConfig, SchedulingStrategy,
    },
    gradient_compression::{CompressionConfig, CompressionMethod, GradientCompressor},
    init_process_group, ProcessGroup,
};
use torsh_tensor::creation::{ones, randn};

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Maximum number of concurrent operations
    pub max_concurrent_ops: usize,
    /// Duration of stress test
    pub test_duration: Duration,
    /// Maximum tensor size to test
    pub max_tensor_size: usize,
    /// Maximum world size to test
    pub max_world_size: u32,
    /// Number of iterations for endurance tests
    pub endurance_iterations: usize,
    /// Memory pressure threshold (GB)
    pub memory_pressure_gb: f64,
    /// Target operations per second
    pub target_ops_per_second: f64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_concurrent_ops: 100,
            test_duration: Duration::from_secs(30),
            max_tensor_size: 1_000_000, // 1M elements
            max_world_size: 16,
            endurance_iterations: 1000,
            memory_pressure_gb: 1.0,
            target_ops_per_second: 100.0,
        }
    }
}

/// Stress test metrics
#[derive(Debug, Clone)]
pub struct StressTestMetrics {
    pub test_name: String,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_duration: Duration,
    pub peak_memory_usage: usize,
    pub average_latency: Duration,
    pub operations_per_second: f64,
    pub error_rate: f64,
}

impl StressTestMetrics {
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::ZERO,
            peak_memory_usage: 0,
            average_latency: Duration::ZERO,
            operations_per_second: 0.0,
            error_rate: 0.0,
        }
    }

    pub fn calculate_derived_metrics(&mut self) {
        if self.total_operations > 0 {
            self.error_rate = self.failed_operations as f64 / self.total_operations as f64;
            self.operations_per_second = if self.total_duration.as_secs_f64() > 0.0 {
                self.total_operations as f64 / self.total_duration.as_secs_f64()
            } else {
                0.0
            };
            self.average_latency = if self.successful_operations > 0 {
                Duration::from_nanos(
                    (self.total_duration.as_nanos() / self.successful_operations as u128)
                        .try_into()
                        .unwrap_or(0),
                )
            } else {
                Duration::ZERO
            };
        }
    }

    pub fn print_summary(&self) {
        println!("\n=== Stress Test Results: {} ===", self.test_name);
        println!("Total Operations: {}", self.total_operations);
        println!("Successful: {}", self.successful_operations);
        println!("Failed: {}", self.failed_operations);
        println!("Error Rate: {:.2}%", self.error_rate * 100.0);
        println!("Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!("Ops/sec: {:.2}", self.operations_per_second);
        println!(
            "Average Latency: {:.2}ms",
            self.average_latency.as_secs_f64() * 1000.0
        );
        println!("Peak Memory: {} KB", self.peak_memory_usage / 1024);
    }

    pub fn assert_performance_thresholds(&self, min_ops_per_sec: f64, max_error_rate: f64) {
        assert!(
            self.operations_per_second >= min_ops_per_sec,
            "Performance too low: {:.2} ops/sec < {:.2} ops/sec",
            self.operations_per_second,
            min_ops_per_sec
        );

        assert!(
            self.error_rate <= max_error_rate,
            "Error rate too high: {:.2}% > {:.2}%",
            self.error_rate * 100.0,
            max_error_rate * 100.0
        );
    }
}

/// Stress test runner
pub struct StressTestRunner {
    config: StressTestConfig,
    metrics: Arc<Mutex<StressTestMetrics>>,
}

impl StressTestRunner {
    pub fn new(config: StressTestConfig, test_name: String) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(StressTestMetrics::new(test_name))),
        }
    }

    /// Run high-concurrency stress test
    pub async fn run_concurrency_stress_test(&self, pg: ProcessGroup) -> Result<()> {
        let start_time = Instant::now();
        let mut handles = Vec::new();
        let pg = Arc::new(pg);

        // Launch concurrent operations
        for i in 0..self.config.max_concurrent_ops {
            let pg_clone = Arc::clone(&pg);
            let metrics = self.metrics.clone();

            let handle = tokio::spawn(async move {
                let tensor_size = 100 + (i % 900); // Varying sizes
                let mut tensor = ones::<f32>(&[tensor_size]).unwrap();

                let op_start = Instant::now();
                let result = all_reduce(&mut tensor, ReduceOp::Sum, &pg_clone).await;
                let op_duration = op_start.elapsed();

                let mut metrics = metrics.lock().expect("lock should not be poisoned");
                metrics.total_operations += 1;

                match result {
                    Ok(()) => {
                        metrics.successful_operations += 1;
                        metrics.total_duration += op_duration;
                    }
                    Err(_) => {
                        metrics.failed_operations += 1;
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all operations to complete
        let results = join_all(handles).await;
        let total_test_duration = start_time.elapsed();

        // Count successful task completions
        let completed_tasks = results.iter().filter(|r| r.is_ok()).count();
        println!(
            "Completed {} out of {} concurrent tasks",
            completed_tasks, self.config.max_concurrent_ops
        );

        {
            let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
            if metrics.total_duration < total_test_duration {
                metrics.total_duration = total_test_duration;
            }
            metrics.calculate_derived_metrics();
        }

        Ok(())
    }

    /// Run memory pressure stress test
    pub async fn run_memory_pressure_test(&self, pg: ProcessGroup) -> Result<()> {
        let target_memory_bytes =
            (self.config.memory_pressure_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let element_size = std::mem::size_of::<f32>();
        let max_elements_per_tensor = target_memory_bytes / (element_size * 10); // Use 1/10th for safety

        let start_time = Instant::now();
        let mut operation_count = 0;

        while start_time.elapsed() < self.config.test_duration {
            // Create large tensor to simulate memory pressure
            let tensor_elements =
                std::cmp::min(max_elements_per_tensor, self.config.max_tensor_size);
            let mut large_tensor = ones::<f32>(&[tensor_elements])?;

            let op_start = Instant::now();
            let result = all_reduce(&mut large_tensor, ReduceOp::Sum, &pg).await;
            let op_duration = op_start.elapsed();

            operation_count += 1;

            {
                let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
                metrics.total_operations += 1;
                metrics.total_duration += op_duration;

                // Estimate memory usage
                let current_memory = tensor_elements * element_size;
                if current_memory > metrics.peak_memory_usage {
                    metrics.peak_memory_usage = current_memory;
                }

                match result {
                    Ok(()) => metrics.successful_operations += 1,
                    Err(_) => metrics.failed_operations += 1,
                }
            }

            // Brief pause to prevent overwhelming the system
            if operation_count % 10 == 0 {
                sleep(Duration::from_millis(10)).await;
            }
        }

        {
            let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
            metrics.calculate_derived_metrics();
        }

        Ok(())
    }

    /// Run endurance test
    pub async fn run_endurance_test(&self, pg: ProcessGroup) -> Result<()> {
        let start_time = Instant::now();

        for i in 0..self.config.endurance_iterations {
            // Vary operation types
            let op_result = match i % 4 {
                0 => {
                    let mut tensor = ones::<f32>(&[1000])?;
                    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await
                }
                1 => {
                    let tensor = ones::<f32>(&[1000])?;
                    let mut output = Vec::new();
                    all_gather(&mut output, &tensor, &pg).await.map(|_| ())
                }
                2 => {
                    let mut tensor = ones::<f32>(&[1000])?;
                    broadcast(&mut tensor, 0, &pg).await
                }
                _ => barrier(&pg).await,
            };

            {
                let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
                metrics.total_operations += 1;

                match op_result {
                    Ok(()) => metrics.successful_operations += 1,
                    Err(_) => metrics.failed_operations += 1,
                }
            }

            // Progress reporting
            if i % 100 == 0 && i > 0 {
                println!(
                    "Endurance test progress: {}/{} iterations",
                    i, self.config.endurance_iterations
                );
            }
        }

        {
            let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
            metrics.total_duration = start_time.elapsed();
            metrics.calculate_derived_metrics();
        }

        Ok(())
    }

    /// Run network saturation test
    pub async fn run_network_saturation_test(&self, pg: ProcessGroup) -> Result<()> {
        let start_time = Instant::now();
        let mut active_operations = 0;
        let max_active = 20; // Limit to prevent overwhelming
        let pg = Arc::new(pg);

        while start_time.elapsed() < self.config.test_duration {
            if active_operations < max_active {
                let pg_clone = Arc::clone(&pg);
                let metrics = self.metrics.clone();

                tokio::spawn(async move {
                    // Use large tensors to saturate network
                    let tensor_size = 10000; // Large tensor for network stress
                    let mut tensor = randn::<f32>(&[tensor_size]).unwrap();

                    let op_start = Instant::now();
                    let result = all_reduce(&mut tensor, ReduceOp::Sum, &pg_clone).await;
                    let op_duration = op_start.elapsed();

                    let mut metrics = metrics.lock().expect("lock should not be poisoned");
                    metrics.total_operations += 1;
                    metrics.total_duration += op_duration;

                    match result {
                        Ok(()) => metrics.successful_operations += 1,
                        Err(_) => metrics.failed_operations += 1,
                    }
                });

                active_operations += 1;
            }

            sleep(Duration::from_millis(1)).await; // Small delay

            // Periodically reduce active count (simulate completion)
            if active_operations >= max_active {
                sleep(Duration::from_millis(100)).await;
                active_operations = std::cmp::max(0, active_operations - 5);
            }
        }

        // Wait for remaining operations
        sleep(Duration::from_secs(2)).await;

        {
            let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
            metrics.calculate_derived_metrics();
        }

        Ok(())
    }

    pub fn get_metrics(&self) -> StressTestMetrics {
        self.metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }
}

#[tokio::test]
async fn test_high_concurrency_stress() -> Result<()> {
    let config = StressTestConfig {
        max_concurrent_ops: 50, // Reduced for test
        test_duration: Duration::from_secs(5),
        ..Default::default()
    };

    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 50000).await?;
    let runner = StressTestRunner::new(config, "High Concurrency".to_string());

    runner.run_concurrency_stress_test(pg).await?;

    let metrics = runner.get_metrics();
    metrics.print_summary();

    // Assert reasonable performance
    metrics.assert_performance_thresholds(10.0, 0.1); // 10 ops/sec, 10% error rate

    Ok(())
}

#[tokio::test]
async fn test_memory_pressure_stress() -> Result<()> {
    let config = StressTestConfig {
        test_duration: Duration::from_secs(10),
        memory_pressure_gb: 0.1,  // 100MB for test
        max_tensor_size: 100_000, // Reduced for test
        ..Default::default()
    };

    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 50010).await?;
    let runner = StressTestRunner::new(config, "Memory Pressure".to_string());

    runner.run_memory_pressure_test(pg).await?;

    let metrics = runner.get_metrics();
    metrics.print_summary();

    // Should handle memory pressure gracefully
    assert!(metrics.peak_memory_usage > 0, "Should have used memory");
    assert!(
        metrics.total_operations > 0,
        "Should have performed operations"
    );

    Ok(())
}

#[tokio::test]
async fn test_endurance_stress() -> Result<()> {
    let config = StressTestConfig {
        endurance_iterations: 200, // Reduced for test
        ..Default::default()
    };

    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 50020).await?;
    let runner = StressTestRunner::new(config, "Endurance".to_string());

    runner.run_endurance_test(pg).await?;

    let metrics = runner.get_metrics();
    metrics.print_summary();

    // Should complete most operations successfully
    assert_eq!(metrics.total_operations, 200);
    metrics.assert_performance_thresholds(50.0, 0.05); // 50 ops/sec, 5% error rate

    Ok(())
}

#[tokio::test]
async fn test_network_saturation_stress() -> Result<()> {
    let config = StressTestConfig {
        test_duration: Duration::from_secs(8),
        ..Default::default()
    };

    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 50030).await?;
    let runner = StressTestRunner::new(config, "Network Saturation".to_string());

    runner.run_network_saturation_test(pg).await?;

    let metrics = runner.get_metrics();
    metrics.print_summary();

    // Should handle network saturation
    assert!(
        metrics.total_operations > 0,
        "Should have performed operations"
    );

    Ok(())
}

#[tokio::test]
async fn test_large_tensor_stress() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 50040).await?;

    // Test with increasingly large tensors
    let sizes = vec![1000, 10000, 100000];
    let mut all_successful = true;

    for size in sizes {
        println!("Testing tensor size: {} elements", size);

        let start = Instant::now();
        let mut large_tensor = ones::<f32>(&[size])?;

        let result = timeout(
            Duration::from_secs(10),
            all_reduce(&mut large_tensor, ReduceOp::Sum, &pg),
        )
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(())) => {
                println!(
                    "Size {} completed in {:.2}ms",
                    size,
                    duration.as_secs_f64() * 1000.0
                );
            }
            Ok(Err(e)) => {
                println!("Size {} failed: {}", size, e);
                all_successful = false;
            }
            Err(_) => {
                println!("Size {} timed out", size);
                all_successful = false;
            }
        }
    }

    assert!(all_successful, "All large tensor operations should succeed");
    Ok(())
}

#[tokio::test]
async fn test_rapid_world_size_changes() -> Result<()> {
    // Test rapidly changing world sizes
    let world_sizes = [2, 4, 8, 4, 2];

    for (i, &world_size) in world_sizes.iter().enumerate() {
        println!("Testing world size: {}", world_size);

        let pg = init_process_group(
            BackendType::Gloo,
            0,
            world_size,
            "127.0.0.1",
            50050 + i as u16,
        )
        .await?;

        // Perform operations with this world size
        for _ in 0..5 {
            let mut tensor = ones::<f32>(&[1000])?;
            all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
        }

        // Barrier to ensure all operations complete
        barrier(&pg).await?;
    }

    println!("Rapid world size changes test completed");
    Ok(())
}

#[tokio::test]
async fn test_mixed_operation_stress() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 50060).await?;
    let test_duration = Duration::from_secs(10);
    let start_time = Instant::now();

    let mut operation_counts = [0u64; 6]; // Track different operation types

    while start_time.elapsed() < test_duration {
        let op_type = random_usize(0, 5); // 0-5 for 6 operations (array indices 0-5)
        let tensor_size = random_usize(500, 2000); // 500-2000 elements

        let result = match op_type {
            0 => {
                let mut tensor = ones::<f32>(&[tensor_size])?;
                all_reduce(&mut tensor, ReduceOp::Sum, &pg).await
            }
            1 => {
                let tensor = ones::<f32>(&[tensor_size])?;
                let mut output = Vec::new();
                all_gather(&mut output, &tensor, &pg).await.map(|_| ())
            }
            2 => {
                let mut tensor = ones::<f32>(&[tensor_size])?;
                broadcast(&mut tensor, 0, &pg).await
            }
            3 => {
                let mut tensor = ones::<f32>(&[tensor_size])?;
                reduce(&mut tensor, 0, ReduceOp::Sum, &pg).await
            }
            4 => {
                let mut output = ones::<f32>(&[tensor_size])?;
                let input_tensors = vec![ones::<f32>(&[tensor_size])?; 4];
                scatter(&mut output, Some(&input_tensors), 0, &pg)
                    .await
                    .map(|_| ())
            }
            _ => barrier(&pg).await,
        };

        operation_counts[op_type] += 1;

        if result.is_err() {
            println!("Operation {} failed", op_type);
        }

        // Small delay to prevent overwhelming
        if operation_counts.iter().sum::<u64>() % 20 == 0 {
            sleep(Duration::from_millis(10)).await;
        }
    }

    let total_ops: u64 = operation_counts.iter().sum();
    println!("Mixed operations completed: {} total", total_ops);
    println!("Operation distribution: {:?}", operation_counts);

    assert!(
        total_ops > 50,
        "Should have completed many mixed operations"
    );
    Ok(())
}

#[tokio::test]
async fn test_compression_under_stress() -> Result<()> {
    let compression_config = CompressionConfig {
        method: CompressionMethod::TopK { k: 0.1 },
        compression_ratio: 0.1,
        error_feedback: true,
        error_feedback_momentum: 0.9,
        memory_efficient: false,
        warmup_steps: 0,
    };

    let mut compressor = GradientCompressor::new(compression_config);
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 50070).await?;

    let test_duration = Duration::from_secs(5);
    let start_time = Instant::now();
    let mut compression_count = 0;

    while start_time.elapsed() < test_duration {
        // Create gradient tensor
        let gradient = randn::<f32>(&[5000])?;

        // Compress and decompress
        let compressed = compressor.compress(&gradient, "test_gradient")?;
        let decompressed = compressor.decompress(&compressed)?;

        // Use decompressed gradient in collective operation
        let mut grad_tensor = decompressed;
        all_reduce(&mut grad_tensor, ReduceOp::Sum, &pg).await?;

        compression_count += 1;

        if compression_count % 10 == 0 {
            sleep(Duration::from_millis(5)).await;
        }
    }

    println!(
        "Compression stress test: {} compressions in {:.2}s",
        compression_count,
        test_duration.as_secs_f64()
    );

    assert!(
        compression_count > 20,
        "Should perform many compressions under stress"
    );
    Ok(())
}

#[tokio::test]
async fn test_scheduler_stress() -> Result<()> {
    let scheduler_config = SchedulerConfig {
        strategy: SchedulingStrategy::PriorityBased,
        max_concurrent_ops: 10,
        bandwidth_limit_bps: 1_000_000_000, // 1 Gbps
        enable_priorities: true,
        adaptive_scheduling: false,
        timeout_ms: 5000,
        enable_compression: false,
        compression_threshold: 1024 * 1024, // 1MB
        #[cfg(feature = "scirs2-simd")]
        enable_simd_optimization: false,
        #[cfg(feature = "scirs2-simd")]
        simd_chunk_size: 1024,
        #[cfg(feature = "scirs2-simd")]
        enable_auto_vectorization: false,
        #[cfg(feature = "scirs2-simd")]
        parallel_execution_strategy: ParallelExecutionStrategy::UniformChunking,
    };

    let scheduler = Arc::new(CommunicationScheduler::new(scheduler_config));
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 50080).await?;
    let pg_arc = Arc::new(pg);

    // Start the scheduler
    scheduler.start().await?;

    let num_operations = 50;
    let mut handles = Vec::new();

    let start = Instant::now();

    // Schedule many operations rapidly using spawn to get concurrent execution
    for i in 0..num_operations {
        let tensor = ones::<f32>(&[1000])?.mul_scalar(i as f32)?;
        let scheduler_clone = scheduler.clone();
        let pg_clone = pg_arc.clone();

        let handle = tokio::spawn(async move {
            scheduler_clone
                .schedule_task(
                    CommunicationOp::AllReduce,
                    tensor,
                    pg_clone,
                    Priority::Normal,
                )
                .await
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let results = join_all(handles).await;

    let duration = start.elapsed();
    let successful = results.iter().filter(|r| matches!(r, Ok(Ok(_)))).count();

    println!(
        "Scheduler stress test: {}/{} operations completed in {:.2}ms",
        successful,
        num_operations,
        duration.as_secs_f64() * 1000.0
    );

    assert!(
        successful >= num_operations * 80 / 100,
        "At least 80% of operations should succeed"
    );

    Ok(())
}
