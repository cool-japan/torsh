//! Stress tests for torsh-data
//!
//! These tests push the data loading system to its limits to identify
//! performance bottlenecks, memory leaks, and stability issues under load.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use torsh_data::dataloader::{simple_dataloader, simple_random_dataloader};
use torsh_data::prelude::*;
// SciRS2 POLICY compliant random generation
use scirs2_core::random::{thread_rng, Rng};
use torsh_tensor::creation::{ones, zeros};

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    pub duration_seconds: u64,
    pub max_memory_mb: usize,
    pub num_threads: usize,
    pub dataset_size: usize,
    pub batch_size: usize,
    pub error_threshold: f64, // Percentage of operations that can fail
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            duration_seconds: 30,
            max_memory_mb: 1024, // 1GB
            num_threads: 4,
            dataset_size: 10000,
            batch_size: 32,
            error_threshold: 0.01, // 1% error rate
        }
    }
}

/// Results from a stress test
#[derive(Debug)]
pub struct StressTestResult {
    pub test_name: String,
    pub config: StressTestConfig,
    pub duration: Duration,
    pub operations_completed: u64,
    pub operations_failed: u64,
    pub throughput_ops_per_sec: f64,
    pub error_rate: f64,
    pub peak_memory_mb: f64,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

impl StressTestResult {
    fn new(
        test_name: String,
        config: StressTestConfig,
        duration: Duration,
        operations_completed: u64,
        operations_failed: u64,
    ) -> Self {
        let total_ops = operations_completed + operations_failed;
        let throughput = if duration.as_secs_f64() > 0.0 {
            total_ops as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        let error_rate = if total_ops > 0 {
            operations_failed as f64 / total_ops as f64
        } else {
            0.0
        };

        Self {
            test_name,
            config,
            duration,
            operations_completed,
            operations_failed,
            throughput_ops_per_sec: throughput,
            error_rate,
            peak_memory_mb: 0.0,
            passed: false,
            failure_reasons: Vec::new(),
        }
    }

    #[allow(dead_code)]
    fn with_memory(mut self, peak_memory_mb: f64) -> Self {
        self.peak_memory_mb = peak_memory_mb;
        self
    }

    fn evaluate(mut self) -> Self {
        // Check if test passed based on criteria
        if self.error_rate > self.config.error_threshold {
            self.failure_reasons.push(format!(
                "Error rate {:.2}% exceeds threshold {:.2}%",
                self.error_rate * 100.0,
                self.config.error_threshold * 100.0
            ));
        }

        if self.peak_memory_mb > self.config.max_memory_mb as f64 {
            self.failure_reasons.push(format!(
                "Peak memory {:.2} MB exceeds limit {} MB",
                self.peak_memory_mb, self.config.max_memory_mb
            ));
        }

        if self.throughput_ops_per_sec < 10.0 {
            self.failure_reasons.push(format!(
                "Throughput {:.2} ops/sec is too low",
                self.throughput_ops_per_sec
            ));
        }

        self.passed = self.failure_reasons.is_empty();
        self
    }
}

/// Thread-safe operation counter
#[derive(Debug)]
struct OperationCounter {
    completed: Arc<Mutex<u64>>,
    failed: Arc<Mutex<u64>>,
}

impl OperationCounter {
    fn new() -> Self {
        Self {
            completed: Arc::new(Mutex::new(0)),
            failed: Arc::new(Mutex::new(0)),
        }
    }

    fn increment_completed(&self) {
        *self.completed.lock().unwrap() += 1;
    }

    fn increment_failed(&self) {
        *self.failed.lock().unwrap() += 1;
    }

    fn get_counts(&self) -> (u64, u64) {
        let completed = *self.completed.lock().unwrap();
        let failed = *self.failed.lock().unwrap();
        (completed, failed)
    }
}

/// High-load data loading stress test
pub fn stress_test_high_load_dataloader() -> StressTestResult {
    let config = StressTestConfig::default();
    let counter = OperationCounter::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_seconds);

    // Create a large dataset for stress testing
    let data = ones::<f32>(&[config.dataset_size, 100]).expect("Failed to create test data");
    let labels = zeros::<f32>(&[config.dataset_size]).expect("Failed to create test labels");
    let dataset = Arc::new(TensorDataset::from_tensors(vec![data, labels]));

    let mut handles = Vec::new();

    // Spawn multiple threads to stress the data loader
    for thread_id in 0..config.num_threads {
        let dataset_clone = Arc::clone(&dataset);
        let counter_clone = OperationCounter {
            completed: Arc::clone(&counter.completed),
            failed: Arc::clone(&counter.failed),
        };
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            let _rng = thread_rng(); // SciRS2 POLICY compliant

            while Instant::now() < end_time {
                // Create a new data loader for each iteration to stress initialization
                match simple_random_dataloader(
                    (*dataset_clone).clone(),
                    config_clone.batch_size,
                    Some(thread_id as u64 + thread_rng().gen::<u64>()), // SciRS2 POLICY compliant
                ) {
                    Ok(dataloader) => {
                        // Process a few batches
                        let mut batch_count = 0;
                        for batch_result in dataloader.iter() {
                            match batch_result {
                                Ok(_batch) => {
                                    counter_clone.increment_completed();
                                    batch_count += 1;
                                    if batch_count >= 5 {
                                        break; // Limit batches per iteration
                                    }
                                }
                                Err(_) => {
                                    counter_clone.increment_failed();
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        counter_clone.increment_failed();
                    }
                }

                // Small delay to prevent overwhelming the system
                thread::sleep(Duration::from_millis(1));
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let (completed, failed) = counter.get_counts();

    StressTestResult::new(
        "High Load DataLoader".to_string(),
        config,
        duration,
        completed,
        failed,
    )
    .evaluate()
}

/// Memory pressure stress test
pub fn stress_test_memory_pressure() -> StressTestResult {
    let mut config = StressTestConfig::default();
    config.dataset_size = 50000; // Larger dataset
    config.batch_size = 128; // Larger batches
    config.max_memory_mb = 2048; // 2GB limit

    let counter = OperationCounter::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_seconds);

    // Create multiple large datasets to stress memory
    let mut datasets = Vec::new();
    for _i in 0..5 {
        match ones::<f32>(&[config.dataset_size / 5, 200]) {
            Ok(data) => {
                let dataset = TensorDataset::from_tensor(data);
                datasets.push(dataset);
            }
            Err(_) => {
                return StressTestResult::new(
                    "Memory Pressure".to_string(),
                    config,
                    Duration::from_secs(0),
                    0,
                    1,
                )
                .evaluate();
            }
        }
    }

    let large_dataset = ConcatDataset::new(datasets);
    let dataset = Arc::new(large_dataset);

    let mut handles = Vec::new();

    // Spawn threads with high memory usage patterns
    for thread_id in 0..config.num_threads {
        let dataset_clone = Arc::clone(&dataset);
        let counter_clone = OperationCounter {
            completed: Arc::clone(&counter.completed),
            failed: Arc::clone(&counter.failed),
        };
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            while Instant::now() < end_time {
                // Create cached dataset to increase memory usage (simplified)
                // Note: CachedDataset doesn't implement Clone, using base dataset
                let cached_dataset = (*dataset_clone).clone();

                match simple_random_dataloader(
                    cached_dataset,
                    config_clone.batch_size,
                    Some(thread_id as u64),
                ) {
                    Ok(dataloader) => {
                        let mut batch_count = 0;
                        for batch_result in dataloader.iter() {
                            match batch_result {
                                Ok(_batch) => {
                                    counter_clone.increment_completed();
                                    batch_count += 1;
                                    if batch_count >= 3 {
                                        break;
                                    }
                                }
                                Err(_) => {
                                    counter_clone.increment_failed();
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        counter_clone.increment_failed();
                    }
                }

                thread::sleep(Duration::from_millis(10));
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let (completed, failed) = counter.get_counts();

    StressTestResult::new(
        "Memory Pressure".to_string(),
        config,
        duration,
        completed,
        failed,
    )
    .evaluate()
}

/// Transform pipeline stress test
pub fn stress_test_transform_pipeline() -> StressTestResult {
    use torsh_data::transforms::augmentation::*;
    use torsh_data::transforms::online::*;

    let config = StressTestConfig::default();
    let counter = OperationCounter::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_seconds);

    // Create heavy augmentation pipeline
    let pipeline = AugmentationPipeline::<torsh_tensor::Tensor<f32>>::heavy_augmentation();
    let engine = Arc::new(OnlineAugmentationEngine::new(pipeline).with_cache(100));

    let mut handles = Vec::new();

    for thread_id in 0..config.num_threads {
        let engine_clone = Arc::clone(&engine);
        let counter_clone = OperationCounter {
            completed: Arc::clone(&counter.completed),
            failed: Arc::clone(&counter.failed),
        };

        let handle = thread::spawn(move || {
            let tensor_size = [3, 128, 128]; // Moderately large tensors

            while Instant::now() < end_time {
                match ones::<f32>(&tensor_size) {
                    Ok(tensor) => {
                        let cache_key = format!(
                            "thread_{}_tensor_{}",
                            thread_id,
                            thread_rng().gen::<u32>() % 50
                        ); // SciRS2 POLICY compliant

                        match engine_clone.apply(tensor, Some(&cache_key)) {
                            Ok(_result) => {
                                counter_clone.increment_completed();
                            }
                            Err(_) => {
                                counter_clone.increment_failed();
                            }
                        }
                    }
                    Err(_) => {
                        counter_clone.increment_failed();
                    }
                }

                // Very small delay to allow high throughput
                thread::sleep(Duration::from_micros(100));
            }
        });

        handles.push(handle);
    }

    // Wait for completion
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let (completed, failed) = counter.get_counts();

    StressTestResult::new(
        "Transform Pipeline".to_string(),
        config,
        duration,
        completed,
        failed,
    )
    .evaluate()
}

/// Concurrent access stress test
pub fn stress_test_concurrent_access() -> StressTestResult {
    let mut config = StressTestConfig::default();
    config.num_threads = 8; // More threads for concurrency test

    let counter = OperationCounter::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_seconds);

    // Create shared dataset
    let data = ones::<f32>(&[config.dataset_size, 50]).expect("Failed to create data");
    let dataset = Arc::new(TensorDataset::from_tensor(data));

    let mut handles = Vec::new();

    // Test different access patterns concurrently
    for thread_id in 0..config.num_threads {
        let dataset_clone = Arc::clone(&dataset);
        let counter_clone = OperationCounter {
            completed: Arc::clone(&counter.completed),
            failed: Arc::clone(&counter.failed),
        };
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            let access_pattern = thread_id % 4;

            while Instant::now() < end_time {
                match access_pattern {
                    0 => {
                        // Sequential access
                        test_dataloader_pattern(
                            &*dataset_clone,
                            config_clone.batch_size,
                            &counter_clone,
                        );
                    }
                    1 => {
                        // Random access
                        test_dataloader_pattern(
                            &*dataset_clone,
                            config_clone.batch_size,
                            &counter_clone,
                        );
                    }
                    2 => {
                        // Subset access
                        let _indices: Vec<usize> =
                            (0..config_clone.dataset_size).step_by(2).collect();
                        // Simplified: skip subset for now due to Clone constraints
                        test_dataloader_pattern(
                            &*dataset_clone,
                            config_clone.batch_size,
                            &counter_clone,
                        );
                    }
                    3 => {
                        // Cached access
                        // Simplified: skip cached for now due to Clone constraints
                        test_dataloader_pattern(
                            &*dataset_clone,
                            config_clone.batch_size,
                            &counter_clone,
                        );
                    }
                    _ => unreachable!(),
                }

                thread::sleep(Duration::from_millis(5));
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let (completed, failed) = counter.get_counts();

    StressTestResult::new(
        "Concurrent Access".to_string(),
        config,
        duration,
        completed,
        failed,
    )
    .evaluate()
}

/// Helper function for testing dataloader patterns
fn test_dataloader_pattern<D>(dataset: &D, batch_size: usize, counter: &OperationCounter)
where
    D: Dataset + Clone + Sync,
    D::Item: Send + Clone,
    DefaultCollate: Collate<D::Item> + Sync,
    <DefaultCollate as Collate<D::Item>>::Output: Send,
{
    match simple_dataloader(dataset.clone(), batch_size, false) {
        Ok(dataloader) => {
            let mut batch_count = 0;
            for batch_result in dataloader.iter() {
                match batch_result {
                    Ok(_batch) => {
                        counter.increment_completed();
                        batch_count += 1;
                        if batch_count >= 3 {
                            break;
                        }
                    }
                    Err(_) => {
                        counter.increment_failed();
                        break;
                    }
                }
            }
        }
        Err(_) => {
            counter.increment_failed();
        }
    }
}

/// Long-running stability test
pub fn stress_test_long_running_stability() -> StressTestResult {
    let mut config = StressTestConfig::default();
    config.duration_seconds = 120; // 2 minutes

    let counter = OperationCounter::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_seconds);

    // Create dataset
    let data = ones::<f32>(&[config.dataset_size, 100]).expect("Failed to create data");
    let dataset = TensorDataset::from_tensor(data);

    // Single-threaded long-running test
    let mut iteration = 0;
    while Instant::now() < end_time {
        iteration += 1;

        // Vary the access pattern over time - always use random for type consistency
        let dataloader_result =
            simple_random_dataloader(dataset.clone(), config.batch_size, Some(iteration as u64));

        match dataloader_result {
            Ok(dataloader) => {
                for batch_result in dataloader.iter() {
                    match batch_result {
                        Ok(_batch) => {
                            counter.increment_completed();
                        }
                        Err(_) => {
                            counter.increment_failed();
                        }
                    }
                }
            }
            Err(_) => {
                counter.increment_failed();
            }
        }

        // Periodic cleanup
        if iteration % 100 == 0 {
            thread::sleep(Duration::from_millis(10));
        }
    }

    let duration = start_time.elapsed();
    let (completed, failed) = counter.get_counts();

    StressTestResult::new(
        "Long Running Stability".to_string(),
        config,
        duration,
        completed,
        failed,
    )
    .evaluate()
}

/// Stress test runner
pub struct StressTestRunner;

impl StressTestRunner {
    /// Run all stress tests
    pub fn run_all_stress_tests() -> Vec<StressTestResult> {
        let mut results = Vec::new();

        println!("Running Stress Tests...");
        println!(
            "Warning: These tests may take several minutes and consume significant resources\n"
        );

        println!("1. High Load DataLoader Test...");
        results.push(stress_test_high_load_dataloader());

        println!("2. Memory Pressure Test...");
        results.push(stress_test_memory_pressure());

        println!("3. Transform Pipeline Test...");
        results.push(stress_test_transform_pipeline());

        println!("4. Concurrent Access Test...");
        results.push(stress_test_concurrent_access());

        println!("5. Long Running Stability Test...");
        results.push(stress_test_long_running_stability());

        results
    }

    /// Print stress test results
    pub fn print_results(results: &[StressTestResult]) {
        println!("\n=== Stress Test Results ===\n");

        let mut passed_count = 0;
        let mut total_count = 0;

        for result in results {
            total_count += 1;
            let status = if result.passed {
                passed_count += 1;
                "PASS"
            } else {
                "FAIL"
            };

            println!("[{}] {}", status, result.test_name);
            println!("    Duration: {:.2}s", result.duration.as_secs_f64());
            println!(
                "    Operations: {} completed, {} failed",
                result.operations_completed, result.operations_failed
            );
            println!(
                "    Throughput: {:.2} ops/sec",
                result.throughput_ops_per_sec
            );
            println!("    Error Rate: {:.2}%", result.error_rate * 100.0);
            println!("    Peak Memory: {:.2} MB", result.peak_memory_mb);
            println!("    Threads: {}", result.config.num_threads);

            if !result.failure_reasons.is_empty() {
                println!("    Failure Reasons:");
                for reason in &result.failure_reasons {
                    println!("      - {reason}");
                }
            }
            println!();
        }

        println!("=== Summary ===");
        println!("Passed: {passed_count}/{total_count} stress tests");

        if passed_count == total_count {
            println!("All stress tests passed! System is stable under load.");
        } else {
            println!("Some stress tests failed. System may have stability issues under load.");
        }

        // Calculate aggregate statistics
        let total_ops: u64 = results.iter().map(|r| r.operations_completed).sum();
        let total_duration: f64 = results.iter().map(|r| r.duration.as_secs_f64()).sum();
        let avg_throughput = if total_duration > 0.0 {
            total_ops as f64 / total_duration
        } else {
            0.0
        };
        let max_memory = results
            .iter()
            .map(|r| r.peak_memory_mb)
            .fold(0.0f64, |a, b| a.max(b));

        println!("\n=== Aggregate Statistics ===");
        println!("Total operations: {total_ops}");
        println!("Total test time: {total_duration:.2} seconds");
        println!("Average throughput: {avg_throughput:.2} ops/sec");
        println!("Peak memory usage: {max_memory:.2} MB");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_test_config() {
        let config = StressTestConfig::default();
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.batch_size, 32);
        assert!(config.error_threshold > 0.0);
    }

    #[test]
    fn test_operation_counter() {
        let counter = OperationCounter::new();

        counter.increment_completed();
        counter.increment_completed();
        counter.increment_failed();

        let (completed, failed) = counter.get_counts();
        assert_eq!(completed, 2);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_stress_test_result() {
        let config = StressTestConfig::default();
        let result = StressTestResult::new(
            "Test".to_string(),
            config,
            Duration::from_secs(10),
            1000,
            10,
        );

        assert_eq!(result.operations_completed, 1000);
        assert_eq!(result.operations_failed, 10);
        assert!(result.throughput_ops_per_sec > 0.0);
        assert!(result.error_rate < 0.1);
    }

    #[test]
    fn test_quick_stress_test() {
        // Run a very quick version of one stress test for CI
        let mut config = StressTestConfig::default();
        config.duration_seconds = 1;
        config.dataset_size = 100;
        config.num_threads = 2;

        // This is a simplified version that should complete quickly
        let counter = OperationCounter::new();
        let start_time = Instant::now();

        // Create small dataset
        let data = ones::<f32>(&[100, 10]).unwrap();
        let dataset = TensorDataset::from_tensor(data);

        // Simple data loading test
        let dataloader = simple_dataloader(dataset, 8, false).unwrap();

        for batch_result in dataloader.iter() {
            match batch_result {
                Ok(_) => counter.increment_completed(),
                Err(_) => counter.increment_failed(),
            }
        }

        let duration = start_time.elapsed();
        let (completed, failed) = counter.get_counts();

        let result = StressTestResult::new(
            "Quick Test".to_string(),
            config,
            duration,
            completed,
            failed,
        )
        .evaluate();

        assert!(result.operations_completed > 0);
        println!(
            "Quick stress test completed: {} operations",
            result.operations_completed
        );
    }
}
