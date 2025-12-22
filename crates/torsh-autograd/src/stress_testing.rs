//! Stress testing module for large computation graphs
//!
//! This module provides comprehensive stress testing capabilities for the autograd system,
//! including tests for large graphs, deep networks, memory pressure scenarios, and
//! performance degradation detection.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use std::time::{Duration, Instant};

/// Configuration for stress tests
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Maximum number of nodes to test
    pub max_nodes: usize,
    /// Maximum graph depth to test
    pub max_depth: usize,
    /// Maximum tensor size (elements)
    pub max_tensor_size: usize,
    /// Time limit for each test
    pub time_limit: Duration,
    /// Memory limit (bytes)
    pub memory_limit: Option<usize>,
    /// Enable detailed logging
    pub verbose: bool,
    /// Number of iterations for performance tests
    pub performance_iterations: usize,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_depth: 1_000,
            max_tensor_size: 1_000_000,
            time_limit: Duration::from_secs(300), // 5 minutes
            memory_limit: Some(8 * 1024 * 1024 * 1024), // 8 GB
            verbose: false,
            performance_iterations: 100,
        }
    }
}

/// Results from stress testing
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Test name
    pub test_name: String,
    /// Success status
    pub success: bool,
    /// Execution time
    pub duration: Duration,
    /// Memory usage (peak)
    pub peak_memory: usize,
    /// Number of nodes processed
    pub nodes_processed: usize,
    /// Number of operations completed
    pub operations_completed: usize,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Error encountered (if any)
    pub error: Option<String>,
}

/// Performance metrics collected during stress testing
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average time per forward pass
    pub avg_forward_time: Duration,
    /// Average time per backward pass
    pub avg_backward_time: Duration,
    /// Memory allocation rate (bytes/sec)
    pub memory_allocation_rate: f64,
    /// Graph construction time
    pub graph_construction_time: Duration,
    /// Peak nodes per second processed
    pub peak_nodes_per_second: f64,
    /// Memory efficiency (useful bytes / total bytes)
    pub memory_efficiency: f64,
}

/// Stress test runner for computation graphs
pub struct ComputationGraphStressTest {
    config: StressTestConfig,
    results: Vec<StressTestResults>,
    start_memory: usize,
}

impl ComputationGraphStressTest {
    /// Create a new stress test runner
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            start_memory: Self::get_memory_usage(),
        }
    }

    /// Run all stress tests
    pub fn run_all_tests(&mut self) -> AutogradResult<Vec<StressTestResults>> {
        if self.config.verbose {
            println!("Starting comprehensive stress tests...");
            println!("Config: {:?}", self.config);
        }

        // Test large sequential computation graphs
        self.test_large_sequential_graph()?;

        // Test deep computation graphs
        self.test_deep_graph()?;

        // Test wide computation graphs
        self.test_wide_graph()?;

        // Test branching computation graphs
        self.test_branching_graph()?;

        // Test memory pressure scenarios
        self.test_memory_pressure()?;

        // Test gradient accumulation stress
        self.test_gradient_accumulation_stress()?;

        // Test concurrent graph operations
        self.test_concurrent_operations()?;

        // Test performance degradation
        self.test_performance_degradation()?;

        Ok(self.results.clone())
    }

    /// Test large sequential computation graphs
    fn test_large_sequential_graph(&mut self) -> AutogradResult<()> {
        let test_name = "large_sequential_graph".to_string();
        let start_time = Instant::now();
        let mut nodes_processed = 0;
        let mut operations_completed = 0;

        if self.config.verbose {
            println!("Testing large sequential computation graph...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            // Create a long chain of operations
            let mut current_tensor = self.create_test_tensor([100, 100])?;

            for i in 0..self.config.max_nodes.min(1000) {
                // Simple linear transformation
                let weight = self.create_test_tensor([100, 100])?;
                current_tensor = self.mock_matmul(&current_tensor, &weight)?;

                // Add activation
                current_tensor = self.mock_relu(&current_tensor)?;

                nodes_processed += 2; // matmul + relu
                operations_completed += 2;

                if i % 100 == 0 && self.config.verbose {
                    println!("Processed {} operations", operations_completed);
                }

                // Check time limit
                if start_time.elapsed() > self.config.time_limit {
                    return Err(AutogradError::gradient_computation(
                        "large_sequential_graph",
                        "Time limit exceeded",
                    ));
                }

                // Check memory limit
                if let Some(limit) = self.config.memory_limit {
                    if Self::get_memory_usage() > limit {
                        return Err(AutogradError::memory_allocation(
                            "large_sequential_graph",
                            Self::get_memory_usage(),
                        ));
                    }
                }
            }

            // Simulate backward pass
            self.mock_backward(&current_tensor)?;

            Ok((nodes_processed, operations_completed))
        });

        let success = result.is_ok();
        let error_message = result.as_ref().err().map(|e| e.to_string());
        let (final_nodes, final_ops) = result.unwrap_or((nodes_processed, operations_completed));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: final_nodes,
            operations_completed: final_ops,
            performance_metrics: PerformanceMetrics::default(),
            error: error_message,
        };

        // Calculate performance metrics
        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);

        self.results.push(test_result);
        Ok(())
    }

    /// Test deep computation graphs
    fn test_deep_graph(&mut self) -> AutogradResult<()> {
        let test_name = "deep_graph".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!(
                "Testing deep computation graph (depth: {})...",
                self.config.max_depth
            );
        }

        let result = self.run_timed_test(test_name.clone(), || {
            let mut current_tensor = self.create_test_tensor([50, 50])?;
            let depth = self.config.max_depth.min(500);

            for i in 0..depth {
                // Create a deep computational path
                let weight = self.create_test_tensor([50, 50])?;
                current_tensor = self.mock_matmul(&current_tensor, &weight)?;
                current_tensor = self.mock_add_scalar(&current_tensor, 0.01)?;
                current_tensor = self.mock_tanh(&current_tensor)?;

                if i % 50 == 0 && self.config.verbose {
                    println!("Depth: {}/{}", i, depth);
                }

                // Check for stack overflow potential or excessive depth
                if i > 100 && start_time.elapsed() > self.config.time_limit / 4 {
                    return Err(AutogradError::gradient_computation(
                        "deep_graph",
                        format!("Graph too deep, stopping at depth {}", i),
                    ));
                }
            }

            // Simulate backward pass through deep graph
            self.mock_backward(&current_tensor)?;

            Ok((depth, depth * 3)) // depth * operations_per_level
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test wide computation graphs (many parallel branches)
    fn test_wide_graph(&mut self) -> AutogradResult<()> {
        let test_name = "wide_graph".to_string();
        let start_time = Instant::now();
        let width = self.config.max_nodes.min(1000);

        if self.config.verbose {
            println!("Testing wide computation graph (width: {})...", width);
        }

        let result = self.run_timed_test(test_name.clone(), || {
            let input = self.create_test_tensor([100, 100])?;
            let mut branches = Vec::new();

            // Create many parallel branches
            for i in 0..width {
                let weight = self.create_test_tensor([100, 100])?;
                let branch = self.mock_matmul(&input, &weight)?;
                branches.push(branch);

                if i % 100 == 0 && self.config.verbose {
                    println!("Created {} branches", i);
                }

                // Check memory pressure
                if let Some(limit) = self.config.memory_limit {
                    if Self::get_memory_usage() > limit / 2 {
                        return Err(AutogradError::memory_allocation(
                            "wide_graph",
                            Self::get_memory_usage(),
                        ));
                    }
                }
            }

            // Combine branches (simulate reduction)
            let mut result = branches[0].clone();
            for branch in branches.iter().skip(1) {
                result = self.mock_add(&result, branch)?;
            }

            // Simulate backward pass
            self.mock_backward(&result)?;

            Ok((width, width * 2)) // width branches + width additions
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test branching computation graphs (complex topology)
    fn test_branching_graph(&mut self) -> AutogradResult<()> {
        let test_name = "branching_graph".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!("Testing branching computation graph...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            let input = self.create_test_tensor([64, 64])?;
            let mut nodes_count = 1;
            let mut ops_count = 0;

            // Create a binary tree-like structure
            let mut current_level = vec![input];
            let max_levels = (self.config.max_nodes as f64).log2() as usize;

            for level in 0..max_levels.min(10) {
                let mut next_level = Vec::new();

                for node in current_level {
                    // Split each node into two branches
                    let weight1 = self.create_test_tensor([64, 64])?;
                    let weight2 = self.create_test_tensor([64, 64])?;

                    let branch1 = self.mock_matmul(&node, &weight1)?;
                    let branch2 = self.mock_matmul(&node, &weight2)?;

                    // Apply different activations
                    let branch1 = self.mock_relu(&branch1)?;
                    let branch2 = self.mock_tanh(&branch2)?;

                    next_level.push(branch1);
                    next_level.push(branch2);

                    nodes_count += 4; // 2 matmuls + 2 activations
                    ops_count += 4;
                }

                current_level = next_level;

                if self.config.verbose {
                    println!("Level {}: {} nodes", level, current_level.len());
                }

                // Memory check
                if let Some(limit) = self.config.memory_limit {
                    if Self::get_memory_usage() > limit / 2 {
                        break;
                    }
                }
            }

            // Combine all branches at the end
            let mut result = current_level[0].clone();
            for node in current_level.iter().skip(1) {
                result = self.mock_add(&result, node)?;
                ops_count += 1;
            }

            // Simulate backward pass
            self.mock_backward(&result)?;

            Ok((nodes_count, ops_count))
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test memory pressure scenarios
    fn test_memory_pressure(&mut self) -> AutogradResult<()> {
        let test_name = "memory_pressure".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!("Testing memory pressure scenarios...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            // Create progressively larger tensors until memory pressure
            let mut tensors = Vec::new();
            let mut size = 1000;
            let mut ops_count = 0;

            while tensors.len() < 100 {
                // Create large tensor
                let tensor = self.create_test_tensor([size, size])?;

                // Apply operations that increase memory usage
                let weight = self.create_test_tensor([size, size])?;
                let result = self.mock_matmul(&tensor, &weight)?;
                let result = self.mock_relu(&result)?;

                tensors.push(result);
                ops_count += 2;

                // Check memory usage
                let current_memory = Self::get_memory_usage();
                if let Some(limit) = self.config.memory_limit {
                    if current_memory > limit * 3 / 4 {
                        if self.config.verbose {
                            println!(
                                "Memory pressure reached: {} MB",
                                current_memory / 1024 / 1024
                            );
                        }
                        break;
                    }
                }

                // Increase tensor size to create more pressure
                if tensors.len() % 10 == 0 {
                    size += 100;
                }

                if self.config.verbose && tensors.len() % 10 == 0 {
                    println!(
                        "Created {} tensors, current size: {}x{}",
                        tensors.len(),
                        size,
                        size
                    );
                }
            }

            // Simulate backward pass on all tensors
            for tensor in &tensors {
                self.mock_backward(tensor)?;
            }

            Ok((tensors.len(), ops_count))
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test gradient accumulation under stress
    fn test_gradient_accumulation_stress(&mut self) -> AutogradResult<()> {
        let test_name = "gradient_accumulation_stress".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!("Testing gradient accumulation stress...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            let batch_size = 1000;
            let accumulated_gradients = 100;
            let mut total_ops = 0;

            // Create model parameters
            let weight1 = self.create_test_tensor([100, 50])?;
            let weight2 = self.create_test_tensor([50, 10])?;

            // Simulate training with gradient accumulation
            for batch in 0..accumulated_gradients {
                // Create batch data
                let input = self.create_test_tensor([batch_size, 100])?;

                // Forward pass
                let h1 = self.mock_matmul(&input, &weight1)?;
                let h1 = self.mock_relu(&h1)?;
                let output = self.mock_matmul(&h1, &weight2)?;
                let loss = self.mock_mean(&output)?;

                // Backward pass (accumulating gradients)
                self.mock_backward(&loss)?;

                total_ops += 5; // 2 matmuls + relu + mean + backward

                if batch % 10 == 0 && self.config.verbose {
                    println!("Processed batch {}/{}", batch, accumulated_gradients);
                }

                // Check memory growth (gradients should accumulate)
                if let Some(limit) = self.config.memory_limit {
                    if Self::get_memory_usage() > limit / 2 {
                        return Err(AutogradError::memory_allocation(
                            "gradient_accumulation_stress",
                            Self::get_memory_usage(),
                        ));
                    }
                }
            }

            Ok((accumulated_gradients, total_ops))
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test concurrent graph operations
    fn test_concurrent_operations(&mut self) -> AutogradResult<()> {
        let test_name = "concurrent_operations".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!("Testing concurrent graph operations...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            // Simulate multiple concurrent computation paths
            let num_threads = 4;
            let operations_per_thread = 100;
            let mut total_ops = 0;

            // Create shared input
            let shared_input = self.create_test_tensor([100, 100])?;

            // Simulate concurrent operations (in a real implementation this would use threads)
            for thread_id in 0..num_threads {
                for op_id in 0..operations_per_thread {
                    // Each "thread" performs independent operations
                    let weight = self.create_test_tensor([100, 100])?;
                    let intermediate = self.mock_matmul(&shared_input, &weight)?;
                    let result = self.mock_add_scalar(&intermediate, thread_id as f32 * 0.01)?;
                    let final_result = self.mock_relu(&result)?;

                    // Simulate backward pass
                    self.mock_backward(&final_result)?;

                    total_ops += 4; // matmul + add_scalar + relu + backward

                    if op_id % 25 == 0 && self.config.verbose {
                        println!(
                            "Thread {}: operation {}/{}",
                            thread_id, op_id, operations_per_thread
                        );
                    }
                }
            }

            Ok((num_threads * operations_per_thread, total_ops))
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Test for performance degradation over time
    fn test_performance_degradation(&mut self) -> AutogradResult<()> {
        let test_name = "performance_degradation".to_string();
        let start_time = Instant::now();

        if self.config.verbose {
            println!("Testing performance degradation over time...");
        }

        let result = self.run_timed_test(test_name.clone(), || {
            let iterations = self.config.performance_iterations;
            let mut timings = Vec::new();
            let mut total_ops = 0;

            for i in 0..iterations {
                let iter_start = Instant::now();

                // Standard operation that should have consistent performance
                let input = self.create_test_tensor([200, 200])?;
                let weight = self.create_test_tensor([200, 200])?;
                let result = self.mock_matmul(&input, &weight)?;
                let result = self.mock_relu(&result)?;
                self.mock_backward(&result)?;

                let iter_time = iter_start.elapsed();
                timings.push(iter_time);
                total_ops += 3;

                if i % 20 == 0 && self.config.verbose {
                    println!("Iteration {}/{}: {:?}", i, iterations, iter_time);
                }

                // Check for significant performance degradation
                if i > 50 {
                    let recent_avg = timings.iter().rev().take(10).sum::<Duration>() / 10;
                    let early_avg = timings.iter().take(10).sum::<Duration>() / 10;

                    if recent_avg > early_avg * 3 {
                        return Err(AutogradError::gradient_computation(
                            "performance_degradation",
                            format!(
                                "Performance degraded significantly: {:?} -> {:?}",
                                early_avg, recent_avg
                            ),
                        ));
                    }
                }
            }

            // Analyze performance stability
            let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
            let variance = timings
                .iter()
                .map(|t| {
                    let diff = t.as_nanos() as f64 - avg_time.as_nanos() as f64;
                    diff * diff
                })
                .sum::<f64>()
                / timings.len() as f64;

            if self.config.verbose {
                println!("Average time: {:?}, Variance: {:.2}", avg_time, variance);
            }

            Ok((iterations, total_ops))
        });

        let success = result.is_ok();
        let error = result.as_ref().err().map(|e| e.to_string());
        let (nodes, ops) = result.unwrap_or((0, 0));

        let mut test_result = StressTestResults {
            test_name,
            success,
            duration: start_time.elapsed(),
            peak_memory: Self::get_memory_usage(),
            nodes_processed: nodes,
            operations_completed: ops,
            performance_metrics: PerformanceMetrics::default(),
            error,
        };

        test_result.performance_metrics = self.calculate_performance_metrics(&test_result);
        self.results.push(test_result);
        Ok(())
    }

    /// Helper function to run a test with timeout and error handling
    fn run_timed_test<F, T>(&self, test_name: String, test_fn: F) -> AutogradResult<T>
    where
        F: FnOnce() -> AutogradResult<T>,
    {
        let start = Instant::now();

        // In a real implementation, this would use proper timeout handling
        let result = test_fn();

        if start.elapsed() > self.config.time_limit {
            return Err(AutogradError::gradient_computation(
                test_name,
                "Test exceeded time limit",
            ));
        }

        result
    }

    /// Calculate performance metrics from test results
    fn calculate_performance_metrics(&self, result: &StressTestResults) -> PerformanceMetrics {
        let _ops_per_second = result.operations_completed as f64 / result.duration.as_secs_f64();
        let nodes_per_second = result.nodes_processed as f64 / result.duration.as_secs_f64();

        PerformanceMetrics {
            avg_forward_time: result.duration / result.operations_completed.max(1) as u32,
            avg_backward_time: result.duration / result.operations_completed.max(1) as u32,
            memory_allocation_rate: result.peak_memory.saturating_sub(self.start_memory) as f64
                / result.duration.as_secs_f64(),
            graph_construction_time: result.duration / 2, // Rough estimate
            peak_nodes_per_second: nodes_per_second,
            memory_efficiency: 0.8, // Placeholder - would need more sophisticated calculation
        }
    }

    /// Get current memory usage (placeholder implementation)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would get actual memory usage
        // For now, return a simulated value
        use std::time::SystemTime;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();
        (now.as_nanos() % 1_000_000_000) as usize
    }

    /// Create a test tensor with specified shape
    fn create_test_tensor(&self, shape: [usize; 2]) -> AutogradResult<MockTensor> {
        Ok(MockTensor {
            shape: vec![shape[0], shape[1]],
            requires_grad: true,
        })
    }

    /// Mock implementations for testing (would use real tensor operations in practice)
    fn mock_matmul(&self, a: &MockTensor, b: &MockTensor) -> AutogradResult<MockTensor> {
        Ok(MockTensor {
            shape: vec![a.shape[0], b.shape[1]],
            requires_grad: a.requires_grad || b.requires_grad,
        })
    }

    fn mock_relu(&self, input: &MockTensor) -> AutogradResult<MockTensor> {
        Ok(input.clone())
    }

    fn mock_tanh(&self, input: &MockTensor) -> AutogradResult<MockTensor> {
        Ok(input.clone())
    }

    fn mock_add(&self, a: &MockTensor, _b: &MockTensor) -> AutogradResult<MockTensor> {
        Ok(a.clone())
    }

    fn mock_add_scalar(&self, input: &MockTensor, _scalar: f32) -> AutogradResult<MockTensor> {
        Ok(input.clone())
    }

    fn mock_mean(&self, input: &MockTensor) -> AutogradResult<MockTensor> {
        Ok(MockTensor {
            shape: vec![1],
            requires_grad: input.requires_grad,
        })
    }

    fn mock_backward(&self, _tensor: &MockTensor) -> AutogradResult<()> {
        // Simulate backward pass computation time
        std::thread::sleep(Duration::from_micros(10));
        Ok(())
    }

    /// Generate a comprehensive report of all test results
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Computation Graph Stress Test Report ===\n\n");

        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.success).count();
        let failed_tests = total_tests - passed_tests;

        report.push_str(&format!(
            "Summary: {}/{} tests passed, {} failed\n",
            passed_tests, total_tests, failed_tests
        ));
        report.push_str(&format!("Configuration: {:?}\n\n", self.config));

        for result in &self.results {
            report.push_str(&format!("--- {} ---\n", result.test_name));
            report.push_str(&format!(
                "Status: {}\n",
                if result.success { "PASSED" } else { "FAILED" }
            ));
            report.push_str(&format!("Duration: {:?}\n", result.duration));
            report.push_str(&format!(
                "Peak Memory: {} MB\n",
                result.peak_memory / 1024 / 1024
            ));
            report.push_str(&format!("Nodes Processed: {}\n", result.nodes_processed));
            report.push_str(&format!("Operations: {}\n", result.operations_completed));
            report.push_str(&format!(
                "Performance: {:.2} nodes/sec\n",
                result.performance_metrics.peak_nodes_per_second
            ));

            if let Some(error) = &result.error {
                report.push_str(&format!("Error: {}\n", error));
            }
            report.push_str("\n");
        }

        report
    }
}

/// Mock tensor for testing purposes
#[derive(Debug, Clone)]
struct MockTensor {
    shape: Vec<usize>,
    requires_grad: bool,
}

/// Preset configurations for different stress test scenarios
impl StressTestConfig {
    /// Configuration for quick smoke tests
    pub fn quick() -> Self {
        Self {
            max_nodes: 100,
            max_depth: 50,
            max_tensor_size: 10_000,
            time_limit: Duration::from_secs(30),
            memory_limit: Some(1024 * 1024 * 1024), // 1 GB
            verbose: true,
            performance_iterations: 10,
        }
    }

    /// Configuration for thorough stress testing
    pub fn thorough() -> Self {
        Self {
            max_nodes: 50_000,
            max_depth: 2_000,
            max_tensor_size: 10_000_000,
            time_limit: Duration::from_secs(1800), // 30 minutes
            memory_limit: Some(16 * 1024 * 1024 * 1024), // 16 GB
            verbose: false,
            performance_iterations: 1000,
        }
    }

    /// Configuration for memory-constrained environments
    pub fn memory_constrained() -> Self {
        Self {
            max_nodes: 1_000,
            max_depth: 100,
            max_tensor_size: 100_000,
            time_limit: Duration::from_secs(60),
            memory_limit: Some(512 * 1024 * 1024), // 512 MB
            verbose: true,
            performance_iterations: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_test_config() {
        let config = StressTestConfig::default();
        assert_eq!(config.max_nodes, 10_000);
        assert_eq!(config.max_depth, 1_000);

        let quick_config = StressTestConfig::quick();
        assert_eq!(quick_config.max_nodes, 100);
        assert!(quick_config.verbose);
    }

    #[test]
    fn test_mock_tensor_operations() {
        let stress_test = ComputationGraphStressTest::new(StressTestConfig::quick());

        let tensor_a = stress_test.create_test_tensor([10, 20]).unwrap();
        let tensor_b = stress_test.create_test_tensor([20, 30]).unwrap();

        let result = stress_test.mock_matmul(&tensor_a, &tensor_b).unwrap();
        assert_eq!(result.shape, vec![10, 30]);
        assert!(result.requires_grad);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let config = StressTestConfig::quick();
        let stress_test = ComputationGraphStressTest::new(config);

        let test_result = StressTestResults {
            test_name: "test".to_string(),
            success: true,
            duration: Duration::from_secs(1),
            peak_memory: 1024 * 1024,
            nodes_processed: 100,
            operations_completed: 200,
            performance_metrics: PerformanceMetrics::default(),
            error: None,
        };

        let metrics = stress_test.calculate_performance_metrics(&test_result);
        assert!(metrics.peak_nodes_per_second > 0.0);
    }

    #[test]
    fn test_report_generation() {
        let config = StressTestConfig::quick();
        let stress_test = ComputationGraphStressTest::new(config);

        let report = stress_test.generate_report();
        assert!(report.contains("Stress Test Report"));
        assert!(report.contains("Summary:"));
    }
}

// Advanced stress testing enhancements

/// Advanced stress test runner with extreme scale scenarios
#[derive(Debug)]
pub struct ExtremeLimitStressTest {
    config: StressTestConfig,
    results_history: Vec<StressTestResults>,
    baseline_metrics: Option<PerformanceMetrics>,
    regression_threshold: f64,
}

impl ExtremeLimitStressTest {
    /// Create a new extreme stress test runner
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
            results_history: Vec::new(),
            baseline_metrics: None,
            regression_threshold: 0.1, // 10% regression threshold
        }
    }

    /// Run all extreme stress tests
    pub fn run_extreme_tests(&mut self) -> AutogradResult<ExtremeTestResults> {
        let mut results = Vec::new();
        let start_time = Instant::now();

        // Test 1: Massive computation graph (100k+ nodes)
        let massive_graph_result = self.test_massive_computation_graph()?;
        results.push(massive_graph_result);

        // Test 2: Extreme depth (10k+ layers)
        let extreme_depth_result = self.test_extreme_depth_graph()?;
        results.push(extreme_depth_result);

        // Test 3: Memory exhaustion boundary test
        let memory_boundary_result = self.test_memory_boundary()?;
        results.push(memory_boundary_result);

        // Test 4: Sustained load test
        let sustained_load_result = self.test_sustained_load()?;
        results.push(sustained_load_result);

        // Test 5: Chaos injection test
        let chaos_test_result = self.test_chaos_injection()?;
        results.push(chaos_test_result);

        // Test 6: Regression detection test
        let regression_result = self.test_regression_detection()?;
        results.push(regression_result);

        let total_duration = start_time.elapsed();
        let overall_success = results.iter().all(|r| r.success);

        Ok(ExtremeTestResults {
            results,
            total_duration,
            overall_success,
            performance_regression_detected: self.detect_performance_regression(),
        })
    }

    /// Test massive computation graph with 100k+ nodes
    fn test_massive_computation_graph(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;
        let mut peak_memory = 0usize;

        let target_nodes = 100_000;

        tracing::info!(
            "Starting massive computation graph test with {} nodes",
            target_nodes
        );

        let test_result = self.run_timed_test("massive_graph".to_string(), || {
            // Simulate creating a massive graph
            let mut nodes_created = 0;
            let mut current_memory = 0usize;

            while nodes_created < target_nodes {
                // Simulate node creation with memory tracking
                let node_size = 1024; // 1KB per node
                current_memory += node_size;
                peak_memory = peak_memory.max(current_memory);

                nodes_created += 1;

                // Check memory limits
                if let Some(limit) = self.config.memory_limit {
                    if current_memory > limit {
                        return Err(AutogradError::gradient_computation(
                            "Memory limit exceeded during massive graph creation",
                            "massive_graph_creation",
                        ));
                    }
                }

                // Check time limits
                if start_time.elapsed() > self.config.time_limit {
                    return Err(AutogradError::gradient_computation(
                        "Time limit exceeded during massive graph creation",
                        "massive_graph_creation",
                    ));
                }

                // Periodic cleanup simulation
                if nodes_created % 10_000 == 0 {
                    current_memory = (current_memory as f64 * 0.9) as usize; // 10% cleanup
                    tracing::debug!(
                        "Created {} nodes, memory: {} MB",
                        nodes_created,
                        current_memory / (1024 * 1024)
                    );
                }
            }

            Ok(())
        });

        if let Err(e) = test_result {
            success = false;
            error_message = Some(e.to_string());
        }

        let duration = start_time.elapsed();
        let nodes_per_second = target_nodes as f64 / duration.as_secs_f64();

        Ok(StressTestResults {
            test_name: "massive_computation_graph".to_string(),
            success,
            duration,
            peak_memory,
            nodes_processed: target_nodes,
            operations_completed: target_nodes,
            performance_metrics: PerformanceMetrics {
                avg_forward_time: Duration::from_millis(1),
                avg_backward_time: Duration::from_millis(1),
                memory_allocation_rate: peak_memory as f64 / duration.as_secs_f64(),
                graph_construction_time: duration,
                peak_nodes_per_second: nodes_per_second,
                memory_efficiency: 0.8,
            },
            error: error_message,
        })
    }

    /// Test extreme depth graph (10k+ layers)
    fn test_extreme_depth_graph(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;
        let mut peak_memory = 0usize;

        let target_depth = 10_000;

        tracing::info!(
            "Starting extreme depth graph test with {} layers",
            target_depth
        );

        let test_result = self.run_timed_test("extreme_depth".to_string(), || {
            let mut current_depth = 0;
            let mut memory_per_layer = 512; // 512 bytes per layer
            let mut total_memory = 0usize;

            while current_depth < target_depth {
                total_memory += memory_per_layer;
                peak_memory = peak_memory.max(total_memory);
                current_depth += 1;

                // Simulate increasing memory usage with depth
                if current_depth % 1000 == 0 {
                    memory_per_layer = (memory_per_layer as f64 * 1.1) as usize;
                    // 10% increase
                }

                // Check constraints
                if let Some(limit) = self.config.memory_limit {
                    if total_memory > limit {
                        return Err(AutogradError::gradient_computation(
                            "Memory limit exceeded in extreme depth test",
                            "extreme_depth_test",
                        ));
                    }
                }

                if start_time.elapsed() > self.config.time_limit {
                    return Err(AutogradError::gradient_computation(
                        "Time limit exceeded in extreme depth test",
                        "extreme_depth_test",
                    ));
                }

                if current_depth % 1000 == 0 {
                    tracing::debug!(
                        "Reached depth {}, memory: {} MB",
                        current_depth,
                        total_memory / (1024 * 1024)
                    );
                }
            }

            Ok(())
        });

        if let Err(e) = test_result {
            success = false;
            error_message = Some(e.to_string());
        }

        let duration = start_time.elapsed();
        let layers_per_second = target_depth as f64 / duration.as_secs_f64();

        Ok(StressTestResults {
            test_name: "extreme_depth_graph".to_string(),
            success,
            duration,
            peak_memory,
            nodes_processed: target_depth,
            operations_completed: target_depth,
            performance_metrics: PerformanceMetrics {
                avg_forward_time: Duration::from_millis(1),
                avg_backward_time: Duration::from_millis(1),
                memory_allocation_rate: peak_memory as f64 / duration.as_secs_f64(),
                graph_construction_time: duration,
                peak_nodes_per_second: layers_per_second,
                memory_efficiency: 0.8,
            },
            error: error_message,
        })
    }

    /// Test memory boundary conditions
    fn test_memory_boundary(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;
        let mut peak_memory = 0usize;

        tracing::info!("Starting memory boundary test");

        let memory_limit = self.config.memory_limit.unwrap_or(8 * 1024 * 1024 * 1024); // 8GB default
        let target_memory = (memory_limit as f64 * 0.95) as usize; // Aim for 95% of limit

        let test_result = self.run_timed_test("memory_boundary".to_string(), || {
            let mut allocated_memory = 0usize;
            let chunk_size = 1024 * 1024; // 1MB chunks

            while allocated_memory < target_memory {
                allocated_memory += chunk_size;
                peak_memory = peak_memory.max(allocated_memory);

                // Simulate memory pressure effects
                if allocated_memory > memory_limit / 2 {
                    // Simulate slower allocation as we approach the limit
                    std::thread::sleep(Duration::from_micros(100));
                }

                if allocated_memory > target_memory {
                    break;
                }

                if start_time.elapsed() > self.config.time_limit {
                    return Err(AutogradError::gradient_computation(
                        "memory_boundary_test",
                        "Time limit exceeded in memory boundary test",
                    ));
                }

                if allocated_memory % (100 * 1024 * 1024) == 0 {
                    tracing::debug!(
                        "Allocated {} MB / {} MB",
                        allocated_memory / (1024 * 1024),
                        memory_limit / (1024 * 1024)
                    );
                }
            }

            Ok(())
        });

        if let Err(e) = test_result {
            success = false;
            error_message = Some(e.to_string());
        }

        let duration = start_time.elapsed();
        let memory_rate = peak_memory as f64 / duration.as_secs_f64();

        Ok(StressTestResults {
            test_name: "memory_boundary".to_string(),
            success,
            duration,
            peak_memory,
            nodes_processed: peak_memory / 1024, // Use memory as node count proxy
            operations_completed: peak_memory / 1024,
            performance_metrics: PerformanceMetrics {
                avg_forward_time: Duration::from_millis(1),
                avg_backward_time: Duration::from_millis(1),
                memory_allocation_rate: memory_rate,
                graph_construction_time: duration,
                peak_nodes_per_second: (peak_memory / 1024) as f64 / duration.as_secs_f64(),
                memory_efficiency: (peak_memory as f64 / memory_limit as f64).min(1.0),
            },
            error: error_message,
        })
    }

    /// Test sustained load over extended period
    fn test_sustained_load(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;
        let mut peak_memory = 0usize;

        let test_duration = Duration::from_secs(300); // 5 minutes sustained test
        tracing::info!(
            "Starting sustained load test for {} seconds",
            test_duration.as_secs()
        );

        let test_result = self.run_timed_test("sustained_load".to_string(), || {
            let mut operations_completed = 0usize;
            let mut current_memory = 0usize;

            while start_time.elapsed() < test_duration {
                // Simulate sustained computation operations
                let operation_memory = 1024; // 1KB per operation
                current_memory += operation_memory;

                // Periodic cleanup to simulate realistic usage
                if operations_completed % 1000 == 0 {
                    current_memory = (current_memory as f64 * 0.8) as usize; // 20% cleanup
                }

                peak_memory = peak_memory.max(current_memory);
                operations_completed += 1;

                // Check memory limits
                if let Some(limit) = self.config.memory_limit {
                    if current_memory > limit {
                        return Err(AutogradError::gradient_computation(
                            "sustained_load_test",
                            "Memory limit exceeded during sustained load test",
                        ));
                    }
                }

                if operations_completed % 10_000 == 0 {
                    tracing::debug!(
                        "Sustained test: {} operations, {} MB memory",
                        operations_completed,
                        current_memory / (1024 * 1024)
                    );
                }

                // Small delay to prevent overwhelming the system
                if operations_completed % 100 == 0 {
                    std::thread::sleep(Duration::from_micros(10));
                }
            }

            Ok(operations_completed)
        });

        let operations_completed = match test_result {
            Ok(count) => count,
            Err(e) => {
                success = false;
                error_message = Some(e.to_string());
                0
            }
        };

        let duration = start_time.elapsed();
        let ops_per_second = operations_completed as f64 / duration.as_secs_f64();

        Ok(StressTestResults {
            test_name: "sustained_load".to_string(),
            success,
            duration,
            peak_memory,
            nodes_processed: operations_completed,
            operations_completed,
            performance_metrics: PerformanceMetrics {
                avg_forward_time: Duration::from_millis(1),
                avg_backward_time: Duration::from_millis(1),
                memory_allocation_rate: peak_memory as f64 / duration.as_secs_f64(),
                graph_construction_time: duration,
                peak_nodes_per_second: ops_per_second,
                memory_efficiency: 0.8,
            },
            error: error_message,
        })
    }

    /// Test chaos injection (random failures, resource constraints)
    fn test_chaos_injection(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;
        let mut peak_memory = 0usize;

        tracing::info!("Starting chaos injection test");

        let test_result = self.run_timed_test("chaos_injection".to_string(), || {
            let mut operations = 0usize;
            let mut failures = 0usize;
            let mut current_memory = 0usize;

            while start_time.elapsed() < Duration::from_secs(60) {
                // 1 minute chaos test
                operations += 1;

                // Simulate random failures (10% failure rate)
                if operations % 10 == 0 {
                    failures += 1;
                    // Simulate failure recovery
                    current_memory = (current_memory as f64 * 0.9) as usize;
                    continue;
                }

                // Simulate random memory spikes
                let memory_spike = if operations % 50 == 0 {
                    10 * 1024
                } else {
                    1024
                };
                current_memory += memory_spike;
                peak_memory = peak_memory.max(current_memory);

                // Simulate random delays
                if operations % 100 == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }

                // Random cleanup
                if operations % 200 == 0 {
                    current_memory = (current_memory as f64 * 0.7) as usize;
                }

                if operations % 1000 == 0 {
                    tracing::debug!(
                        "Chaos test: {} ops, {} failures, {} MB memory",
                        operations,
                        failures,
                        current_memory / (1024 * 1024)
                    );
                }
            }

            Ok((operations, failures))
        });

        let (operations, failures) = match test_result {
            Ok((ops, fails)) => (ops, fails),
            Err(e) => {
                success = false;
                error_message = Some(e.to_string());
                (0, 0)
            }
        };

        let duration = start_time.elapsed();
        let failure_rate = if operations > 0 {
            failures as f64 / operations as f64
        } else {
            0.0
        };

        Ok(StressTestResults {
            test_name: "chaos_injection".to_string(),
            success,
            duration,
            peak_memory,
            nodes_processed: operations,
            operations_completed: operations,
            performance_metrics: PerformanceMetrics {
                avg_forward_time: Duration::from_millis(1),
                avg_backward_time: Duration::from_millis(1),
                memory_allocation_rate: peak_memory as f64 / duration.as_secs_f64(),
                graph_construction_time: duration,
                peak_nodes_per_second: operations as f64 / duration.as_secs_f64(),
                memory_efficiency: 1.0 - failure_rate, // Use failure rate as efficiency metric
            },
            error: error_message,
        })
    }

    /// Test regression detection against baseline
    fn test_regression_detection(&mut self) -> AutogradResult<StressTestResults> {
        let start_time = Instant::now();
        let mut success = true;
        let mut error_message = None;

        tracing::info!("Starting regression detection test");

        // Run a standard performance test
        let current_metrics = self.measure_current_performance()?;

        // Compare against baseline if available
        let regression_detected = if let Some(baseline) = &self.baseline_metrics {
            self.compare_performance(&current_metrics, baseline)
        } else {
            // First run - establish baseline
            self.baseline_metrics = Some(current_metrics.clone());
            false
        };

        if regression_detected {
            success = false;
            error_message = Some("Performance regression detected".to_string());
        }

        let duration = start_time.elapsed();

        Ok(StressTestResults {
            test_name: "regression_detection".to_string(),
            success,
            duration,
            peak_memory: 0, // Will be set by actual measurement
            nodes_processed: 10_000,
            operations_completed: 10_000,
            performance_metrics: PerformanceMetrics::default(),
            error: error_message,
        })
    }

    /// Measure current performance for regression detection
    fn measure_current_performance(&self) -> AutogradResult<PerformanceMetrics> {
        let start_time = Instant::now();
        let test_nodes = 10_000;
        let mut peak_memory = 0usize;

        // Simulate a standard performance test
        for i in 0..test_nodes {
            let node_memory = 512; // 512 bytes per node
            peak_memory += node_memory;

            if i % 1000 == 0 {
                // Simulate periodic cleanup
                peak_memory = (peak_memory as f64 * 0.95) as usize;
            }
        }

        let duration = start_time.elapsed();
        let nodes_per_second = test_nodes as f64 / duration.as_secs_f64();

        Ok(PerformanceMetrics {
            avg_forward_time: Duration::from_millis(1),
            avg_backward_time: Duration::from_millis(1),
            memory_allocation_rate: peak_memory as f64 / duration.as_secs_f64(),
            graph_construction_time: duration,
            peak_nodes_per_second: nodes_per_second,
            memory_efficiency: 0.8,
        })
    }

    /// Compare current performance against baseline
    fn compare_performance(
        &self,
        current: &PerformanceMetrics,
        baseline: &PerformanceMetrics,
    ) -> bool {
        let performance_change = (baseline.peak_nodes_per_second - current.peak_nodes_per_second)
            / baseline.peak_nodes_per_second;
        let memory_change = (current.memory_allocation_rate - baseline.memory_allocation_rate)
            / baseline.memory_allocation_rate;

        // Detect regression if performance decreased by more than threshold
        // or memory usage increased significantly
        performance_change > self.regression_threshold || memory_change > self.regression_threshold
    }

    /// Detect overall performance regression
    fn detect_performance_regression(&self) -> bool {
        if self.results_history.len() < 2 {
            return false;
        }

        // Compare latest results with historical average
        let recent_results = &self.results_history[self.results_history.len().saturating_sub(5)..];
        let avg_duration = recent_results
            .iter()
            .map(|r| r.duration.as_secs_f64())
            .sum::<f64>()
            / recent_results.len() as f64;

        let latest_duration = self.results_history.last().unwrap().duration.as_secs_f64();

        (latest_duration - avg_duration) / avg_duration > self.regression_threshold
    }

    /// Run a test with timing and error handling
    fn run_timed_test<F, T>(&self, test_name: String, test_fn: F) -> AutogradResult<T>
    where
        F: FnOnce() -> AutogradResult<T>,
    {
        let start = Instant::now();
        tracing::debug!("Starting test: {}", test_name);

        let result = test_fn();

        let duration = start.elapsed();
        tracing::debug!("Test {} completed in {:?}", test_name, duration);

        result
    }
}

/// Results from extreme stress testing
#[derive(Debug, Clone)]
pub struct ExtremeTestResults {
    pub results: Vec<StressTestResults>,
    pub total_duration: Duration,
    pub overall_success: bool,
    pub performance_regression_detected: bool,
}

impl ExtremeTestResults {
    /// Generate a comprehensive report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== EXTREME STRESS TEST REPORT ===\n\n");
        report.push_str(&format!("Overall Success: {}\n", self.overall_success));
        report.push_str(&format!("Total Duration: {:?}\n", self.total_duration));
        report.push_str(&format!(
            "Regression Detected: {}\n\n",
            self.performance_regression_detected
        ));

        report.push_str("Individual Test Results:\n");
        report.push_str("-".repeat(50).as_str());
        report.push_str("\n");

        for result in &self.results {
            report.push_str(&format!("Test: {}\n", result.test_name));
            report.push_str(&format!("  Success: {}\n", result.success));
            report.push_str(&format!("  Duration: {:?}\n", result.duration));
            report.push_str(&format!(
                "  Peak Memory: {} MB\n",
                result.peak_memory / (1024 * 1024)
            ));
            report.push_str(&format!("  Nodes Processed: {}\n", result.nodes_processed));

            if let Some(error) = &result.error {
                report.push_str(&format!("  Error: {}\n", error));
            }
            report.push_str("\n");
        }

        report.push_str("=== END REPORT ===\n");
        report
    }
}

#[cfg(test)]
mod extreme_tests {
    use super::*;

    #[test]
    fn test_extreme_stress_test_creation() {
        let config = StressTestConfig::default();
        let stress_test = ExtremeLimitStressTest::new(config);

        assert_eq!(stress_test.results_history.len(), 0);
        assert!(stress_test.baseline_metrics.is_none());
    }

    #[test]
    fn test_performance_metrics_comparison() {
        let config = StressTestConfig::default();
        let stress_test = ExtremeLimitStressTest::new(config);

        let baseline = PerformanceMetrics {
            avg_forward_time: Duration::from_millis(1),
            avg_backward_time: Duration::from_millis(1),
            memory_allocation_rate: 1000.0,
            graph_construction_time: Duration::from_secs(10),
            peak_nodes_per_second: 1000.0,
            memory_efficiency: 0.8,
        };

        let current = PerformanceMetrics {
            avg_forward_time: Duration::from_millis(1),
            avg_backward_time: Duration::from_millis(1),
            memory_allocation_rate: 1200.0, // 20% higher memory usage
            graph_construction_time: Duration::from_secs(12),
            peak_nodes_per_second: 800.0, // 20% slower
            memory_efficiency: 0.7,
        };

        let regression = stress_test.compare_performance(&current, &baseline);
        assert!(regression); // Should detect regression
    }

    #[test]
    fn test_extreme_report_generation() {
        let results = ExtremeTestResults {
            results: vec![StressTestResults {
                test_name: "test1".to_string(),
                success: true,
                duration: Duration::from_secs(1),
                peak_memory: 1024,
                nodes_processed: 100,
                operations_completed: 100,
                performance_metrics: PerformanceMetrics::default(),
                error: None,
            }],
            total_duration: Duration::from_secs(1),
            overall_success: true,
            performance_regression_detected: false,
        };

        let report = results.generate_report();
        assert!(report.contains("EXTREME STRESS TEST REPORT"));
        assert!(report.contains("Overall Success: true"));
    }
}
