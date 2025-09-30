//! Benchmarking utilities for FX graph operations and transformations
//!
//! This module provides comprehensive benchmarking capabilities for measuring
//! performance of graph operations, transformations, and code generation.

use crate::{FxGraph, TorshResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark results for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub execution_time: Duration,
    pub memory_usage: Option<usize>,
    pub iterations: usize,
    pub success_rate: f64,
}

/// Comprehensive benchmark suite for graph operations
#[derive(Debug)]
pub struct GraphBenchmarkSuite {
    results: HashMap<String, Vec<BenchmarkResult>>,
    warmup_iterations: usize,
    benchmark_iterations: usize,
}

impl GraphBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            warmup_iterations: 10,
            benchmark_iterations: 100,
        }
    }

    /// Set the number of warmup iterations
    pub fn with_warmup_iterations(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set the number of benchmark iterations
    pub fn with_benchmark_iterations(mut self, iterations: usize) -> Self {
        self.benchmark_iterations = iterations;
        self
    }

    /// Benchmark graph creation operations
    pub fn benchmark_graph_creation(&mut self) -> TorshResult<()> {
        // Benchmark single operation graph creation
        let result = self.benchmark_operation("single_op_creation", || {
            let _graph = FxGraph::single_op("relu", vec!["input".to_string()]);
            Ok(())
        })?;

        // Benchmark sequential operations graph creation
        let result_seq = self.benchmark_operation("sequential_ops_creation", || {
            let _graph = FxGraph::sequential_ops(&["relu", "sigmoid", "tanh"]);
            Ok(())
        })?;

        // Benchmark large graph creation
        let result_large = self.benchmark_operation("large_graph_creation", || {
            let ops: Vec<&str> = (0..100)
                .map(|i| {
                    if i % 3 == 0 {
                        "relu"
                    } else if i % 3 == 1 {
                        "sigmoid"
                    } else {
                        "tanh"
                    }
                })
                .collect();
            let _graph = FxGraph::sequential_ops(&ops);
            Ok(())
        })?;

        self.results
            .entry("graph_creation".to_string())
            .or_insert_with(Vec::new)
            .extend([result, result_seq, result_large]);

        Ok(())
    }

    /// Benchmark graph serialization operations
    pub fn benchmark_serialization(&mut self) -> TorshResult<()> {
        let test_graph = FxGraph::sequential_ops(&["relu", "sigmoid", "tanh", "softmax"]);

        // Benchmark JSON serialization
        let json_serialize = self.benchmark_operation("json_serialize", || {
            let _json = test_graph.to_json()?;
            Ok(())
        })?;

        // Benchmark binary serialization
        let binary_serialize = self.benchmark_operation("binary_serialize", || {
            let _binary = test_graph.to_binary()?;
            Ok(())
        })?;

        // Benchmark JSON deserialization
        let json_data = test_graph.to_json()?;
        let json_deserialize = self.benchmark_operation("json_deserialize", || {
            let _graph = FxGraph::from_json(&json_data)?;
            Ok(())
        })?;

        // Benchmark binary deserialization
        let binary_data = test_graph.to_binary()?;
        let binary_deserialize = self.benchmark_operation("binary_deserialize", || {
            let _graph = FxGraph::from_binary(&binary_data)?;
            Ok(())
        })?;

        self.results
            .entry("serialization".to_string())
            .or_insert_with(Vec::new)
            .extend([
                json_serialize,
                binary_serialize,
                json_deserialize,
                binary_deserialize,
            ]);

        Ok(())
    }

    /// Benchmark graph analysis operations
    pub fn benchmark_analysis(&mut self) -> TorshResult<()> {
        let test_graph =
            FxGraph::sequential_ops(&["relu", "sigmoid", "tanh", "softmax", "dropout"]);

        // Benchmark validation
        let validation = self.benchmark_operation("graph_validation", || {
            let _result = test_graph.validate()?;
            Ok(())
        })?;

        // Benchmark node filtering
        let node_filtering = self.benchmark_operation("node_filtering", || {
            let _inputs = test_graph.input_nodes();
            let _outputs = test_graph.output_nodes();
            let _calls = test_graph.call_nodes();
            Ok(())
        })?;

        // Benchmark summary generation
        let summary = self.benchmark_operation("summary_generation", || {
            let _summary = test_graph.summary();
            Ok(())
        })?;

        self.results
            .entry("analysis".to_string())
            .or_insert_with(Vec::new)
            .extend([validation, node_filtering, summary]);

        Ok(())
    }

    /// Benchmark code generation operations
    pub fn benchmark_codegen(&mut self) -> TorshResult<()> {
        let test_graph = FxGraph::sequential_ops(&["relu", "sigmoid", "tanh"]);

        // Benchmark Python code generation
        let python_codegen = self.benchmark_operation("python_codegen", || {
            let _code = test_graph.to_python()?;
            Ok(())
        })?;

        // Benchmark C++ code generation
        let cpp_codegen = self.benchmark_operation("cpp_codegen", || {
            let _code = test_graph.to_cpp()?;
            Ok(())
        })?;

        self.results
            .entry("codegen".to_string())
            .or_insert_with(Vec::new)
            .extend([python_codegen, cpp_codegen]);

        Ok(())
    }

    /// Generic method to benchmark any operation
    pub fn benchmark_operation<F>(
        &self,
        name: &str,
        mut operation: F,
    ) -> TorshResult<BenchmarkResult>
    where
        F: FnMut() -> TorshResult<()>,
    {
        // Warmup phase
        for _ in 0..self.warmup_iterations {
            let _ = operation();
        }

        // Benchmark phase
        let mut total_time = Duration::ZERO;
        let mut successful_runs = 0;

        for _ in 0..self.benchmark_iterations {
            let start = Instant::now();
            match operation() {
                Ok(_) => {
                    total_time += start.elapsed();
                    successful_runs += 1;
                }
                Err(_) => {} // Count failures but continue
            }
        }

        let avg_time = if successful_runs > 0 {
            total_time / successful_runs as u32
        } else {
            Duration::ZERO
        };

        let success_rate = successful_runs as f64 / self.benchmark_iterations as f64;

        Ok(BenchmarkResult {
            operation_name: name.to_string(),
            execution_time: avg_time,
            memory_usage: None, // Could be extended to measure memory usage
            iterations: self.benchmark_iterations,
            success_rate,
        })
    }

    /// Run a comprehensive benchmark suite
    pub fn run_comprehensive_benchmark(&mut self) -> TorshResult<()> {
        println!("Running comprehensive FX graph benchmark suite...");

        self.benchmark_graph_creation()?;
        self.benchmark_serialization()?;
        self.benchmark_analysis()?;
        self.benchmark_codegen()?;

        Ok(())
    }

    /// Get benchmark results for a specific category
    pub fn get_results(&self, category: &str) -> Option<&Vec<BenchmarkResult>> {
        self.results.get(category)
    }

    /// Get all benchmark results
    pub fn get_all_results(&self) -> &HashMap<String, Vec<BenchmarkResult>> {
        &self.results
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("FX Graph Performance Benchmark Report\n");
        report.push_str("=====================================\n\n");

        for (category, results) in &self.results {
            report.push_str(&format!("Category: {category}\n"));
            report.push_str("----------------------------\n");

            for result in results {
                report.push_str(&format!(
                    "  Operation: {}\n    Time: {:?}\n    Iterations: {}\n    Success Rate: {:.2}%\n\n",
                    result.operation_name,
                    result.execution_time,
                    result.iterations,
                    result.success_rate * 100.0
                ));
            }
            report.push('\n');
        }

        report
    }

    /// Compare performance against baseline benchmarks
    pub fn compare_with_baseline(&self, baseline: &GraphBenchmarkSuite) -> String {
        let mut comparison = String::new();
        comparison.push_str("Performance Comparison with Baseline\n");
        comparison.push_str("===================================\n\n");

        for (category, results) in &self.results {
            if let Some(baseline_results) = baseline.get_results(category) {
                comparison.push_str(&format!("Category: {category}\n"));
                comparison.push_str("----------------------------\n");

                for (current, baseline_result) in results.iter().zip(baseline_results.iter()) {
                    if current.operation_name == baseline_result.operation_name {
                        let ratio = if baseline_result.execution_time.as_nanos() > 0 {
                            current.execution_time.as_nanos() as f64
                                / baseline_result.execution_time.as_nanos() as f64
                        } else {
                            1.0
                        };

                        let performance_change = if ratio < 1.0 {
                            let speedup = 1.0 / ratio;
                            format!("FASTER by {speedup:.2}x")
                        } else if ratio > 1.0 {
                            format!("SLOWER by {ratio:.2}x")
                        } else {
                            "SAME".to_string()
                        };

                        comparison.push_str(&format!(
                            "  {}: {} (Current: {:?}, Baseline: {:?})\n",
                            current.operation_name,
                            performance_change,
                            current.execution_time,
                            baseline_result.execution_time
                        ));
                    }
                }
                comparison.push('\n');
            }
        }

        comparison
    }
}

/// Performance regression testing utilities
pub struct RegressionTester {
    threshold: f64, // Allowed performance degradation (e.g., 1.1 = 10% slower is acceptable)
}

impl RegressionTester {
    /// Create a new regression tester with a specified threshold
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Test for performance regressions
    pub fn test_regression(
        &self,
        current: &GraphBenchmarkSuite,
        baseline: &GraphBenchmarkSuite,
    ) -> Vec<String> {
        let mut regressions = Vec::new();

        for (category, current_results) in current.get_all_results() {
            if let Some(baseline_results) = baseline.get_results(category) {
                for (current_result, baseline_result) in
                    current_results.iter().zip(baseline_results.iter())
                {
                    if current_result.operation_name == baseline_result.operation_name {
                        let ratio = if baseline_result.execution_time.as_nanos() > 0 {
                            current_result.execution_time.as_nanos() as f64
                                / baseline_result.execution_time.as_nanos() as f64
                        } else {
                            1.0
                        };

                        if ratio > self.threshold {
                            regressions.push(format!(
                                "REGRESSION in {}/{}: {:.2}x slower than baseline (threshold: {:.2}x)",
                                category,
                                current_result.operation_name,
                                ratio,
                                self.threshold
                            ));
                        }
                    }
                }
            }
        }

        regressions
    }
}

/// Simple benchmark macro for quick measurements
#[macro_export]
macro_rules! benchmark {
    ($name:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();
        println!("Benchmark '{}': {:?}", $name, duration);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = GraphBenchmarkSuite::new()
            .with_warmup_iterations(5)
            .with_benchmark_iterations(50);

        assert_eq!(suite.warmup_iterations, 5);
        assert_eq!(suite.benchmark_iterations, 50);
    }

    #[test]
    fn test_simple_benchmark() {
        let suite = GraphBenchmarkSuite::new()
            .with_warmup_iterations(1)
            .with_benchmark_iterations(5);

        let result = suite
            .benchmark_operation("test_op", || {
                // Simulate some work
                std::thread::sleep(std::time::Duration::from_millis(1));
                Ok(())
            })
            .unwrap();

        assert_eq!(result.operation_name, "test_op");
        assert_eq!(result.iterations, 5);
        assert_eq!(result.success_rate, 1.0);
        assert!(result.execution_time > Duration::ZERO);
    }

    #[test]
    fn test_graph_creation_benchmark() {
        let mut suite = GraphBenchmarkSuite::new()
            .with_warmup_iterations(1)
            .with_benchmark_iterations(10);

        suite.benchmark_graph_creation().unwrap();

        let results = suite.get_results("graph_creation").unwrap();
        assert_eq!(results.len(), 3); // single_op, sequential_ops, large_graph

        for result in results {
            assert_eq!(result.success_rate, 1.0);
            assert!(result.execution_time > Duration::ZERO);
        }
    }

    #[test]
    fn test_serialization_benchmark() {
        let mut suite = GraphBenchmarkSuite::new()
            .with_warmup_iterations(1)
            .with_benchmark_iterations(5);

        suite.benchmark_serialization().unwrap();

        let results = suite.get_results("serialization").unwrap();
        assert_eq!(results.len(), 4); // json_serialize, binary_serialize, json_deserialize, binary_deserialize
    }

    #[test]
    fn test_report_generation() {
        let mut suite = GraphBenchmarkSuite::new()
            .with_warmup_iterations(1)
            .with_benchmark_iterations(5);

        suite.benchmark_graph_creation().unwrap();

        let report = suite.generate_report();
        assert!(report.contains("FX Graph Performance Benchmark Report"));
        assert!(report.contains("graph_creation"));
        assert!(report.contains("single_op_creation"));
    }

    #[test]
    fn test_regression_tester() {
        let tester = RegressionTester::new(1.5); // 50% degradation threshold

        // Create mock benchmark suites
        let mut baseline = GraphBenchmarkSuite::new();
        baseline.results.insert(
            "test".to_string(),
            vec![BenchmarkResult {
                operation_name: "fast_op".to_string(),
                execution_time: Duration::from_millis(10),
                memory_usage: None,
                iterations: 100,
                success_rate: 1.0,
            }],
        );

        let mut current = GraphBenchmarkSuite::new();
        current.results.insert(
            "test".to_string(),
            vec![BenchmarkResult {
                operation_name: "fast_op".to_string(),
                execution_time: Duration::from_millis(20), // 2x slower
                memory_usage: None,
                iterations: 100,
                success_rate: 1.0,
            }],
        );

        let regressions = tester.test_regression(&current, &baseline);
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].contains("REGRESSION"));
        assert!(regressions[0].contains("2.00x slower"));
    }

    #[test]
    fn test_benchmark_macro() {
        let result = benchmark!("test_operation", {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);
    }
}
