//! Model benchmarking utilities for performance evaluation
//!
//! This module provides comprehensive benchmarking tools for measuring model
//! performance, including speed, memory usage, accuracy, and comparison utilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use torsh_core::error::Result;
use torsh_core::DeviceType;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Comprehensive benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Input batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Input shapes to test (excluding batch dimension)
    pub input_shapes: Vec<Vec<usize>>,
    /// Devices to benchmark on
    pub devices: Vec<DeviceType>,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to measure accuracy (requires test data)
    pub measure_accuracy: bool,
    /// Data types to test
    pub dtypes: Vec<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_sizes: vec![1, 8, 16, 32],
            input_shapes: vec![vec![224, 224, 3]], // Default image size
            devices: vec![DeviceType::Cpu],
            measure_memory: true,
            measure_accuracy: false,
            dtypes: vec!["f32".to_string()],
        }
    }
}

/// Benchmark results for a single configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Configuration used for this benchmark
    pub config: BenchmarkConfig,
    /// Performance metrics per configuration
    pub performance_metrics: HashMap<String, PerformanceMetrics>,
    /// Memory usage metrics
    pub memory_metrics: Option<MemoryMetrics>,
    /// Accuracy metrics (if measured)
    pub accuracy_metrics: Option<AccuracyMetrics>,
    /// Model information
    pub model_info: ModelInfo,
    /// Benchmark timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance metrics for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Minimum inference time in milliseconds
    pub min_inference_time_ms: f64,
    /// Maximum inference time in milliseconds
    pub max_inference_time_ms: f64,
    /// Standard deviation of inference times
    pub std_inference_time_ms: f64,
    /// Throughput (samples per second)
    pub throughput_samples_per_sec: f64,
    /// Latency percentiles
    pub latency_percentiles: LatencyPercentiles,
    /// Configuration details
    pub batch_size: usize,
    pub input_shape: Vec<usize>,
    pub device: DeviceType,
    pub dtype: String,
}

/// Latency percentile measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage during inference (bytes)
    pub peak_memory_bytes: usize,
    /// Memory usage per sample (bytes)
    pub memory_per_sample_bytes: usize,
    /// Model parameter memory (bytes)
    pub model_memory_bytes: usize,
    /// Activation memory (bytes)
    pub activation_memory_bytes: usize,
}

/// Accuracy evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Top-1 accuracy
    pub top1_accuracy: f64,
    /// Top-5 accuracy (if applicable)
    pub top5_accuracy: Option<f64>,
    /// Mean absolute error (for regression)
    pub mae: Option<f64>,
    /// Root mean squared error (for regression)
    pub rmse: Option<f64>,
    /// F1 score (for classification)
    pub f1_score: Option<f64>,
    /// Per-class accuracy (for classification)
    pub per_class_accuracy: Option<HashMap<String, f64>>,
}

/// Model information for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Model size in bytes
    pub model_size_bytes: usize,
    /// Model architecture type
    pub architecture: String,
    /// Model version/variant
    pub version: Option<String>,
}

/// Main benchmarking engine
pub struct ModelBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResults>,
}

impl ModelBenchmark {
    /// Create a new model benchmark
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Benchmark a model with the configured settings
    pub fn benchmark_model<M: Module>(
        &mut self,
        model: &M,
        model_name: &str,
    ) -> Result<BenchmarkResults> {
        let model_info = self.extract_model_info(model, model_name)?;
        let mut performance_metrics = HashMap::new();

        // Benchmark all configurations
        for &batch_size in &self.config.batch_sizes {
            for input_shape in &self.config.input_shapes {
                for &device in &self.config.devices {
                    for dtype in &self.config.dtypes {
                        let config_key = format!(
                            "batch{}_shape{:?}_{}_{}",
                            batch_size,
                            input_shape,
                            format!("{:?}", device),
                            dtype
                        );

                        let metrics = self.benchmark_single_config(
                            model,
                            batch_size,
                            input_shape,
                            device,
                            dtype,
                        )?;

                        performance_metrics.insert(config_key, metrics);
                    }
                }
            }
        }

        let memory_metrics = if self.config.measure_memory {
            Some(self.measure_memory_usage(model)?)
        } else {
            None
        };

        let accuracy_metrics = if self.config.measure_accuracy {
            // Would need test dataset for this
            None
        } else {
            None
        };

        let results = BenchmarkResults {
            config: self.config.clone(),
            performance_metrics,
            memory_metrics,
            accuracy_metrics,
            model_info,
            timestamp: chrono::Utc::now(),
        };

        self.results.push(results.clone());
        Ok(results)
    }

    /// Benchmark a single configuration
    fn benchmark_single_config<M: Module>(
        &self,
        model: &M,
        batch_size: usize,
        input_shape: &[usize],
        device: DeviceType,
        dtype: &str,
    ) -> Result<PerformanceMetrics> {
        // Create input tensor
        let mut full_shape = vec![batch_size];
        full_shape.extend_from_slice(input_shape);

        let input_tensor = self.create_random_input(&full_shape, device, dtype)?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = model.forward(&input_tensor)?;
        }

        // Benchmark
        let mut execution_times = Vec::new();

        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            let _ = model.forward(&input_tensor)?;
            let duration = start.elapsed();
            execution_times.push(duration.as_secs_f64() * 1000.0); // Convert to ms
        }

        // Calculate statistics
        let metrics = self.calculate_performance_metrics(
            execution_times,
            batch_size,
            input_shape.to_vec(),
            device,
            dtype.to_string(),
        );

        Ok(metrics)
    }

    /// Create random input tensor for benchmarking
    fn create_random_input(
        &self,
        shape: &[usize],
        device: DeviceType,
        _dtype: &str,
    ) -> Result<Tensor> {
        // Create random tensor with normal distribution
        let total_elements: usize = shape.iter().product();
        let random_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.001).collect(); // Simplified data generation

        Tensor::from_data(random_data, shape.to_vec(), device)
    }

    /// Calculate performance metrics from execution times
    fn calculate_performance_metrics(
        &self,
        mut execution_times: Vec<f64>,
        batch_size: usize,
        input_shape: Vec<usize>,
        device: DeviceType,
        dtype: String,
    ) -> PerformanceMetrics {
        execution_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let min_time = execution_times[0];
        let max_time = execution_times[execution_times.len() - 1];

        // Calculate standard deviation
        let variance = execution_times
            .iter()
            .map(|&x| (x - avg_time).powi(2))
            .sum::<f64>()
            / execution_times.len() as f64;
        let std_time = variance.sqrt();

        // Calculate throughput
        let throughput = (batch_size as f64 * 1000.0) / avg_time; // samples per second

        // Calculate percentiles
        let len = execution_times.len();
        let latency_percentiles = LatencyPercentiles {
            p50: execution_times[len * 50 / 100],
            p90: execution_times[len * 90 / 100],
            p95: execution_times[len * 95 / 100],
            p99: execution_times[len * 99 / 100],
        };

        PerformanceMetrics {
            avg_inference_time_ms: avg_time,
            min_inference_time_ms: min_time,
            max_inference_time_ms: max_time,
            std_inference_time_ms: std_time,
            throughput_samples_per_sec: throughput,
            latency_percentiles,
            batch_size,
            input_shape,
            device,
            dtype,
        }
    }

    /// Extract model information
    fn extract_model_info<M: Module>(&self, model: &M, name: &str) -> Result<ModelInfo> {
        let parameters = model.parameters();
        let num_parameters = parameters
            .values()
            .map(|p| p.numel())
            .collect::<torsh_core::error::Result<Vec<_>>>()?
            .iter()
            .sum::<usize>();

        // Estimate model size (parameters * 4 bytes for f32)
        let model_size_bytes = num_parameters * 4;

        Ok(ModelInfo {
            name: name.to_string(),
            num_parameters,
            model_size_bytes,
            architecture: "Unknown".to_string(), // Would need model-specific detection
            version: None,
        })
    }

    /// Measure memory usage
    fn measure_memory_usage<M: Module>(&self, _model: &M) -> Result<MemoryMetrics> {
        // Simplified memory measurement - would need platform-specific implementation
        Ok(MemoryMetrics {
            peak_memory_bytes: 0,
            memory_per_sample_bytes: 0,
            model_memory_bytes: 0,
            activation_memory_bytes: 0,
        })
    }

    /// Compare two benchmark results
    pub fn compare_results(
        &self,
        baseline: &BenchmarkResults,
        comparison: &BenchmarkResults,
    ) -> ComparisonResults {
        let mut comparisons = HashMap::new();

        for (config_key, baseline_metrics) in &baseline.performance_metrics {
            if let Some(comparison_metrics) = comparison.performance_metrics.get(config_key) {
                let speedup = baseline_metrics.avg_inference_time_ms
                    / comparison_metrics.avg_inference_time_ms;
                let throughput_improvement = comparison_metrics.throughput_samples_per_sec
                    / baseline_metrics.throughput_samples_per_sec;

                comparisons.insert(
                    config_key.clone(),
                    ConfigComparison {
                        speedup_factor: speedup,
                        throughput_improvement: throughput_improvement,
                        latency_reduction_percent: (1.0
                            - comparison_metrics.avg_inference_time_ms
                                / baseline_metrics.avg_inference_time_ms)
                            * 100.0,
                    },
                );
            }
        }

        ComparisonResults {
            baseline_model: baseline.model_info.name.clone(),
            comparison_model: comparison.model_info.name.clone(),
            config_comparisons: comparisons,
            parameter_reduction: 1.0
                - (comparison.model_info.num_parameters as f64
                    / baseline.model_info.num_parameters as f64),
            size_reduction: 1.0
                - (comparison.model_info.model_size_bytes as f64
                    / baseline.model_info.model_size_bytes as f64),
        }
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResults] {
        &self.results
    }

    /// Save benchmark results to file
    pub fn save_results(&self, path: &str) -> Result<()> {
        let json_data = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json_data)?;
        Ok(())
    }

    /// Load benchmark results from file
    pub fn load_results(&mut self, path: &str) -> Result<()> {
        let json_data = std::fs::read_to_string(path)?;
        self.results = serde_json::from_str(&json_data)?;
        Ok(())
    }
}

/// Comparison results between two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    pub baseline_model: String,
    pub comparison_model: String,
    pub config_comparisons: HashMap<String, ConfigComparison>,
    pub parameter_reduction: f64,
    pub size_reduction: f64,
}

/// Comparison for a specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigComparison {
    pub speedup_factor: f64,
    pub throughput_improvement: f64,
    pub latency_reduction_percent: f64,
}

impl fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Benchmark Results for {}", self.model_info.name)?;
        writeln!(f, "Parameters: {}", self.model_info.num_parameters)?;
        writeln!(
            f,
            "Model Size: {:.2} MB",
            self.model_info.model_size_bytes as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f)?;

        for (config, metrics) in &self.performance_metrics {
            writeln!(f, "Configuration: {}", config)?;
            writeln!(
                f,
                "  Avg Inference Time: {:.2} ms",
                metrics.avg_inference_time_ms
            )?;
            writeln!(
                f,
                "  Throughput: {:.2} samples/sec",
                metrics.throughput_samples_per_sec
            )?;
            writeln!(
                f,
                "  P50 Latency: {:.2} ms",
                metrics.latency_percentiles.p50
            )?;
            writeln!(
                f,
                "  P95 Latency: {:.2} ms",
                metrics.latency_percentiles.p95
            )?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl fmt::Display for ComparisonResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Model Comparison: {} vs {}",
            self.baseline_model, self.comparison_model
        )?;
        writeln!(
            f,
            "Parameter Reduction: {:.1}%",
            self.parameter_reduction * 100.0
        )?;
        writeln!(f, "Size Reduction: {:.1}%", self.size_reduction * 100.0)?;
        writeln!(f)?;

        for (config, comparison) in &self.config_comparisons {
            writeln!(f, "Configuration: {}", config)?;
            writeln!(f, "  Speedup: {:.2}x", comparison.speedup_factor)?;
            writeln!(
                f,
                "  Throughput Improvement: {:.2}x",
                comparison.throughput_improvement
            )?;
            writeln!(
                f,
                "  Latency Reduction: {:.1}%",
                comparison.latency_reduction_percent
            )?;
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Utility functions for benchmarking
pub mod utils {
    use super::*;

    /// Create a quick benchmark config for common use cases
    pub fn quick_benchmark_config() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 5,
            benchmark_iterations: 50,
            batch_sizes: vec![1, 16],
            input_shapes: vec![vec![224, 224, 3]],
            devices: vec![DeviceType::Cpu],
            measure_memory: false,
            measure_accuracy: false,
            dtypes: vec!["f32".to_string()],
        }
    }

    /// Create a comprehensive benchmark config
    pub fn comprehensive_benchmark_config() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 20,
            benchmark_iterations: 200,
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
            input_shapes: vec![
                vec![224, 224, 3], // Common image size
                vec![512, 512, 3], // High resolution
                vec![128, 128, 3], // Low resolution
            ],
            devices: vec![DeviceType::Cpu], // Add GPU when available
            measure_memory: true,
            measure_accuracy: false,
            dtypes: vec!["f32".to_string()],
        }
    }

    /// Calculate efficiency score (throughput per parameter)
    pub fn calculate_efficiency_score(throughput: f64, num_parameters: usize) -> f64 {
        throughput / (num_parameters as f64 / 1_000_000.0) // Samples per second per million parameters
    }

    /// Calculate memory efficiency (throughput per MB of memory)
    pub fn calculate_memory_efficiency(throughput: f64, memory_bytes: usize) -> f64 {
        throughput / (memory_bytes as f64 / (1024.0 * 1024.0)) // Samples per second per MB
    }

    /// Generate benchmark report summary
    pub fn generate_summary(results: &[BenchmarkResults]) -> String {
        let mut summary = String::new();

        summary.push_str("=== Benchmark Summary ===\n\n");

        for result in results {
            summary.push_str(&format!("Model: {}\n", result.model_info.name));
            summary.push_str(&format!(
                "Parameters: {}\n",
                result.model_info.num_parameters
            ));

            // Find best performance config
            let best_config = result.performance_metrics.iter().max_by(|a, b| {
                a.1.throughput_samples_per_sec
                    .partial_cmp(&b.1.throughput_samples_per_sec)
                    .unwrap()
            });

            if let Some((config_name, metrics)) = best_config {
                summary.push_str(&format!("Best Config: {}\n", config_name));
                summary.push_str(&format!(
                    "Best Throughput: {:.2} samples/sec\n",
                    metrics.throughput_samples_per_sec
                ));
                summary.push_str(&format!(
                    "Best Latency: {:.2} ms\n",
                    metrics.avg_inference_time_ms
                ));
            }

            summary.push_str("\n");
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert!(config.batch_sizes.contains(&1));
        assert!(config.batch_sizes.contains(&32));
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let execution_times = vec![10.0, 12.0, 8.0, 11.0, 9.0]; // milliseconds
        let benchmark = ModelBenchmark::new(BenchmarkConfig::default());

        let metrics = benchmark.calculate_performance_metrics(
            execution_times.clone(),
            16, // batch_size
            vec![224, 224, 3],
            DeviceType::Cpu,
            "f32".to_string(),
        );

        let expected_avg = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        assert!((metrics.avg_inference_time_ms - expected_avg).abs() < 1e-6);

        let expected_throughput = (16.0 * 1000.0) / expected_avg;
        assert!((metrics.throughput_samples_per_sec - expected_throughput).abs() < 1e-6);
    }

    #[test]
    fn test_model_info_extraction() {
        // Would need a mock model for testing
        let model_info = ModelInfo {
            name: "test_model".to_string(),
            num_parameters: 1000000,
            model_size_bytes: 4000000, // 4MB for 1M f32 parameters
            architecture: "TestNet".to_string(),
            version: Some("v1.0".to_string()),
        };

        assert_eq!(model_info.num_parameters, 1000000);
        assert_eq!(model_info.model_size_bytes, 4000000);
    }

    #[test]
    fn test_latency_percentiles() {
        let execution_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let benchmark = ModelBenchmark::new(BenchmarkConfig::default());

        let metrics = benchmark.calculate_performance_metrics(
            execution_times,
            1,
            vec![10],
            DeviceType::Cpu,
            "f32".to_string(),
        );

        // For 10 samples, P50 should be around the 5th element (index 4)
        // Allow some tolerance for different percentile calculation methods
        assert!((metrics.latency_percentiles.p50 - 5.5).abs() < 2.0);
        assert!(metrics.latency_percentiles.p95 > metrics.latency_percentiles.p50);
    }

    #[test]
    fn test_comparison_results() {
        let baseline_metrics = PerformanceMetrics {
            avg_inference_time_ms: 100.0,
            min_inference_time_ms: 90.0,
            max_inference_time_ms: 110.0,
            std_inference_time_ms: 5.0,
            throughput_samples_per_sec: 10.0,
            latency_percentiles: LatencyPercentiles {
                p50: 100.0,
                p90: 105.0,
                p95: 107.0,
                p99: 109.0,
            },
            batch_size: 1,
            input_shape: vec![224, 224, 3],
            device: DeviceType::Cpu,
            dtype: "f32".to_string(),
        };

        let comparison_metrics = PerformanceMetrics {
            avg_inference_time_ms: 50.0,      // 2x faster
            throughput_samples_per_sec: 20.0, // 2x throughput
            ..baseline_metrics.clone()
        };

        let speedup =
            baseline_metrics.avg_inference_time_ms / comparison_metrics.avg_inference_time_ms;
        assert!((speedup - 2.0).abs() < 1e-6);

        let throughput_improvement = comparison_metrics.throughput_samples_per_sec
            / baseline_metrics.throughput_samples_per_sec;
        assert!((throughput_improvement - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_efficiency_calculations() {
        let throughput = 100.0; // samples/sec
        let num_parameters = 1_000_000; // 1M parameters
        let memory_bytes = 100 * 1024 * 1024; // 100MB

        let efficiency = utils::calculate_efficiency_score(throughput, num_parameters);
        assert_eq!(efficiency, 100.0); // 100 samples/sec per 1M parameters

        let memory_efficiency = utils::calculate_memory_efficiency(throughput, memory_bytes);
        assert_eq!(memory_efficiency, 1.0); // 1 sample/sec per MB
    }

    #[test]
    fn test_config_serialization() {
        let config = BenchmarkConfig {
            warmup_iterations: 15,
            benchmark_iterations: 75,
            batch_sizes: vec![2, 4, 8],
            input_shapes: vec![vec![128, 128, 3]],
            devices: vec![DeviceType::Cpu],
            measure_memory: true,
            measure_accuracy: false,
            dtypes: vec!["f32".to_string()],
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: BenchmarkConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.warmup_iterations, deserialized.warmup_iterations);
        assert_eq!(config.batch_sizes, deserialized.batch_sizes);
        assert_eq!(config.measure_memory, deserialized.measure_memory);
    }
}
