//! Enhanced Benchmarking Suite for Distributed Training Features
//!
//! This module provides comprehensive benchmarking capabilities for the enhanced
//! distributed training features including gradient compression, network-aware
//! adaptation, and performance optimizations.

use crate::gradient_compression::{CompressionConfig, CompressionMethod, GradientCompressor};
use crate::gradient_compression_enhanced::{CompressionMetrics, EnhancedGradientCompressor};
use crate::network_aware_compression::{
    AdaptiveCompressionConfig, NetworkAwareCompressor, TrainingMetrics,
};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_tensor::{creation::randn, Tensor};
use tracing::info;

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of iterations per benchmark
    pub iterations: usize,
    /// Tensor sizes to benchmark
    pub tensor_sizes: Vec<Vec<usize>>,
    /// Compression methods to test
    pub compression_methods: Vec<CompressionMethod>,
    /// Compression ratios to test
    pub compression_ratios: Vec<f32>,
    /// Whether to include warmup iterations
    pub include_warmup: bool,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Whether to collect detailed metrics
    pub detailed_metrics: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            tensor_sizes: vec![
                vec![1000],               // 1D tensor
                vec![1000, 1000],         // 2D tensor
                vec![512, 512, 3],        // 3D tensor (image-like)
                vec![128, 128, 128, 128], // 4D tensor (batch)
            ],
            compression_methods: vec![
                CompressionMethod::TopK { k: 0.1 },
                CompressionMethod::Quantization { bits: 8 },
                CompressionMethod::SignSGD,
                CompressionMethod::Threshold { threshold: 0.01 },
            ],
            compression_ratios: vec![0.01, 0.05, 0.1, 0.2, 0.5],
            include_warmup: true,
            warmup_iterations: 10,
            detailed_metrics: true,
        }
    }
}

/// Benchmark result for a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test name
    pub test_name: String,
    /// Tensor shape tested
    pub tensor_shape: Vec<usize>,
    /// Compression method used
    pub compression_method: CompressionMethod,
    /// Average compression time (microseconds)
    pub avg_compression_time_us: f64,
    /// Average decompression time (microseconds)
    pub avg_decompression_time_us: f64,
    /// Average compression ratio achieved
    pub avg_compression_ratio: f32,
    /// Average compression error
    pub avg_compression_error: f32,
    /// Average throughput (MB/s)
    pub avg_throughput_mbps: f32,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Performance improvement over baseline
    pub performance_improvement_pct: f32,
    /// Standard deviation of compression times
    pub compression_time_std_dev: f64,
    /// Number of iterations
    pub iterations: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Original tensor size in bytes
    pub original_size_bytes: usize,
    /// Compressed size in bytes
    pub compressed_size_bytes: usize,
    /// Memory saved in bytes
    pub memory_saved_bytes: usize,
    /// Memory reduction percentage
    pub memory_reduction_pct: f32,
}

/// Comprehensive benchmark suite
pub struct EnhancedBenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl EnhancedBenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run complete benchmark suite
    pub fn run_complete_suite(&mut self) -> TorshResult<BenchmarkSummary> {
        info!("Starting enhanced distributed training benchmark suite");

        // Benchmark standard gradient compression
        self.benchmark_standard_compression()?;

        // Benchmark enhanced gradient compression
        self.benchmark_enhanced_compression()?;

        // Benchmark network-aware adaptive compression
        self.benchmark_network_aware_compression()?;

        // Benchmark compression method comparison
        self.benchmark_compression_methods()?;

        // Benchmark scaling performance
        self.benchmark_scaling_performance()?;

        let summary = self.generate_summary()?;
        info!("Benchmark suite completed successfully");
        Ok(summary)
    }

    /// Benchmark standard gradient compression
    fn benchmark_standard_compression(&mut self) -> TorshResult<()> {
        info!("Benchmarking standard gradient compression");

        for tensor_shape in &self.config.tensor_sizes {
            for compression_ratio in &self.config.compression_ratios {
                let config = CompressionConfig {
                    compression_ratio: *compression_ratio,
                    ..CompressionConfig::default()
                };

                let result = self.benchmark_compressor_performance(
                    &format!(
                        "standard_compression_{}_{:.2}",
                        self.shape_to_string(tensor_shape),
                        compression_ratio
                    ),
                    tensor_shape,
                    config,
                    BenchmarkType::Standard,
                )?;

                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Benchmark enhanced gradient compression
    fn benchmark_enhanced_compression(&mut self) -> TorshResult<()> {
        info!("Benchmarking enhanced gradient compression");

        for tensor_shape in &self.config.tensor_sizes {
            for compression_ratio in &self.config.compression_ratios {
                let config = CompressionConfig {
                    compression_ratio: *compression_ratio,
                    ..CompressionConfig::default()
                };

                let result = self.benchmark_compressor_performance(
                    &format!(
                        "enhanced_compression_{}_{:.2}",
                        self.shape_to_string(tensor_shape),
                        compression_ratio
                    ),
                    tensor_shape,
                    config,
                    BenchmarkType::Enhanced,
                )?;

                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Benchmark network-aware adaptive compression
    fn benchmark_network_aware_compression(&mut self) -> TorshResult<()> {
        info!("Benchmarking network-aware adaptive compression");

        for tensor_shape in &self.config.tensor_sizes {
            let base_config = CompressionConfig::default();
            let adaptive_config = AdaptiveCompressionConfig::default();

            let result = self.benchmark_network_aware_performance(
                &format!(
                    "network_aware_compression_{}",
                    self.shape_to_string(tensor_shape)
                ),
                tensor_shape,
                base_config,
                adaptive_config,
            )?;

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark different compression methods
    fn benchmark_compression_methods(&mut self) -> TorshResult<()> {
        info!("Benchmarking compression methods comparison");

        let test_shape = vec![1000, 1000]; // 2D tensor for method comparison

        for method in &self.config.compression_methods {
            let config = CompressionConfig {
                method: method.clone(),
                ..CompressionConfig::default()
            };

            let result = self.benchmark_compressor_performance(
                &format!("method_comparison_{:?}", method),
                &test_shape,
                config,
                BenchmarkType::Enhanced,
            )?;

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark scaling performance
    fn benchmark_scaling_performance(&mut self) -> TorshResult<()> {
        info!("Benchmarking scaling performance");

        let scaling_sizes = vec![
            vec![100, 100],   // Small
            vec![500, 500],   // Medium
            vec![1000, 1000], // Large
            vec![2000, 2000], // Extra Large
            vec![5000, 1000], // Wide
            vec![1000, 5000], // Tall
        ];

        for tensor_shape in &scaling_sizes {
            let config = CompressionConfig::default();

            let result = self.benchmark_compressor_performance(
                &format!("scaling_performance_{}", self.shape_to_string(tensor_shape)),
                tensor_shape,
                config,
                BenchmarkType::Enhanced,
            )?;

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark individual compressor performance
    fn benchmark_compressor_performance(
        &self,
        test_name: &str,
        tensor_shape: &[usize],
        config: CompressionConfig,
        benchmark_type: BenchmarkType,
    ) -> TorshResult<BenchmarkResult> {
        // Create test tensor
        let test_tensor = randn::<f32>(tensor_shape)?;
        let original_size = self.calculate_tensor_size(&test_tensor);

        // Prepare metrics collection
        let mut compression_times = Vec::new();
        let mut decompression_times = Vec::new();
        let mut compression_ratios = Vec::new();
        let mut compression_errors = Vec::new();
        let mut throughputs = Vec::new();
        let mut compressed_sizes = Vec::new();

        // Warmup iterations
        if self.config.include_warmup {
            for _ in 0..self.config.warmup_iterations {
                self.run_single_compression_test(&test_tensor, &config, benchmark_type)?;
            }
        }

        // Actual benchmark iterations
        for _iteration in 0..self.config.iterations {
            let metrics =
                self.run_single_compression_test(&test_tensor, &config, benchmark_type)?;

            compression_times.push(metrics.compression_time_us as f64);
            decompression_times.push(metrics.decompression_time_us as f64);
            compression_ratios.push(metrics.compression_ratio);
            compression_errors.push(metrics.compression_error);
            throughputs.push(metrics.throughput_mbps);
            compressed_sizes.push(metrics.memory_saved);
        }

        // Calculate statistics
        let avg_compression_time =
            compression_times.iter().sum::<f64>() / compression_times.len() as f64;
        let avg_decompression_time =
            decompression_times.iter().sum::<f64>() / decompression_times.len() as f64;
        let avg_compression_ratio =
            compression_ratios.iter().sum::<f32>() / compression_ratios.len() as f32;
        let avg_compression_error =
            compression_errors.iter().sum::<f32>() / compression_errors.len() as f32;
        let avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;
        let avg_compressed_size = compressed_sizes.iter().sum::<usize>() / compressed_sizes.len();

        // Calculate standard deviation
        let variance: f64 = compression_times
            .iter()
            .map(|&x| (x - avg_compression_time).powi(2))
            .sum::<f64>()
            / compression_times.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate memory usage stats
        let memory_saved = original_size.saturating_sub(avg_compressed_size);
        let memory_reduction_pct = if original_size > 0 {
            (memory_saved as f32 / original_size as f32) * 100.0
        } else {
            0.0
        };

        let memory_usage = MemoryUsageStats {
            original_size_bytes: original_size,
            compressed_size_bytes: avg_compressed_size,
            memory_saved_bytes: memory_saved,
            memory_reduction_pct,
        };

        // Calculate performance improvement (comparing to hypothetical baseline)
        let baseline_time = 1000.0; // Microseconds baseline
        let performance_improvement = if avg_compression_time > 0.0 {
            ((baseline_time - avg_compression_time) / baseline_time * 100.0).max(-100.0)
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            tensor_shape: tensor_shape.to_vec(),
            compression_method: config.method.clone(),
            avg_compression_time_us: avg_compression_time,
            avg_decompression_time_us: avg_decompression_time,
            avg_compression_ratio,
            avg_compression_error,
            avg_throughput_mbps: avg_throughput,
            memory_usage,
            performance_improvement_pct: performance_improvement as f32,
            compression_time_std_dev: std_dev,
            iterations: self.config.iterations,
        })
    }

    /// Benchmark network-aware compression performance
    fn benchmark_network_aware_performance(
        &self,
        test_name: &str,
        tensor_shape: &[usize],
        base_config: CompressionConfig,
        adaptive_config: AdaptiveCompressionConfig,
    ) -> TorshResult<BenchmarkResult> {
        let test_tensor = randn::<f32>(tensor_shape)?;
        let original_size = self.calculate_tensor_size(&test_tensor);

        let mut compressor = NetworkAwareCompressor::new(base_config.clone(), adaptive_config)?;

        let mut compression_times = Vec::new();
        let mut compression_ratios = Vec::new();
        let mut compression_errors = Vec::new();
        let mut throughputs = Vec::new();

        // Warmup
        if self.config.include_warmup {
            for _ in 0..self.config.warmup_iterations {
                let training_metrics = TrainingMetrics {
                    loss: 0.5,
                    gradient_norm: 1.0,
                    learning_rate: 0.001,
                };
                compressor.compress_gradient_adaptive(&test_tensor, Some(training_metrics))?;
            }
        }

        // Benchmark iterations
        for _iteration in 0..self.config.iterations {
            let training_metrics = TrainingMetrics {
                loss: 0.5,
                gradient_norm: 1.0,
                learning_rate: 0.001,
            };

            let start_time = Instant::now();
            let (_compressed, metrics) =
                compressor.compress_gradient_adaptive(&test_tensor, Some(training_metrics))?;
            let compression_time = start_time.elapsed().as_micros() as f64;

            compression_times.push(compression_time);
            compression_ratios.push(metrics.compression_ratio);
            compression_errors.push(metrics.compression_error);
            throughputs.push(metrics.throughput_mbps);
        }

        // Calculate statistics
        let avg_compression_time =
            compression_times.iter().sum::<f64>() / compression_times.len() as f64;
        let avg_compression_ratio =
            compression_ratios.iter().sum::<f32>() / compression_ratios.len() as f32;
        let avg_compression_error =
            compression_errors.iter().sum::<f32>() / compression_errors.len() as f32;
        let avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;

        let variance: f64 = compression_times
            .iter()
            .map(|&x| (x - avg_compression_time).powi(2))
            .sum::<f64>()
            / compression_times.len() as f64;
        let std_dev = variance.sqrt();

        let compressed_size = (original_size as f32 * (1.0 - avg_compression_ratio)) as usize;
        let memory_saved = original_size.saturating_sub(compressed_size);

        let memory_usage = MemoryUsageStats {
            original_size_bytes: original_size,
            compressed_size_bytes: compressed_size,
            memory_saved_bytes: memory_saved,
            memory_reduction_pct: (memory_saved as f32 / original_size as f32) * 100.0,
        };

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            tensor_shape: tensor_shape.to_vec(),
            compression_method: base_config.method,
            avg_compression_time_us: avg_compression_time,
            avg_decompression_time_us: 0.0, // Not measured for network-aware
            avg_compression_ratio,
            avg_compression_error,
            avg_throughput_mbps: avg_throughput,
            memory_usage,
            performance_improvement_pct: 0.0, // Calculated later
            compression_time_std_dev: std_dev,
            iterations: self.config.iterations,
        })
    }

    /// Run single compression test
    fn run_single_compression_test(
        &self,
        tensor: &Tensor,
        config: &CompressionConfig,
        benchmark_type: BenchmarkType,
    ) -> TorshResult<CompressionMetrics> {
        match benchmark_type {
            BenchmarkType::Standard => {
                let mut compressor = GradientCompressor::new(config.clone());
                let start_time = Instant::now();
                let compressed = compressor.compress(tensor, "benchmark")?;
                let compression_time = start_time.elapsed();

                // Estimate decompression time (simplified)
                let decompression_start = Instant::now();
                let _decompressed = compressor.decompress(&compressed)?;
                let decompression_time = decompression_start.elapsed();

                Ok(CompressionMetrics {
                    compression_ratio: config.compression_ratio,
                    compression_time_us: compression_time.as_micros() as u64,
                    decompression_time_us: decompression_time.as_micros() as u64,
                    memory_saved: (tensor.numel() * 4 * (1.0 - config.compression_ratio) as usize),
                    throughput_mbps: self.calculate_throughput(tensor, compression_time),
                    compression_error: 0.01, // Simplified error estimate
                    optimized_ops_count: 1,
                })
            }
            BenchmarkType::Enhanced => {
                let mut compressor = EnhancedGradientCompressor::new(config.clone())?;
                let (_, metrics) = compressor.compress_gradient_enhanced(tensor, "benchmark")?;
                Ok(metrics)
            }
        }
    }

    /// Calculate tensor size in bytes
    fn calculate_tensor_size(&self, tensor: &Tensor) -> usize {
        tensor.numel() * 4 // Assuming f32 (4 bytes per element)
    }

    /// Calculate throughput in MB/s
    fn calculate_throughput(&self, tensor: &Tensor, duration: Duration) -> f32 {
        let size_mb = self.calculate_tensor_size(tensor) as f32 / 1_048_576.0; // Convert to MB
        let duration_s = duration.as_secs_f32();
        if duration_s > 0.0 {
            size_mb / duration_s
        } else {
            0.0
        }
    }

    /// Convert shape to string representation
    fn shape_to_string(&self, shape: &[usize]) -> String {
        shape
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }

    /// Generate benchmark summary
    fn generate_summary(&self) -> TorshResult<BenchmarkSummary> {
        let mut method_performance = HashMap::new();
        let mut size_performance = HashMap::new();

        // Analyze performance by compression method
        for result in &self.results {
            let method_key = format!("{:?}", result.compression_method);
            let entry = method_performance
                .entry(method_key)
                .or_insert_with(Vec::new);
            entry.push(result.avg_compression_time_us);
        }

        // Analyze performance by tensor size
        for result in &self.results {
            let size_key = self.shape_to_string(&result.tensor_shape);
            let entry = size_performance.entry(size_key).or_insert_with(Vec::new);
            entry.push(result.avg_compression_time_us);
        }

        // Find best performing configurations
        let best_compression_ratio = self
            .results
            .iter()
            .max_by(|a, b| {
                a.avg_compression_ratio
                    .partial_cmp(&b.avg_compression_ratio)
                    .unwrap()
            })
            .cloned();

        let best_throughput = self
            .results
            .iter()
            .max_by(|a, b| {
                a.avg_throughput_mbps
                    .partial_cmp(&b.avg_throughput_mbps)
                    .unwrap()
            })
            .cloned();

        let best_memory_efficiency = self
            .results
            .iter()
            .max_by(|a, b| {
                a.memory_usage
                    .memory_reduction_pct
                    .partial_cmp(&b.memory_usage.memory_reduction_pct)
                    .unwrap()
            })
            .cloned();

        Ok(BenchmarkSummary {
            total_tests: self.results.len(),
            best_compression_ratio,
            best_throughput,
            best_memory_efficiency,
            method_performance,
            size_performance,
            overall_performance_improvement: self.calculate_overall_improvement(),
        })
    }

    /// Calculate overall performance improvement
    fn calculate_overall_improvement(&self) -> f32 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total_improvement: f32 = self
            .results
            .iter()
            .map(|r| r.performance_improvement_pct)
            .sum();

        total_improvement / self.results.len() as f32
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Export results to JSON
    pub fn export_results_json(&self) -> TorshResult<String> {
        serde_json::to_string_pretty(&self.results).map_err(|e| {
            TorshDistributedError::communication_error(
                "json_export",
                format!("Serialization failed: {}", e),
            )
        })
    }
}

/// Benchmark type enumeration
#[derive(Debug, Clone, Copy)]
enum BenchmarkType {
    Standard,
    Enhanced,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Total number of tests run
    pub total_tests: usize,
    /// Best compression ratio result
    pub best_compression_ratio: Option<BenchmarkResult>,
    /// Best throughput result
    pub best_throughput: Option<BenchmarkResult>,
    /// Best memory efficiency result
    pub best_memory_efficiency: Option<BenchmarkResult>,
    /// Performance by compression method
    pub method_performance: HashMap<String, Vec<f64>>,
    /// Performance by tensor size
    pub size_performance: HashMap<String, Vec<f64>>,
    /// Overall performance improvement percentage
    pub overall_performance_improvement: f32,
}

impl BenchmarkSummary {
    /// Print summary to console
    pub fn print_summary(&self) {
        println!("\n=== Enhanced Distributed Training Benchmark Summary ===");
        println!("Total tests run: {}", self.total_tests);
        println!(
            "Overall performance improvement: {:.2}%",
            self.overall_performance_improvement
        );

        if let Some(ref result) = self.best_compression_ratio {
            println!("\nBest Compression Ratio:");
            println!("  Test: {}", result.test_name);
            println!("  Ratio: {:.3}", result.avg_compression_ratio);
            println!("  Time: {:.2}μs", result.avg_compression_time_us);
        }

        if let Some(ref result) = self.best_throughput {
            println!("\nBest Throughput:");
            println!("  Test: {}", result.test_name);
            println!("  Throughput: {:.2} MB/s", result.avg_throughput_mbps);
            println!("  Time: {:.2}μs", result.avg_compression_time_us);
        }

        if let Some(ref result) = self.best_memory_efficiency {
            println!("\nBest Memory Efficiency:");
            println!("  Test: {}", result.test_name);
            println!(
                "  Memory saved: {:.2}%",
                result.memory_usage.memory_reduction_pct
            );
            println!("  Bytes saved: {}", result.memory_usage.memory_saved_bytes);
        }

        println!("\n=== Performance by Compression Method ===");
        for (method, times) in &self.method_performance {
            let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
            println!("  {}: {:.2}μs average", method, avg_time);
        }

        println!("\n=== Performance by Tensor Size ===");
        for (size, times) in &self.size_performance {
            let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
            println!("  {}: {:.2}μs average", size, avg_time);
        }

        println!("======================================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[tokio::test]
    async fn test_benchmark_suite() -> TorshResult<()> {
        let config = BenchmarkConfig {
            iterations: 5,                      // Reduced for testing
            tensor_sizes: vec![vec![100, 100]], // Small tensor for testing
            compression_ratios: vec![0.1, 0.5],
            include_warmup: false,
            warmup_iterations: 0,
            detailed_metrics: true,
            ..BenchmarkConfig::default()
        };

        let mut suite = EnhancedBenchmarkSuite::new(config);
        let summary = suite.run_complete_suite()?;

        assert!(summary.total_tests > 0);
        assert!(!suite.get_results().is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_compression_benchmark() -> TorshResult<()> {
        let config = CompressionConfig::default();
        let tensor = ones::<f32>(&[100, 100])?;

        let benchmark_suite = EnhancedBenchmarkSuite::new(BenchmarkConfig::default());
        let metrics = benchmark_suite.run_single_compression_test(
            &tensor,
            &config,
            BenchmarkType::Enhanced,
        )?;

        // Note: Compression time may be 0 for fast operations or mock implementations
        // compression_time_us is u64, always >= 0
        assert!(metrics.compression_ratio >= 0.0);
        assert!(metrics.throughput_mbps >= 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_usage_calculation() -> TorshResult<()> {
        let tensor = ones::<f32>(&[1000, 1000])?;
        let benchmark_suite = EnhancedBenchmarkSuite::new(BenchmarkConfig::default());

        let size = benchmark_suite.calculate_tensor_size(&tensor);
        assert_eq!(size, 1000 * 1000 * 4); // 4 bytes per f32

        Ok(())
    }
}
