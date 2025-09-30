//! Performance benchmarking utilities for quantization analysis

use crate::{QScheme, QuantConfig, TorshResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance benchmarking utilities for quantization analysis
#[derive(Debug, Clone)]
pub struct QuantizationBenchmarker {
    /// Configuration for benchmarking
    pub config: BenchmarkConfig,
    /// Collected benchmark metrics
    pub metrics: Vec<BenchmarkResult>,
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target batch size for benchmarking
    pub batch_size: usize,
    /// Include memory usage measurements
    pub measure_memory: bool,
    /// Include accuracy measurements
    pub measure_accuracy: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            batch_size: 32,
            measure_memory: true,
            measure_accuracy: true,
        }
    }
}

impl Default for QuantizationBenchmarker {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

impl QuantizationBenchmarker {
    /// Create a new benchmarker with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            metrics: Vec::new(),
        }
    }

    /// Benchmark a quantization scheme
    pub fn benchmark_scheme(
        &mut self,
        scheme: QScheme,
        operation: impl Fn() -> TorshResult<()>,
    ) -> TorshResult<BenchmarkResult> {
        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            operation()?;
        }

        // Measurement phase
        let start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            operation()?;
        }
        let duration = start.elapsed();

        let avg_duration = duration / self.config.measurement_iterations as u32;
        let throughput = self.calculate_throughput(avg_duration);

        let result = BenchmarkResult {
            scheme,
            avg_latency_ms: avg_duration.as_millis() as f32,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: self.estimate_memory_usage(scheme),
            accuracy_preservation: self.estimate_accuracy_preservation(scheme),
            compression_ratio: self.estimate_compression_ratio(scheme),
        };

        self.metrics.push(result.clone());
        Ok(result)
    }

    /// Benchmark multiple schemes and compare
    pub fn benchmark_comparison(
        &mut self,
        schemes: &[QScheme],
        operation_factory: impl Fn(QScheme) -> Box<dyn Fn() -> TorshResult<()>>,
    ) -> TorshResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for &scheme in schemes {
            let operation = operation_factory(scheme);
            let result = self.benchmark_scheme(scheme, || operation())?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Quantization Benchmarking Report\n");
        report.push_str(&"=".repeat(80));
        report.push('\n');

        report.push_str(&format!(
            "{:<20} | {:>12} | {:>12} | {:>10} | {:>10}\n",
            "Scheme", "Latency (ms)", "Throughput", "Memory", "Accuracy"
        ));
        report.push_str(&"-".repeat(80));
        report.push('\n');

        for metric in &self.metrics {
            report.push_str(&format!(
                "{:<20} | {:>10.2} | {:>10.0} | {:>8.1}MB | {:>8.3}\n",
                format!("{:?}", metric.scheme),
                metric.avg_latency_ms,
                metric.throughput_ops_per_sec,
                metric.memory_usage_mb,
                metric.accuracy_preservation
            ));
        }

        report.push('\n');
        report.push_str(&format!(
            "Benchmark Configuration:\n\
             - Warmup iterations: {}\n\
             - Measurement iterations: {}\n\
             - Batch size: {}",
            self.config.warmup_iterations,
            self.config.measurement_iterations,
            self.config.batch_size
        ));

        report
    }

    /// Find the best scheme based on criteria
    pub fn find_best_scheme(&self, criteria: OptimizationCriteria) -> Option<QScheme> {
        if self.metrics.is_empty() {
            return None;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_scheme = None;

        for metric in &self.metrics {
            let score = criteria.calculate_score(metric);
            if score > best_score {
                best_score = score;
                best_scheme = Some(metric.scheme);
            }
        }

        best_scheme
    }

    // Helper methods
    fn calculate_throughput(&self, avg_duration: Duration) -> f32 {
        self.config.batch_size as f32 / avg_duration.as_secs_f32()
    }

    fn estimate_memory_usage(&self, scheme: QScheme) -> f32 {
        // Simplified memory estimation
        match scheme {
            QScheme::Binary => 0.5,
            QScheme::Ternary => 1.0,
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 2.0,
            QScheme::PerTensorAffine | QScheme::PerChannelAffine => 4.0,
            QScheme::PerTensorSymmetric | QScheme::PerChannelSymmetric => 4.0,
            QScheme::MixedPrecision => 8.0,
            QScheme::GroupWise => 3.0,
        }
    }

    fn estimate_accuracy_preservation(&self, scheme: QScheme) -> f32 {
        match scheme {
            QScheme::PerTensorAffine => 0.98,
            QScheme::PerChannelAffine => 0.99,
            QScheme::PerTensorSymmetric => 0.97,
            QScheme::PerChannelSymmetric => 0.98,
            QScheme::Int4PerTensor => 0.93,
            QScheme::Int4PerChannel => 0.95,
            QScheme::MixedPrecision => 0.99,
            QScheme::Binary => 0.75,
            QScheme::Ternary => 0.85,
            QScheme::GroupWise => 0.96,
        }
    }

    fn estimate_compression_ratio(&self, scheme: QScheme) -> f32 {
        match scheme {
            QScheme::PerTensorAffine => 4.0,
            QScheme::PerChannelAffine => 3.8,
            QScheme::PerTensorSymmetric => 4.0,
            QScheme::PerChannelSymmetric => 3.8,
            QScheme::Int4PerTensor => 8.0,
            QScheme::Int4PerChannel => 7.5,
            QScheme::MixedPrecision => 5.0,
            QScheme::Binary => 32.0,
            QScheme::Ternary => 16.0,
            QScheme::GroupWise => 6.0,
        }
    }

    /// Clear all collected metrics
    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
    }

    /// Get all collected metrics
    pub fn get_metrics(&self) -> &[BenchmarkResult] {
        &self.metrics
    }
}

/// Benchmark result for a specific quantization scheme
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Quantization scheme tested
    pub scheme: QScheme,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Accuracy preservation ratio (0.0 to 1.0)
    pub accuracy_preservation: f32,
    /// Compression ratio compared to FP32
    pub compression_ratio: f32,
}

/// Optimization criteria for selecting best quantization scheme
#[derive(Debug, Clone)]
pub struct OptimizationCriteria {
    /// Weight for latency optimization (lower is better)
    pub latency_weight: f32,
    /// Weight for throughput optimization (higher is better)
    pub throughput_weight: f32,
    /// Weight for memory optimization (lower is better)
    pub memory_weight: f32,
    /// Weight for accuracy preservation (higher is better)
    pub accuracy_weight: f32,
    /// Weight for compression ratio (higher is better)
    pub compression_weight: f32,
}

impl OptimizationCriteria {
    /// Create criteria optimized for speed
    pub fn optimize_for_speed() -> Self {
        Self {
            latency_weight: 0.4,
            throughput_weight: 0.4,
            memory_weight: 0.1,
            accuracy_weight: 0.1,
            compression_weight: 0.0,
        }
    }

    /// Create criteria optimized for accuracy
    pub fn optimize_for_accuracy() -> Self {
        Self {
            latency_weight: 0.1,
            throughput_weight: 0.1,
            memory_weight: 0.1,
            accuracy_weight: 0.7,
            compression_weight: 0.0,
        }
    }

    /// Create criteria optimized for size
    pub fn optimize_for_size() -> Self {
        Self {
            latency_weight: 0.1,
            throughput_weight: 0.1,
            memory_weight: 0.3,
            accuracy_weight: 0.2,
            compression_weight: 0.3,
        }
    }

    /// Calculate weighted score for a benchmark result
    pub fn calculate_score(&self, result: &BenchmarkResult) -> f32 {
        // Normalize metrics to 0-1 range and apply weights
        let latency_score = (1.0 / result.avg_latency_ms.max(0.001)) * self.latency_weight;
        let throughput_score =
            (result.throughput_ops_per_sec / 10000.0).min(1.0) * self.throughput_weight;
        let memory_score = (1.0 / result.memory_usage_mb.max(0.1)) * self.memory_weight;
        let accuracy_score = result.accuracy_preservation * self.accuracy_weight;
        let compression_score =
            (result.compression_ratio / 32.0).min(1.0) * self.compression_weight;

        latency_score + throughput_score + memory_score + accuracy_score + compression_score
    }
}
