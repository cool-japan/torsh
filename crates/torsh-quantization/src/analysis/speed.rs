//! Speed analysis for quantized operations

use crate::QScheme;
use std::time::{Duration, Instant};

/// Speed analysis for quantized operations
pub struct SpeedAnalyzer;

impl SpeedAnalyzer {
    /// Estimate speed improvement for different quantization schemes
    pub fn estimate_speed_improvement(scheme: QScheme) -> f32 {
        match scheme {
            QScheme::Binary => 8.0,  // Binary operations are very fast
            QScheme::Ternary => 6.0, // Ternary operations are fast
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 4.0, // 4-bit operations
            QScheme::PerTensorAffine
            | QScheme::PerChannelAffine
            | QScheme::PerTensorSymmetric
            | QScheme::PerChannelSymmetric => 2.0, // 8-bit operations
            QScheme::MixedPrecision => 1.5, // Mixed precision
            QScheme::GroupWise => 2.5, // Group-wise quantization
        }
    }

    /// Benchmark operation speed
    pub fn benchmark_operation<F>(operation: F, iterations: usize) -> Duration
    where
        F: Fn(),
    {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        start.elapsed()
    }

    /// Calculate throughput (operations per second)
    pub fn calculate_throughput(duration: Duration, operations: usize) -> f64 {
        operations as f64 / duration.as_secs_f64()
    }

    /// Compare speed between different quantization schemes
    pub fn compare_schemes(
        _num_operations: usize,
        _baseline_duration: Duration,
    ) -> std::collections::HashMap<QScheme, f32> {
        let mut comparison = std::collections::HashMap::new();

        let schemes = vec![
            QScheme::Binary,
            QScheme::Ternary,
            QScheme::Int4PerTensor,
            QScheme::PerTensorAffine,
            QScheme::PerChannelAffine,
            QScheme::MixedPrecision,
            QScheme::GroupWise,
        ];

        for scheme in schemes {
            let estimated_improvement = Self::estimate_speed_improvement(scheme);
            comparison.insert(scheme, estimated_improvement);
        }

        comparison
    }

    /// Generate speed analysis report
    pub fn generate_speed_report(
        baseline_duration: Duration,
        quantized_duration: Duration,
        operations: usize,
    ) -> String {
        let speedup = baseline_duration.as_secs_f64() / quantized_duration.as_secs_f64();
        let baseline_throughput = Self::calculate_throughput(baseline_duration, operations);
        let quantized_throughput = Self::calculate_throughput(quantized_duration, operations);

        format!(
            "Speed Analysis Report:\n\
             - Baseline Duration: {:.3}ms\n\
             - Quantized Duration: {:.3}ms\n\
             - Speed Improvement: {:.2}x\n\
             - Baseline Throughput: {:.0} ops/sec\n\
             - Quantized Throughput: {:.0} ops/sec\n\
             - Efficiency Gain: {:.2}%",
            baseline_duration.as_millis(),
            quantized_duration.as_millis(),
            speedup,
            baseline_throughput,
            quantized_throughput,
            (speedup - 1.0) * 100.0
        )
    }
}
