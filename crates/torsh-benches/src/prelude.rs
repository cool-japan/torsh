//! Prelude module for convenient imports
//!
//! This module re-exports commonly used items for easy access.

// Core benchmarking framework
pub use crate::benchmarks::*;
pub use crate::comparisons::*;
pub use crate::metrics::*;
pub use crate::utils::*;

// Specialized benchmarks
pub use crate::model_benchmarks::*;
pub use crate::hardware_benchmarks::*;
pub use crate::precision_benchmarks::*;
pub use crate::distributed_training::*;
pub use crate::edge_deployment::*;
pub use crate::mobile_benchmarks::*;
pub use crate::wasm_benchmarks::*;
pub use crate::custom_ops_benchmarks::*;

// Analysis and reporting
pub use crate::benchmark_analysis::*;
pub use crate::benchmark_cache::{BenchmarkCache, CacheEntry, CacheStats};
pub use crate::benchmark_comparison::*;
pub use crate::cached_runner::{BatchCachedRunner, CachedBenchRunner};
pub use crate::system_info::*;
pub use crate::html_reporting::*;
pub use crate::performance_dashboards::*;
pub use crate::regression_detection::*;
pub use crate::visualization::*;
pub use crate::ci_integration::*;
pub use crate::advanced_analysis::*;
pub use crate::benchmark_validation::*;

// PyTorch comparisons (feature gated)
#[cfg(feature = "pytorch")]
pub use crate::pytorch_comparisons::*;

// Common types and traits
pub use std::time::{Duration, Instant};
pub use serde::{Serialize, Deserialize};
pub use crate::{BenchConfig, BenchResult, BenchRunner, Benchmarkable};

/// Common benchmark configuration presets
pub mod presets {
    use super::*;
    
    /// Quick benchmark configuration for fast testing
    pub fn quick_bench_config() -> BenchConfig {
        BenchConfig {
            name: "quick".to_string(),
            warmup_time: Duration::from_millis(50),
            measurement_time: Duration::from_millis(100),
            sizes: vec![64, 256],
            ..Default::default()
        }
    }
    
    /// Standard benchmark configuration for normal use
    pub fn standard_bench_config() -> BenchConfig {
        BenchConfig {
            name: "standard".to_string(),
            warmup_time: Duration::from_millis(100),
            measurement_time: Duration::from_secs(1),
            sizes: vec![64, 256, 1024],
            ..Default::default()
        }
    }
    
    /// Comprehensive benchmark configuration for detailed analysis
    pub fn comprehensive_bench_config() -> BenchConfig {
        BenchConfig {
            name: "comprehensive".to_string(),
            warmup_time: Duration::from_secs(1),
            measurement_time: Duration::from_secs(10),
            sizes: vec![64, 256, 1024, 4096],
            ..Default::default()
        }
    }
    
    /// Production benchmark configuration for CI/CD
    pub fn production_bench_config() -> BenchConfig {
        BenchConfig {
            name: "production".to_string(),
            warmup_time: Duration::from_millis(500),
            measurement_time: Duration::from_secs(5),
            sizes: vec![256, 1024, 4096],
            ..Default::default()
        }
    }
}

/// Utility functions for common benchmarking tasks
pub mod utils {
    use super::*;
    
    /// Create a range of sizes for scalability testing
    pub fn create_size_range(start: usize, end: usize, steps: usize) -> Vec<usize> {
        if steps <= 1 {
            return vec![start];
        }
        
        let mut sizes = Vec::with_capacity(steps);
        for i in 0..steps {
            let ratio = i as f64 / (steps - 1) as f64;
            let size = start + ((end - start) as f64 * ratio) as usize;
            sizes.push(size);
        }
        sizes
    }
    
    /// Create logarithmic size range for wide-range testing
    pub fn create_log_size_range(start: usize, end: usize, steps: usize) -> Vec<usize> {
        if steps <= 1 {
            return vec![start];
        }
        
        let log_start = (start as f64).ln();
        let log_end = (end as f64).ln();
        let step_size = (log_end - log_start) / (steps - 1) as f64;
        
        (0..steps)
            .map(|i| (log_start + i as f64 * step_size).exp() as usize)
            .collect()
    }
    
    /// Format duration for human-readable output
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos();
        
        if nanos < 1000 {
            format!("{} ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.2} μs", nanos as f64 / 1000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.2} ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
        }
    }
    
    /// Format throughput for human-readable output
    pub fn format_throughput(ops_per_sec: f64) -> String {
        if ops_per_sec < 1000.0 {
            format!("{:.1} ops/s", ops_per_sec)
        } else if ops_per_sec < 1_000_000.0 {
            format!("{:.1} Kops/s", ops_per_sec / 1000.0)
        } else if ops_per_sec < 1_000_000_000.0 {
            format!("{:.1} Mops/s", ops_per_sec / 1_000_000.0)
        } else {
            format!("{:.1} Gops/s", ops_per_sec / 1_000_000_000.0)
        }
    }
    
    /// Format memory size for human-readable output
    pub fn format_memory_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }
    
    /// Calculate confidence interval for measurements
    pub fn confidence_interval(values: &[f64], confidence: f64) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let std_error = std_dev / n.sqrt();
        
        // Approximate critical value for 95% confidence
        let critical_value = if confidence >= 0.95 { 1.96 } else { 1.645 };
        let margin_of_error = critical_value * std_error;
        
        (mean - margin_of_error, mean + margin_of_error)
    }
}