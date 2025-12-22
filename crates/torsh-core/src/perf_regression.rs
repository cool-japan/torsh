//! Performance regression testing framework
//!
//! This module provides infrastructure for detecting performance regressions across releases.
//! It enables tracking of operation performance, establishing baselines, and detecting when
//! performance degrades beyond acceptable thresholds.
//!
//! # Features
//! - Performance benchmark recording with statistical analysis
//! - Baseline establishment and comparison
//! - Regression detection with configurable thresholds
//! - Historical performance tracking
//! - Multi-platform performance profiles

use crate::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

/// Performance measurement for a single operation
#[derive(Debug, Clone)]
pub struct PerfMeasurement {
    /// Operation name
    pub operation: String,

    /// Duration in nanoseconds
    pub duration_ns: u64,

    /// Throughput (operations per second)
    pub throughput: Option<f64>,

    /// Memory used in bytes
    pub memory_bytes: Option<usize>,

    /// Metadata (e.g., input size, configuration)
    pub metadata: HashMap<String, String>,
}

impl PerfMeasurement {
    /// Create a new performance measurement
    pub fn new(operation: String, duration_ns: u64) -> Self {
        Self {
            operation,
            duration_ns,
            throughput: None,
            memory_bytes: None,
            metadata: HashMap::new(),
        }
    }

    /// Set throughput
    pub fn with_throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput = Some(ops_per_sec);
        self
    }

    /// Set memory usage
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000.0
    }

    /// Get duration in seconds
    pub fn duration_secs(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000_000.0
    }
}

/// Statistical summary of multiple measurements
#[derive(Debug, Clone)]
pub struct PerfStatistics {
    /// Number of samples
    pub count: usize,

    /// Mean duration in nanoseconds
    pub mean_ns: f64,

    /// Median duration in nanoseconds
    pub median_ns: f64,

    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,

    /// Minimum duration in nanoseconds
    pub min_ns: u64,

    /// Maximum duration in nanoseconds
    pub max_ns: u64,

    /// 95th percentile in nanoseconds
    pub p95_ns: f64,

    /// 99th percentile in nanoseconds
    pub p99_ns: f64,
}

impl PerfStatistics {
    /// Calculate statistics from measurements
    pub fn from_measurements(measurements: &[PerfMeasurement]) -> Self {
        if measurements.is_empty() {
            return Self {
                count: 0,
                mean_ns: 0.0,
                median_ns: 0.0,
                std_dev_ns: 0.0,
                min_ns: 0,
                max_ns: 0,
                p95_ns: 0.0,
                p99_ns: 0.0,
            };
        }

        let mut durations: Vec<u64> = measurements.iter().map(|m| m.duration_ns).collect();
        durations.sort_unstable();

        let count = durations.len();
        let sum: u64 = durations.iter().sum();
        let mean = sum as f64 / count as f64;

        let median = if count % 2 == 0 {
            (durations[count / 2 - 1] + durations[count / 2]) as f64 / 2.0
        } else {
            durations[count / 2] as f64
        };

        let variance = durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev = variance.sqrt();

        let min = durations[0];
        let max = durations[count - 1];

        let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
        let p99_idx = ((count as f64 * 0.99) as usize).min(count - 1);

        Self {
            count,
            mean_ns: mean,
            median_ns: median,
            std_dev_ns: std_dev,
            min_ns: min,
            max_ns: max,
            p95_ns: durations[p95_idx] as f64,
            p99_ns: durations[p99_idx] as f64,
        }
    }

    /// Get coefficient of variation (CV = std_dev / mean)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean_ns > 0.0 {
            self.std_dev_ns / self.mean_ns
        } else {
            0.0
        }
    }

    /// Check if measurements are stable (low variance)
    pub fn is_stable(&self, cv_threshold: f64) -> bool {
        self.coefficient_of_variation() < cv_threshold
    }
}

/// Performance baseline for an operation
#[derive(Debug, Clone)]
pub struct PerfBaseline {
    /// Operation name
    pub operation: String,

    /// Platform identifier (e.g., "x86_64-linux", "aarch64-macos")
    pub platform: String,

    /// Version when baseline was established
    pub version: String,

    /// Statistical baseline
    pub statistics: PerfStatistics,

    /// When baseline was created
    pub timestamp: u64,
}

impl PerfBaseline {
    /// Create a new baseline
    pub fn new(
        operation: String,
        platform: String,
        version: String,
        statistics: PerfStatistics,
    ) -> Self {
        Self {
            operation,
            platform,
            version,
            statistics,
            timestamp: current_timestamp(),
        }
    }

    /// Compare current measurements against baseline
    pub fn compare(&self, current: &PerfStatistics) -> PerfComparison {
        let mean_ratio = current.mean_ns / self.statistics.mean_ns;
        let median_ratio = current.median_ns / self.statistics.median_ns;
        let p95_ratio = current.p95_ns / self.statistics.p95_ns;

        let mean_change_pct = (mean_ratio - 1.0) * 100.0;
        let median_change_pct = (median_ratio - 1.0) * 100.0;
        let p95_change_pct = (p95_ratio - 1.0) * 100.0;

        PerfComparison {
            baseline: self.clone(),
            current: current.clone(),
            mean_ratio,
            median_ratio,
            p95_ratio,
            mean_change_pct,
            median_change_pct,
            p95_change_pct,
        }
    }
}

/// Comparison between baseline and current performance
#[derive(Debug, Clone)]
pub struct PerfComparison {
    /// Baseline performance
    pub baseline: PerfBaseline,

    /// Current performance
    pub current: PerfStatistics,

    /// Ratio of current to baseline (mean)
    pub mean_ratio: f64,

    /// Ratio of current to baseline (median)
    pub median_ratio: f64,

    /// Ratio of current to baseline (p95)
    pub p95_ratio: f64,

    /// Percentage change (mean)
    pub mean_change_pct: f64,

    /// Percentage change (median)
    pub median_change_pct: f64,

    /// Percentage change (p95)
    pub p95_change_pct: f64,
}

impl PerfComparison {
    /// Check if there's a regression based on threshold
    pub fn is_regression(&self, threshold_pct: f64) -> bool {
        self.mean_change_pct > threshold_pct
            || self.median_change_pct > threshold_pct
            || self.p95_change_pct > threshold_pct
    }

    /// Check if there's an improvement
    pub fn is_improvement(&self, threshold_pct: f64) -> bool {
        // Use epsilon for floating point comparison
        const EPSILON: f64 = 1e-10;
        self.mean_change_pct <= -threshold_pct + EPSILON
            && self.median_change_pct <= -threshold_pct + EPSILON
            && self.p95_change_pct <= -threshold_pct + EPSILON
    }

    /// Get severity of regression
    pub fn regression_severity(&self) -> RegressionSeverity {
        let max_change = self
            .mean_change_pct
            .max(self.median_change_pct)
            .max(self.p95_change_pct);

        if max_change < 5.0 {
            RegressionSeverity::None
        } else if max_change < 10.0 {
            RegressionSeverity::Minor
        } else if max_change < 25.0 {
            RegressionSeverity::Moderate
        } else if max_change < 50.0 {
            RegressionSeverity::Major
        } else {
            RegressionSeverity::Critical
        }
    }
}

/// Severity of performance regression
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegressionSeverity {
    None,
    Minor,    // 5-10%
    Moderate, // 10-25%
    Major,    // 25-50%
    Critical, // >50%
}

impl RegressionSeverity {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            Self::None => "No regression",
            Self::Minor => "Minor regression (5-10%)",
            Self::Moderate => "Moderate regression (10-25%)",
            Self::Major => "Major regression (25-50%)",
            Self::Critical => "Critical regression (>50%)",
        }
    }
}

/// Performance regression tracker
#[derive(Debug, Clone)]
pub struct RegressionTracker {
    /// Baselines by operation and platform
    baselines: HashMap<String, HashMap<String, PerfBaseline>>,

    /// Regression threshold percentage
    threshold_pct: f64,

    /// Platform identifier
    platform: String,

    /// Version identifier
    version: String,
}

impl RegressionTracker {
    /// Create a new regression tracker
    pub fn new(platform: String, version: String) -> Self {
        Self {
            baselines: HashMap::new(),
            threshold_pct: 10.0, // 10% by default
            platform,
            version,
        }
    }

    /// Set regression threshold
    pub fn with_threshold(mut self, threshold_pct: f64) -> Self {
        self.threshold_pct = threshold_pct;
        self
    }

    /// Add a baseline
    pub fn add_baseline(&mut self, baseline: PerfBaseline) {
        self.baselines
            .entry(baseline.operation.clone())
            .or_insert_with(HashMap::new)
            .insert(baseline.platform.clone(), baseline);
    }

    /// Check for regression
    pub fn check_regression(
        &self,
        operation: &str,
        current: &PerfStatistics,
    ) -> Option<PerfComparison> {
        let baseline = self
            .baselines
            .get(operation)
            .and_then(|platforms| platforms.get(&self.platform))?;

        Some(baseline.compare(current))
    }

    /// Get all regressions above threshold
    pub fn find_regressions(
        &self,
        measurements: &[(String, PerfStatistics)],
    ) -> Vec<PerfComparison> {
        let mut regressions = Vec::new();

        for (operation, stats) in measurements {
            if let Some(comparison) = self.check_regression(operation, stats) {
                if comparison.is_regression(self.threshold_pct) {
                    regressions.push(comparison);
                }
            }
        }

        regressions
    }

    /// Generate regression report
    pub fn generate_report(&self, measurements: &[(String, PerfStatistics)]) -> RegressionReport {
        let comparisons: Vec<_> = measurements
            .iter()
            .filter_map(|(op, stats)| self.check_regression(op, stats))
            .collect();

        let regressions: Vec<_> = comparisons
            .iter()
            .filter(|c| c.is_regression(self.threshold_pct))
            .cloned()
            .collect();

        let improvements: Vec<_> = comparisons
            .iter()
            .filter(|c| c.is_improvement(self.threshold_pct))
            .cloned()
            .collect();

        RegressionReport {
            platform: self.platform.clone(),
            version: self.version.clone(),
            total_operations: measurements.len(),
            regressions,
            improvements,
            threshold_pct: self.threshold_pct,
            timestamp: current_timestamp(),
        }
    }
}

/// Regression detection report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Platform identifier
    pub platform: String,

    /// Version identifier
    pub version: String,

    /// Total number of operations tested
    pub total_operations: usize,

    /// Detected regressions
    pub regressions: Vec<PerfComparison>,

    /// Detected improvements
    pub improvements: Vec<PerfComparison>,

    /// Threshold used
    pub threshold_pct: f64,

    /// Report timestamp
    pub timestamp: u64,
}

impl RegressionReport {
    /// Check if any regressions were found
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// Get worst regression
    pub fn worst_regression(&self) -> Option<&PerfComparison> {
        self.regressions
            .iter()
            .max_by(|a, b| a.mean_change_pct.partial_cmp(&b.mean_change_pct).unwrap())
    }

    /// Get best improvement
    pub fn best_improvement(&self) -> Option<&PerfComparison> {
        self.improvements
            .iter()
            .min_by(|a, b| a.mean_change_pct.partial_cmp(&b.mean_change_pct).unwrap())
    }

    /// Count regressions by severity
    pub fn regressions_by_severity(&self) -> HashMap<RegressionSeverity, usize> {
        let mut counts = HashMap::new();
        for regression in &self.regressions {
            let severity = regression.regression_severity();
            *counts.entry(severity).or_insert(0) += 1;
        }
        counts
    }
}

/// Performance benchmark runner
#[cfg(feature = "std")]
pub struct BenchmarkRunner {
    /// Number of warmup iterations
    warmup_iterations: usize,

    /// Number of measurement iterations
    measurement_iterations: usize,

    /// Minimum sample count
    min_samples: usize,
}

#[cfg(feature = "std")]
impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            min_samples: 50,
        }
    }

    /// Set warmup iterations
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set measurement iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Run benchmark
    pub fn run<F>(&self, operation: &str, mut f: F) -> Result<PerfMeasurement>
    where
        F: FnMut() -> Result<()>,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f()?;
        }

        // Measure
        let start = Instant::now();
        for _ in 0..self.measurement_iterations {
            f()?;
        }
        let duration = start.elapsed();

        let duration_ns = duration.as_nanos() as u64;
        let avg_duration_ns = duration_ns / self.measurement_iterations as u64;

        let throughput = 1_000_000_000.0 / avg_duration_ns as f64;

        Ok(
            PerfMeasurement::new(operation.to_string(), avg_duration_ns)
                .with_throughput(throughput),
        )
    }

    /// Run benchmark with multiple samples
    pub fn run_with_samples<F>(&self, operation: &str, mut f: F) -> Result<Vec<PerfMeasurement>>
    where
        F: FnMut() -> Result<()>,
    {
        let mut measurements = Vec::with_capacity(self.min_samples);

        for _ in 0..self.min_samples {
            measurements.push(self.run(operation, &mut f)?);
        }

        Ok(measurements)
    }
}

#[cfg(feature = "std")]
impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp (seconds since epoch)
fn current_timestamp() -> u64 {
    #[cfg(feature = "std")]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[cfg(not(feature = "std"))]
    {
        0 // Placeholder for no_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_measurement() {
        let measurement = PerfMeasurement::new("test_op".to_string(), 1_000_000)
            .with_throughput(1000.0)
            .with_memory(1024);

        assert_eq!(measurement.duration_ms(), 1.0);
        assert_eq!(measurement.throughput, Some(1000.0));
        assert_eq!(measurement.memory_bytes, Some(1024));
    }

    #[test]
    fn test_perf_statistics() {
        let measurements = vec![
            PerfMeasurement::new("op".to_string(), 100),
            PerfMeasurement::new("op".to_string(), 200),
            PerfMeasurement::new("op".to_string(), 150),
            PerfMeasurement::new("op".to_string(), 180),
            PerfMeasurement::new("op".to_string(), 120),
        ];

        let stats = PerfStatistics::from_measurements(&measurements);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_ns, 100);
        assert_eq!(stats.max_ns, 200);
        assert_eq!(stats.median_ns, 150.0);
    }

    #[test]
    fn test_regression_detection() {
        let baseline_stats = PerfStatistics {
            count: 100,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            std_dev_ns: 50.0,
            min_ns: 900,
            max_ns: 1100,
            p95_ns: 1050.0,
            p99_ns: 1080.0,
        };

        let baseline = PerfBaseline::new(
            "test_op".to_string(),
            "test_platform".to_string(),
            "1.0.0".to_string(),
            baseline_stats,
        );

        // Test regression
        let current_stats = PerfStatistics {
            count: 100,
            mean_ns: 1200.0, // 20% slower
            median_ns: 1200.0,
            std_dev_ns: 60.0,
            min_ns: 1000,
            max_ns: 1300,
            p95_ns: 1260.0,
            p99_ns: 1290.0,
        };

        let comparison = baseline.compare(&current_stats);
        assert!(comparison.is_regression(10.0));
        assert_eq!(
            comparison.regression_severity(),
            RegressionSeverity::Moderate
        );
    }

    #[test]
    fn test_regression_tracker() {
        let mut tracker = RegressionTracker::new("test_platform".to_string(), "1.0.0".to_string())
            .with_threshold(15.0);

        let baseline_stats = PerfStatistics {
            count: 100,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            std_dev_ns: 50.0,
            min_ns: 900,
            max_ns: 1100,
            p95_ns: 1050.0,
            p99_ns: 1080.0,
        };

        let baseline = PerfBaseline::new(
            "test_op".to_string(),
            "test_platform".to_string(),
            "0.9.0".to_string(),
            baseline_stats,
        );

        tracker.add_baseline(baseline);

        let current_stats = PerfStatistics {
            count: 100,
            mean_ns: 1100.0, // 10% slower (below 15% threshold)
            median_ns: 1100.0,
            std_dev_ns: 55.0,
            min_ns: 950,
            max_ns: 1200,
            p95_ns: 1155.0,
            p99_ns: 1188.0,
        };

        let comparison = tracker.check_regression("test_op", &current_stats).unwrap();
        assert!(!comparison.is_regression(15.0)); // Should not be flagged

        // Test with worse performance
        let worse_stats = PerfStatistics {
            count: 100,
            mean_ns: 1200.0, // 20% slower (above threshold)
            median_ns: 1200.0,
            std_dev_ns: 60.0,
            min_ns: 1000,
            max_ns: 1300,
            p95_ns: 1260.0,
            p99_ns: 1290.0,
        };

        let comparison = tracker.check_regression("test_op", &worse_stats).unwrap();
        assert!(comparison.is_regression(15.0));
    }

    #[test]
    fn test_regression_report() {
        let mut tracker = RegressionTracker::new("test_platform".to_string(), "1.0.0".to_string());

        let baseline_stats = PerfStatistics {
            count: 100,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            std_dev_ns: 50.0,
            min_ns: 900,
            max_ns: 1100,
            p95_ns: 1050.0,
            p99_ns: 1080.0,
        };

        tracker.add_baseline(PerfBaseline::new(
            "op1".to_string(),
            "test_platform".to_string(),
            "0.9.0".to_string(),
            baseline_stats.clone(),
        ));

        tracker.add_baseline(PerfBaseline::new(
            "op2".to_string(),
            "test_platform".to_string(),
            "0.9.0".to_string(),
            baseline_stats,
        ));

        let measurements = vec![
            (
                "op1".to_string(),
                PerfStatistics {
                    count: 100,
                    mean_ns: 1200.0, // Regression
                    median_ns: 1200.0,
                    std_dev_ns: 60.0,
                    min_ns: 1000,
                    max_ns: 1300,
                    p95_ns: 1260.0,
                    p99_ns: 1290.0,
                },
            ),
            (
                "op2".to_string(),
                PerfStatistics {
                    count: 100,
                    mean_ns: 900.0, // Improvement
                    median_ns: 900.0,
                    std_dev_ns: 45.0,
                    min_ns: 800,
                    max_ns: 1000,
                    p95_ns: 945.0,
                    p99_ns: 972.0,
                },
            ),
        ];

        let report = tracker.generate_report(&measurements);
        assert!(report.has_regressions());
        assert_eq!(report.regressions.len(), 1);
        assert_eq!(report.improvements.len(), 1);
        assert!(report.worst_regression().is_some());
        assert!(report.best_improvement().is_some());
    }
}
