// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Performance Regression Detection
//!
//! This module provides comprehensive performance regression detection for autograd operations,
//! tracking metrics over time and alerting when performance degrades beyond acceptable thresholds.
//!
//! # Features
//!
//! - **Baseline Tracking**: Establish performance baselines from historical data
//! - **Regression Detection**: Detect when performance degrades beyond thresholds
//! - **Statistical Analysis**: Use statistical methods to identify significant regressions
//! - **Multi-metric Tracking**: Track multiple metrics (time, memory, throughput, etc.)
//! - **Automated Alerts**: Trigger alerts when regressions are detected
//! - **Regression Reports**: Generate detailed reports on detected regressions
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_autograd::performance_regression::{RegressionDetector, RegressionConfig};
//!
//! // Create detector
//! let detector = RegressionDetector::new(RegressionConfig::default());
//!
//! // Record baseline performance
//! detector.record("backward_pass", PerformanceMetrics {
//!     execution_time_ms: 100.0,
//!     memory_usage_mb: 512.0,
//!     throughput: 1000.0,
//! });
//!
//! // Later, check for regressions
//! let result = detector.check_regression("backward_pass", PerformanceMetrics {
//!     execution_time_ms: 200.0,  // 2x slower - potential regression!
//!     memory_usage_mb: 512.0,
//!     throughput: 500.0,
//! });
//! ```

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Performance regression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Enable regression detection
    pub enabled: bool,

    /// Threshold for time regression (e.g., 1.5 = 50% slower is a regression)
    pub time_regression_threshold: f64,

    /// Threshold for memory regression
    pub memory_regression_threshold: f64,

    /// Threshold for throughput regression (e.g., 0.7 = 30% lower throughput is a regression)
    pub throughput_regression_threshold: f64,

    /// Number of samples to keep for baseline calculation
    pub baseline_window_size: usize,

    /// Minimum number of samples before regression detection starts
    pub min_samples_for_detection: usize,

    /// Use statistical methods (T-test, etc.) for detection
    pub use_statistical_methods: bool,

    /// Statistical significance level (p-value threshold)
    pub significance_level: f64,

    /// Enable automated alerts
    pub enable_alerts: bool,

    /// Alert cooldown period (avoid alert fatigue)
    pub alert_cooldown_seconds: u64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_regression_threshold: 1.3,       // 30% slower
            memory_regression_threshold: 1.5,     // 50% more memory
            throughput_regression_threshold: 0.7, // 30% lower throughput
            baseline_window_size: 100,
            min_samples_for_detection: 10,
            use_statistical_methods: true,
            significance_level: 0.05,
            enable_alerts: true,
            alert_cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,

    /// Memory usage in megabytes
    pub memory_usage_mb: f64,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,

    /// Timestamp when measured
    #[serde(with = "system_time_serde")]
    pub timestamp: SystemTime,
}

mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_mb: 0.0,
            throughput: 0.0,
            custom_metrics: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }
}

impl PerformanceMetrics {
    /// Create new metrics
    pub fn new(execution_time_ms: f64, memory_usage_mb: f64, throughput: f64) -> Self {
        Self {
            execution_time_ms,
            memory_usage_mb,
            throughput,
            custom_metrics: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Add custom metric
    pub fn with_custom_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(name.into(), value);
        self
    }
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Whether regression was detected
    pub regression_detected: bool,

    /// Operation name
    pub operation: String,

    /// Regressed metrics
    pub regressed_metrics: Vec<RegressedMetric>,

    /// Current metrics
    pub current_metrics: PerformanceMetrics,

    /// Baseline metrics
    pub baseline_metrics: BaselineMetrics,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Suggested actions
    pub suggestions: Vec<String>,
}

/// Information about a regressed metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressedMetric {
    /// Metric name
    pub name: String,

    /// Current value
    pub current_value: f64,

    /// Baseline value
    pub baseline_value: f64,

    /// Regression ratio (current / baseline)
    pub regression_ratio: f64,

    /// Severity (1.0 = threshold, higher = worse)
    pub severity: f64,

    /// Statistical p-value (if available)
    pub p_value: Option<f64>,
}

/// Baseline performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Average execution time
    pub avg_execution_time_ms: f64,

    /// Standard deviation of execution time
    pub std_execution_time_ms: f64,

    /// Average memory usage
    pub avg_memory_usage_mb: f64,

    /// Standard deviation of memory
    pub std_memory_usage_mb: f64,

    /// Average throughput
    pub avg_throughput: f64,

    /// Standard deviation of throughput
    pub std_throughput: f64,

    /// Number of samples in baseline
    pub sample_count: usize,

    /// Custom metric baselines
    pub custom_baselines: HashMap<String, (f64, f64)>, // (mean, std)
}

/// Historical performance data
#[derive(Debug)]
struct PerformanceHistory {
    /// Historical metrics
    samples: VecDeque<PerformanceMetrics>,

    /// Cached baseline (invalidated when samples change)
    cached_baseline: Option<BaselineMetrics>,

    /// Last alert time
    last_alert_time: Option<Instant>,
}

impl PerformanceHistory {
    fn new(window_size: usize) -> Self {
        let mut samples = VecDeque::new();
        samples.reserve(window_size);
        Self {
            samples,
            cached_baseline: None,
            last_alert_time: None,
        }
    }

    fn add_sample(&mut self, metrics: PerformanceMetrics, window_size: usize) {
        self.samples.push_back(metrics);
        if self.samples.len() > window_size {
            self.samples.pop_front();
        }
        // Invalidate cache
        self.cached_baseline = None;
    }

    fn compute_baseline(&mut self) -> BaselineMetrics {
        if let Some(ref baseline) = self.cached_baseline {
            return baseline.clone();
        }

        if self.samples.is_empty() {
            return BaselineMetrics {
                avg_execution_time_ms: 0.0,
                std_execution_time_ms: 0.0,
                avg_memory_usage_mb: 0.0,
                std_memory_usage_mb: 0.0,
                avg_throughput: 0.0,
                std_throughput: 0.0,
                sample_count: 0,
                custom_baselines: HashMap::new(),
            };
        }

        let n = self.samples.len() as f64;

        // Compute means
        let avg_time = self
            .samples
            .iter()
            .map(|s| s.execution_time_ms)
            .sum::<f64>()
            / n;
        let avg_memory = self.samples.iter().map(|s| s.memory_usage_mb).sum::<f64>() / n;
        let avg_throughput = self.samples.iter().map(|s| s.throughput).sum::<f64>() / n;

        // Compute standard deviations
        let std_time = (self
            .samples
            .iter()
            .map(|s| (s.execution_time_ms - avg_time).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        let std_memory = (self
            .samples
            .iter()
            .map(|s| (s.memory_usage_mb - avg_memory).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        let std_throughput = (self
            .samples
            .iter()
            .map(|s| (s.throughput - avg_throughput).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        // Compute custom metric baselines
        let mut custom_baselines = HashMap::new();
        let mut all_custom_keys: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for sample in &self.samples {
            for key in sample.custom_metrics.keys() {
                all_custom_keys.insert(key.clone());
            }
        }

        for key in all_custom_keys {
            let values: Vec<f64> = self
                .samples
                .iter()
                .filter_map(|s| s.custom_metrics.get(&key).copied())
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64)
                    .sqrt();
                custom_baselines.insert(key, (mean, std));
            }
        }

        let baseline = BaselineMetrics {
            avg_execution_time_ms: avg_time,
            std_execution_time_ms: std_time,
            avg_memory_usage_mb: avg_memory,
            std_memory_usage_mb: std_memory,
            avg_throughput: avg_throughput,
            std_throughput: std_throughput,
            sample_count: self.samples.len(),
            custom_baselines,
        };

        self.cached_baseline = Some(baseline.clone());
        baseline
    }

    fn should_alert(&mut self, cooldown: Duration) -> bool {
        if let Some(last_time) = self.last_alert_time {
            if last_time.elapsed() < cooldown {
                return false;
            }
        }
        self.last_alert_time = Some(Instant::now());
        true
    }
}

/// Performance regression detector
#[derive(Clone)]
pub struct RegressionDetector {
    config: Arc<RwLock<RegressionConfig>>,
    history: Arc<Mutex<HashMap<String, PerformanceHistory>>>,
    alert_callbacks: Arc<RwLock<Vec<Arc<dyn Fn(&RegressionResult) + Send + Sync>>>>,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            history: Arc::new(Mutex::new(HashMap::new())),
            alert_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record performance metrics
    pub fn record(&self, operation: impl Into<String>, metrics: PerformanceMetrics) {
        let config = self.config.read();
        if !config.enabled {
            return;
        }

        let operation = operation.into();
        let window_size = config.baseline_window_size;

        let mut history = self.history.lock();
        let entry = history
            .entry(operation)
            .or_insert_with(|| PerformanceHistory::new(window_size));

        entry.add_sample(metrics, window_size);
    }

    /// Check for performance regression
    pub fn check_regression(
        &self,
        operation: impl Into<String>,
        current: PerformanceMetrics,
    ) -> RegressionResult {
        let config = self.config.read();
        let operation = operation.into();

        let mut history = self.history.lock();
        let entry = match history.get_mut(&operation) {
            Some(e) => e,
            None => {
                // No baseline yet
                return RegressionResult {
                    regression_detected: false,
                    operation,
                    regressed_metrics: Vec::new(),
                    current_metrics: current,
                    baseline_metrics: BaselineMetrics {
                        avg_execution_time_ms: 0.0,
                        std_execution_time_ms: 0.0,
                        avg_memory_usage_mb: 0.0,
                        std_memory_usage_mb: 0.0,
                        avg_throughput: 0.0,
                        std_throughput: 0.0,
                        sample_count: 0,
                        custom_baselines: HashMap::new(),
                    },
                    confidence: 0.0,
                    suggestions: vec!["Insufficient baseline data".into()],
                };
            }
        };

        if entry.samples.len() < config.min_samples_for_detection {
            return RegressionResult {
                regression_detected: false,
                operation,
                regressed_metrics: Vec::new(),
                current_metrics: current,
                baseline_metrics: entry.compute_baseline(),
                confidence: 0.0,
                suggestions: vec![format!(
                    "Need {} more samples for detection",
                    config.min_samples_for_detection - entry.samples.len()
                )],
            };
        }

        let baseline = entry.compute_baseline();
        let mut regressed_metrics = Vec::new();

        // Check execution time
        if baseline.avg_execution_time_ms > 0.0 {
            let ratio = current.execution_time_ms / baseline.avg_execution_time_ms;
            if ratio >= config.time_regression_threshold {
                let p_value = if config.use_statistical_methods {
                    Some(self.compute_t_test(
                        current.execution_time_ms,
                        baseline.avg_execution_time_ms,
                        baseline.std_execution_time_ms,
                        baseline.sample_count,
                    ))
                } else {
                    None
                };

                regressed_metrics.push(RegressedMetric {
                    name: "execution_time_ms".into(),
                    current_value: current.execution_time_ms,
                    baseline_value: baseline.avg_execution_time_ms,
                    regression_ratio: ratio,
                    severity: ratio / config.time_regression_threshold,
                    p_value,
                });
            }
        }

        // Check memory usage
        if baseline.avg_memory_usage_mb > 0.0 {
            let ratio = current.memory_usage_mb / baseline.avg_memory_usage_mb;
            if ratio >= config.memory_regression_threshold {
                let p_value = if config.use_statistical_methods {
                    Some(self.compute_t_test(
                        current.memory_usage_mb,
                        baseline.avg_memory_usage_mb,
                        baseline.std_memory_usage_mb,
                        baseline.sample_count,
                    ))
                } else {
                    None
                };

                regressed_metrics.push(RegressedMetric {
                    name: "memory_usage_mb".into(),
                    current_value: current.memory_usage_mb,
                    baseline_value: baseline.avg_memory_usage_mb,
                    regression_ratio: ratio,
                    severity: ratio / config.memory_regression_threshold,
                    p_value,
                });
            }
        }

        // Check throughput (lower is worse)
        if baseline.avg_throughput > 0.0 {
            let ratio = current.throughput / baseline.avg_throughput;
            if ratio <= config.throughput_regression_threshold {
                let p_value = if config.use_statistical_methods {
                    Some(self.compute_t_test(
                        current.throughput,
                        baseline.avg_throughput,
                        baseline.std_throughput,
                        baseline.sample_count,
                    ))
                } else {
                    None
                };

                regressed_metrics.push(RegressedMetric {
                    name: "throughput".into(),
                    current_value: current.throughput,
                    baseline_value: baseline.avg_throughput,
                    regression_ratio: ratio,
                    severity: config.throughput_regression_threshold / ratio,
                    p_value,
                });
            }
        }

        // Generate suggestions
        let suggestions = self.generate_suggestions(&regressed_metrics);

        // Compute overall confidence
        let confidence = if regressed_metrics.is_empty() {
            0.0
        } else if config.use_statistical_methods {
            // Average (1 - p_value) for statistical confidence
            let p_values: Vec<f64> = regressed_metrics.iter().filter_map(|m| m.p_value).collect();
            if p_values.is_empty() {
                0.5 // Default confidence
            } else {
                p_values.iter().map(|p| 1.0 - p).sum::<f64>() / p_values.len() as f64
            }
        } else {
            // Simple confidence based on severity
            let avg_severity = regressed_metrics.iter().map(|m| m.severity).sum::<f64>()
                / regressed_metrics.len() as f64;
            (avg_severity - 1.0).min(1.0).max(0.0)
        };

        let regression_detected = !regressed_metrics.is_empty()
            && (!config.use_statistical_methods || confidence >= (1.0 - config.significance_level));

        let result = RegressionResult {
            regression_detected,
            operation: operation.clone(),
            regressed_metrics,
            current_metrics: current.clone(),
            baseline_metrics: baseline,
            confidence,
            suggestions,
        };

        // Trigger alerts if enabled
        if regression_detected
            && config.enable_alerts
            && entry.should_alert(Duration::from_secs(config.alert_cooldown_seconds))
        {
            let callbacks = self.alert_callbacks.read();
            for callback in callbacks.iter() {
                callback(&result);
            }
        }

        result
    }

    /// Compute t-test p-value (one-sample)
    fn compute_t_test(&self, sample: f64, pop_mean: f64, pop_std: f64, n: usize) -> f64 {
        if pop_std == 0.0 || n == 0 {
            return 1.0; // No significant difference
        }

        let t = ((sample - pop_mean) / (pop_std / (n as f64).sqrt())).abs();

        // Simplified p-value approximation using normal distribution
        // For more accurate results, use a proper statistical library
        let p = 2.0 * (1.0 - self.normal_cdf(t));
        p.max(0.0).min(1.0)
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Generate suggestions based on regressed metrics
    fn generate_suggestions(&self, regressed: &[RegressedMetric]) -> Vec<String> {
        let mut suggestions = Vec::new();

        for metric in regressed {
            match metric.name.as_str() {
                "execution_time_ms" => {
                    suggestions.push(format!(
                        "Execution time increased by {:.1}%. Consider profiling and optimizing hot paths.",
                        (metric.regression_ratio - 1.0) * 100.0
                    ));
                }
                "memory_usage_mb" => {
                    suggestions.push(format!(
                        "Memory usage increased by {:.1}%. Check for memory leaks or inefficient allocations.",
                        (metric.regression_ratio - 1.0) * 100.0
                    ));
                }
                "throughput" => {
                    suggestions.push(format!(
                        "Throughput decreased by {:.1}%. Investigate bottlenecks in the pipeline.",
                        (1.0 - metric.regression_ratio) * 100.0
                    ));
                }
                _ => {
                    suggestions.push(format!(
                        "Metric '{}' regressed by {:.1}%",
                        metric.name,
                        (metric.regression_ratio - 1.0).abs() * 100.0
                    ));
                }
            }
        }

        if suggestions.is_empty() {
            suggestions.push("No regressions detected".into());
        }

        suggestions
    }

    /// Add alert callback
    pub fn add_alert_callback(&self, callback: Arc<dyn Fn(&RegressionResult) + Send + Sync>) {
        self.alert_callbacks.write().push(callback);
    }

    /// Clear alert callbacks
    pub fn clear_alert_callbacks(&self) {
        self.alert_callbacks.write().clear();
    }

    /// Get baseline for operation
    pub fn get_baseline(&self, operation: &str) -> Option<BaselineMetrics> {
        let mut history = self.history.lock();
        history.get_mut(operation).map(|h| h.compute_baseline())
    }

    /// Clear history for operation
    pub fn clear_history(&self, operation: &str) {
        self.history.lock().remove(operation);
    }

    /// Clear all history
    pub fn clear_all_history(&self) {
        self.history.lock().clear();
    }

    /// Get all tracked operations
    pub fn tracked_operations(&self) -> Vec<String> {
        self.history.lock().keys().cloned().collect()
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(RegressionConfig::default())
    }
}

/// Global regression detector instance
static GLOBAL_DETECTOR: once_cell::sync::Lazy<RegressionDetector> =
    once_cell::sync::Lazy::new(|| RegressionDetector::default());

/// Get the global regression detector
pub fn global_detector() -> &'static RegressionDetector {
    &GLOBAL_DETECTOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_regression_detection() {
        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            use_statistical_methods: false, // Disable for simpler threshold-based detection
            ..Default::default()
        });

        // Record baseline performance
        for _ in 0..10 {
            detector.record("test_op", PerformanceMetrics::new(100.0, 512.0, 1000.0));
        }

        // Check with similar performance - no regression
        let result =
            detector.check_regression("test_op", PerformanceMetrics::new(105.0, 520.0, 980.0));
        assert!(!result.regression_detected);

        // Check with regressed performance
        let result = detector.check_regression(
            "test_op",
            PerformanceMetrics::new(200.0, 512.0, 1000.0), // 2x slower
        );
        assert!(result.regression_detected);
        assert!(!result.regressed_metrics.is_empty());
    }

    #[test]
    fn test_memory_regression() {
        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            memory_regression_threshold: 1.5,
            use_statistical_methods: false,
            ..Default::default()
        });

        // Baseline
        for _ in 0..10 {
            detector.record("test_op", PerformanceMetrics::new(100.0, 512.0, 1000.0));
        }

        // Memory regression
        let result = detector.check_regression(
            "test_op",
            PerformanceMetrics::new(100.0, 1024.0, 1000.0), // 2x memory
        );

        assert!(result.regression_detected);
        let memory_metric = result
            .regressed_metrics
            .iter()
            .find(|m| m.name == "memory_usage_mb");
        assert!(memory_metric.is_some());
    }

    #[test]
    fn test_throughput_regression() {
        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            throughput_regression_threshold: 0.7,
            use_statistical_methods: false,
            ..Default::default()
        });

        // Baseline
        for _ in 0..10 {
            detector.record("test_op", PerformanceMetrics::new(100.0, 512.0, 1000.0));
        }

        // Throughput regression
        let result = detector.check_regression(
            "test_op",
            PerformanceMetrics::new(100.0, 512.0, 500.0), // Half throughput
        );

        assert!(result.regression_detected);
        let throughput_metric = result
            .regressed_metrics
            .iter()
            .find(|m| m.name == "throughput");
        assert!(throughput_metric.is_some());
    }

    #[test]
    fn test_custom_metrics() {
        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            ..Default::default()
        });

        // Record baseline with custom metric
        for _ in 0..10 {
            let metrics = PerformanceMetrics::new(100.0, 512.0, 1000.0)
                .with_custom_metric("cache_hit_rate", 0.95);
            detector.record("test_op", metrics);
        }

        let baseline = detector.get_baseline("test_op").unwrap();
        assert!(baseline.custom_baselines.contains_key("cache_hit_rate"));
    }

    #[test]
    fn test_alert_callbacks() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            enable_alerts: true,
            use_statistical_methods: false,
            ..Default::default()
        });

        let alert_triggered = Arc::new(AtomicBool::new(false));
        let alert_clone = alert_triggered.clone();

        detector.add_alert_callback(Arc::new(move |_result| {
            alert_clone.store(true, Ordering::Relaxed);
        }));

        // Baseline
        for _ in 0..10 {
            detector.record("test_op", PerformanceMetrics::new(100.0, 512.0, 1000.0));
        }

        // Trigger regression
        detector.check_regression("test_op", PerformanceMetrics::new(200.0, 512.0, 1000.0));

        assert!(alert_triggered.load(Ordering::Relaxed));
    }

    #[test]
    fn test_statistical_detection() {
        let detector = RegressionDetector::new(RegressionConfig {
            min_samples_for_detection: 5,
            use_statistical_methods: true,
            significance_level: 0.05,
            ..Default::default()
        });

        // Baseline with some variance
        let times = vec![
            100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 102.0,
        ];
        for time in times {
            detector.record("test_op", PerformanceMetrics::new(time, 512.0, 1000.0));
        }

        // Significant regression
        let result =
            detector.check_regression("test_op", PerformanceMetrics::new(150.0, 512.0, 1000.0));

        assert!(result.regression_detected);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_global_detector() {
        let detector = global_detector();
        detector.clear_all_history();

        for _ in 0..10 {
            detector.record("global_test", PerformanceMetrics::new(100.0, 512.0, 1000.0));
        }

        let ops = detector.tracked_operations();
        assert!(ops.contains(&"global_test".to_string()));
    }
}
