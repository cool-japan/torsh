//! Performance regression detection system
//!
//! This module provides automated detection of performance regressions by comparing
//! current profiling results with historical baselines, statistical analysis of
//! performance trends, and alerting for significant degradations.

use crate::{ProfileEvent, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::TorshError;

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Minimum number of samples required for baseline
    pub min_baseline_samples: usize,
    /// Maximum age of baseline data in days
    pub max_baseline_age_days: u32,
    /// Threshold for detecting performance regression (percentage)
    pub regression_threshold_percent: f64,
    /// Threshold for detecting performance improvement (percentage)
    pub improvement_threshold_percent: f64,
    /// Statistical significance level (p-value)
    pub significance_level: f64,
    /// Whether to use adaptive thresholds based on historical variance
    pub adaptive_thresholds: bool,
    /// Window size for rolling baseline updates
    pub rolling_window_size: usize,
    /// Whether to enable outlier detection and filtering
    pub outlier_detection: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            min_baseline_samples: 10,
            max_baseline_age_days: 30,
            regression_threshold_percent: 5.0,  // 5% slowdown
            improvement_threshold_percent: 5.0, // 5% speedup
            significance_level: 0.05,           // 95% confidence
            adaptive_thresholds: true,
            rolling_window_size: 100,
            outlier_detection: true,
        }
    }
}

/// Performance baseline for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub metric_name: String,
    pub category: String,
    pub samples: Vec<f64>,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub sample_count: usize,
}

impl PerformanceBaseline {
    /// Create a new baseline from samples
    pub fn new(metric_name: String, category: String, samples: Vec<f64>) -> Self {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_samples.len() % 2 == 0 {
            (sorted_samples[sorted_samples.len() / 2 - 1]
                + sorted_samples[sorted_samples.len() / 2])
                / 2.0
        } else {
            sorted_samples[sorted_samples.len() / 2]
        };

        let percentile_95 = sorted_samples[(sorted_samples.len() as f64 * 0.95) as usize - 1];
        let percentile_99 = sorted_samples[(sorted_samples.len() as f64 * 0.99) as usize - 1];

        Self {
            metric_name,
            category,
            samples,
            mean,
            std_dev,
            median,
            percentile_95,
            percentile_99,
            last_updated: chrono::Utc::now(),
            sample_count: sorted_samples.len(),
        }
    }

    /// Update baseline with new samples
    pub fn update(&mut self, new_samples: Vec<f64>, rolling_window_size: usize) {
        self.samples.extend(new_samples);

        // Keep only the latest samples within the rolling window
        if self.samples.len() > rolling_window_size {
            let excess = self.samples.len() - rolling_window_size;
            self.samples.drain(0..excess);
        }

        // Recalculate statistics
        let mean = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
        let variance = self.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.samples.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_samples = self.samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_samples.len() % 2 == 0 {
            (sorted_samples[sorted_samples.len() / 2 - 1]
                + sorted_samples[sorted_samples.len() / 2])
                / 2.0
        } else {
            sorted_samples[sorted_samples.len() / 2]
        };

        let percentile_95 = sorted_samples[(sorted_samples.len() as f64 * 0.95) as usize - 1];
        let percentile_99 = sorted_samples[(sorted_samples.len() as f64 * 0.99) as usize - 1];

        self.mean = mean;
        self.std_dev = std_dev;
        self.median = median;
        self.percentile_95 = percentile_95;
        self.percentile_99 = percentile_99;
        self.last_updated = chrono::Utc::now();
        self.sample_count = self.samples.len();
    }

    /// Check if baseline is stale
    pub fn is_stale(&self, max_age_days: u32) -> bool {
        let age = chrono::Utc::now().signed_duration_since(self.last_updated);
        age.num_days() > max_age_days as i64
    }

    /// Remove outliers from samples using IQR method
    pub fn remove_outliers(&mut self) {
        if self.samples.len() < 4 {
            return; // Need at least 4 samples for IQR
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted.len() / 4;
        let q3_idx = (3 * sorted.len()) / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        self.samples
            .retain(|&x| x >= lower_bound && x <= upper_bound);
    }
}

/// Result of regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub metric_name: String,
    pub category: String,
    pub baseline_mean: f64,
    pub current_mean: f64,
    pub change_percent: f64,
    pub is_regression: bool,
    pub is_improvement: bool,
    pub is_significant: bool,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub severity: RegressionSeverity,
    pub recommendation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Severity levels for regressions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Critical,    // > 20% degradation
    High,        // 10-20% degradation
    Medium,      // 5-10% degradation
    Low,         // < 5% degradation
    None,        // No significant change
    Improvement, // Performance improvement
}

impl RegressionSeverity {
    /// Determine severity based on change percentage
    pub fn from_change_percent(change_percent: f64) -> Self {
        if change_percent < -5.0 {
            Self::Improvement
        } else if change_percent < 5.0 {
            Self::None
        } else if change_percent < 10.0 {
            Self::Low
        } else if change_percent < 20.0 {
            Self::Medium
        } else if change_percent < 50.0 {
            Self::High
        } else {
            Self::Critical
        }
    }
}

/// Regression detection system
pub struct RegressionDetector {
    config: RegressionConfig,
    baselines: HashMap<String, PerformanceBaseline>,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
        }
    }

    /// Create a new regression detector with default configuration
    pub fn with_defaults() -> Self {
        let config = RegressionConfig {
            min_baseline_samples: 5,
            max_baseline_age_days: 30,
            regression_threshold_percent: 10.0,
            improvement_threshold_percent: 5.0,
            significance_level: 0.05,
            adaptive_thresholds: true,
            rolling_window_size: 100,
            outlier_detection: true,
        };
        Self::new(config)
    }

    /// Add or update a baseline for a metric
    pub fn update_baseline(
        &mut self,
        metric_name: &str,
        category: &str,
        samples: Vec<f64>,
    ) -> TorshResult<()> {
        let key = format!("{category}::{metric_name}");

        match self.baselines.get_mut(&key) {
            Some(baseline) => {
                baseline.update(samples, self.config.rolling_window_size);
                if self.config.outlier_detection {
                    baseline.remove_outliers();
                }
            }
            None => {
                let mut baseline = PerformanceBaseline::new(
                    metric_name.to_string(),
                    category.to_string(),
                    samples,
                );
                if self.config.outlier_detection {
                    baseline.remove_outliers();
                }
                self.baselines.insert(key, baseline);
            }
        }

        Ok(())
    }

    /// Load baselines from a file
    pub fn load_baselines(&mut self, filename: &str) -> TorshResult<()> {
        let data = std::fs::read_to_string(filename)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to read file: {e}")))?;

        let baselines: HashMap<String, PerformanceBaseline> = serde_json::from_str(&data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to parse baselines: {e}")))?;

        self.baselines = baselines;
        Ok(())
    }

    /// Save baselines to a file
    pub fn save_baselines(&self, filename: &str) -> TorshResult<()> {
        let data = serde_json::to_string_pretty(&self.baselines).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize baselines: {e}"))
        })?;

        std::fs::write(filename, data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    /// Detect regressions in profiling events
    pub fn detect_regressions(
        &self,
        events: &[ProfileEvent],
    ) -> TorshResult<Vec<RegressionResult>> {
        let mut results = Vec::new();

        // Group events by category and metric
        let mut metrics: HashMap<String, Vec<f64>> = HashMap::new();

        for event in events {
            // Duration metrics
            let duration_key = format!("{}::duration_us", event.category);
            metrics
                .entry(duration_key)
                .or_default()
                .push(event.duration_us as f64);

            // FLOPS metrics (if available)
            if let Some(flops) = event.flops {
                if flops > 0 {
                    let flops_key = format!("{}::flops", event.category);
                    metrics.entry(flops_key).or_default().push(flops as f64);
                }
            }

            // Bytes transferred metrics (if available)
            if let Some(bytes_transferred) = event.bytes_transferred {
                if bytes_transferred > 0 {
                    let bytes_key = format!("{}::bytes_transferred", event.category);
                    metrics
                        .entry(bytes_key)
                        .or_default()
                        .push(bytes_transferred as f64);
                }
            }
        }

        // Check each metric against its baseline
        for (metric_key, values) in metrics.iter() {
            if let Some(baseline) = self.baselines.get(metric_key) {
                if baseline.sample_count >= self.config.min_baseline_samples
                    && !baseline.is_stale(self.config.max_baseline_age_days)
                {
                    let result = self.analyze_metric(baseline, values)?;
                    results.push(result);
                }
            }
        }

        results.sort_by(|a, b| {
            // Sort by severity, then by change magnitude

            match (a.severity, b.severity) {
                (RegressionSeverity::Critical, _) => std::cmp::Ordering::Less,
                (_, RegressionSeverity::Critical) => std::cmp::Ordering::Greater,
                (RegressionSeverity::High, RegressionSeverity::High) => {
                    b.change_percent.partial_cmp(&a.change_percent).unwrap()
                }
                (RegressionSeverity::High, _) => std::cmp::Ordering::Less,
                (_, RegressionSeverity::High) => std::cmp::Ordering::Greater,
                _ => b.change_percent.partial_cmp(&a.change_percent).unwrap(),
            }
        });

        Ok(results)
    }

    /// Analyze a specific metric against its baseline
    fn analyze_metric(
        &self,
        baseline: &PerformanceBaseline,
        current_values: &[f64],
    ) -> TorshResult<RegressionResult> {
        if current_values.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No current values provided".to_string(),
            ));
        }

        let current_mean = current_values.iter().sum::<f64>() / current_values.len() as f64;
        let change_percent = ((current_mean - baseline.mean) / baseline.mean) * 100.0;

        // Perform statistical significance test (Welch's t-test)
        let current_variance = current_values
            .iter()
            .map(|x| (x - current_mean).powi(2))
            .sum::<f64>()
            / current_values.len() as f64;
        let current_std_dev = current_variance.sqrt();

        let (is_significant, p_value) = self.welch_t_test(
            baseline.mean,
            baseline.std_dev,
            baseline.sample_count,
            current_mean,
            current_std_dev,
            current_values.len(),
        );

        // Determine if this is a regression or improvement
        let threshold = if self.config.adaptive_thresholds {
            // Use baseline variance to adjust thresholds
            let cv = baseline.std_dev / baseline.mean; // Coefficient of variation
            self.config.regression_threshold_percent * (1.0 + cv)
        } else {
            self.config.regression_threshold_percent
        };

        let is_regression = change_percent > threshold && is_significant;
        let is_improvement =
            change_percent < -self.config.improvement_threshold_percent && is_significant;

        let severity = RegressionSeverity::from_change_percent(change_percent);

        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(
            current_mean,
            current_std_dev,
            current_values.len(),
            1.0 - self.config.significance_level,
        );

        // Generate recommendation
        let recommendation =
            self.generate_recommendation(baseline, current_mean, change_percent, severity);

        Ok(RegressionResult {
            metric_name: baseline.metric_name.clone(),
            category: baseline.category.clone(),
            baseline_mean: baseline.mean,
            current_mean,
            change_percent,
            is_regression,
            is_improvement,
            is_significant,
            p_value,
            confidence_interval,
            severity,
            recommendation,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Perform Welch's t-test for unequal variances
    fn welch_t_test(
        &self,
        mean1: f64,
        std1: f64,
        n1: usize,
        mean2: f64,
        std2: f64,
        n2: usize,
    ) -> (bool, f64) {
        if n1 < 2 || n2 < 2 {
            return (false, 1.0);
        }

        let var1 = std1.powi(2);
        let var2 = std2.powi(2);

        let se = ((var1 / n1 as f64) + (var2 / n2 as f64)).sqrt();

        if se == 0.0 {
            return (false, 1.0);
        }

        let t_stat = (mean1 - mean2).abs() / se;

        // Welch-Satterthwaite equation for degrees of freedom
        let df_num = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let df_denom = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64
            + (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let _df = df_num / df_denom;

        // Approximate p-value calculation (simplified)
        // In a real implementation, you would use a proper t-distribution CDF
        let p_value = if t_stat > 2.0 {
            0.01 // Highly significant
        } else if t_stat > 1.96 {
            0.05 // Significant at 5% level
        } else if t_stat > 1.0 {
            0.15 // Somewhat significant
        } else {
            0.5 // Not significant
        };

        let is_significant = p_value < self.config.significance_level;

        (is_significant, p_value)
    }

    /// Calculate confidence interval for the mean
    fn calculate_confidence_interval(
        &self,
        mean: f64,
        std_dev: f64,
        n: usize,
        confidence: f64,
    ) -> (f64, f64) {
        if n < 2 {
            return (mean, mean);
        }

        // Use t-distribution critical value (approximation)
        let t_critical = if confidence >= 0.99 {
            2.576
        } else if confidence >= 0.95 {
            1.96
        } else {
            1.645
        };

        let margin_of_error = t_critical * (std_dev / (n as f64).sqrt());

        (mean - margin_of_error, mean + margin_of_error)
    }

    /// Generate recommendation based on regression analysis
    fn generate_recommendation(
        &self,
        baseline: &PerformanceBaseline,
        current_mean: f64,
        change_percent: f64,
        severity: RegressionSeverity,
    ) -> String {
        match severity {
            RegressionSeverity::Critical => {
                format!(
                    "CRITICAL REGRESSION: {} performance degraded by {:.1}% (from {:.2} to {:.2}). Immediate investigation required. Consider reverting recent changes.",
                    baseline.metric_name, change_percent, baseline.mean, current_mean
                )
            }
            RegressionSeverity::High => {
                format!(
                    "HIGH REGRESSION: {} performance degraded by {:.1}% (from {:.2} to {:.2}). Review recent optimizations and profile hotspots.",
                    baseline.metric_name, change_percent, baseline.mean, current_mean
                )
            }
            RegressionSeverity::Medium => {
                format!(
                    "MEDIUM REGRESSION: {} performance degraded by {:.1}% (from {:.2} to {:.2}). Monitor trend and consider optimization.",
                    baseline.metric_name, change_percent, baseline.mean, current_mean
                )
            }
            RegressionSeverity::Low => {
                format!(
                    "MINOR REGRESSION: {} performance degraded by {:.1}% (from {:.2} to {:.2}). Monitor for trend continuation.",
                    baseline.metric_name, change_percent, baseline.mean, current_mean
                )
            }
            RegressionSeverity::Improvement => {
                format!(
                    "IMPROVEMENT: {} performance improved by {:.1}% (from {:.2} to {:.2}). Excellent work!",
                    baseline.metric_name, change_percent.abs(), baseline.mean, current_mean
                )
            }
            RegressionSeverity::None => {
                format!(
                    "NO CHANGE: {} performance is stable (change: {:.1}%). Continue monitoring.",
                    baseline.metric_name, change_percent
                )
            }
        }
    }

    /// Get summary of all baselines
    pub fn get_baseline_summary(&self) -> Vec<&PerformanceBaseline> {
        self.baselines.values().collect()
    }

    /// Clean up stale baselines
    pub fn cleanup_stale_baselines(&mut self) -> usize {
        let initial_count = self.baselines.len();
        self.baselines
            .retain(|_, baseline| !baseline.is_stale(self.config.max_baseline_age_days));
        initial_count - self.baselines.len()
    }
}

/// Create a new regression detector with default configuration
pub fn create_regression_detector() -> RegressionDetector {
    RegressionDetector::new(RegressionConfig::default())
}

/// Create a new regression detector with custom configuration
pub fn create_regression_detector_with_config(config: RegressionConfig) -> RegressionDetector {
    RegressionDetector::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileEvent;

    #[test]
    fn test_baseline_creation() {
        let samples = vec![100.0, 105.0, 95.0, 110.0, 98.0];
        let baseline = PerformanceBaseline::new(
            "test_metric".to_string(),
            "test_category".to_string(),
            samples,
        );

        assert_eq!(baseline.metric_name, "test_metric");
        assert_eq!(baseline.sample_count, 5);
        assert!((baseline.mean - 101.6).abs() < 0.1);
    }

    #[test]
    fn test_baseline_update() {
        let mut baseline = PerformanceBaseline::new(
            "test_metric".to_string(),
            "test_category".to_string(),
            vec![100.0, 105.0, 95.0],
        );

        baseline.update(vec![110.0, 98.0], 10);
        assert_eq!(baseline.sample_count, 5);
    }

    #[test]
    fn test_outlier_removal() {
        let mut baseline = PerformanceBaseline::new(
            "test_metric".to_string(),
            "test_category".to_string(),
            vec![100.0, 102.0, 98.0, 101.0, 1000.0], // 1000.0 is an outlier
        );

        baseline.remove_outliers();
        assert!(baseline.samples.len() < 5); // Outlier should be removed
        assert!(!baseline.samples.contains(&1000.0));
    }

    #[test]
    fn test_regression_detection() {
        let config = RegressionConfig {
            min_baseline_samples: 3,
            regression_threshold_percent: 10.0,
            ..Default::default()
        };

        let mut detector = RegressionDetector::new(config);

        // Add baseline
        detector
            .update_baseline("duration_us", "test_category", vec![100.0, 105.0, 95.0])
            .unwrap();

        // Create events that show regression - need multiple samples for statistical significance
        let events = vec![
            ProfileEvent {
                name: "test_event".to_string(),
                category: "test_category".to_string(),
                start_us: 0,
                duration_us: 120, // 20% slower than baseline
                thread_id: 1,
                operation_count: Some(1),
                flops: Some(0),
                bytes_transferred: Some(0),
                stack_trace: Some("test trace".to_string()),
            },
            ProfileEvent {
                name: "test_event".to_string(),
                category: "test_category".to_string(),
                start_us: 0,
                duration_us: 125, // 25% slower than baseline
                thread_id: 1,
                operation_count: Some(1),
                flops: Some(0),
                bytes_transferred: Some(0),
                stack_trace: Some("test trace".to_string()),
            },
        ];

        let results = detector.detect_regressions(&events).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_regression);
        assert!(results[0].change_percent > 10.0);
    }

    #[test]
    fn test_improvement_detection() {
        let config = RegressionConfig {
            min_baseline_samples: 3,
            improvement_threshold_percent: 5.0,
            ..Default::default()
        };

        let mut detector = RegressionDetector::new(config);

        // Add baseline
        detector
            .update_baseline("duration_us", "test_category", vec![100.0, 105.0, 95.0])
            .unwrap();

        // Create events that show improvement - need multiple samples for statistical significance
        let events = vec![
            ProfileEvent {
                name: "test_event".to_string(),
                category: "test_category".to_string(),
                start_us: 0,
                duration_us: 85, // 15% faster than baseline
                thread_id: 1,
                operation_count: Some(1),
                flops: Some(0),
                bytes_transferred: Some(0),
                stack_trace: Some("test trace".to_string()),
            },
            ProfileEvent {
                name: "test_event".to_string(),
                category: "test_category".to_string(),
                start_us: 0,
                duration_us: 80, // 20% faster than baseline
                thread_id: 1,
                operation_count: Some(1),
                flops: Some(0),
                bytes_transferred: Some(0),
                stack_trace: Some("test trace".to_string()),
            },
        ];

        let results = detector.detect_regressions(&events).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_improvement);
        assert!(results[0].change_percent < -5.0);
    }

    #[test]
    fn test_save_load_baselines() {
        let mut detector = create_regression_detector();
        detector
            .update_baseline("test_metric", "test_category", vec![100.0, 105.0, 95.0])
            .unwrap();

        let temp_file = "/tmp/test_baselines.json";
        detector.save_baselines(temp_file).unwrap();

        let mut new_detector = create_regression_detector();
        new_detector.load_baselines(temp_file).unwrap();

        assert_eq!(new_detector.baselines.len(), 1);
        assert!(new_detector
            .baselines
            .contains_key("test_category::test_metric"));

        // Clean up
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_severity_classification() {
        assert!(matches!(
            RegressionSeverity::from_change_percent(25.0),
            RegressionSeverity::High
        ));
        assert!(matches!(
            RegressionSeverity::from_change_percent(7.0),
            RegressionSeverity::Low
        ));
        assert!(matches!(
            RegressionSeverity::from_change_percent(2.0),
            RegressionSeverity::None
        ));
        assert!(matches!(
            RegressionSeverity::from_change_percent(-10.0),
            RegressionSeverity::Improvement
        ));
        assert!(matches!(
            RegressionSeverity::from_change_percent(60.0),
            RegressionSeverity::Critical
        ));
    }
}
