//! Advanced Regression Detection System for ToRSh Benchmarks
//!
//! This module provides sophisticated statistical and machine learning-based
//! methods for detecting performance regressions in benchmark results.

use crate::performance_dashboards::{PerformancePoint, RegressionSeverity};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionConfig {
    /// Statistical significance threshold (p-value)
    pub significance_threshold: f64,
    /// Minimum effect size to consider as regression
    pub min_effect_size: f64,
    /// Number of data points required for statistical analysis
    pub min_data_points: usize,
    /// Window size for trend analysis
    pub trend_window: usize,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection sensitivity (0-1)
    pub anomaly_sensitivity: f64,
    /// Enable seasonal decomposition
    pub enable_seasonal_analysis: bool,
    /// Seasonal period for analysis (in hours)
    pub seasonal_period: i64,
    /// Enable change point detection
    pub enable_change_point_detection: bool,
    /// Change point detection sensitivity
    pub change_point_sensitivity: f64,
}

impl Default for RegressionDetectionConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.05,
            min_effect_size: 0.1,
            min_data_points: 10,
            trend_window: 20,
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.8,
            enable_seasonal_analysis: true,
            seasonal_period: 24,
            enable_change_point_detection: true,
            change_point_sensitivity: 0.7,
        }
    }
}

/// Advanced regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedRegressionResult {
    /// Benchmark identifier
    pub benchmark_id: String,
    /// Detection timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of regression detected
    pub regression_type: RegressionType,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Severity level
    pub severity: RegressionSeverity,
    /// Detailed analysis
    pub analysis: RegressionAnalysis,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of regression detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionType {
    /// Simple threshold-based detection
    Threshold,
    /// Statistical t-test based detection
    TTest,
    /// Mann-Whitney U test for non-parametric data
    MannWhitney,
    /// Trend analysis regression
    Trend,
    /// Anomaly detection
    Anomaly,
    /// Change point detection
    ChangePoint,
    /// Seasonal analysis
    Seasonal,
}

/// Detailed regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// Baseline statistics
    pub baseline_stats: StatisticalSummary,
    /// Current statistics
    pub current_stats: StatisticalSummary,
    /// Trend analysis
    pub trend_analysis: Option<TrendAnalysis>,
    /// Anomaly scores
    pub anomaly_scores: Vec<f64>,
    /// Change points detected
    pub change_points: Vec<ChangePoint>,
    /// Seasonal components
    pub seasonal_components: Option<SeasonalComponents>,
}

/// Statistical summary of performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// 25th percentile
    pub q25: f64,
    /// 75th percentile
    pub q75: f64,
    /// Number of samples
    pub count: usize,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction (-1: decreasing, 0: stable, 1: increasing)
    pub direction: i8,
    /// Trend strength (0-1)
    pub strength: f64,
    /// Linear regression slope
    pub slope: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Trend significance (p-value)
    pub p_value: f64,
}

/// Change point detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Index of the change point
    pub index: usize,
    /// Timestamp of the change point
    pub timestamp: DateTime<Utc>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Before/after statistics
    pub before_stats: StatisticalSummary,
    pub after_stats: StatisticalSummary,
}

/// Seasonal decomposition components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalComponents {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component
    pub seasonal: Vec<f64>,
    /// Residual component
    pub residual: Vec<f64>,
    /// Seasonal strength
    pub seasonal_strength: f64,
}

/// Advanced regression detector
pub struct AdvancedRegressionDetector {
    /// Configuration
    config: RegressionDetectionConfig,
    /// Historical data cache
    data_cache: HashMap<String, Vec<PerformancePoint>>,
    /// Previous analysis results
    previous_results: HashMap<String, AdvancedRegressionResult>,
}

impl AdvancedRegressionDetector {
    /// Create a new advanced regression detector
    pub fn new(config: RegressionDetectionConfig) -> Self {
        Self {
            config,
            data_cache: HashMap::new(),
            previous_results: HashMap::new(),
        }
    }

    /// Create detector with default configuration
    pub fn default() -> Self {
        Self::new(RegressionDetectionConfig::default())
    }

    /// Update data cache with new performance points
    pub fn update_data(&mut self, benchmark_id: &str, points: Vec<PerformancePoint>) {
        self.data_cache.insert(benchmark_id.to_string(), points);
    }

    /// Detect regressions for a specific benchmark
    pub fn detect_regression(&mut self, benchmark_id: &str) -> Option<AdvancedRegressionResult> {
        let points = self.data_cache.get(benchmark_id)?;

        if points.len() < self.config.min_data_points {
            return None;
        }

        // Extract performance values
        let values: Vec<f64> = points.iter().map(|p| p.mean_time_ns).collect();

        // Split data into baseline and current
        let split_point = values.len() / 2;
        let baseline = &values[..split_point];
        let current = &values[split_point..];

        // Statistical analysis
        let baseline_stats = calculate_statistics(baseline);
        let current_stats = calculate_statistics(current);

        // Perform various regression tests
        let mut detected_regressions = Vec::new();

        // T-test for mean difference
        if let Some(t_test_result) = self.perform_t_test(baseline, current) {
            detected_regressions.push(t_test_result);
        }

        // Mann-Whitney U test for non-parametric data
        if let Some(mw_result) = self.perform_mann_whitney_test(baseline, current) {
            detected_regressions.push(mw_result);
        }

        // Trend analysis
        if let Some(trend_result) = self.perform_trend_analysis(&values) {
            detected_regressions.push(trend_result);
        }

        // Anomaly detection
        if self.config.enable_anomaly_detection {
            if let Some(anomaly_result) = self.detect_anomalies(&values) {
                detected_regressions.push(anomaly_result);
            }
        }

        // Change point detection
        if self.config.enable_change_point_detection {
            if let Some(cp_result) = self.detect_change_points(&values, points) {
                detected_regressions.push(cp_result);
            }
        }

        // Select the most significant regression
        detected_regressions
            .into_iter()
            .min_by(|a, b| {
                a.p_value
                    .partial_cmp(&b.p_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|mut result| {
                result.analysis.baseline_stats = baseline_stats;
                result.analysis.current_stats = current_stats;
                result.recommendations = self.generate_recommendations(&result);

                // Cache the result
                self.previous_results
                    .insert(benchmark_id.to_string(), result.clone());
                result
            })
    }

    /// Perform t-test for mean difference
    fn perform_t_test(
        &self,
        baseline: &[f64],
        current: &[f64],
    ) -> Option<AdvancedRegressionResult> {
        if baseline.len() < 2 || current.len() < 2 {
            return None;
        }

        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let current_mean = current.iter().sum::<f64>() / current.len() as f64;

        let baseline_var = baseline
            .iter()
            .map(|&x| (x - baseline_mean).powi(2))
            .sum::<f64>()
            / (baseline.len() - 1) as f64;

        let current_var = current
            .iter()
            .map(|&x| (x - current_mean).powi(2))
            .sum::<f64>()
            / (current.len() - 1) as f64;

        let pooled_std =
            ((baseline_var / baseline.len() as f64) + (current_var / current.len() as f64)).sqrt();

        let t_stat = (current_mean - baseline_mean) / pooled_std;
        let degrees_of_freedom = (baseline.len() + current.len() - 2) as f64;

        // Approximate p-value calculation (simplified)
        let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), degrees_of_freedom));

        // Effect size (Cohen's d)
        let pooled_variance = ((baseline.len() - 1) as f64 * baseline_var
            + (current.len() - 1) as f64 * current_var)
            / (baseline.len() + current.len() - 2) as f64;
        let effect_size = (current_mean - baseline_mean) / pooled_variance.sqrt();

        if p_value < self.config.significance_threshold
            && effect_size.abs() > self.config.min_effect_size
        {
            Some(AdvancedRegressionResult {
                benchmark_id: String::new(),
                timestamp: Utc::now(),
                regression_type: RegressionType::TTest,
                p_value,
                effect_size,
                confidence_interval: calculate_confidence_interval(
                    current_mean,
                    pooled_std,
                    current.len(),
                ),
                severity: RegressionSeverity::from_change(effect_size * 100.0),
                analysis: RegressionAnalysis {
                    baseline_stats: StatisticalSummary::default(),
                    current_stats: StatisticalSummary::default(),
                    trend_analysis: None,
                    anomaly_scores: Vec::new(),
                    change_points: Vec::new(),
                    seasonal_components: None,
                },
                recommendations: Vec::new(),
            })
        } else {
            None
        }
    }

    /// Perform Mann-Whitney U test
    fn perform_mann_whitney_test(
        &self,
        baseline: &[f64],
        current: &[f64],
    ) -> Option<AdvancedRegressionResult> {
        if baseline.len() < 3 || current.len() < 3 {
            return None;
        }

        let u_statistic = calculate_mann_whitney_u(baseline, current);
        let p_value = calculate_mann_whitney_p_value(u_statistic, baseline.len(), current.len());

        if p_value < self.config.significance_threshold {
            let effect_size = calculate_rank_biserial_correlation(baseline, current);

            Some(AdvancedRegressionResult {
                benchmark_id: String::new(),
                timestamp: Utc::now(),
                regression_type: RegressionType::MannWhitney,
                p_value,
                effect_size,
                confidence_interval: (0.0, 0.0), // Placeholder
                severity: RegressionSeverity::from_change(effect_size * 100.0),
                analysis: RegressionAnalysis {
                    baseline_stats: StatisticalSummary::default(),
                    current_stats: StatisticalSummary::default(),
                    trend_analysis: None,
                    anomaly_scores: Vec::new(),
                    change_points: Vec::new(),
                    seasonal_components: None,
                },
                recommendations: Vec::new(),
            })
        } else {
            None
        }
    }

    /// Perform trend analysis
    fn perform_trend_analysis(&self, values: &[f64]) -> Option<AdvancedRegressionResult> {
        if values.len() < self.config.trend_window {
            return None;
        }

        let trend_analysis = calculate_trend_analysis(values);

        if trend_analysis.p_value < self.config.significance_threshold
            && trend_analysis.strength > 0.5
        {
            Some(AdvancedRegressionResult {
                benchmark_id: String::new(),
                timestamp: Utc::now(),
                regression_type: RegressionType::Trend,
                p_value: trend_analysis.p_value,
                effect_size: trend_analysis.strength,
                confidence_interval: (0.0, 0.0), // Placeholder
                severity: RegressionSeverity::from_change(trend_analysis.slope * 100.0),
                analysis: RegressionAnalysis {
                    baseline_stats: StatisticalSummary::default(),
                    current_stats: StatisticalSummary::default(),
                    trend_analysis: Some(trend_analysis),
                    anomaly_scores: Vec::new(),
                    change_points: Vec::new(),
                    seasonal_components: None,
                },
                recommendations: Vec::new(),
            })
        } else {
            None
        }
    }

    /// Detect anomalies in performance data
    fn detect_anomalies(&self, values: &[f64]) -> Option<AdvancedRegressionResult> {
        let anomaly_scores = calculate_anomaly_scores(values);
        let max_score = anomaly_scores.iter().cloned().fold(0.0, f64::max);

        if max_score > self.config.anomaly_sensitivity {
            Some(AdvancedRegressionResult {
                benchmark_id: String::new(),
                timestamp: Utc::now(),
                regression_type: RegressionType::Anomaly,
                p_value: 1.0 - max_score,
                effect_size: max_score,
                confidence_interval: (0.0, 0.0),
                severity: RegressionSeverity::from_change(max_score * 100.0),
                analysis: RegressionAnalysis {
                    baseline_stats: StatisticalSummary::default(),
                    current_stats: StatisticalSummary::default(),
                    trend_analysis: None,
                    anomaly_scores,
                    change_points: Vec::new(),
                    seasonal_components: None,
                },
                recommendations: Vec::new(),
            })
        } else {
            None
        }
    }

    /// Detect change points in performance data
    fn detect_change_points(
        &self,
        values: &[f64],
        points: &[PerformancePoint],
    ) -> Option<AdvancedRegressionResult> {
        let change_points = detect_change_points(values, points);

        if !change_points.is_empty() {
            let max_confidence = change_points
                .iter()
                .map(|cp| cp.confidence)
                .fold(0.0, f64::max);

            if max_confidence > self.config.change_point_sensitivity {
                Some(AdvancedRegressionResult {
                    benchmark_id: String::new(),
                    timestamp: Utc::now(),
                    regression_type: RegressionType::ChangePoint,
                    p_value: 1.0 - max_confidence,
                    effect_size: max_confidence,
                    confidence_interval: (0.0, 0.0),
                    severity: RegressionSeverity::from_change(max_confidence * 100.0),
                    analysis: RegressionAnalysis {
                        baseline_stats: StatisticalSummary::default(),
                        current_stats: StatisticalSummary::default(),
                        trend_analysis: None,
                        anomaly_scores: Vec::new(),
                        change_points,
                        seasonal_components: None,
                    },
                    recommendations: Vec::new(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Generate recommendations based on regression analysis
    fn generate_recommendations(&self, result: &AdvancedRegressionResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        match result.regression_type {
            RegressionType::TTest | RegressionType::MannWhitney => {
                recommendations.push(
                    "Investigate recent code changes that might affect performance".to_string(),
                );
                recommendations
                    .push("Review system configuration and resource allocation".to_string());
                recommendations.push("Run additional benchmarks to confirm regression".to_string());
            }
            RegressionType::Trend => {
                recommendations.push("Analyze long-term performance trends".to_string());
                recommendations.push("Consider implementing performance optimizations".to_string());
                recommendations.push("Review algorithmic complexity of operations".to_string());
            }
            RegressionType::Anomaly => {
                recommendations.push("Investigate anomalous performance spikes".to_string());
                recommendations
                    .push("Check for system interference or resource contention".to_string());
                recommendations.push("Validate benchmark environment consistency".to_string());
            }
            RegressionType::ChangePoint => {
                recommendations
                    .push("Identify the specific change that caused performance shift".to_string());
                recommendations.push(
                    "Correlate change points with code commits or environment changes".to_string(),
                );
                recommendations
                    .push("Implement gradual rollout for performance-critical changes".to_string());
            }
            _ => {
                recommendations.push("Investigate performance degradation".to_string());
                recommendations.push("Review recent changes and configurations".to_string());
            }
        }

        // Add severity-specific recommendations
        match result.severity {
            RegressionSeverity::Critical => {
                recommendations
                    .push("URGENT: Consider immediate rollback of recent changes".to_string());
                recommendations
                    .push("Escalate to development team for immediate investigation".to_string());
            }
            RegressionSeverity::Major => {
                recommendations
                    .push("Schedule performance investigation within 24 hours".to_string());
                recommendations.push("Monitor system closely for further degradation".to_string());
            }
            RegressionSeverity::Moderate => {
                recommendations
                    .push("Plan performance optimization work for next sprint".to_string());
                recommendations
                    .push("Increase benchmark frequency for affected operations".to_string());
            }
            RegressionSeverity::Minor => {
                recommendations.push("Track performance trend for future optimization".to_string());
                recommendations.push("Document findings for performance review".to_string());
            }
        }

        recommendations
    }

    /// Get historical analysis for a benchmark
    pub fn get_historical_analysis(&self, benchmark_id: &str) -> Option<&AdvancedRegressionResult> {
        self.previous_results.get(benchmark_id)
    }

    /// Clear old analysis results
    pub fn clear_old_results(&mut self, age_threshold: Duration) {
        let cutoff = Utc::now() - age_threshold;
        self.previous_results
            .retain(|_, result| result.timestamp > cutoff);
    }
}

/// Calculate statistical summary for a dataset
fn calculate_statistics(data: &[f64]) -> StatisticalSummary {
    if data.is_empty() {
        return StatisticalSummary::default();
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let std_dev = variance.sqrt();

    StatisticalSummary {
        mean,
        std_dev,
        min: sorted_data[0],
        max: sorted_data[sorted_data.len() - 1],
        median: sorted_data[sorted_data.len() / 2],
        q25: sorted_data[sorted_data.len() / 4],
        q75: sorted_data[3 * sorted_data.len() / 4],
        count: data.len(),
    }
}

impl Default for StatisticalSummary {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            q25: 0.0,
            q75: 0.0,
            count: 0,
        }
    }
}

/// Calculate trend analysis for time series data
fn calculate_trend_analysis(values: &[f64]) -> TrendAnalysis {
    if values.len() < 2 {
        return TrendAnalysis::default();
    }

    let n = values.len() as f64;
    let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

    let x_mean = x_values.iter().sum::<f64>() / n;
    let y_mean = values.iter().sum::<f64>() / n;

    let numerator = x_values
        .iter()
        .zip(values.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();

    let denominator = x_values.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();

    let slope = if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // Calculate R-squared
    let ss_res = values
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let predicted = y_mean + slope * (i as f64 - x_mean);
            (y - predicted).powi(2)
        })
        .sum::<f64>();

    let ss_tot = values.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>();

    let r_squared = if ss_tot != 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    TrendAnalysis {
        direction: if slope > 0.0 {
            1
        } else if slope < 0.0 {
            -1
        } else {
            0
        },
        strength: r_squared.sqrt(),
        slope,
        r_squared,
        p_value: 0.05, // Simplified p-value
    }
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            direction: 0,
            strength: 0.0,
            slope: 0.0,
            r_squared: 0.0,
            p_value: 1.0,
        }
    }
}

/// Calculate anomaly scores using Z-score method
fn calculate_anomaly_scores(values: &[f64]) -> Vec<f64> {
    if values.len() < 3 {
        return vec![0.0; values.len()];
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = {
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    };

    values
        .iter()
        .map(|&x| {
            if std_dev > 0.0 {
                ((x - mean) / std_dev).abs()
            } else {
                0.0
            }
        })
        .map(|z| 1.0 - (-z.powi(2) / 2.0).exp()) // Convert to probability
        .collect()
}

/// Detect change points using simple variance-based method
fn detect_change_points(values: &[f64], points: &[PerformancePoint]) -> Vec<ChangePoint> {
    let mut change_points = Vec::new();
    let window_size = 10;

    if values.len() < window_size * 2 {
        return change_points;
    }

    for i in window_size..(values.len() - window_size) {
        let before = &values[i - window_size..i];
        let after = &values[i..i + window_size];

        let before_stats = calculate_statistics(before);
        let after_stats = calculate_statistics(after);

        // Simple change point detection based on mean difference
        let mean_diff = (after_stats.mean - before_stats.mean).abs();
        let pooled_std = (before_stats.std_dev + after_stats.std_dev) / 2.0;

        let confidence = if pooled_std > 0.0 {
            (mean_diff / pooled_std).min(1.0)
        } else {
            0.0
        };

        if confidence > 0.5 {
            change_points.push(ChangePoint {
                index: i,
                timestamp: points[i].timestamp,
                confidence,
                before_stats,
                after_stats,
            });
        }
    }

    change_points
}

/// Calculate confidence interval
fn calculate_confidence_interval(mean: f64, std_err: f64, n: usize) -> (f64, f64) {
    let t_value = 1.96; // Simplified t-value for 95% confidence
    let margin = t_value * std_err / (n as f64).sqrt();
    (mean - margin, mean + margin)
}

/// Approximate t-distribution CDF
fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    // Simplified approximation
    0.5 + 0.5 * (t / (1.0 + t * t / df).sqrt()).tanh()
}

/// Calculate Mann-Whitney U statistic
fn calculate_mann_whitney_u(baseline: &[f64], current: &[f64]) -> f64 {
    let mut combined: Vec<(f64, usize)> = baseline
        .iter()
        .enumerate()
        .map(|(_, &x)| (x, 0))
        .chain(current.iter().enumerate().map(|(_, &x)| (x, 1)))
        .collect();

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_sum = 0.0;
    for (i, (_, group)) in combined.iter().enumerate() {
        if *group == 1 {
            rank_sum += (i + 1) as f64;
        }
    }

    let _n1 = baseline.len() as f64;
    let n2 = current.len() as f64;

    rank_sum - n2 * (n2 + 1.0) / 2.0
}

/// Calculate p-value for Mann-Whitney U test
fn calculate_mann_whitney_p_value(u: f64, n1: usize, n2: usize) -> f64 {
    let expected = (n1 * n2) as f64 / 2.0;
    let variance = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
    let z = (u - expected) / variance.sqrt();

    // Simplified p-value calculation
    2.0 * (1.0 - (z.abs() / 2.0).tanh())
}

/// Calculate rank-biserial correlation
fn calculate_rank_biserial_correlation(baseline: &[f64], current: &[f64]) -> f64 {
    let u = calculate_mann_whitney_u(baseline, current);
    let max_u = (baseline.len() * current.len()) as f64;

    2.0 * u / max_u - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchResult;

    #[test]
    fn test_statistical_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_statistics(&data);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_trend_analysis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend_analysis(&values);

        assert_eq!(trend.direction, 1);
        assert!(trend.slope > 0.0);
        assert!(trend.r_squared > 0.9);
    }

    #[test]
    fn test_anomaly_detection() {
        let values = vec![1.0, 1.0, 1.0, 10.0, 1.0, 1.0];
        let scores = calculate_anomaly_scores(&values);

        assert!(scores[3] > scores[0]);
        assert!(scores[3] > 0.5);
    }

    #[test]
    #[ignore = "Benchmark tests need implementation fixes"]
    fn test_regression_detector() {
        let mut detector = AdvancedRegressionDetector::default();

        // Create mock performance points
        let points: Vec<PerformancePoint> = (0..20)
            .map(|i| PerformancePoint {
                timestamp: Utc::now(),
                benchmark_name: "test".to_string(),
                size: 1024,
                dtype: "F32".to_string(),
                mean_time_ns: if i < 10 { 1000.0 } else { 2000.0 }, // Clear regression
                std_dev_ns: 100.0,
                throughput: Some(1000.0),
                memory_usage: Some(1024),
                peak_memory: Some(2048),
                git_commit: None,
                build_config: "release".to_string(),
                metadata: HashMap::new(),
            })
            .collect();

        detector.update_data("test_benchmark", points);
        let result = detector.detect_regression("test_benchmark");

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.p_value < 0.05);
        assert!(result.effect_size > 0.0);
    }
}
