//! Performance trend analysis functionality

use super::super::reporting::PerformanceReport;
use std::collections::HashMap;

/// Trend analyzer for performance time-series analysis
///
/// Analyzes performance trends over time to identify performance
/// regressions, improvements, and patterns in sparse tensor operations.
#[derive(Debug)]
pub struct TrendAnalyzer {
    /// Historical performance reports
    reports: Vec<PerformanceReport>,
    /// Trend analysis cache
    trend_cache: HashMap<String, TrendAnalysis>,
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalyzer {
    /// Create a new trend analyzer
    pub fn new() -> Self {
        Self {
            reports: Vec::new(),
            trend_cache: HashMap::new(),
        }
    }

    /// Add a performance report for trend analysis
    pub fn add_report(&mut self, report: PerformanceReport) {
        self.reports.push(report);
        self.trend_cache.clear(); // Invalidate cache
    }

    /// Analyze performance trend for a specific operation
    pub fn analyze_operation_trend(&self, operation: &str) -> Option<TrendAnalysis> {
        if let Some(cached) = self.trend_cache.get(operation) {
            return Some(cached.clone());
        }

        if self.reports.len() < 2 {
            return None;
        }

        let mut performance_points = Vec::new();
        for (i, report) in self.reports.iter().enumerate() {
            if let Some(stats) = report.operation_statistics.get(operation) {
                performance_points.push((i, stats.avg_time().as_secs_f64() * 1000.0));
            }
        }

        if performance_points.len() < 2 {
            return None;
        }

        let trend_direction = self.calculate_trend_direction(&performance_points);
        let trend_strength = self.calculate_trend_strength(&performance_points);
        let performance_change = self.calculate_performance_change(&performance_points);

        let analysis = TrendAnalysis {
            operation: operation.to_string(),
            trend_direction,
            trend_strength,
            performance_change_percent: performance_change,
            data_points: performance_points.len(),
            confidence: self.calculate_trend_confidence(&performance_points),
        };

        Some(analysis)
    }

    /// Get comprehensive trend summary for all operations
    pub fn get_trend_summary(&self) -> Vec<TrendAnalysis> {
        let mut trends = Vec::new();

        if self.reports.is_empty() {
            return trends;
        }

        // Get all unique operations from the latest report
        let latest_report = &self.reports[self.reports.len() - 1];
        for operation in latest_report.operation_statistics.keys() {
            if let Some(trend) = self.analyze_operation_trend(operation) {
                trends.push(trend);
            }
        }

        trends
    }

    /// Detect performance regressions
    pub fn detect_regressions(&self, threshold_percent: f64) -> Vec<String> {
        let mut regressions = Vec::new();

        for trend in self.get_trend_summary() {
            if trend.trend_direction == TrendDirection::Declining
                && trend.performance_change_percent < -threshold_percent
                && trend.confidence > 0.7
            {
                regressions.push(format!(
                    "Operation '{}' shows {:.1}% performance regression",
                    trend.operation, -trend.performance_change_percent
                ));
            }
        }

        regressions
    }

    /// Detect performance improvements
    pub fn detect_improvements(&self, threshold_percent: f64) -> Vec<String> {
        let mut improvements = Vec::new();

        for trend in self.get_trend_summary() {
            if trend.trend_direction == TrendDirection::Improving
                && trend.performance_change_percent > threshold_percent
                && trend.confidence > 0.7
            {
                improvements.push(format!(
                    "Operation '{}' shows {:.1}% performance improvement",
                    trend.operation, trend.performance_change_percent
                ));
            }
        }

        improvements
    }

    /// Calculate trend direction from performance points
    fn calculate_trend_direction(&self, points: &[(usize, f64)]) -> TrendDirection {
        if points.len() < 2 {
            return TrendDirection::Stable;
        }

        let first = points[0].1;
        let last = points[points.len() - 1].1;
        let change_percent = ((last - first) / first) * 100.0;

        if change_percent > 5.0 {
            TrendDirection::Declining // Higher time = worse performance
        } else if change_percent < -5.0 {
            TrendDirection::Improving // Lower time = better performance
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate trend strength (0-1)
    fn calculate_trend_strength(&self, points: &[(usize, f64)]) -> f64 {
        if points.len() < 3 {
            return 0.5;
        }

        // Calculate linear regression coefficient of determination (R²)
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| *x as f64).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| (*x as f64) * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        let sum_y2: f64 = points.iter().map(|(_, y)| y.powi(2)).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator == 0.0 {
            return 0.0;
        }

        let correlation = numerator / denominator;
        correlation.abs().min(1.0) // R value, strength is absolute correlation
    }

    /// Calculate overall performance change percentage
    fn calculate_performance_change(&self, points: &[(usize, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        let first = points[0].1;
        let last = points[points.len() - 1].1;

        if first == 0.0 {
            return 0.0;
        }

        // For performance, lower time = better, so we invert the calculation
        -((last - first) / first) * 100.0
    }

    /// Calculate confidence in trend analysis
    fn calculate_trend_confidence(&self, points: &[(usize, f64)]) -> f64 {
        let data_points_score = (points.len() as f64 / 10.0).min(1.0);
        let trend_strength = self.calculate_trend_strength(points);

        (data_points_score + trend_strength) / 2.0
    }
}

/// Performance trend analysis result
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Operation being analyzed
    pub operation: String,
    /// Direction of the performance trend
    pub trend_direction: TrendDirection,
    /// Strength of the trend (0-1)
    pub trend_strength: f64,
    /// Overall performance change percentage
    pub performance_change_percent: f64,
    /// Number of data points used
    pub data_points: usize,
    /// Confidence in the analysis (0-1)
    pub confidence: f64,
}

/// Direction of performance trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Performance is improving over time
    Improving,
    /// Performance is declining over time
    Declining,
    /// Performance is stable with no clear trend
    Stable,
}

impl TrendAnalysis {
    /// Get a human-readable description of the trend
    pub fn description(&self) -> String {
        let direction_desc = match self.trend_direction {
            TrendDirection::Improving => "improving",
            TrendDirection::Declining => "declining",
            TrendDirection::Stable => "stable",
        };

        format!(
            "Operation '{}' shows {} performance ({:+.1}% change, {:.0}% confidence)",
            self.operation,
            direction_desc,
            self.performance_change_percent,
            self.confidence * 100.0
        )
    }

    /// Check if this trend indicates a significant change
    pub fn is_significant(&self, threshold_percent: f64, min_confidence: f64) -> bool {
        self.performance_change_percent.abs() > threshold_percent
            && self.confidence > min_confidence
    }
}
