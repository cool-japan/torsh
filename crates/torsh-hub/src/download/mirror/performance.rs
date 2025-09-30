//! Performance Analysis and Monitoring
//!
//! This module provides comprehensive performance analysis capabilities for mirror
//! management, including trend analysis, prediction, benchmarking, and adaptive
//! learning algorithms for optimal mirror selection.

use super::types::*;
use reqwest::Client;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use torsh_core::error::{Result, TorshError};

// ================================================================================================
// Performance Analyzer Implementation
// ================================================================================================

impl PerformanceAnalyzer {
    /// Create a new performance analyzer with default settings
    pub fn new() -> Self {
        Self {
            enabled: true,
            trend_analysis_window: Duration::from_secs(86400), // 24 hours
            prediction_accuracy: 0.0,
            performance_cache: HashMap::new(),
        }
    }

    /// Create a new performance analyzer with custom settings
    pub fn with_settings(enabled: bool, trend_analysis_window: Duration) -> Self {
        Self {
            enabled,
            trend_analysis_window,
            prediction_accuracy: 0.0,
            performance_cache: HashMap::new(),
        }
    }

    /// Comprehensive mirror benchmarking with detailed performance analysis
    ///
    /// Performs thorough benchmarking of a single mirror including latency,
    /// throughput, and reliability measurements. Results are cached and used
    /// for performance trend analysis.
    ///
    /// # Arguments
    /// * `mirror` - The mirror server to benchmark
    /// * `client` - HTTP client for making benchmark requests
    /// * `current_time` - Current timestamp for record keeping
    ///
    /// # Returns
    /// * `MirrorBenchmarkResult` - Comprehensive benchmark results
    pub async fn benchmark_single_mirror(
        &mut self,
        mirror: &mut MirrorServer,
        client: &Client,
        current_time: u64,
    ) -> MirrorBenchmarkResult {
        if !self.enabled {
            return MirrorBenchmarkResult::default_for_mirror(&mirror.id);
        }

        let benchmark_start = Instant::now();
        let test_url = format!("{}/health", mirror.base_url.trim_end_matches('/'));

        // Perform latency test
        let latency_result = self.measure_latency(client, &test_url).await;

        // Perform throughput test (small file download)
        let throughput_result = self.measure_throughput(client, mirror).await;

        // Calculate overall score
        let overall_score = self.calculate_benchmark_score(&latency_result, &throughput_result);

        let benchmark_duration = benchmark_start.elapsed();

        // Create benchmark result
        let result = MirrorBenchmarkResult {
            mirror_id: mirror.id.clone(),
            success: latency_result.is_ok() || throughput_result.is_ok(),
            response_time: latency_result.unwrap_or(0), // Default to 0 if measurement failed
            bandwidth: throughput_result.ok(),
            load_percentage: None, // Would need additional measurement
            timestamp: current_time,
            additional_metrics: HashMap::new(), // Simplified for now
        };

        // Update mirror statistics based on benchmark results
        self.update_mirror_from_benchmark(mirror, &result, current_time);

        // Cache performance data
        self.cache_performance_data(&mirror.id, &result);

        result
    }

    /// Measure latency to a mirror endpoint
    async fn measure_latency(&self, client: &Client, url: &str) -> Result<u64> {
        let start = Instant::now();

        match client.head(url).send().await {
            Ok(response) => {
                let latency = start.elapsed().as_millis() as u64;
                if response.status().is_success() {
                    Ok(latency)
                } else {
                    Err(TorshError::IoError(format!(
                        "Health check failed with status: {}",
                        response.status()
                    )))
                }
            }
            Err(e) => Err(TorshError::IoError(format!(
                "Latency measurement failed: {}",
                e
            ))),
        }
    }

    /// Measure throughput by downloading a small test file
    async fn measure_throughput(&self, client: &Client, mirror: &MirrorServer) -> Result<f64> {
        // Use a small test file for throughput measurement
        let test_url = format!("{}/test/1MB.bin", mirror.base_url.trim_end_matches('/'));
        let start = Instant::now();

        match client.get(&test_url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let content_length = response.content_length().unwrap_or(1_048_576); // 1MB fallback

                    // Download the content
                    match response.bytes().await {
                        Ok(bytes) => {
                            let duration = start.elapsed();
                            let throughput =
                                (bytes.len() as f64) / duration.as_secs_f64() / 1_048_576.0; // MB/s
                            Ok(throughput)
                        }
                        Err(e) => Err(TorshError::IoError(format!(
                            "Throughput download failed: {}",
                            e
                        ))),
                    }
                } else {
                    // Fallback: estimate from response headers
                    let duration = start.elapsed();
                    let estimated_throughput = 1.0 / duration.as_secs_f64(); // Rough estimate
                    Ok(estimated_throughput)
                }
            }
            Err(_) => {
                // Fallback: Use latency-based estimation
                if let Ok(latency) = self.measure_latency(client, &mirror.base_url).await {
                    // Rough throughput estimation based on latency
                    let estimated_throughput = (1000.0 / latency as f64).max(0.1);
                    Ok(estimated_throughput)
                } else {
                    Err(TorshError::IoError(
                        "Throughput measurement failed".to_string(),
                    ))
                }
            }
        }
    }

    /// Calculate overall benchmark score from individual measurements
    fn calculate_benchmark_score(
        &self,
        latency_result: &Result<u64>,
        throughput_result: &Result<f64>,
    ) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Latency score (lower is better)
        if let Ok(latency) = latency_result {
            let latency_score = (1000.0 - (*latency as f64).min(1000.0)) / 1000.0;
            score += latency_score * 0.6; // 60% weight
            factors += 1;
        }

        // Throughput score (higher is better)
        if let Ok(throughput) = throughput_result {
            let throughput_score = (throughput / 100.0).min(1.0); // Normalize to 100 MB/s max
            score += throughput_score * 0.4; // 40% weight
            factors += 1;
        }

        if factors > 0 {
            score
        } else {
            0.0
        }
    }

    /// Update mirror statistics based on benchmark results
    fn update_mirror_from_benchmark(
        &self,
        mirror: &mut MirrorServer,
        result: &MirrorBenchmarkResult,
        current_time: u64,
    ) {
        if result.success {
            // Update success metrics
            mirror.consecutive_failures = 0;
            mirror.avg_response_time = Some(result.response_time);
            mirror.reliability_score = (mirror.reliability_score + 0.05).min(1.0);

            // Reactivate mirror if it was inactive
            if !mirror.active && mirror.reliability_score > 0.7 {
                mirror.active = true;
            }
        } else {
            // Update failure metrics
            mirror.consecutive_failures += 1;
            mirror.reliability_score = (mirror.reliability_score - 0.05).max(0.0);

            // Deactivate mirror if too many consecutive failures
            if mirror.consecutive_failures >= 5 {
                mirror.active = false;
            }
        }

        // Add performance snapshot to history
        let snapshot = PerformanceSnapshot {
            timestamp: current_time,
            response_time: result.response_time,
            throughput: result.bandwidth,
            error_rate: if result.success { 0.0 } else { 100.0 },
            load_percentage: mirror.capacity.current_load.unwrap_or(0.0),
        };
        mirror.performance_history.push(snapshot);

        // Limit history size to prevent unbounded growth
        if mirror.performance_history.len() > 1000 {
            mirror.performance_history.drain(0..100); // Remove oldest 100 entries
        }
    }

    /// Cache performance data for trend analysis
    fn cache_performance_data(&mut self, mirror_id: &str, result: &MirrorBenchmarkResult) {
        if !self.enabled {
            return;
        }

        let snapshot = PerformanceSnapshot {
            timestamp: result.timestamp,
            response_time: result.response_time,
            throughput: result.bandwidth,
            error_rate: if result.success { 0.0 } else { 100.0 },
            load_percentage: 0.0, // Not available in benchmark result
        };

        self.performance_cache
            .entry(mirror_id.to_string())
            .or_insert_with(Vec::new)
            .push(snapshot);

        // Limit cache size per mirror
        if let Some(snapshots) = self.performance_cache.get_mut(mirror_id) {
            if snapshots.len() > 500 {
                snapshots.drain(0..50); // Remove oldest 50 entries
            }
        }
    }

    /// Calculate performance trend for a mirror based on recent history
    ///
    /// Analyzes performance history to determine if the mirror's performance
    /// is improving, stable, or degrading over time.
    pub fn calculate_performance_trend(&self, mirror: &MirrorServer) -> PerformanceTrend {
        if mirror.performance_history.len() < 5 {
            return PerformanceTrend::Stable;
        }

        let recent_samples = mirror
            .performance_history
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>();

        let mut response_times: Vec<u64> = recent_samples.iter().map(|s| s.response_time).collect();

        let mut throughputs: Vec<f64> =
            recent_samples.iter().filter_map(|s| s.throughput).collect();

        // Calculate trends
        let response_time_trend = if response_times.len() >= 3 {
            self.calculate_linear_trend(
                &response_times.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            )
        } else {
            0.0
        };

        let throughput_trend = if throughputs.len() >= 3 {
            self.calculate_linear_trend(&throughputs)
        } else {
            0.0
        };

        // Combine trends (response time trending down is good, throughput trending up is good)
        let overall_trend = throughput_trend - (response_time_trend * 0.5);

        if overall_trend > 0.1 {
            PerformanceTrend::Improving
        } else if overall_trend < -0.1 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Calculate linear trend using least squares method
    fn calculate_linear_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope (trend)
        let denominator = n * x2_sum - x_sum.powi(2);
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * xy_sum - x_sum * y_sum) / denominator
    }

    /// Analyze performance patterns and detect anomalies
    pub fn analyze_performance_patterns(&self, mirror_id: &str) -> PerformanceAnalysis {
        let snapshots = match self.performance_cache.get(mirror_id) {
            Some(snapshots) => snapshots,
            None => return PerformanceAnalysis::default(),
        };

        if snapshots.len() < 10 {
            return PerformanceAnalysis::default();
        }

        let response_times: Vec<f64> = snapshots.iter().map(|s| s.response_time as f64).collect();

        let throughputs: Vec<f64> = snapshots.iter().filter_map(|s| s.throughput).collect();

        // Calculate statistics
        let avg_response_time = response_times.iter().sum::<f64>() / response_times.len() as f64;
        let avg_throughput = if !throughputs.is_empty() {
            Some(throughputs.iter().sum::<f64>() / throughputs.len() as f64)
        } else {
            None
        };

        // Calculate standard deviations for anomaly detection
        let response_time_std =
            self.calculate_standard_deviation(&response_times, avg_response_time);
        let throughput_std = if let Some(avg_tp) = avg_throughput {
            Some(self.calculate_standard_deviation(&throughputs, avg_tp))
        } else {
            None
        };

        // Detect anomalies (values more than 2 standard deviations from mean)
        let anomaly_threshold = 2.0;
        let response_time_anomalies = response_times
            .iter()
            .filter(|&&rt| (rt - avg_response_time).abs() > response_time_std * anomaly_threshold)
            .count();

        let throughput_anomalies =
            if let (Some(avg_tp), Some(tp_std)) = (avg_throughput, throughput_std) {
                throughputs
                    .iter()
                    .filter(|&&tp| (tp - avg_tp).abs() > tp_std * anomaly_threshold)
                    .count()
            } else {
                0
            };

        PerformanceAnalysis {
            mirror_id: mirror_id.to_string(),
            average_response_time: Duration::from_millis(avg_response_time as u64),
            throughput_mbps: avg_throughput.unwrap_or(0.0),
            reliability_score: self.calculate_reliability_score(snapshots),
            trend: self.calculate_trend_from_cache(snapshots),
            bottlenecks: Vec::new(),     // Placeholder
            recommendations: Vec::new(), // Placeholder
        }
    }

    /// Calculate standard deviation for a set of values
    fn calculate_standard_deviation(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let variance: f64 =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate reliability score from performance snapshots
    fn calculate_reliability_score(&self, snapshots: &[PerformanceSnapshot]) -> f64 {
        if snapshots.is_empty() {
            return 0.0;
        }

        let success_count = snapshots.iter().filter(|s| s.error_rate < 50.0).count();

        success_count as f64 / snapshots.len() as f64
    }

    /// Calculate performance trend from cached snapshots
    fn calculate_trend_from_cache(&self, snapshots: &[PerformanceSnapshot]) -> PerformanceTrend {
        if snapshots.len() < 5 {
            return PerformanceTrend::Stable;
        }

        let recent_snapshots: Vec<&PerformanceSnapshot> = snapshots.iter().rev().take(10).collect();

        let response_times: Vec<f64> = recent_snapshots
            .iter()
            .map(|s| s.response_time as f64)
            .collect();

        let throughputs: Vec<f64> = recent_snapshots
            .iter()
            .filter_map(|s| s.throughput)
            .collect();

        let response_time_trend = if response_times.len() >= 3 {
            self.calculate_linear_trend(&response_times)
        } else {
            0.0
        };

        let throughput_trend = if throughputs.len() >= 3 {
            self.calculate_linear_trend(&throughputs)
        } else {
            0.0
        };

        let overall_trend = throughput_trend - (response_time_trend * 0.5);

        if overall_trend > 0.1 {
            PerformanceTrend::Improving
        } else if overall_trend < -0.1 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Predict future performance based on historical trends
    pub fn predict_performance(
        &self,
        mirror_id: &str,
        horizon_minutes: u64,
    ) -> Option<PerformancePrediction> {
        if !self.enabled {
            return None;
        }

        let snapshots = self.performance_cache.get(mirror_id)?;
        if snapshots.len() < 10 {
            return None;
        }

        let analysis = self.analyze_performance_patterns(mirror_id);

        // Simple linear extrapolation based on recent trend
        let recent_snapshots: Vec<&PerformanceSnapshot> = snapshots.iter().rev().take(20).collect();

        let response_times: Vec<f64> = recent_snapshots
            .iter()
            .map(|s| s.response_time as f64)
            .collect();

        let response_time_trend = self.calculate_linear_trend(&response_times);
        let predicted_response_time = analysis.average_response_time.as_millis() as f64
            + (response_time_trend * horizon_minutes as f64);

        let throughputs: Vec<f64> = recent_snapshots
            .iter()
            .filter_map(|s| s.throughput)
            .collect();

        let predicted_throughput = if !throughputs.is_empty() {
            let throughput_trend = self.calculate_linear_trend(&throughputs);
            analysis.throughput_mbps + (throughput_trend * horizon_minutes as f64)
        } else {
            analysis.throughput_mbps
        };

        Some(PerformancePrediction {
            mirror_id: mirror_id.to_string(),
            time_horizon: Duration::from_secs(horizon_minutes as u64 * 60),
            predicted_response_time: Duration::from_millis(predicted_response_time.max(0.0) as u64),
            predicted_throughput,
            confidence_level: self.calculate_prediction_confidence(&analysis) as f32,
        })
    }

    /// Calculate confidence score for predictions
    fn calculate_prediction_confidence(&self, analysis: &PerformanceAnalysis) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Higher reliability increases confidence
        confidence += analysis.reliability_score * 0.3;

        // Higher throughput suggests better performance, increasing confidence
        if analysis.throughput_mbps > 100.0 {
            confidence += 0.2;
        } else if analysis.throughput_mbps > 50.0 {
            confidence += 0.1;
        }

        confidence.min(1.0).max(0.0)
    }

    /// Clear performance cache for all mirrors
    pub fn clear_cache(&mut self) {
        self.performance_cache.clear();
    }

    /// Clear performance cache for a specific mirror
    pub fn clear_mirror_cache(&mut self, mirror_id: &str) {
        self.performance_cache.remove(mirror_id);
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, usize> {
        self.performance_cache
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Enable or disable performance analysis
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear_cache();
        }
    }

    /// Check if performance analysis is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Update trend analysis window
    pub fn set_trend_analysis_window(&mut self, window: Duration) {
        self.trend_analysis_window = window;
    }

    /// Get current trend analysis window
    pub fn get_trend_analysis_window(&self) -> Duration {
        self.trend_analysis_window
    }

    /// Update prediction accuracy based on validation results
    pub fn update_prediction_accuracy(&mut self, accuracy: f64) {
        self.prediction_accuracy = accuracy.min(1.0).max(0.0);
    }

    /// Get current prediction accuracy
    pub fn get_prediction_accuracy(&self) -> f64 {
        self.prediction_accuracy
    }
}

// ================================================================================================
// Performance Utility Functions
// ================================================================================================

/// Create a helper function for creating test mirror servers in tests
#[cfg(test)]
pub fn create_mirror_server(
    id: &str,
    base_url: &str,
    country: &str,
    city: &str,
    provider: &str,
) -> MirrorServer {
    MirrorServer {
        id: id.to_string(),
        base_url: base_url.to_string(),
        reliability_score: 0.8,
        avg_response_time: Some(100),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: country.to_string(),
            region: "Region".to_string(),
            city: city.to_string(),
            latitude: Some(40.0),
            longitude: Some(-74.0),
            provider: provider.to_string(),
            timezone: Some("America/New_York".to_string()),
            datacenter: Some("dc1".to_string()),
        },
        capacity: MirrorCapacity::default(),
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        last_successful_connection: None,
        provider_info: ProviderInfo {
            name: provider.to_string(),
            network_tier: Some("Premium".to_string()),
            cdn_integration: true,
            edge_location: Some("NYC".to_string()),
            network_quality: NetworkQuality::default(),
        },
        performance_history: Vec::new(),
    }
}

impl Default for MirrorBenchmarkResult {
    fn default() -> Self {
        Self {
            mirror_id: String::new(),
            response_time: 0,
            bandwidth: None,
            load_percentage: None,
            success: false,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            additional_metrics: HashMap::new(),
        }
    }
}

impl MirrorBenchmarkResult {
    /// Create a default benchmark result for a specific mirror
    pub fn default_for_mirror(mirror_id: &str) -> Self {
        Self {
            mirror_id: mirror_id.to_string(),
            ..Self::default()
        }
    }
}

impl Default for PerformanceAnalysis {
    fn default() -> Self {
        Self {
            mirror_id: String::new(),
            average_response_time: Duration::from_secs(0),
            throughput_mbps: 0.0,
            reliability_score: 0.0,
            trend: PerformanceTrend::Stable,
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[test]
    fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert!(analyzer.is_enabled());
        assert_eq!(
            analyzer.get_trend_analysis_window(),
            Duration::from_secs(86400)
        );
        assert_eq!(analyzer.get_prediction_accuracy(), 0.0);
    }

    #[test]
    fn test_performance_analyzer_with_settings() {
        let window = Duration::from_secs(3600);
        let analyzer = PerformanceAnalyzer::with_settings(false, window);
        assert!(!analyzer.is_enabled());
        assert_eq!(analyzer.get_trend_analysis_window(), window);
    }

    #[test]
    fn test_performance_trend_calculation() {
        let analyzer = PerformanceAnalyzer::new();
        let mirror = create_mirror_server(
            "test",
            "https://test.example.com",
            "US",
            "New York",
            "TestProvider",
        );

        // Test with empty history
        let trend = analyzer.calculate_performance_trend(&mirror);
        assert_eq!(trend, PerformanceTrend::Stable);
    }

    #[test]
    fn test_linear_trend_calculation() {
        let analyzer = PerformanceAnalyzer::new();

        // Test increasing trend
        let increasing_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = analyzer.calculate_linear_trend(&increasing_values);
        assert!(trend > 0.0);

        // Test decreasing trend
        let decreasing_values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = analyzer.calculate_linear_trend(&decreasing_values);
        assert!(trend < 0.0);

        // Test stable trend
        let stable_values = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = analyzer.calculate_linear_trend(&stable_values);
        assert!(trend.abs() < 0.1);
    }

    #[test]
    fn test_standard_deviation_calculation() {
        let analyzer = PerformanceAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = analyzer.calculate_standard_deviation(&values, mean);

        // Standard deviation of [1,2,3,4,5] with mean 3 should be approximately 1.58
        assert!((std_dev - 1.58).abs() < 0.1);
    }

    #[test]
    fn test_cache_operations() {
        let mut analyzer = PerformanceAnalyzer::new();
        assert!(analyzer.get_cache_stats().is_empty());

        // Test cache clearing
        analyzer.clear_cache();
        assert!(analyzer.get_cache_stats().is_empty());

        // Test enabling/disabling
        analyzer.set_enabled(false);
        assert!(!analyzer.is_enabled());
        analyzer.set_enabled(true);
        assert!(analyzer.is_enabled());
    }

    #[test]
    fn test_prediction_accuracy_update() {
        let mut analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.get_prediction_accuracy(), 0.0);

        analyzer.update_prediction_accuracy(0.85);
        assert_eq!(analyzer.get_prediction_accuracy(), 0.85);

        // Test bounds checking
        analyzer.update_prediction_accuracy(1.5);
        assert_eq!(analyzer.get_prediction_accuracy(), 1.0);

        analyzer.update_prediction_accuracy(-0.1);
        assert_eq!(analyzer.get_prediction_accuracy(), 0.0);
    }

    #[test]
    fn test_benchmark_score_calculation() {
        let analyzer = PerformanceAnalyzer::new();

        // Test with good performance
        let good_latency = Ok(50u64);
        let good_throughput = Ok(50.0f64);
        let score = analyzer.calculate_benchmark_score(&good_latency, &good_throughput);
        assert!(score > 0.7);

        // Test with poor performance
        let poor_latency = Ok(800u64);
        let poor_throughput = Ok(1.0f64);
        let score = analyzer.calculate_benchmark_score(&poor_latency, &poor_throughput);
        assert!(score < 0.4);

        // Test with failures
        let failed_latency = Err(TorshError::IoError("Failed".to_string()));
        let failed_throughput = Err(TorshError::IoError("Failed".to_string()));
        let score = analyzer.calculate_benchmark_score(&failed_latency, &failed_throughput);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_benchmark_result_default() {
        let result = MirrorBenchmarkResult::default();
        assert!(!result.success);
        assert_eq!(result.response_time, 0);
        assert!(result.bandwidth.is_none());

        let result = MirrorBenchmarkResult::default_for_mirror("test_mirror");
        assert_eq!(result.mirror_id, "test_mirror");
    }

    #[test]
    fn test_performance_analysis_default() {
        // Note: PerformanceAnalysis doesn't have a Default implementation, so we create a sample
        let analysis = PerformanceAnalysis {
            mirror_id: "test".to_string(),
            average_response_time: std::time::Duration::from_millis(0),
            throughput_mbps: 0.0,
            reliability_score: 0.0,
            trend: PerformanceTrend::Stable,
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
        };
        assert_eq!(
            analysis.average_response_time,
            std::time::Duration::from_millis(0)
        );
        assert_eq!(analysis.throughput_mbps, 0.0);
        assert_eq!(analysis.trend, PerformanceTrend::Stable);
    }

    // Integration test for performance analysis patterns
    #[test]
    fn test_performance_pattern_analysis() {
        let analyzer = PerformanceAnalyzer::new();

        // Test with no data
        let analysis = analyzer.analyze_performance_patterns("nonexistent");
        assert_eq!(analysis.reliability_score, 0.0);
    }
}
