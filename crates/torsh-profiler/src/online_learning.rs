//! Online Learning Module for Real-time Performance Analysis
//!
//! This module provides online (incremental) learning algorithms for real-time
//! anomaly detection, performance prediction, and adaptive model updates without
//! requiring full dataset retraining.
//!
//! # Features
//!
//! - Online anomaly detection using incremental statistics
//! - Streaming K-means for dynamic clustering
//! - Online gradient descent for real-time prediction
//! - Adaptive threshold adjustment based on data drift
//! - Exponentially weighted moving averages for trend analysis
//! - Sliding window analysis for concept drift detection

use crate::{ProfileEvent, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use torsh_core::TorshError;

// ✅ SciRS2 Policy Compliance - Using scirs2-core for random number generation
use scirs2_core::random::{thread_rng, Normal, Random};

/// Configuration for online learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Window size for sliding window analysis
    pub window_size: usize,
    /// Learning rate for online gradient descent
    pub learning_rate: f64,
    /// Decay factor for exponentially weighted moving averages (0.0 to 1.0)
    pub ewma_decay: f64,
    /// Number of clusters for streaming K-means
    pub num_clusters: usize,
    /// Anomaly threshold in standard deviations
    pub anomaly_threshold: f64,
    /// Minimum samples before making predictions
    pub min_samples: usize,
    /// Enable concept drift detection
    pub drift_detection: bool,
    /// Drift detection sensitivity (0.0 to 1.0)
    pub drift_sensitivity: f64,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            learning_rate: 0.01,
            ewma_decay: 0.9,
            num_clusters: 5,
            anomaly_threshold: 3.0,
            min_samples: 30,
            drift_detection: true,
            drift_sensitivity: 0.05,
        }
    }
}

/// Online statistics tracker using Welford's algorithm for numerical stability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineStats {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
    min: f64,
    max: f64,
}

impl OnlineStats {
    /// Create a new online statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Update statistics with a new value using Welford's algorithm
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get the current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the current variance
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the current standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the current minimum
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get the current maximum
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Get the sample count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if a value is an anomaly based on z-score
    pub fn is_anomaly(&self, value: f64, threshold: f64) -> bool {
        if self.count < 2 {
            return false;
        }
        let z_score = (value - self.mean()).abs() / self.std_dev();
        z_score > threshold
    }

    /// Calculate z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        if self.count < 2 || self.std_dev() == 0.0 {
            return 0.0;
        }
        (value - self.mean()) / self.std_dev()
    }
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponentially Weighted Moving Average (EWMA) for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWMA {
    value: f64,
    decay: f64,
    initialized: bool,
}

impl EWMA {
    /// Create a new EWMA with given decay factor (0.0 to 1.0)
    pub fn new(decay: f64) -> Self {
        Self {
            value: 0.0,
            decay: decay.clamp(0.0, 1.0),
            initialized: false,
        }
    }

    /// Update EWMA with a new value
    pub fn update(&mut self, value: f64) {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = self.decay * self.value + (1.0 - self.decay) * value;
        }
    }

    /// Get the current EWMA value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Streaming centroid for online K-means
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCentroid {
    pub center: Vec<f64>,
    pub count: usize,
    pub sum: Vec<f64>,
}

impl StreamingCentroid {
    /// Create a new streaming centroid
    pub fn new(dimensions: usize) -> Self {
        Self {
            center: vec![0.0; dimensions],
            count: 0,
            sum: vec![0.0; dimensions],
        }
    }

    /// Update centroid with a new point
    pub fn update(&mut self, point: &[f64]) {
        self.count += 1;
        for (i, &val) in point.iter().enumerate() {
            self.sum[i] += val;
            self.center[i] = self.sum[i] / self.count as f64;
        }
    }

    /// Calculate distance to a point
    pub fn distance(&self, point: &[f64]) -> f64 {
        self.center
            .iter()
            .zip(point.iter())
            .map(|(c, p)| (c - p).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Online K-means clustering for streaming data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingKMeans {
    centroids: Vec<StreamingCentroid>,
    dimensions: usize,
}

impl StreamingKMeans {
    /// Create a new streaming K-means clusterer
    pub fn new(num_clusters: usize, dimensions: usize) -> Self {
        let centroids = (0..num_clusters)
            .map(|_| StreamingCentroid::new(dimensions))
            .collect();

        Self {
            centroids,
            dimensions,
        }
    }

    /// Update clustering with a new data point
    pub fn update(&mut self, point: &[f64]) -> TorshResult<usize> {
        if point.len() != self.dimensions {
            return Err(TorshError::InvalidArgument(format!(
                "Point dimension mismatch: expected {}, got {}",
                self.dimensions,
                point.len()
            )));
        }

        // Find nearest centroid
        let (cluster_id, _distance) = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.distance(point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .ok_or_else(|| TorshError::operation_error("No centroids available"))?;

        // Update the nearest centroid
        self.centroids[cluster_id].update(point);

        Ok(cluster_id)
    }

    /// Get cluster assignment for a point
    pub fn predict(&self, point: &[f64]) -> TorshResult<usize> {
        if point.len() != self.dimensions {
            return Err(TorshError::InvalidArgument(format!(
                "Point dimension mismatch: expected {}, got {}",
                self.dimensions,
                point.len()
            )));
        }

        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.distance(point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| TorshError::operation_error("No centroids available"))
    }

    /// Get centroids
    pub fn centroids(&self) -> &[StreamingCentroid] {
        &self.centroids
    }
}

/// Online anomaly detector using incremental statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineAnomalyDetector {
    config: OnlineLearningConfig,
    duration_stats: OnlineStats,
    memory_stats: OnlineStats,
    flops_stats: OnlineStats,
    duration_ewma: EWMA,
    memory_ewma: EWMA,
    recent_anomalies: VecDeque<AnomalyEvent>,
    total_samples: usize,
    anomaly_count: usize,
}

/// Anomaly event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub anomaly_type: AnomalyType,
    pub severity: f64,
    pub z_score: f64,
    pub expected_value: f64,
    pub actual_value: f64,
    pub explanation: String,
}

/// Type of anomaly detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    DurationSpike,
    MemorySpike,
    FlopsAnomaly,
    TrendDeviation,
}

impl std::fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyType::DurationSpike => write!(f, "Duration Spike"),
            AnomalyType::MemorySpike => write!(f, "Memory Spike"),
            AnomalyType::FlopsAnomaly => write!(f, "FLOPS Anomaly"),
            AnomalyType::TrendDeviation => write!(f, "Trend Deviation"),
        }
    }
}

impl OnlineAnomalyDetector {
    /// Create a new online anomaly detector
    pub fn new(config: OnlineLearningConfig) -> Self {
        Self {
            config: config.clone(),
            duration_stats: OnlineStats::new(),
            memory_stats: OnlineStats::new(),
            flops_stats: OnlineStats::new(),
            duration_ewma: EWMA::new(config.ewma_decay),
            memory_ewma: EWMA::new(config.ewma_decay),
            recent_anomalies: VecDeque::with_capacity(config.window_size),
            total_samples: 0,
            anomaly_count: 0,
        }
    }

    /// Process a new profile event and detect anomalies
    pub fn process_event(&mut self, event: &ProfileEvent) -> TorshResult<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();
        self.total_samples += 1;

        // Update statistics
        let duration = event.duration_us as f64;
        self.duration_stats.update(duration);
        self.duration_ewma.update(duration);

        if let Some(bytes) = event.bytes_transferred {
            let memory = bytes as f64;
            self.memory_stats.update(memory);
            self.memory_ewma.update(memory);
        }

        if let Some(flops) = event.flops {
            let flops_val = flops as f64;
            self.flops_stats.update(flops_val);
        }

        // Only detect anomalies after minimum samples
        if self.total_samples < self.config.min_samples {
            return Ok(anomalies);
        }

        // Check duration anomaly
        if self
            .duration_stats
            .is_anomaly(duration, self.config.anomaly_threshold)
        {
            let z_score = self.duration_stats.z_score(duration);
            let severity = z_score.abs() / self.config.anomaly_threshold;

            anomalies.push(AnomalyEvent {
                timestamp: chrono::Utc::now(),
                anomaly_type: AnomalyType::DurationSpike,
                severity,
                z_score,
                expected_value: self.duration_stats.mean(),
                actual_value: duration,
                explanation: format!(
                    "Operation duration ({:.2}μs) is {:.1}σ from mean ({:.2}μs)",
                    duration,
                    z_score.abs(),
                    self.duration_stats.mean()
                ),
            });
            self.anomaly_count += 1;
        }

        // Check memory anomaly
        if let Some(bytes) = event.bytes_transferred {
            let memory = bytes as f64;
            if self
                .memory_stats
                .is_anomaly(memory, self.config.anomaly_threshold)
            {
                let z_score = self.memory_stats.z_score(memory);
                let severity = z_score.abs() / self.config.anomaly_threshold;

                anomalies.push(AnomalyEvent {
                    timestamp: chrono::Utc::now(),
                    anomaly_type: AnomalyType::MemorySpike,
                    severity,
                    z_score,
                    expected_value: self.memory_stats.mean(),
                    actual_value: memory,
                    explanation: format!(
                        "Memory usage ({:.2} bytes) is {:.1}σ from mean ({:.2} bytes)",
                        memory,
                        z_score.abs(),
                        self.memory_stats.mean()
                    ),
                });
                self.anomaly_count += 1;
            }
        }

        // Check EWMA trend deviation
        if self.duration_ewma.is_initialized() {
            let ewma_value = self.duration_ewma.value();
            let deviation = (duration - ewma_value).abs() / ewma_value;
            if deviation > 0.5 {
                // 50% deviation from trend
                anomalies.push(AnomalyEvent {
                    timestamp: chrono::Utc::now(),
                    anomaly_type: AnomalyType::TrendDeviation,
                    severity: deviation,
                    z_score: 0.0,
                    expected_value: ewma_value,
                    actual_value: duration,
                    explanation: format!(
                        "Duration ({:.2}μs) deviates {:.1}% from recent trend ({:.2}μs)",
                        duration,
                        deviation * 100.0,
                        ewma_value
                    ),
                });
            }
        }

        // Store recent anomalies
        for anomaly in &anomalies {
            if self.recent_anomalies.len() >= self.config.window_size {
                self.recent_anomalies.pop_front();
            }
            self.recent_anomalies.push_back(anomaly.clone());
        }

        Ok(anomalies)
    }

    /// Get statistics summary
    pub fn get_stats(&self) -> OnlineAnomalyStats {
        OnlineAnomalyStats {
            total_samples: self.total_samples,
            anomaly_count: self.anomaly_count,
            anomaly_rate: if self.total_samples > 0 {
                self.anomaly_count as f64 / self.total_samples as f64
            } else {
                0.0
            },
            duration_mean: self.duration_stats.mean(),
            duration_std: self.duration_stats.std_dev(),
            memory_mean: self.memory_stats.mean(),
            memory_std: self.memory_stats.std_dev(),
            recent_anomaly_count: self.recent_anomalies.len(),
        }
    }

    /// Get recent anomalies
    pub fn recent_anomalies(&self) -> &VecDeque<AnomalyEvent> {
        &self.recent_anomalies
    }

    /// Export anomaly data
    pub fn export_json(&self) -> TorshResult<String> {
        serde_json::to_string_pretty(&self.get_stats())
            .map_err(|e| TorshError::operation_error(&format!("JSON export failed: {}", e)))
    }
}

/// Statistics summary for online anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineAnomalyStats {
    pub total_samples: usize,
    pub anomaly_count: usize,
    pub anomaly_rate: f64,
    pub duration_mean: f64,
    pub duration_std: f64,
    pub memory_mean: f64,
    pub memory_std: f64,
    pub recent_anomaly_count: usize,
}

/// Online gradient descent for real-time performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlinePredictor {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    sample_count: usize,
    loss_history: VecDeque<f64>,
    window_size: usize,
}

impl OnlinePredictor {
    /// Create a new online predictor
    pub fn new(num_features: usize, learning_rate: f64, window_size: usize) -> Self {
        Self {
            weights: vec![0.0; num_features],
            bias: 0.0,
            learning_rate,
            sample_count: 0,
            loss_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Update model with a new training sample
    pub fn update(&mut self, features: &[f64], target: f64) -> TorshResult<f64> {
        if features.len() != self.weights.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.weights.len(),
                features.len()
            )));
        }

        // Make prediction
        let prediction = self.predict(features)?;

        // Calculate error
        let error = target - prediction;

        // Update weights using gradient descent
        for (i, &feature) in features.iter().enumerate() {
            self.weights[i] += self.learning_rate * error * feature;
        }

        // Update bias
        self.bias += self.learning_rate * error;

        // Track loss
        let loss = error.powi(2);
        if self.loss_history.len() >= self.window_size {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);

        self.sample_count += 1;

        Ok(loss)
    }

    /// Make a prediction
    pub fn predict(&self, features: &[f64]) -> TorshResult<f64> {
        if features.len() != self.weights.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.weights.len(),
                features.len()
            )));
        }

        let prediction: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum::<f64>()
            + self.bias;

        Ok(prediction)
    }

    /// Get average recent loss
    pub fn average_loss(&self) -> f64 {
        if self.loss_history.is_empty() {
            return 0.0;
        }
        self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64
    }

    /// Get model statistics
    pub fn get_stats(&self) -> PredictorStats {
        PredictorStats {
            sample_count: self.sample_count,
            average_loss: self.average_loss(),
            weights: self.weights.clone(),
            bias: self.bias,
        }
    }
}

/// Statistics for online predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorStats {
    pub sample_count: usize,
    pub average_loss: f64,
    pub weights: Vec<f64>,
    pub bias: f64,
}

/// Concept drift detector using ADWIN (Adaptive Windowing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetector {
    window: VecDeque<f64>,
    sensitivity: f64,
    drift_detected: bool,
    last_drift_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl DriftDetector {
    /// Create a new drift detector
    pub fn new(sensitivity: f64) -> Self {
        Self {
            window: VecDeque::new(),
            sensitivity: sensitivity.clamp(0.0, 1.0),
            drift_detected: false,
            last_drift_time: None,
        }
    }

    /// Add a new value and check for drift
    pub fn add_value(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        self.drift_detected = self.detect_drift();

        if self.drift_detected {
            self.last_drift_time = Some(chrono::Utc::now());
            // Reset window on drift detection
            let keep_size = self.window.len() / 2;
            self.window.drain(0..keep_size);
        }

        self.drift_detected
    }

    /// Detect drift using statistical test
    fn detect_drift(&self) -> bool {
        if self.window.len() < 30 {
            return false;
        }

        // Split window in half and compare means
        let mid = self.window.len() / 2;
        let first_half: Vec<f64> = self.window.iter().take(mid).copied().collect();
        let second_half: Vec<f64> = self.window.iter().skip(mid).copied().collect();

        let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let var1 =
            first_half.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / first_half.len() as f64;
        let var2 =
            second_half.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / second_half.len() as f64;

        // Check if means are significantly different
        let threshold = self.sensitivity * (var1 + var2).sqrt();
        (mean1 - mean2).abs() > threshold
    }

    /// Check if drift was recently detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get last drift time
    pub fn last_drift_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.last_drift_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::new();

        // Add some values
        for value in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(*value);
        }

        assert_eq!(stats.mean(), 3.0);
        assert!((stats.variance() - 2.5).abs() < 0.01);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 5.0);
    }

    #[test]
    fn test_online_stats_anomaly_detection() {
        let mut stats = OnlineStats::new();

        // Add normal values
        for value in &[10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8] {
            stats.update(*value);
        }

        // Normal value should not be anomaly
        assert!(!stats.is_anomaly(10.0, 3.0));

        // Value far from mean should be anomaly
        assert!(stats.is_anomaly(30.0, 3.0));
    }

    #[test]
    fn test_ewma() {
        let mut ewma = EWMA::new(0.9);
        assert!(!ewma.is_initialized());

        ewma.update(10.0);
        assert!(ewma.is_initialized());
        assert_eq!(ewma.value(), 10.0);

        ewma.update(20.0);
        assert!((ewma.value() - 11.0).abs() < 0.1); // 0.9*10 + 0.1*20 = 11
    }

    #[test]
    fn test_streaming_kmeans() {
        let mut kmeans = StreamingKMeans::new(2, 2);

        // Add some points
        let points = vec![
            vec![1.0, 1.0],
            vec![1.5, 1.5],
            vec![10.0, 10.0],
            vec![10.5, 10.5],
        ];

        for point in &points {
            kmeans.update(point).unwrap();
        }

        // Points close to each other should be in the same cluster
        let cluster1 = kmeans.predict(&vec![1.0, 1.0]).unwrap();
        let cluster2 = kmeans.predict(&vec![1.5, 1.5]).unwrap();
        assert_eq!(cluster1, cluster2);

        let cluster3 = kmeans.predict(&vec![10.0, 10.0]).unwrap();
        let cluster4 = kmeans.predict(&vec![10.5, 10.5]).unwrap();
        assert_eq!(cluster3, cluster4);

        // Distant clusters should be different
        assert_ne!(cluster1, cluster3);
    }

    #[test]
    fn test_online_predictor() {
        let mut predictor = OnlinePredictor::new(2, 0.01, 100);

        // Train with simple linear relationship: y = 2*x1 + 3*x2
        for _ in 0..100 {
            let x1 = 1.0;
            let x2 = 2.0;
            let y = 2.0 * x1 + 3.0 * x2; // = 8.0
            predictor.update(&[x1, x2], y).unwrap();
        }

        // Make prediction
        let prediction = predictor.predict(&[1.0, 2.0]).unwrap();
        assert!((prediction - 8.0).abs() < 1.0); // Should be close to 8.0
    }

    #[test]
    fn test_drift_detector() {
        let mut detector = DriftDetector::new(0.05);

        // Add values from one distribution
        for _ in 0..30 {
            detector.add_value(10.0);
        }

        assert!(!detector.is_drift_detected());

        // Add values from different distribution
        for _ in 0..30 {
            if detector.add_value(20.0) {
                break;
            }
        }

        // Drift should be detected at some point
        assert!(detector.is_drift_detected());
    }

    #[test]
    fn test_online_anomaly_detector() {
        let config = OnlineLearningConfig::default();
        let mut detector = OnlineAnomalyDetector::new(config);

        // Get thread_id as usize
        let thread_id = format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_numeric())
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(1);

        // Create normal events
        for i in 0..50 {
            let event = ProfileEvent {
                name: format!("op_{}", i),
                category: "test".to_string(),
                thread_id,
                start_us: i as u64 * 1000,
                duration_us: 100, // Normal duration
                operation_count: Some(10),
                flops: Some(1000),
                bytes_transferred: Some(1024),
                stack_trace: None,
            };

            let _anomalies = detector.process_event(&event).unwrap();
        }

        // Create anomalous event
        let anomalous_event = ProfileEvent {
            name: "anomalous_op".to_string(),
            category: "test".to_string(),
            thread_id,
            start_us: 50000,
            duration_us: 1000, // Anomalous duration (10x normal)
            operation_count: Some(10),
            flops: Some(1000),
            bytes_transferred: Some(1024),
            stack_trace: None,
        };

        let anomalies = detector.process_event(&anomalous_event).unwrap();
        assert!(!anomalies.is_empty());

        let stats = detector.get_stats();
        assert!(stats.total_samples > 0);
        assert!(stats.anomaly_count > 0);
    }
}
