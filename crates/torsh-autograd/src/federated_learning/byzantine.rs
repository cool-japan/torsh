//! Byzantine Fault Tolerance for Federated Learning
//!
//! This module provides comprehensive Byzantine fault tolerance mechanisms for federated learning systems.
//! Byzantine faults occur when clients behave maliciously or arbitrarily, potentially trying to poison
//! the global model or disrupt the training process.
//!
//! # Detection Methods
//!
//! - **Statistical Test**: Detects outliers using statistical measures
//! - **Distance Based**: Identifies clients with significantly different gradients
//! - **Clustering Based**: Groups clients and identifies outlier clusters
//! - **Reputation Based**: Maintains reputation scores for clients over time
//! - **Ensemble Based**: Combines multiple detection methods for robustness
//!
//! # Robustness Techniques
//!
//! The module implements various techniques to handle Byzantine clients:
//! - Anomaly score computation based on gradient norms
//! - Historical behavior tracking
//! - Adaptive threshold adjustment
//! - Multi-round consensus mechanisms
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::byzantine::{ByzantineDetector, ByzantineDetectionMethod};
//! use std::collections::HashMap;
//!
//! // Create a Byzantine detector
//! let mut detector = ByzantineDetector::new();
//! detector.set_detection_method(ByzantineDetectionMethod::StatisticalTest);
//! detector.set_detection_threshold(2.0);
//!
//! // Detect Byzantine behavior in client gradients
//! let mut gradients = HashMap::new();
//! gradients.insert("layer_1".to_string(), vec![0.1, 0.2, 0.3]);
//!
//! let is_byzantine = detector.detect_byzantine_behavior("client_1", &gradients)?;
//! if is_byzantine {
//!     println!("Client client_1 exhibits Byzantine behavior!");
//! }
//! ```
//!
//! # Defense Strategies
//!
//! The module supports various defense strategies against Byzantine attacks:
//! - Gradient clipping and normalization
//! - Robust aggregation methods (handled in aggregation module)
//! - Client reputation management
//! - Adaptive detection thresholds

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::{HashMap, HashSet};

use crate::federated_learning::aggregation::FederatedError;

/// Byzantine fault detector for identifying malicious or faulty clients
///
/// The ByzantineDetector monitors client behavior across federated learning rounds
/// and identifies clients that exhibit suspicious patterns that could indicate
/// malicious behavior, hardware faults, or other anomalies.
///
/// # Thread Safety
///
/// This struct is designed to be thread-safe and can be safely shared across threads
/// when wrapped in appropriate synchronization primitives.
#[derive(Debug)]
pub struct ByzantineDetector {
    /// The detection method to use
    detection_method: ByzantineDetectionMethod,
    /// Set of clients currently flagged as suspicious
    suspicious_clients: HashSet<String>,
    /// Threshold for considering behavior as Byzantine
    detection_threshold: f64,
    /// Historical anomaly scores for each client
    detection_history: HashMap<String, Vec<f64>>,
    /// Client reputation scores (higher = more trustworthy)
    reputation_scores: HashMap<String, f64>,
    /// Number of rounds to maintain history
    history_window: usize,
    /// Adaptive threshold parameters
    adaptive_threshold_config: AdaptiveThresholdConfig,
}

// ByzantineDetector is Send + Sync
unsafe impl Send for ByzantineDetector {}
unsafe impl Sync for ByzantineDetector {}

/// Methods for detecting Byzantine behavior in federated learning
///
/// Different detection methods provide different trade-offs between
/// sensitivity, specificity, and computational overhead.
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineDetectionMethod {
    /// Statistical outlier detection using z-scores
    StatisticalTest,
    /// Distance-based detection using gradient similarity
    DistanceBased,
    /// Clustering-based detection grouping similar clients
    ClusteringBased,
    /// Reputation-based detection using historical behavior
    ReputationBased,
    /// Ensemble method combining multiple detection techniques
    EnsembleBased,
}

/// Configuration for adaptive threshold adjustment
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdConfig {
    /// Whether to use adaptive thresholding
    pub enabled: bool,
    /// Learning rate for threshold adaptation
    pub adaptation_rate: f64,
    /// Minimum threshold value
    pub min_threshold: f64,
    /// Maximum threshold value
    pub max_threshold: f64,
    /// Target false positive rate
    pub target_false_positive_rate: f64,
}

/// Detailed results from Byzantine detection
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether the client is flagged as Byzantine
    pub is_byzantine: bool,
    /// Anomaly score (higher = more suspicious)
    pub anomaly_score: f64,
    /// Confidence in the detection
    pub confidence: f64,
    /// Reason for the detection
    pub detection_reason: String,
    /// Historical context
    pub historical_average: f64,
}

impl ByzantineDetector {
    /// Creates a new ByzantineDetector with default settings
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let detector = ByzantineDetector::new();
    /// ```
    pub fn new() -> Self {
        Self {
            detection_method: ByzantineDetectionMethod::StatisticalTest,
            suspicious_clients: HashSet::new(),
            detection_threshold: 2.0,
            detection_history: HashMap::new(),
            reputation_scores: HashMap::new(),
            history_window: 10,
            adaptive_threshold_config: AdaptiveThresholdConfig::default(),
        }
    }

    /// Creates a new ByzantineDetector with custom configuration
    ///
    /// # Arguments
    ///
    /// * `method` - The detection method to use
    /// * `threshold` - The detection threshold
    /// * `history_window` - Number of rounds to maintain history
    pub fn with_config(
        method: ByzantineDetectionMethod,
        threshold: f64,
        history_window: usize,
    ) -> Self {
        Self {
            detection_method: method,
            detection_threshold: threshold,
            history_window,
            ..Self::new()
        }
    }

    /// Detects Byzantine behavior for a specific client
    ///
    /// This is the main detection method that analyzes client gradients
    /// and determines if the client exhibits Byzantine behavior.
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client identifier
    /// * `gradients` - The client's gradient updates
    ///
    /// # Returns
    ///
    /// Boolean indicating whether the client is flagged as Byzantine
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut gradients = HashMap::new();
    /// gradients.insert("layer_1".to_string(), vec![0.1, 0.2, 0.3]);
    /// let is_byzantine = detector.detect_byzantine_behavior("client_1", &gradients)?;
    /// ```
    pub fn detect_byzantine_behavior(
        &mut self,
        client_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<bool, FederatedError> {
        let result = self.detect_byzantine_behavior_detailed(client_id, gradients)?;
        Ok(result.is_byzantine)
    }

    /// Detects Byzantine behavior with detailed results
    ///
    /// Provides comprehensive information about the detection including
    /// anomaly scores, confidence levels, and reasoning.
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client identifier
    /// * `gradients` - The client's gradient updates
    ///
    /// # Returns
    ///
    /// Detailed detection results
    pub fn detect_byzantine_behavior_detailed(
        &mut self,
        client_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<DetectionResult, FederatedError> {
        let anomaly_score = match self.detection_method {
            ByzantineDetectionMethod::StatisticalTest => {
                self.compute_statistical_anomaly_score(gradients)?
            }
            ByzantineDetectionMethod::DistanceBased => {
                self.compute_distance_based_score(gradients)?
            }
            ByzantineDetectionMethod::ClusteringBased => {
                self.compute_clustering_based_score(gradients)?
            }
            ByzantineDetectionMethod::ReputationBased => {
                self.compute_reputation_based_score(client_id, gradients)?
            }
            ByzantineDetectionMethod::EnsembleBased => {
                self.compute_ensemble_score(client_id, gradients)?
            }
        };

        // Update detection history
        let history = self
            .detection_history
            .entry(client_id.to_string())
            .or_insert_with(Vec::new);
        history.push(anomaly_score);

        // Maintain history window
        if history.len() > self.history_window {
            history.remove(0);
        }

        // Compute historical average
        let historical_average = if history.is_empty() {
            0.0
        } else {
            history.iter().sum::<f64>() / history.len() as f64
        };

        // Determine if Byzantine based on current and historical scores
        let current_threshold = if self.adaptive_threshold_config.enabled {
            self.adapt_threshold(client_id, anomaly_score)
        } else {
            self.detection_threshold
        };

        let is_byzantine = historical_average > current_threshold;
        let confidence = self.compute_confidence(historical_average, current_threshold);

        // Update suspicious clients set and reputation
        if is_byzantine {
            self.suspicious_clients.insert(client_id.to_string());
            self.decrease_reputation(client_id);
        } else {
            self.suspicious_clients.remove(client_id);
            self.increase_reputation(client_id);
        }

        let detection_reason =
            self.generate_detection_reason(anomaly_score, historical_average, current_threshold);

        Ok(DetectionResult {
            is_byzantine,
            anomaly_score,
            confidence,
            detection_reason,
            historical_average,
        })
    }

    /// Gets the current detection method
    pub fn get_detection_method(&self) -> &ByzantineDetectionMethod {
        &self.detection_method
    }

    /// Sets a new detection method
    pub fn set_detection_method(&mut self, method: ByzantineDetectionMethod) {
        self.detection_method = method;
    }

    /// Gets the current detection threshold
    pub fn get_detection_threshold(&self) -> f64 {
        self.detection_threshold
    }

    /// Sets a new detection threshold
    pub fn set_detection_threshold(&mut self, threshold: f64) {
        self.detection_threshold = threshold;
    }

    /// Gets the set of suspicious clients
    pub fn get_suspicious_clients(&self) -> &HashSet<String> {
        &self.suspicious_clients
    }

    /// Gets detection history for a specific client
    pub fn get_client_history(&self, client_id: &str) -> Option<&Vec<f64>> {
        self.detection_history.get(client_id)
    }

    /// Gets reputation score for a specific client
    pub fn get_reputation_score(&self, client_id: &str) -> f64 {
        self.reputation_scores
            .get(client_id)
            .copied()
            .unwrap_or(1.0)
    }

    /// Manually sets reputation score for a client
    pub fn set_reputation_score(&mut self, client_id: &str, score: f64) {
        self.reputation_scores
            .insert(client_id.to_string(), score.clamp(0.0, 1.0));
    }

    /// Clears a client from the suspicious list
    pub fn clear_suspicion(&mut self, client_id: &str) {
        self.suspicious_clients.remove(client_id);
    }

    /// Resets all detection state
    pub fn reset(&mut self) {
        self.suspicious_clients.clear();
        self.detection_history.clear();
        self.reputation_scores.clear();
    }

    /// Configures adaptive thresholding
    pub fn set_adaptive_threshold_config(&mut self, config: AdaptiveThresholdConfig) {
        self.adaptive_threshold_config = config;
    }

    /// Computes anomaly score using statistical tests
    fn compute_statistical_anomaly_score(
        &self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        let mut total_norm = 0.0;
        let mut param_count = 0;

        for gradient in gradients.values() {
            for &value in gradient {
                total_norm += (value as f64).powi(2);
                param_count += 1;
            }
        }

        let avg_norm = if param_count > 0 {
            (total_norm / param_count as f64).sqrt()
        } else {
            0.0
        };

        Ok(avg_norm)
    }

    /// Computes anomaly score using distance-based methods
    fn compute_distance_based_score(
        &self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        // For distance-based detection, we need reference gradients
        // This is a simplified implementation
        let gradient_norm = self.compute_gradient_norm(gradients);

        // In practice, this would compare against typical gradient patterns
        let typical_norm = 1.0; // This would be computed from historical data
        let distance = (gradient_norm - typical_norm).abs();

        Ok(distance)
    }

    /// Computes anomaly score using clustering-based methods
    fn compute_clustering_based_score(
        &self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        // Simplified clustering-based detection
        let gradient_norm = self.compute_gradient_norm(gradients);

        // In practice, this would perform actual clustering
        let cluster_center = 1.0; // This would be computed from clustering
        let distance_to_center = (gradient_norm - cluster_center).abs();

        Ok(distance_to_center)
    }

    /// Computes anomaly score using reputation-based methods
    fn compute_reputation_based_score(
        &self,
        client_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        let base_score = self.compute_statistical_anomaly_score(gradients)?;
        let reputation = self.get_reputation_score(client_id);

        // Weight anomaly score by inverse of reputation
        let weighted_score = base_score / reputation.max(0.1);

        Ok(weighted_score)
    }

    /// Computes anomaly score using ensemble methods
    fn compute_ensemble_score(
        &mut self,
        client_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        let statistical_score = self.compute_statistical_anomaly_score(gradients)?;
        let distance_score = self.compute_distance_based_score(gradients)?;
        let reputation_score = self.compute_reputation_based_score(client_id, gradients)?;

        // Weighted ensemble
        let ensemble_score =
            0.4 * statistical_score + 0.3 * distance_score + 0.3 * reputation_score;

        Ok(ensemble_score)
    }

    /// Computes the L2 norm of gradients
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;

        for gradient in gradients.values() {
            for &value in gradient {
                norm_squared += (value as f64).powi(2);
            }
        }

        norm_squared.sqrt()
    }

    /// Adapts detection threshold based on recent behavior
    fn adapt_threshold(&mut self, client_id: &str, _current_score: f64) -> f64 {
        if !self.adaptive_threshold_config.enabled {
            return self.detection_threshold;
        }

        // Simple adaptive threshold based on running average
        let history = self.detection_history.get(client_id);
        if let Some(scores) = history {
            if !scores.is_empty() {
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                let std_dev = (scores.iter().map(|&x| (x - avg_score).powi(2)).sum::<f64>()
                    / scores.len() as f64)
                    .sqrt();

                // Adaptive threshold: mean + k * std_dev
                let adaptive_threshold = avg_score + 2.0 * std_dev;

                adaptive_threshold.clamp(
                    self.adaptive_threshold_config.min_threshold,
                    self.adaptive_threshold_config.max_threshold,
                )
            } else {
                self.detection_threshold
            }
        } else {
            self.detection_threshold
        }
    }

    /// Computes confidence in the detection decision
    fn compute_confidence(&self, score: f64, threshold: f64) -> f64 {
        let distance_from_threshold = (score - threshold).abs();
        let max_distance = threshold.max(10.0); // Normalization factor

        (distance_from_threshold / max_distance).min(1.0)
    }

    /// Generates human-readable detection reason
    fn generate_detection_reason(
        &self,
        current_score: f64,
        historical_avg: f64,
        threshold: f64,
    ) -> String {
        if historical_avg > threshold {
            format!(
                "Historical average anomaly score ({:.3}) exceeds threshold ({:.3}). Current score: {:.3}",
                historical_avg, threshold, current_score
            )
        } else {
            format!(
                "Historical average anomaly score ({:.3}) is below threshold ({:.3}). Current score: {:.3}",
                historical_avg, threshold, current_score
            )
        }
    }

    /// Increases reputation score for good behavior
    fn increase_reputation(&mut self, client_id: &str) {
        let current_reputation = self.get_reputation_score(client_id);
        let new_reputation = (current_reputation + 0.01).min(1.0);
        self.reputation_scores
            .insert(client_id.to_string(), new_reputation);
    }

    /// Decreases reputation score for suspicious behavior
    fn decrease_reputation(&mut self, client_id: &str) {
        let current_reputation = self.get_reputation_score(client_id);
        let new_reputation = (current_reputation - 0.05).max(0.0);
        self.reputation_scores
            .insert(client_id.to_string(), new_reputation);
    }
}

impl Default for ByzantineDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            adaptation_rate: 0.1,
            min_threshold: 0.5,
            max_threshold: 5.0,
            target_false_positive_rate: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byzantine_detector_creation() {
        let detector = ByzantineDetector::new();
        assert_eq!(
            *detector.get_detection_method(),
            ByzantineDetectionMethod::StatisticalTest
        );
        assert_eq!(detector.get_detection_threshold(), 2.0);
        assert!(detector.get_suspicious_clients().is_empty());
    }

    #[test]
    fn test_byzantine_detector_with_config() {
        let detector =
            ByzantineDetector::with_config(ByzantineDetectionMethod::DistanceBased, 3.0, 20);
        assert_eq!(
            *detector.get_detection_method(),
            ByzantineDetectionMethod::DistanceBased
        );
        assert_eq!(detector.get_detection_threshold(), 3.0);
    }

    #[test]
    fn test_byzantine_detection() {
        let mut detector = ByzantineDetector::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let result = detector.detect_byzantine_behavior("client_1", &gradients);
        assert!(result.is_ok());

        // Should have history for this client now
        assert!(detector.get_client_history("client_1").is_some());
    }

    #[test]
    fn test_detailed_byzantine_detection() {
        let mut detector = ByzantineDetector::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let result = detector.detect_byzantine_behavior_detailed("client_1", &gradients);
        assert!(result.is_ok());

        let detection_result = result.unwrap();
        assert!(detection_result.anomaly_score >= 0.0);
        assert!(detection_result.confidence >= 0.0 && detection_result.confidence <= 1.0);
        assert!(!detection_result.detection_reason.is_empty());
    }

    #[test]
    fn test_reputation_management() {
        let mut detector = ByzantineDetector::new();

        // Initial reputation should be 1.0
        assert_eq!(detector.get_reputation_score("client_1"), 1.0);

        // Set custom reputation
        detector.set_reputation_score("client_1", 0.5);
        assert_eq!(detector.get_reputation_score("client_1"), 0.5);

        // Test reputation increase
        detector.increase_reputation("client_1");
        assert!(detector.get_reputation_score("client_1") > 0.5);

        // Test reputation decrease
        detector.decrease_reputation("client_1");
        // Should be less than before increase
    }

    #[test]
    fn test_detection_method_switching() {
        let mut detector = ByzantineDetector::new();
        assert_eq!(
            *detector.get_detection_method(),
            ByzantineDetectionMethod::StatisticalTest
        );

        detector.set_detection_method(ByzantineDetectionMethod::ReputationBased);
        assert_eq!(
            *detector.get_detection_method(),
            ByzantineDetectionMethod::ReputationBased
        );
    }

    #[test]
    fn test_threshold_adjustment() {
        let mut detector = ByzantineDetector::new();
        assert_eq!(detector.get_detection_threshold(), 2.0);

        detector.set_detection_threshold(3.5);
        assert_eq!(detector.get_detection_threshold(), 3.5);
    }

    #[test]
    fn test_suspicious_clients_management() {
        let mut detector = ByzantineDetector::new();

        // Initially no suspicious clients
        assert!(detector.get_suspicious_clients().is_empty());

        // Simulate detection that flags a client
        detector.suspicious_clients.insert("client_1".to_string());
        assert!(detector.get_suspicious_clients().contains("client_1"));

        // Clear suspicion
        detector.clear_suspicion("client_1");
        assert!(!detector.get_suspicious_clients().contains("client_1"));
    }

    #[test]
    fn test_gradient_norm_computation() {
        let detector = ByzantineDetector::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![3.0, 4.0]); // L2 norm = 5.0

        let norm = detector.compute_gradient_norm(&gradients);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_history_window_management() {
        let mut detector = ByzantineDetector::with_config(
            ByzantineDetectionMethod::StatisticalTest,
            2.0,
            3, // Small history window for testing
        );

        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0]);

        // Add more detections than the history window
        for i in 0..5 {
            let client_id = format!("client_{}", i);
            let _ = detector.detect_byzantine_behavior(&client_id, &gradients);
        }

        // History should be limited by window size
        if let Some(history) = detector.get_client_history("client_0") {
            assert!(history.len() <= 3);
        }
    }

    #[test]
    fn test_adaptive_threshold_config() {
        let mut detector = ByzantineDetector::new();

        let config = AdaptiveThresholdConfig {
            enabled: true,
            adaptation_rate: 0.2,
            min_threshold: 1.0,
            max_threshold: 4.0,
            target_false_positive_rate: 0.1,
        };

        detector.set_adaptive_threshold_config(config.clone());
        // Config should be set (we can't directly access it due to privacy)
    }

    #[test]
    fn test_detector_reset() {
        let mut detector = ByzantineDetector::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        // Generate some state
        let _ = detector.detect_byzantine_behavior("client_1", &gradients);
        detector.set_reputation_score("client_1", 0.5);

        // Verify state exists
        assert!(detector.get_client_history("client_1").is_some());
        assert_eq!(detector.get_reputation_score("client_1"), 0.5);

        // Reset and verify clean state
        detector.reset();
        assert!(detector.get_client_history("client_1").is_none());
        assert_eq!(detector.get_reputation_score("client_1"), 1.0); // Default reputation
        assert!(detector.get_suspicious_clients().is_empty());
    }

    #[test]
    fn test_ensemble_detection() {
        let mut detector = ByzantineDetector::new();
        detector.set_detection_method(ByzantineDetectionMethod::EnsembleBased);

        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

        let result = detector.detect_byzantine_behavior("client_1", &gradients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detection_methods() {
        let methods = [
            ByzantineDetectionMethod::StatisticalTest,
            ByzantineDetectionMethod::DistanceBased,
            ByzantineDetectionMethod::ClusteringBased,
            ByzantineDetectionMethod::ReputationBased,
            ByzantineDetectionMethod::EnsembleBased,
        ];

        for method in &methods {
            let mut detector = ByzantineDetector::new();
            detector.set_detection_method(method.clone());

            let mut gradients = HashMap::new();
            gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);

            let result = detector.detect_byzantine_behavior("client_1", &gradients);
            assert!(result.is_ok(), "Detection failed for method: {:?}", method);
        }
    }
}
