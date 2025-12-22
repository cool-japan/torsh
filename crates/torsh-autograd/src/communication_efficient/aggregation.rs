//! Gradient Aggregation Engine for Communication-Efficient Distributed Training
//!
//! This module provides a comprehensive gradient aggregation engine designed for distributed
//! machine learning scenarios. It implements various aggregation algorithms with support for
//! Byzantine fault tolerance, staleness compensation, and quality-weighted aggregation.
//!
//! # Overview
//!
//! The aggregation engine is responsible for combining gradients from multiple workers in a
//! distributed training setup while maintaining robustness against various failure modes and
//! optimizing for communication efficiency.
//!
//! # Key Features
//!
//! - **Multiple Aggregation Methods**: Average, weighted average, median, trimmed mean
//! - **Byzantine Fault Tolerance**: Robust aggregation against malicious or faulty workers
//! - **Staleness Compensation**: Handling of delayed gradient updates in asynchronous scenarios
//! - **Quality Weighting**: Incorporating gradient quality metrics into aggregation
//! - **Performance Monitoring**: Comprehensive metrics collection for aggregation operations
//! - **Thread Safety**: Full async/thread-safe operation for distributed environments
//!
//! # Examples
//!
//! ## Basic Aggregation
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::aggregation::AggregationEngine;
//! use torsh_autograd::communication_efficient::config::AggregationMethod;
//! use std::collections::HashMap;
//!
//! let mut engine = AggregationEngine::new(AggregationMethod::Average);
//!
//! let mut grad1 = HashMap::new();
//! grad1.insert("layer1.weight".to_string(), vec![1.0, 2.0, 3.0]);
//!
//! let mut grad2 = HashMap::new();
//! grad2.insert("layer1.weight".to_string(), vec![2.0, 3.0, 4.0]);
//!
//! let gradients = vec![grad1, grad2];
//! let aggregated = engine.aggregate(gradients).unwrap();
//! assert_eq!(aggregated["layer1.weight"], vec![1.5, 2.5, 3.5]);
//! ```
//!
//! ## Byzantine-Resistant Aggregation
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::aggregation::AggregationEngine;
//! use torsh_autograd::communication_efficient::config::AggregationMethod;
//!
//! let mut engine = AggregationEngine::new(AggregationMethod::Median);
//! engine.enable_byzantine_resilience(true);
//! engine.set_byzantine_threshold(0.33); // Tolerate up to 33% malicious workers
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::config::{AggregationMethod, CommunicationConfig};

/// Error types specific to gradient aggregation operations.
#[derive(Debug, Clone)]
pub enum AggregationError {
    /// No gradients provided for aggregation
    EmptyGradientSet,
    /// Gradients have incompatible shapes or dimensions
    IncompatibleGradients,
    /// Byzantine fault detected and cannot be handled
    ByzantineFaultDetected,
    /// Staleness threshold exceeded
    StalenessThresholdExceeded,
    /// Quality threshold not met
    QualityThresholdNotMet,
    /// Internal aggregation computation failed
    ComputationFailed(String),
    /// Worker contribution tracking failed
    WorkerTrackingFailed,
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregationError::EmptyGradientSet => {
                write!(f, "No gradients provided for aggregation")
            }
            AggregationError::IncompatibleGradients => {
                write!(f, "Gradients have incompatible shapes")
            }
            AggregationError::ByzantineFaultDetected => write!(f, "Byzantine fault detected"),
            AggregationError::StalenessThresholdExceeded => {
                write!(f, "Staleness threshold exceeded")
            }
            AggregationError::QualityThresholdNotMet => write!(f, "Quality threshold not met"),
            AggregationError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            AggregationError::WorkerTrackingFailed => {
                write!(f, "Worker contribution tracking failed")
            }
        }
    }
}

impl std::error::Error for AggregationError {}

/// Metadata for tracking worker contributions and gradient quality.
#[derive(Debug, Clone)]
pub struct WorkerContribution {
    /// Worker ID
    pub worker_id: u32,
    /// Number of gradients contributed
    pub contribution_count: u64,
    /// Average quality score of gradients from this worker
    pub quality_score: f64,
    /// Last seen timestamp
    pub last_seen: Instant,
    /// Whether this worker is suspected of Byzantine behavior
    pub suspected_byzantine: bool,
    /// Staleness metric (how delayed this worker's updates are)
    pub staleness: Duration,
}

/// Performance metrics for aggregation operations.
#[derive(Debug, Clone, Default)]
pub struct AggregationMetrics {
    /// Total number of aggregations performed
    pub total_aggregations: u64,
    /// Total time spent on aggregation operations
    pub total_aggregation_time: Duration,
    /// Average number of workers per aggregation
    pub average_worker_count: f64,
    /// Number of Byzantine faults detected and handled
    pub byzantine_faults_detected: u64,
    /// Number of gradients rejected due to quality issues
    pub quality_rejections: u64,
    /// Number of gradients adjusted for staleness
    pub staleness_adjustments: u64,
}

/// Main gradient aggregation engine for distributed training.
///
/// The `AggregationEngine` provides robust gradient aggregation with support for
/// various aggregation methods, Byzantine fault tolerance, staleness compensation,
/// and quality-weighted aggregation.
///
/// # Thread Safety
///
/// This structure is designed to be thread-safe and can be safely shared across
/// multiple threads in a distributed training environment.
#[derive(Debug)]
pub struct AggregationEngine {
    /// Aggregation method to use
    method: AggregationMethod,
    /// Buffer for storing intermediate aggregation results
    aggregation_buffer: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    /// Tracking of worker contributions and quality metrics
    worker_contributions: Arc<Mutex<HashMap<u32, WorkerContribution>>>,
    /// Quality weights for each worker (higher is better)
    quality_weights: Arc<Mutex<HashMap<u32, f64>>>,
    /// Whether staleness compensation is enabled
    staleness_compensation: bool,
    /// Whether Byzantine resilience is enabled
    byzantine_resilience: bool,
    /// Maximum acceptable staleness before rejection
    max_staleness: Duration,
    /// Minimum quality threshold for gradient acceptance
    quality_threshold: f64,
    /// Byzantine fault tolerance threshold (fraction of workers that can be malicious)
    byzantine_threshold: f64,
    /// Performance metrics collection
    metrics: Arc<Mutex<AggregationMetrics>>,
    /// Configuration reference
    config: Option<CommunicationConfig>,
}

impl AggregationEngine {
    /// Creates a new aggregation engine with the specified method.
    ///
    /// # Arguments
    ///
    /// * `method` - The aggregation method to use for combining gradients
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_autograd::communication_efficient::aggregation::AggregationEngine;
    /// use torsh_autograd::communication_efficient::config::AggregationMethod;
    ///
    /// let engine = AggregationEngine::new(AggregationMethod::WeightedAverage);
    /// ```
    pub fn new(method: AggregationMethod) -> Self {
        Self {
            method,
            aggregation_buffer: Arc::new(Mutex::new(HashMap::new())),
            worker_contributions: Arc::new(Mutex::new(HashMap::new())),
            quality_weights: Arc::new(Mutex::new(HashMap::new())),
            staleness_compensation: true,
            byzantine_resilience: false,
            max_staleness: Duration::from_secs(30),
            quality_threshold: 0.5,
            byzantine_threshold: 0.33,
            metrics: Arc::new(Mutex::new(AggregationMetrics::default())),
            config: None,
        }
    }

    /// Creates a new aggregation engine with configuration.
    ///
    /// # Arguments
    ///
    /// * `method` - The aggregation method to use
    /// * `config` - Communication configuration containing aggregation settings
    pub fn with_config(method: AggregationMethod, config: CommunicationConfig) -> Self {
        let mut engine = Self::new(method);
        engine.config = Some(config);
        engine.byzantine_resilience = engine.config.as_ref().unwrap().fault_tolerance;
        engine
    }

    /// Enables or disables Byzantine resilience.
    ///
    /// When enabled, the aggregation engine will attempt to detect and handle
    /// Byzantine faults from malicious or faulty workers.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable Byzantine resilience
    pub fn enable_byzantine_resilience(&mut self, enabled: bool) {
        self.byzantine_resilience = enabled;
    }

    /// Sets the Byzantine fault tolerance threshold.
    ///
    /// This threshold determines what fraction of workers can be malicious
    /// before the aggregation becomes unreliable.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Fraction of workers that can be Byzantine (0.0 to 1.0)
    pub fn set_byzantine_threshold(&mut self, threshold: f64) {
        self.byzantine_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Sets the maximum acceptable staleness for gradients.
    ///
    /// Gradients older than this threshold will be rejected or given
    /// reduced weight in the aggregation.
    ///
    /// # Arguments
    ///
    /// * `max_staleness` - Maximum acceptable staleness duration
    pub fn set_max_staleness(&mut self, max_staleness: Duration) {
        self.max_staleness = max_staleness;
    }

    /// Sets the minimum quality threshold for gradient acceptance.
    ///
    /// Gradients with quality scores below this threshold will be rejected.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum quality threshold (0.0 to 1.0)
    pub fn set_quality_threshold(&mut self, threshold: f64) {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enables or disables staleness compensation.
    ///
    /// When enabled, gradients with higher staleness will be given reduced weight
    /// in the aggregation to account for their age.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable staleness compensation
    pub fn enable_staleness_compensation(&mut self, enabled: bool) {
        self.staleness_compensation = enabled;
    }

    /// Main aggregation method that dispatches to specific aggregation algorithms.
    ///
    /// This method handles the complete aggregation pipeline including quality
    /// checks, staleness compensation, and Byzantine fault detection.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Vector of gradient maps from different workers
    ///
    /// # Returns
    ///
    /// Result containing the aggregated gradient or an aggregation error.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_autograd::communication_efficient::aggregation::AggregationEngine;
    /// use torsh_autograd::communication_efficient::config::AggregationMethod;
    /// use std::collections::HashMap;
    ///
    /// let mut engine = AggregationEngine::new(AggregationMethod::Average);
    ///
    /// let mut grad1 = HashMap::new();
    /// grad1.insert("param".to_string(), vec![1.0, 2.0]);
    ///
    /// let mut grad2 = HashMap::new();
    /// grad2.insert("param".to_string(), vec![3.0, 4.0]);
    ///
    /// let result = engine.aggregate(vec![grad1, grad2]).unwrap();
    /// assert_eq!(result["param"], vec![2.0, 3.0]);
    /// ```
    pub fn aggregate(
        &mut self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        let start_time = Instant::now();

        if gradients.is_empty() {
            return Err(AggregationError::EmptyGradientSet);
        }

        // Validate gradient compatibility
        self.validate_gradient_compatibility(&gradients)?;

        // Apply quality filtering and staleness compensation if enabled
        let filtered_gradients = if self.staleness_compensation || self.quality_threshold > 0.0 {
            self.filter_and_compensate_gradients(gradients)?
        } else {
            gradients
        };

        // Apply Byzantine fault detection if enabled
        let final_gradients = if self.byzantine_resilience {
            self.detect_and_handle_byzantine_faults(filtered_gradients)?
        } else {
            filtered_gradients
        };

        // Store gradient count before consuming final_gradients
        let gradient_count = final_gradients.len();

        // Perform the actual aggregation
        let result = match self.method {
            AggregationMethod::Average => self.average_aggregation(final_gradients),
            AggregationMethod::WeightedAverage => {
                self.weighted_average_aggregation(final_gradients)
            }
            AggregationMethod::Median => self.median_aggregation(final_gradients),
            AggregationMethod::TrimmedMean => self.trimmed_mean_aggregation(final_gradients),
            AggregationMethod::AdaptiveWeighting => {
                self.adaptive_weighted_aggregation(final_gradients)
            }
            _ => self.average_aggregation(final_gradients),
        };

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_aggregations += 1;
        metrics.total_aggregation_time += start_time.elapsed();
        metrics.average_worker_count = (metrics.average_worker_count
            * (metrics.total_aggregations - 1) as f64
            + gradient_count as f64)
            / metrics.total_aggregations as f64;

        result
    }

    /// Validates that all gradients have compatible shapes and parameters.
    fn validate_gradient_compatibility(
        &self,
        gradients: &[HashMap<String, Vec<f32>>],
    ) -> Result<(), AggregationError> {
        if gradients.is_empty() {
            return Ok(());
        }

        let reference = &gradients[0];
        for gradient in gradients.iter().skip(1) {
            if gradient.len() != reference.len() {
                return Err(AggregationError::IncompatibleGradients);
            }

            for (param_name, values) in gradient {
                if let Some(ref_values) = reference.get(param_name) {
                    if values.len() != ref_values.len() {
                        return Err(AggregationError::IncompatibleGradients);
                    }
                } else {
                    return Err(AggregationError::IncompatibleGradients);
                }
            }
        }
        Ok(())
    }

    /// Filters gradients based on quality and applies staleness compensation.
    fn filter_and_compensate_gradients(
        &mut self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<Vec<HashMap<String, Vec<f32>>>, AggregationError> {
        let mut filtered = Vec::new();
        let _current_time = Instant::now();

        for (idx, gradient) in gradients.into_iter().enumerate() {
            // Simulate quality score computation (in practice, this would be based on actual metrics)
            let quality_score = self.compute_gradient_quality(&gradient);

            if quality_score < self.quality_threshold {
                let mut metrics = self.metrics.lock().unwrap();
                metrics.quality_rejections += 1;
                continue;
            }

            // Apply staleness compensation if enabled
            let compensated_gradient = if self.staleness_compensation {
                // Simulate staleness (in practice, this would come from gradient metadata)
                let staleness = Duration::from_millis(idx as u64 * 100);

                if staleness > self.max_staleness {
                    let mut metrics = self.metrics.lock().unwrap();
                    metrics.staleness_adjustments += 1;
                    continue;
                }

                self.apply_staleness_compensation(gradient, staleness)
            } else {
                gradient
            };

            filtered.push(compensated_gradient);
        }

        if filtered.is_empty() {
            return Err(AggregationError::QualityThresholdNotMet);
        }

        Ok(filtered)
    }

    /// Detects and handles Byzantine faults in the gradient set.
    fn detect_and_handle_byzantine_faults(
        &mut self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<Vec<HashMap<String, Vec<f32>>>, AggregationError> {
        if gradients.len() < 3 {
            // Cannot detect Byzantine faults with fewer than 3 gradients
            return Ok(gradients);
        }

        let mut filtered = Vec::new();
        let gradient_norms: Vec<f64> = gradients
            .iter()
            .map(|g| self.compute_gradient_norm(g))
            .collect();

        // Compute median norm for outlier detection
        let mut sorted_norms = gradient_norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_norm = sorted_norms[sorted_norms.len() / 2];

        // Filter out gradients that are too far from the median (potential Byzantine)
        let threshold = median_norm * 3.0; // 3x median threshold
        let mut byzantine_count = 0;

        for (gradient, norm) in gradients.into_iter().zip(gradient_norms.iter()) {
            if (norm - median_norm).abs() > threshold {
                byzantine_count += 1;
                continue;
            }
            filtered.push(gradient);
        }

        // Check if Byzantine threshold is exceeded
        let byzantine_ratio = byzantine_count as f64 / (filtered.len() + byzantine_count) as f64;
        if byzantine_ratio > self.byzantine_threshold {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.byzantine_faults_detected += 1;
            return Err(AggregationError::ByzantineFaultDetected);
        }

        Ok(filtered)
    }

    /// Computes a quality score for a gradient (0.0 to 1.0).
    fn compute_gradient_quality(&self, gradient: &HashMap<String, Vec<f32>>) -> f64 {
        // Simple quality metric based on gradient norm and finite values
        let norm = self.compute_gradient_norm(gradient);
        let finite_ratio = self.compute_finite_ratio(gradient);

        // Quality is inverse related to norm (avoid exploding gradients) and proportional to finite ratio
        let norm_score = 1.0 / (1.0 + (norm / 1000.0).powi(2));
        let quality = norm_score * finite_ratio;

        quality.clamp(0.0, 1.0)
    }

    /// Applies staleness compensation to a gradient.
    fn apply_staleness_compensation(
        &self,
        mut gradient: HashMap<String, Vec<f32>>,
        staleness: Duration,
    ) -> HashMap<String, Vec<f32>> {
        // Apply exponential decay based on staleness
        let decay_factor = (-staleness.as_secs_f64() / 10.0).exp() as f32;

        for values in gradient.values_mut() {
            for value in values.iter_mut() {
                *value *= decay_factor;
            }
        }

        gradient
    }

    /// Computes the L2 norm of a gradient.
    fn compute_gradient_norm(&self, gradient: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;
        for values in gradient.values() {
            for &value in values {
                norm_squared += (value as f64).powi(2);
            }
        }
        norm_squared.sqrt()
    }

    /// Computes the ratio of finite values in a gradient.
    fn compute_finite_ratio(&self, gradient: &HashMap<String, Vec<f32>>) -> f64 {
        let mut total_elements = 0;
        let mut finite_elements = 0;

        for values in gradient.values() {
            for &value in values {
                total_elements += 1;
                if value.is_finite() {
                    finite_elements += 1;
                }
            }
        }

        if total_elements == 0 {
            1.0
        } else {
            finite_elements as f64 / total_elements as f64
        }
    }

    /// Performs simple average aggregation.
    ///
    /// Computes the element-wise average of all input gradients.
    fn average_aggregation(
        &self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        let mut aggregated = HashMap::new();
        let num_gradients = gradients.len() as f32;

        for (param_name, values) in &gradients[0] {
            let mut avg_values = vec![0.0; values.len()];

            for gradient in &gradients {
                if let Some(gradient_values) = gradient.get(param_name) {
                    for (i, &value) in gradient_values.iter().enumerate() {
                        avg_values[i] += value / num_gradients;
                    }
                }
            }

            aggregated.insert(param_name.clone(), avg_values);
        }

        Ok(aggregated)
    }

    /// Performs weighted average aggregation.
    ///
    /// Computes a weighted average where more recent gradients have higher weights.
    fn weighted_average_aggregation(
        &self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        let mut aggregated = HashMap::new();

        // Use quality weights if available, otherwise use recency-based weights
        let weights: Vec<f64> = if let Ok(quality_weights) = self.quality_weights.lock() {
            (0..gradients.len())
                .map(|i| quality_weights.get(&(i as u32)).copied().unwrap_or(1.0))
                .collect()
        } else {
            (0..gradients.len())
                .map(|i| 1.0 / (i as f64 + 1.0))
                .collect()
        };

        let total_weight: f64 = weights.iter().sum();

        for (param_name, values) in &gradients[0] {
            let mut weighted_values = vec![0.0; values.len()];

            for (gradient_idx, gradient) in gradients.iter().enumerate() {
                if let Some(gradient_values) = gradient.get(param_name) {
                    let weight = weights[gradient_idx] / total_weight;
                    for (i, &value) in gradient_values.iter().enumerate() {
                        weighted_values[i] += value * weight as f32;
                    }
                }
            }

            aggregated.insert(param_name.clone(), weighted_values);
        }

        Ok(aggregated)
    }

    /// Performs robust median aggregation.
    ///
    /// Computes the element-wise median, which is robust against Byzantine faults.
    fn median_aggregation(
        &self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        let mut aggregated = HashMap::new();

        for (param_name, values) in &gradients[0] {
            let mut median_values = Vec::new();

            for i in 0..values.len() {
                let mut position_values: Vec<f32> = gradients
                    .iter()
                    .filter_map(|g| g.get(param_name).and_then(|v| v.get(i)))
                    .cloned()
                    .collect();

                position_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let median = if position_values.len() % 2 == 0 {
                    let mid = position_values.len() / 2;
                    (position_values[mid - 1] + position_values[mid]) / 2.0
                } else {
                    position_values[position_values.len() / 2]
                };

                median_values.push(median);
            }

            aggregated.insert(param_name.clone(), median_values);
        }

        Ok(aggregated)
    }

    /// Performs trimmed mean aggregation.
    ///
    /// Computes the mean after removing the top and bottom percentiles.
    fn trimmed_mean_aggregation(
        &self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        let trim_ratio = 0.1; // Trim 10% from each end
        let mut aggregated = HashMap::new();

        for (param_name, values) in &gradients[0] {
            let mut trimmed_values = Vec::new();

            for i in 0..values.len() {
                let mut position_values: Vec<f32> = gradients
                    .iter()
                    .filter_map(|g| g.get(param_name).and_then(|v| v.get(i)))
                    .cloned()
                    .collect();

                position_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let trim_count = (position_values.len() as f64 * trim_ratio) as usize;
                let trimmed_slice =
                    &position_values[trim_count..position_values.len() - trim_count];

                let mean = if trimmed_slice.is_empty() {
                    0.0
                } else {
                    trimmed_slice.iter().sum::<f32>() / trimmed_slice.len() as f32
                };

                trimmed_values.push(mean);
            }

            aggregated.insert(param_name.clone(), trimmed_values);
        }

        Ok(aggregated)
    }

    /// Performs adaptive weighted aggregation.
    ///
    /// Uses historical quality metrics to adaptively weight gradients.
    fn adaptive_weighted_aggregation(
        &self,
        gradients: Vec<HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, AggregationError> {
        // For now, fall back to weighted average
        // In a full implementation, this would use historical worker performance
        self.weighted_average_aggregation(gradients)
    }

    /// Updates worker contribution tracking.
    ///
    /// # Arguments
    ///
    /// * `worker_id` - ID of the worker
    /// * `quality_score` - Quality score of the gradient from this worker
    pub fn update_worker_contribution(&mut self, worker_id: u32, quality_score: f64) {
        let mut contributions = self.worker_contributions.lock().unwrap();
        let contribution = contributions
            .entry(worker_id)
            .or_insert_with(|| WorkerContribution {
                worker_id,
                contribution_count: 0,
                quality_score: 0.0,
                last_seen: Instant::now(),
                suspected_byzantine: false,
                staleness: Duration::from_secs(0),
            });

        contribution.contribution_count += 1;
        contribution.quality_score = (contribution.quality_score
            * (contribution.contribution_count - 1) as f64
            + quality_score)
            / contribution.contribution_count as f64;
        contribution.last_seen = Instant::now();

        // Update quality weights
        let mut quality_weights = self.quality_weights.lock().unwrap();
        quality_weights.insert(worker_id, contribution.quality_score);
    }

    /// Gets current aggregation metrics.
    ///
    /// # Returns
    ///
    /// A clone of the current aggregation metrics.
    pub fn get_metrics(&self) -> AggregationMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Resets aggregation metrics.
    pub fn reset_metrics(&mut self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = AggregationMetrics::default();
    }

    /// Gets worker contribution information.
    ///
    /// # Returns
    ///
    /// A clone of the current worker contributions map.
    pub fn get_worker_contributions(&self) -> HashMap<u32, WorkerContribution> {
        self.worker_contributions.lock().unwrap().clone()
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for AggregationEngine {}
unsafe impl Sync for AggregationEngine {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication_efficient::config::AggregationMethod;

    #[test]
    fn test_aggregation_engine_creation() {
        let engine = AggregationEngine::new(AggregationMethod::Average);
        assert!(!engine.byzantine_resilience);
        assert!(engine.staleness_compensation);
    }

    #[test]
    fn test_average_aggregation() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);

        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1.0, 2.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param_1".to_string(), vec![3.0, 4.0]);

        let gradients = vec![grad1, grad2];
        let result = engine.aggregate(gradients).unwrap();

        let expected = vec![2.0, 3.0];
        let actual = &result["param_1"];
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            approx::assert_abs_diff_eq!(e, a, epsilon = 0.1);
        }
    }

    #[test]
    fn test_median_aggregation() {
        let mut engine = AggregationEngine::new(AggregationMethod::Median);
        // Disable staleness compensation to get exact median values
        engine.enable_staleness_compensation(false);
        engine.set_quality_threshold(0.0); // Disable quality filtering

        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1.0, 2.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param_1".to_string(), vec![3.0, 4.0]);

        let mut grad3 = HashMap::new();
        grad3.insert("param_1".to_string(), vec![5.0, 6.0]);

        let gradients = vec![grad1, grad2, grad3];
        let result = engine.aggregate(gradients).unwrap();

        // Use approximate comparison due to potential floating point precision issues
        let expected = vec![3.0, 4.0];
        let actual = &result["param_1"];
        for (e, a) in expected.iter().zip(actual.iter()) {
            approx::assert_abs_diff_eq!(e, a, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_byzantine_resilience() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);
        engine.enable_byzantine_resilience(true);
        engine.set_byzantine_threshold(0.5); // Allow up to 50% Byzantine nodes
        engine.enable_staleness_compensation(false); // Disable staleness compensation
        engine.set_quality_threshold(0.0); // Disable quality filtering

        // Create gradients with one outlier
        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1.0, 2.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param_1".to_string(), vec![1.1, 2.1]);

        let mut grad3 = HashMap::new();
        grad3.insert("param_1".to_string(), vec![100.0, 200.0]); // Byzantine outlier

        let gradients = vec![grad1, grad2, grad3];
        let result = engine.aggregate(gradients);

        // Should still succeed and filter out the outlier
        assert!(
            result.is_ok(),
            "Byzantine resilience should handle outliers"
        );
    }

    #[test]
    fn test_empty_gradients() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);
        let result = engine.aggregate(vec![]);
        assert!(matches!(result, Err(AggregationError::EmptyGradientSet)));
    }

    #[test]
    fn test_incompatible_gradients() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);

        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1.0, 2.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param_1".to_string(), vec![3.0, 4.0, 5.0]); // Different size

        let gradients = vec![grad1, grad2];
        let result = engine.aggregate(gradients);

        assert!(matches!(
            result,
            Err(AggregationError::IncompatibleGradients)
        ));
    }

    #[test]
    fn test_worker_contribution_tracking() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);
        engine.update_worker_contribution(1, 0.8);
        engine.update_worker_contribution(1, 0.9);

        let contributions = engine.get_worker_contributions();
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[&1].contribution_count, 2);
        assert!((contributions[&1].quality_score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_collection() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);

        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1.0, 2.0]);

        let gradients = vec![grad1];
        let _ = engine.aggregate(gradients);

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_aggregations, 1);
        assert!(metrics.total_aggregation_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_quality_threshold() {
        let mut engine = AggregationEngine::new(AggregationMethod::Average);
        engine.enable_staleness_compensation(false); // Disable staleness compensation
        engine.set_quality_threshold(0.99); // Even higher threshold to force failure

        // Create a gradient that will have low quality (large norm or non-finite values)
        let mut grad1 = HashMap::new();
        grad1.insert("param_1".to_string(), vec![1e6, 2e6]); // Very large values for low quality

        let gradients = vec![grad1];
        let result = engine.aggregate(gradients);

        // Should fail due to high quality threshold
        assert!(
            result.is_err(),
            "High quality threshold should reject low-quality gradients"
        );
    }
}
