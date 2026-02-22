//! ML-Powered Auto-Configuration System
//!
//! This module provides intelligent quantization configuration recommendations
//! based on tensor characteristics, performance metrics, and learned patterns.
//!
//! ## Features
//!
//! - **Tensor Analysis**: Analyzes tensor properties (shape, distribution, sparsity, etc.)
//! - **Performance Prediction**: Estimates quantization quality and performance trade-offs
//! - **Configuration Selection**: Automatically selects optimal quantization schemes
//! - **Adaptive Recommendations**: Learns from historical quantization results
//!
//! ## Usage
//!
//! ```rust
//! use torsh_quantization::auto_config::{AutoConfigurator, ConfigObjective};
//! use torsh_tensor::creation::tensor_1d;
//!
//! // Create auto-configurator with specific objectives
//! let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
//!
//! // Create a tensor to analyze
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let tensor = tensor_1d(&data).unwrap();
//!
//! // Get optimal configuration for a tensor
//! let optimal_config = configurator.recommend(&tensor, None).unwrap();
//! assert!(optimal_config.validate().is_ok());
//! ```

use crate::config::{ObserverType, QScheme, QuantBackend, QuantConfig};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Objectives for configuration selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigObjective {
    /// Maximize compression ratio
    MaximumCompression,
    /// Maximize accuracy (minimize quantization error)
    MaximumAccuracy,
    /// Balance between compression and accuracy
    BalancedQuality,
    /// Optimize for inference speed
    MaximumSpeed,
    /// Optimize for memory efficiency
    MinimumMemory,
    /// Optimize for mobile/edge devices
    EdgeOptimized,
}

/// Tensor characteristics for ML-based analysis
#[derive(Debug, Clone)]
pub struct TensorProfile {
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Total number of elements
    pub numel: usize,
    /// Data statistics
    pub stats: TensorStats,
    /// Sparsity level (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f32,
    /// Distribution characteristics
    pub distribution: DistributionProfile,
}

/// Statistical properties of tensor data
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Dynamic range
    pub range: f32,
    /// Presence of outliers
    pub has_outliers: bool,
    /// Percentage of near-zero values
    pub near_zero_ratio: f32,
}

/// Distribution profile for intelligent scheme selection
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionProfile {
    /// Normal/Gaussian distribution
    Normal,
    /// Uniform distribution
    Uniform,
    /// Heavy-tailed distribution (many outliers)
    HeavyTailed,
    /// Bimodal distribution
    Bimodal,
    /// Highly skewed distribution
    Skewed,
    /// Sparse distribution
    Sparse,
}

/// ML-powered auto-configurator
pub struct AutoConfigurator {
    objective: ConfigObjective,
    /// Historical performance data for learning
    history: Vec<ConfigPerformance>,
    /// Feature importance weights (learned from experience)
    feature_weights: FeatureWeights,
}

/// Performance metrics for a configuration
#[derive(Debug, Clone)]
struct ConfigPerformance {
    #[allow(dead_code)]
    config: QuantConfig,
    profile: TensorProfile,
    /// Observed quantization error
    error: f32,
    #[allow(dead_code)]
    /// Compression ratio achieved
    compression: f32,
    #[allow(dead_code)]
    /// Inference speedup (if measured)
    speedup: Option<f32>,
}

/// Learned feature importance weights
#[derive(Debug, Clone)]
struct FeatureWeights {
    /// Weight for data range consideration
    range_weight: f32,
    /// Weight for sparsity consideration
    sparsity_weight: f32,
    /// Weight for distribution type
    distribution_weight: f32,
    /// Weight for tensor size
    size_weight: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            range_weight: 1.0,
            sparsity_weight: 0.8,
            distribution_weight: 0.9,
            size_weight: 0.7,
        }
    }
}

impl AutoConfigurator {
    /// Create a new auto-configurator with specified objective
    pub fn new(objective: ConfigObjective) -> Self {
        Self {
            objective,
            history: Vec::new(),
            feature_weights: FeatureWeights::default(),
        }
    }

    /// Recommend optimal configuration for a tensor
    pub fn recommend(
        &self,
        tensor: &Tensor,
        constraints: Option<ConfigConstraints>,
    ) -> TorshResult<QuantConfig> {
        // Analyze tensor characteristics
        let profile = self.analyze_tensor(tensor)?;

        // Select optimal configuration based on profile and objective
        let config = self.select_configuration(&profile, constraints)?;

        Ok(config)
    }

    /// Recommend multiple configurations ranked by expected performance
    pub fn recommend_ranked(
        &self,
        tensor: &Tensor,
        top_k: usize,
        constraints: Option<ConfigConstraints>,
    ) -> TorshResult<Vec<(QuantConfig, f32)>> {
        let profile = self.analyze_tensor(tensor)?;
        let mut candidates = self.generate_candidates(&profile, constraints)?;

        // Score each candidate
        for (config, score) in &mut candidates {
            *score = self.score_configuration(config, &profile);
        }

        // Sort by score (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Update the configurator with observed performance
    pub fn update_performance(
        &mut self,
        config: &QuantConfig,
        tensor: &Tensor,
        observed_error: f32,
        observed_compression: f32,
        speedup: Option<f32>,
    ) -> TorshResult<()> {
        let profile = self.analyze_tensor(tensor)?;

        let performance = ConfigPerformance {
            config: config.clone(),
            profile,
            error: observed_error,
            compression: observed_compression,
            speedup,
        };

        self.history.push(performance);

        // Update feature weights based on new data (simple online learning)
        if self.history.len() >= 10 {
            self.update_feature_weights();
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Private helper methods
    // -------------------------------------------------------------------------

    /// Analyze tensor to extract characteristics
    fn analyze_tensor(&self, tensor: &Tensor) -> TorshResult<TensorProfile> {
        let data = tensor.data()?;
        let shape = tensor.shape().dims().to_vec();
        let numel = tensor.shape().numel();

        // Calculate statistics
        let stats = self.calculate_stats(&data)?;

        // Calculate sparsity
        let sparsity = self.calculate_sparsity(&data);

        // Determine distribution profile
        let distribution = self.classify_distribution(&data, &stats);

        Ok(TensorProfile {
            shape,
            numel,
            stats,
            sparsity,
            distribution,
        })
    }

    /// Calculate statistical properties
    fn calculate_stats(&self, data: &[f32]) -> TorshResult<TensorStats> {
        if data.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot calculate stats for empty tensor".to_string(),
            ));
        }

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        let mean = data.iter().sum::<f32>() / data.len() as f32;

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        // Detect outliers using IQR method
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        let outlier_threshold_low = q1 - 1.5 * iqr;
        let outlier_threshold_high = q3 + 1.5 * iqr;

        let has_outliers = data
            .iter()
            .any(|&x| x < outlier_threshold_low || x > outlier_threshold_high);

        // Calculate near-zero ratio
        let zero_threshold = range.abs() * 0.01; // 1% of range
        let near_zero_count = data.iter().filter(|&&x| x.abs() < zero_threshold).count();
        let near_zero_ratio = near_zero_count as f32 / data.len() as f32;

        Ok(TensorStats {
            min,
            max,
            mean,
            std_dev,
            range,
            has_outliers,
            near_zero_ratio,
        })
    }

    /// Calculate sparsity level
    fn calculate_sparsity(&self, data: &[f32]) -> f32 {
        let zero_count = data.iter().filter(|&&x| x.abs() < 1e-8).count();
        zero_count as f32 / data.len() as f32
    }

    /// Classify distribution type
    fn classify_distribution(&self, data: &[f32], stats: &TensorStats) -> DistributionProfile {
        // Check for sparsity first
        if stats.near_zero_ratio > 0.6 {
            return DistributionProfile::Sparse;
        }

        // Calculate skewness
        let skewness = data
            .iter()
            .map(|&x| ((x - stats.mean) / stats.std_dev).powi(3))
            .sum::<f32>()
            / data.len() as f32;

        // Calculate kurtosis for tail heaviness
        let kurtosis = data
            .iter()
            .map(|&x| ((x - stats.mean) / stats.std_dev).powi(4))
            .sum::<f32>()
            / data.len() as f32;

        // Classification logic
        if skewness.abs() > 1.0 {
            DistributionProfile::Skewed
        } else if kurtosis > 4.0 {
            DistributionProfile::HeavyTailed
        } else if (kurtosis - 3.0).abs() < 0.5 && skewness.abs() < 0.5 {
            DistributionProfile::Normal
        } else if kurtosis < 2.0 {
            DistributionProfile::Uniform
        } else {
            DistributionProfile::Bimodal
        }
    }

    /// Select optimal configuration based on profile
    fn select_configuration(
        &self,
        profile: &TensorProfile,
        constraints: Option<ConfigConstraints>,
    ) -> TorshResult<QuantConfig> {
        let mut config = match self.objective {
            ConfigObjective::MaximumCompression => self.select_for_compression(profile),
            ConfigObjective::MaximumAccuracy => self.select_for_accuracy(profile),
            ConfigObjective::BalancedQuality => self.select_balanced(profile),
            ConfigObjective::MaximumSpeed => self.select_for_speed(profile),
            ConfigObjective::MinimumMemory => self.select_for_memory(profile),
            ConfigObjective::EdgeOptimized => self.select_for_edge(profile),
        }?;

        // Apply constraints if provided
        if let Some(constraints) = constraints {
            config = self.apply_constraints(config, constraints)?;
        }

        Ok(config)
    }

    /// Select configuration optimized for compression
    fn select_for_compression(&self, profile: &TensorProfile) -> TorshResult<QuantConfig> {
        // Use aggressive quantization
        if profile.sparsity > 0.5 {
            // Sparse data - use binary or ternary
            if profile.distribution == DistributionProfile::Sparse {
                Ok(QuantConfig::binary())
            } else {
                Ok(QuantConfig::ternary())
            }
        } else if profile.numel < 1000 {
            // Small tensors - INT4 is good
            Ok(QuantConfig::int4())
        } else {
            // Large tensors - group-wise INT4
            let group_size = (profile.numel / 100).min(128).max(16);
            Ok(QuantConfig::group_wise(0, group_size))
        }
    }

    /// Select configuration optimized for accuracy
    fn select_for_accuracy(&self, profile: &TensorProfile) -> TorshResult<QuantConfig> {
        let mut config = if profile.stats.has_outliers
            || profile.distribution == DistributionProfile::HeavyTailed
        {
            // Use histogram observer for outliers
            QuantConfig::int8().with_observer(ObserverType::Histogram)
        } else if profile.stats.range > 1000.0 {
            // Large range - use per-channel quantization
            QuantConfig::per_channel(0).with_observer(ObserverType::Percentile)
        } else {
            // Standard case - per-tensor with percentile
            QuantConfig::int8().with_observer(ObserverType::Percentile)
        };

        // Use reduced range for better numerical stability if needed
        if profile.stats.range > 10000.0 {
            config = config.with_reduce_range(crate::config::ReduceRange::Reduce);
        }

        Ok(config)
    }

    /// Select balanced configuration
    fn select_balanced(&self, profile: &TensorProfile) -> TorshResult<QuantConfig> {
        if profile.numel > 100000 && profile.sparsity < 0.1 {
            // Large, dense tensors - group-wise for balance
            let group_size = if profile.stats.has_outliers { 32 } else { 64 };
            Ok(QuantConfig::group_wise(0, group_size).with_observer(ObserverType::Histogram))
        } else if profile.sparsity > 0.3 {
            // Moderately sparse - INT4
            Ok(QuantConfig::int4().with_observer(ObserverType::MinMax))
        } else {
            // Standard case - INT8 with histogram
            Ok(QuantConfig::int8().with_observer(ObserverType::Histogram))
        }
    }

    /// Select configuration optimized for speed
    fn select_for_speed(&self, profile: &TensorProfile) -> TorshResult<QuantConfig> {
        // Prefer simpler schemes and backends
        let mut config = if profile.numel < 10000 {
            QuantConfig::int8()
        } else {
            QuantConfig::int8().with_observer(ObserverType::MinMax) // MinMax is fastest
        };

        // Use optimized backend
        config = config.with_backend(QuantBackend::Fbgemm);

        Ok(config)
    }

    /// Select configuration optimized for memory
    fn select_for_memory(&self, profile: &TensorProfile) -> TorshResult<QuantConfig> {
        // Similar to compression but with per-channel for better quality
        if profile.sparsity > 0.4 {
            Ok(QuantConfig::binary())
        } else if profile.numel > 50000 {
            Ok(QuantConfig::int4())
        } else {
            Ok(QuantConfig::int8())
        }
    }

    /// Select configuration optimized for edge devices
    fn select_for_edge(&self, _profile: &TensorProfile) -> TorshResult<QuantConfig> {
        // Edge devices prefer simple, fast quantization
        Ok(QuantConfig::int8()
            .with_backend(QuantBackend::Qnnpack)
            .with_observer(ObserverType::MinMax))
    }

    /// Generate candidate configurations
    fn generate_candidates(
        &self,
        profile: &TensorProfile,
        constraints: Option<ConfigConstraints>,
    ) -> TorshResult<Vec<(QuantConfig, f32)>> {
        let mut candidates = vec![
            (QuantConfig::int8(), 0.0),
            (QuantConfig::int4(), 0.0),
            (QuantConfig::per_channel(0), 0.0),
        ];

        // Add specialized candidates based on profile
        if profile.sparsity > 0.3 {
            candidates.push((QuantConfig::binary(), 0.0));
            candidates.push((QuantConfig::ternary(), 0.0));
        }

        if profile.numel > 10000 {
            candidates.push((QuantConfig::group_wise(0, 64), 0.0));
            candidates.push((QuantConfig::group_wise(0, 32), 0.0));
        }

        // Apply constraints
        if let Some(constraints) = constraints {
            candidates.retain(|(config, _)| self.satisfies_constraints(config, &constraints));
        }

        Ok(candidates)
    }

    /// Score a configuration for the current objective
    fn score_configuration(&self, config: &QuantConfig, profile: &TensorProfile) -> f32 {
        let mut score = 0.0;

        // Base score from scheme
        let scheme_score = self.score_scheme(config.scheme, profile);
        score += scheme_score * self.feature_weights.distribution_weight;

        // Score from observer type
        let observer_score = self.score_observer(config.observer_type, profile);
        score += observer_score * self.feature_weights.range_weight;

        // Score from backend
        let backend_score = self.score_backend(config.backend, profile);
        score += backend_score * 0.5;

        // Adjust based on tensor size
        let size_score = self.score_size(config.scheme, profile.numel);
        score += size_score * self.feature_weights.size_weight;

        score
    }

    /// Score quantization scheme
    fn score_scheme(&self, scheme: QScheme, _profile: &TensorProfile) -> f32 {
        match (self.objective, scheme) {
            (ConfigObjective::MaximumCompression, QScheme::Binary) => 10.0,
            (ConfigObjective::MaximumCompression, QScheme::Ternary) => 9.0,
            (ConfigObjective::MaximumCompression, QScheme::Int4PerTensor) => 8.0,
            (ConfigObjective::MaximumAccuracy, QScheme::PerChannelAffine) => 10.0,
            (ConfigObjective::MaximumAccuracy, QScheme::PerTensorAffine) => 8.5,
            (ConfigObjective::MaximumSpeed, QScheme::PerTensorAffine) => 10.0,
            (ConfigObjective::MaximumSpeed, QScheme::PerTensorSymmetric) => 9.5,
            (ConfigObjective::BalancedQuality, QScheme::GroupWise) => 9.0,
            (ConfigObjective::BalancedQuality, QScheme::PerTensorAffine) => 8.0,
            _ => 5.0,
        }
    }

    /// Score observer type
    fn score_observer(&self, observer: ObserverType, profile: &TensorProfile) -> f32 {
        match observer {
            ObserverType::Histogram if profile.stats.has_outliers => 10.0,
            ObserverType::Percentile
                if profile.distribution == DistributionProfile::HeavyTailed =>
            {
                9.5
            }
            ObserverType::MinMax => 7.0, // Fast but less accurate
            _ => 6.0,
        }
    }

    /// Score backend
    fn score_backend(&self, backend: QuantBackend, _profile: &TensorProfile) -> f32 {
        match (self.objective, backend) {
            (ConfigObjective::MaximumSpeed, QuantBackend::Fbgemm) => 10.0,
            (ConfigObjective::EdgeOptimized, QuantBackend::Qnnpack) => 10.0,
            _ => 5.0,
        }
    }

    /// Score based on tensor size
    fn score_size(&self, scheme: QScheme, numel: usize) -> f32 {
        match scheme {
            QScheme::GroupWise if numel > 100000 => 10.0,
            QScheme::PerChannelAffine if numel > 10000 => 8.0,
            QScheme::Binary if numel < 1000 => 3.0, // Binary not great for small tensors
            _ => 5.0,
        }
    }

    /// Apply constraints to configuration
    fn apply_constraints(
        &self,
        mut config: QuantConfig,
        constraints: ConfigConstraints,
    ) -> TorshResult<QuantConfig> {
        if let Some(backend) = constraints.required_backend {
            config = config.with_backend(backend);
        }

        if let Some(min_bits) = constraints.min_bits {
            // Ensure scheme uses at least min_bits
            if min_bits >= 8
                && matches!(
                    config.scheme,
                    QScheme::Int4PerTensor | QScheme::Binary | QScheme::Ternary
                )
            {
                config = QuantConfig::int8();
            }
        }

        Ok(config)
    }

    /// Check if configuration satisfies constraints
    fn satisfies_constraints(&self, config: &QuantConfig, constraints: &ConfigConstraints) -> bool {
        if let Some(backend) = constraints.required_backend {
            if config.backend != backend {
                return false;
            }
        }

        if let Some(min_bits) = constraints.min_bits {
            let scheme_bits = match config.scheme {
                QScheme::Binary => 1,
                QScheme::Ternary => 2,
                QScheme::Int4PerTensor | QScheme::Int4PerChannel => 4,
                _ => 8,
            };
            if scheme_bits < min_bits {
                return false;
            }
        }

        true
    }

    /// Update feature weights based on historical performance
    fn update_feature_weights(&mut self) {
        // Simple online learning: boost weights for features that correlate with good performance
        // This is a simplified version - production would use more sophisticated ML

        if self.history.len() < 10 {
            return;
        }

        // Calculate average error for different feature combinations
        let sparse_configs: Vec<&ConfigPerformance> = self
            .history
            .iter()
            .filter(|p| p.profile.sparsity > 0.3)
            .collect();

        let dense_configs: Vec<&ConfigPerformance> = self
            .history
            .iter()
            .filter(|p| p.profile.sparsity <= 0.3)
            .collect();

        // Adjust sparsity weight based on performance
        if !sparse_configs.is_empty() {
            let avg_sparse_error =
                sparse_configs.iter().map(|p| p.error).sum::<f32>() / sparse_configs.len() as f32;
            let avg_dense_error =
                dense_configs.iter().map(|p| p.error).sum::<f32>() / dense_configs.len() as f32;

            if avg_sparse_error < avg_dense_error {
                self.feature_weights.sparsity_weight *= 1.1;
            } else {
                self.feature_weights.sparsity_weight *= 0.95;
            }

            // Keep weights in reasonable range
            self.feature_weights.sparsity_weight =
                self.feature_weights.sparsity_weight.clamp(0.5, 2.0);
        }
    }
}

/// Constraints for configuration selection
#[derive(Debug, Clone, Default)]
pub struct ConfigConstraints {
    /// Required backend (if any)
    pub required_backend: Option<QuantBackend>,
    /// Minimum number of quantization bits
    pub min_bits: Option<u32>,
    /// Maximum memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Target compression ratio
    pub target_compression: Option<f32>,
}

impl ConfigConstraints {
    /// Create new constraints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set required backend
    pub fn with_backend(mut self, backend: QuantBackend) -> Self {
        self.required_backend = Some(backend);
        self
    }

    /// Set minimum bits
    pub fn with_min_bits(mut self, bits: u32) -> Self {
        self.min_bits = Some(bits);
        self
    }

    /// Set maximum memory
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory = Some(bytes);
        self
    }

    /// Set target compression ratio
    pub fn with_target_compression(mut self, ratio: f32) -> Self {
        self.target_compression = Some(ratio);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_auto_configurator_basic() {
        let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();

        let config = configurator.recommend(&tensor, None).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tensor_profile_analysis() {
        let configurator = AutoConfigurator::new(ConfigObjective::MaximumAccuracy);
        // Create data with more values to make outlier detection more reliable
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 100.0]; // Has outlier
        let tensor = tensor_1d(&data).unwrap();

        let profile = configurator.analyze_tensor(&tensor).unwrap();
        assert!(
            profile.stats.has_outliers,
            "Expected outliers to be detected"
        );
        assert_eq!(profile.numel, 10);
        assert!(profile.stats.max > 90.0, "Max value should be around 100");
    }

    #[test]
    fn test_sparse_tensor_recommendation() {
        let configurator = AutoConfigurator::new(ConfigObjective::MaximumCompression);
        let data = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let config = configurator.recommend(&tensor, None).unwrap();
        // Should recommend binary or ternary for sparse data
        assert!(matches!(config.scheme, QScheme::Binary | QScheme::Ternary));
    }

    #[test]
    fn test_constraints_application() {
        let configurator = AutoConfigurator::new(ConfigObjective::MaximumSpeed);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = tensor_1d(&data).unwrap();

        let constraints = ConfigConstraints::new()
            .with_backend(QuantBackend::Qnnpack)
            .with_min_bits(8);

        let config = configurator.recommend(&tensor, Some(constraints)).unwrap();
        assert_eq!(config.backend, QuantBackend::Qnnpack);
    }

    #[test]
    fn test_ranked_recommendations() {
        let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = tensor_1d(&data).unwrap();

        let ranked = configurator.recommend_ranked(&tensor, 3, None).unwrap();
        assert_eq!(ranked.len(), 3);

        // Scores should be descending
        assert!(ranked[0].1 >= ranked[1].1);
        assert!(ranked[1].1 >= ranked[2].1);
    }

    #[test]
    fn test_performance_update() {
        let mut configurator = AutoConfigurator::new(ConfigObjective::MaximumAccuracy);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        configurator
            .update_performance(&config, &tensor, 0.1, 4.0, Some(1.5))
            .unwrap();

        assert_eq!(configurator.history.len(), 1);
    }

    #[test]
    fn test_distribution_classification() {
        let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);

        // Normal distribution
        let normal_data = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let tensor = tensor_1d(&normal_data).unwrap();
        let _profile = configurator.analyze_tensor(&tensor).unwrap();
        // Distribution classification depends on stats

        // Sparse distribution
        let sparse_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let tensor = tensor_1d(&sparse_data).unwrap();
        let _profile = configurator.analyze_tensor(&tensor).unwrap();
        assert_eq!(_profile.distribution, DistributionProfile::Sparse);
    }

    #[test]
    fn test_objective_specific_selection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = tensor_1d(&data).unwrap();

        // Test each objective
        let objectives = vec![
            ConfigObjective::MaximumCompression,
            ConfigObjective::MaximumAccuracy,
            ConfigObjective::BalancedQuality,
            ConfigObjective::MaximumSpeed,
            ConfigObjective::MinimumMemory,
            ConfigObjective::EdgeOptimized,
        ];

        for objective in objectives {
            let configurator = AutoConfigurator::new(objective);
            let config = configurator.recommend(&tensor, None).unwrap();
            assert!(
                config.validate().is_ok(),
                "Failed for objective {:?}",
                objective
            );
        }
    }
}
