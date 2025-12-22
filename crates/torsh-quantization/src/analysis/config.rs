//! Configuration and core types for quantization analysis

use crate::QScheme;
use std::collections::HashMap;

/// Configuration parameters for sensitivity analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Threshold for considering a layer as high sensitivity (default: 0.05)
    pub sensitivity_threshold: f32,
    /// Threshold for keeping layers in FP32 (default: 0.05)
    pub fp32_threshold: f32,
    /// Threshold for aggressive quantization candidates (default: 0.01)
    pub aggressive_threshold: f32,
    /// Maximum acceptable accuracy drop percentage (default: 5.0%)
    pub max_accuracy_drop_percent: f32,
    /// Weights for efficiency score calculation
    pub efficiency_weights: EfficiencyWeights,
    /// Normalization factors for efficiency score
    pub normalization_factors: NormalizationFactors,
}

/// Weights for efficiency score calculation
#[derive(Debug, Clone)]
pub struct EfficiencyWeights {
    /// Weight for accuracy in efficiency score (default: 0.5)
    pub accuracy: f32,
    /// Weight for size reduction in efficiency score (default: 0.3)
    pub size: f32,
    /// Weight for speed improvement in efficiency score (default: 0.2)
    pub speed: f32,
}

/// Normalization factors for efficiency score
#[derive(Debug, Clone)]
pub struct NormalizationFactors {
    /// Maximum expected size reduction factor (default: 8.0)
    pub max_size_reduction: f32,
    /// Maximum expected speed improvement factor (default: 10.0)
    pub max_speed_improvement: f32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            sensitivity_threshold: 0.05,
            fp32_threshold: 0.05,
            aggressive_threshold: 0.01,
            max_accuracy_drop_percent: 5.0,
            efficiency_weights: EfficiencyWeights::default(),
            normalization_factors: NormalizationFactors::default(),
        }
    }
}

impl Default for EfficiencyWeights {
    fn default() -> Self {
        Self {
            accuracy: 0.5,
            size: 0.3,
            speed: 0.2,
        }
    }
}

impl Default for NormalizationFactors {
    fn default() -> Self {
        Self {
            max_size_reduction: 8.0,
            max_speed_improvement: 10.0,
        }
    }
}

impl AnalysisConfig {
    /// Create a new analysis configuration with custom sensitivity thresholds
    pub fn with_sensitivity_thresholds(
        sensitivity_threshold: f32,
        fp32_threshold: f32,
        aggressive_threshold: f32,
    ) -> Self {
        Self {
            sensitivity_threshold,
            fp32_threshold,
            aggressive_threshold,
            ..Default::default()
        }
    }

    /// Create a new analysis configuration with custom efficiency weights
    pub fn with_efficiency_weights(accuracy: f32, size: f32, speed: f32) -> Self {
        Self {
            efficiency_weights: EfficiencyWeights {
                accuracy,
                size,
                speed,
            },
            ..Default::default()
        }
    }

    /// Create a conservative analysis configuration (higher thresholds)
    pub fn conservative() -> Self {
        Self {
            sensitivity_threshold: 0.02,
            fp32_threshold: 0.02,
            aggressive_threshold: 0.005,
            max_accuracy_drop_percent: 2.0,
            ..Default::default()
        }
    }

    /// Create an aggressive analysis configuration (lower thresholds)
    pub fn aggressive() -> Self {
        Self {
            sensitivity_threshold: 0.1,
            fp32_threshold: 0.1,
            aggressive_threshold: 0.05,
            max_accuracy_drop_percent: 10.0,
            ..Default::default()
        }
    }
}

/// Results of sensitivity analysis for a single layer
#[derive(Debug, Clone)]
pub struct LayerSensitivityResult {
    /// Layer name or identifier
    pub layer_name: String,
    /// Original accuracy (before quantization)
    pub original_accuracy: f32,
    /// Accuracy after quantizing this layer
    pub quantized_accuracy: f32,
    /// Sensitivity score (accuracy drop)
    pub sensitivity_score: f32,
    /// Recommended quantization scheme for this layer
    pub recommended_scheme: QScheme,
    /// Whether this layer should be kept in full precision
    pub keep_fp32: bool,
}

impl LayerSensitivityResult {
    /// Create a new sensitivity result
    pub fn new(layer_name: String, original_accuracy: f32, quantized_accuracy: f32) -> Self {
        Self::new_with_config(
            layer_name,
            original_accuracy,
            quantized_accuracy,
            &AnalysisConfig::default(),
        )
    }

    /// Create a new sensitivity result with custom analysis configuration
    pub fn new_with_config(
        layer_name: String,
        original_accuracy: f32,
        quantized_accuracy: f32,
        config: &AnalysisConfig,
    ) -> Self {
        let sensitivity_score = original_accuracy - quantized_accuracy;
        let keep_fp32 = sensitivity_score > config.fp32_threshold;
        let recommended_scheme = Self::determine_recommended_scheme(sensitivity_score, config);

        Self {
            layer_name,
            original_accuracy,
            quantized_accuracy,
            sensitivity_score,
            recommended_scheme,
            keep_fp32,
        }
    }

    /// Determine the recommended quantization scheme based on sensitivity and configuration
    fn determine_recommended_scheme(sensitivity_score: f32, config: &AnalysisConfig) -> QScheme {
        if sensitivity_score > config.fp32_threshold {
            // High sensitivity - use conservative quantization
            QScheme::PerTensorAffine
        } else if sensitivity_score > config.aggressive_threshold {
            // Medium sensitivity - use per-channel for better accuracy
            QScheme::PerChannelAffine
        } else if sensitivity_score > config.aggressive_threshold / 2.0 {
            // Low sensitivity - can use INT4
            QScheme::Int4PerTensor
        } else {
            // Very low sensitivity - can use aggressive quantization
            QScheme::Int4PerChannel
        }
    }

    /// Get the accuracy drop percentage
    pub fn accuracy_drop_percentage(&self) -> f32 {
        (self.sensitivity_score / self.original_accuracy) * 100.0
    }

    /// Check if this layer is highly sensitive to quantization
    pub fn is_high_sensitivity(&self) -> bool {
        self.is_high_sensitivity_with_config(&AnalysisConfig::default())
    }

    /// Check if this layer is highly sensitive to quantization with custom config
    pub fn is_high_sensitivity_with_config(&self, config: &AnalysisConfig) -> bool {
        self.sensitivity_score > config.sensitivity_threshold
            || self.accuracy_drop_percentage() > config.max_accuracy_drop_percent
    }
}

/// Comprehensive sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResults {
    /// Results for individual layers
    pub layer_results: Vec<LayerSensitivityResult>,
    /// Overall model sensitivity summary
    pub overall_sensitivity: f32,
    /// Most sensitive layers (top 10% by sensitivity score)
    pub most_sensitive_layers: Vec<String>,
    /// Least sensitive layers (suitable for aggressive quantization)
    pub least_sensitive_layers: Vec<String>,
    /// Recommended mixed precision configuration
    pub recommended_config: HashMap<String, QScheme>,
}

impl SensitivityAnalysisResults {
    /// Create a new sensitivity analysis results
    pub fn new(layer_results: Vec<LayerSensitivityResult>) -> Self {
        let overall_sensitivity = if layer_results.is_empty() {
            0.0
        } else {
            layer_results
                .iter()
                .map(|r| r.sensitivity_score)
                .sum::<f32>()
                / layer_results.len() as f32
        };

        // Sort layers by sensitivity
        let mut sorted_results = layer_results.clone();
        sorted_results.sort_by(|a, b| {
            b.sensitivity_score
                .partial_cmp(&a.sensitivity_score)
                .unwrap()
        });

        let num_layers = sorted_results.len();
        let top_10_percent = (num_layers as f32 * 0.1).ceil() as usize;
        let bottom_10_percent = (num_layers as f32 * 0.1).ceil() as usize;

        let most_sensitive_layers = sorted_results
            .iter()
            .take(top_10_percent)
            .map(|r| r.layer_name.clone())
            .collect();

        let least_sensitive_layers = sorted_results
            .iter()
            .rev()
            .take(bottom_10_percent)
            .map(|r| r.layer_name.clone())
            .collect();

        // Generate recommended configuration
        let mut recommended_config = HashMap::new();
        for result in &layer_results {
            recommended_config.insert(result.layer_name.clone(), result.recommended_scheme);
        }

        Self {
            layer_results,
            overall_sensitivity,
            most_sensitive_layers,
            least_sensitive_layers,
            recommended_config,
        }
    }

    /// Get layers that should be kept in FP32
    pub fn get_fp32_layers(&self) -> Vec<&String> {
        self.layer_results
            .iter()
            .filter(|r| r.keep_fp32)
            .map(|r| &r.layer_name)
            .collect()
    }

    /// Get the average sensitivity score
    pub fn average_sensitivity(&self) -> f32 {
        self.overall_sensitivity
    }

    /// Get layers suitable for aggressive quantization (INT4 or lower)
    pub fn get_aggressive_quantization_candidates(&self) -> Vec<&String> {
        self.get_aggressive_quantization_candidates_with_config(&AnalysisConfig::default())
    }

    /// Get layers suitable for aggressive quantization with custom config
    pub fn get_aggressive_quantization_candidates_with_config(
        &self,
        config: &AnalysisConfig,
    ) -> Vec<&String> {
        self.layer_results
            .iter()
            .filter(|r| r.sensitivity_score < config.aggressive_threshold)
            .map(|r| &r.layer_name)
            .collect()
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        format!(
            "Sensitivity Analysis Summary:\n\
             - Total layers analyzed: {}\n\
             - Average sensitivity: {:.4}\n\
             - Most sensitive layers ({}):\n{}\n\
             - Least sensitive layers ({}):\n{}\n\
             - Layers recommended for FP32: {}",
            self.layer_results.len(),
            self.overall_sensitivity,
            self.most_sensitive_layers.len(),
            self.most_sensitive_layers
                .iter()
                .map(|name| format!("  - {}", name))
                .collect::<Vec<_>>()
                .join("\n"),
            self.least_sensitive_layers.len(),
            self.least_sensitive_layers
                .iter()
                .map(|name| format!("  - {}", name))
                .collect::<Vec<_>>()
                .join("\n"),
            self.get_fp32_layers().len()
        )
    }
}

/// Accuracy comparison between quantized and original models
#[derive(Debug, Clone)]
pub struct AccuracyComparison {
    /// Original model accuracy
    pub original_accuracy: f32,
    /// Quantized model accuracy
    pub quantized_accuracy: f32,
    /// Accuracy drop (original - quantized)
    pub accuracy_drop: f32,
    /// Accuracy drop as percentage
    pub accuracy_drop_percentage: f32,
    /// Whether the accuracy drop is acceptable
    pub is_acceptable: bool,
    /// Additional metrics for detailed comparison
    pub detailed_metrics: HashMap<String, f32>,
}

impl AccuracyComparison {
    /// Create a new accuracy comparison
    pub fn new(original_accuracy: f32, quantized_accuracy: f32) -> Self {
        Self::new_with_threshold(original_accuracy, quantized_accuracy, 5.0)
    }

    /// Create a new accuracy comparison with custom acceptable threshold
    pub fn new_with_threshold(
        original_accuracy: f32,
        quantized_accuracy: f32,
        acceptable_drop_percentage: f32,
    ) -> Self {
        let accuracy_drop = original_accuracy - quantized_accuracy;
        let accuracy_drop_percentage = (accuracy_drop / original_accuracy) * 100.0;
        let is_acceptable = accuracy_drop_percentage <= acceptable_drop_percentage;

        Self {
            original_accuracy,
            quantized_accuracy,
            accuracy_drop,
            accuracy_drop_percentage,
            is_acceptable,
            detailed_metrics: HashMap::new(),
        }
    }

    /// Add a detailed metric for comparison
    pub fn add_metric(&mut self, name: String, value: f32) {
        self.detailed_metrics.insert(name, value);
    }

    /// Get the efficiency score based on accuracy preservation
    pub fn efficiency_score(&self) -> f32 {
        if self.original_accuracy == 0.0 {
            0.0
        } else {
            self.quantized_accuracy / self.original_accuracy
        }
    }

    /// Check if quantization is recommended based on accuracy
    pub fn is_quantization_recommended(&self) -> bool {
        self.is_acceptable && self.efficiency_score() > 0.95
    }

    /// Generate a comparison report
    pub fn report(&self) -> String {
        let mut report = format!(
            "Accuracy Comparison Report:\n\
             - Original Accuracy: {:.4} ({:.2}%)\n\
             - Quantized Accuracy: {:.4} ({:.2}%)\n\
             - Accuracy Drop: {:.4} ({:.2}%)\n\
             - Efficiency Score: {:.4}\n\
             - Acceptable: {}\n\
             - Quantization Recommended: {}",
            self.original_accuracy,
            self.original_accuracy * 100.0,
            self.quantized_accuracy,
            self.quantized_accuracy * 100.0,
            self.accuracy_drop,
            self.accuracy_drop_percentage,
            self.efficiency_score(),
            if self.is_acceptable { "Yes" } else { "No" },
            if self.is_quantization_recommended() {
                "Yes"
            } else {
                "No"
            }
        );

        if !self.detailed_metrics.is_empty() {
            report.push_str("\n\nDetailed Metrics:");
            for (name, value) in &self.detailed_metrics {
                report.push_str(&format!("\n  - {}: {:.4}", name, value));
            }
        }

        report
    }
}
