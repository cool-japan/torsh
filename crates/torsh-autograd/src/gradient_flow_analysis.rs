// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient Flow Analysis and Reporting
//!
//! This module provides comprehensive analysis of gradient flow through
//! computation graphs, identifying issues like vanishing/exploding gradients,
//! dead neurons, and optimization bottlenecks.
//!
//! # Features
//!
//! - **Gradient Magnitude Analysis**: Track gradient magnitudes across layers
//! - **Flow Path Analysis**: Identify critical paths for gradient flow
//! - **Anomaly Detection**: Detect vanishing, exploding, and dead gradients
//! - **Layer-wise Statistics**: Per-layer gradient statistics and health metrics
//! - **Temporal Analysis**: Track gradient evolution over training steps
//! - **Visualization**: Generate gradient flow visualizations and reports

use crate::common_utils::*;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

/// Gradient flow analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowAnalysis {
    /// Timestamp of analysis
    pub timestamp: DateTime<Utc>,

    /// Training step/iteration
    pub step: usize,

    /// Layer-wise gradient statistics
    pub layer_statistics: Vec<LayerGradientStats>,

    /// Detected anomalies
    pub anomalies: Vec<GradientAnomaly>,

    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Critical paths (layers with issues)
    pub critical_paths: Vec<String>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Per-layer gradient statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientStats {
    /// Layer name/identifier
    pub layer_name: String,

    /// Layer type (conv, linear, etc.)
    pub layer_type: String,

    /// Layer depth in network
    pub depth: usize,

    /// Gradient magnitude statistics
    pub magnitude: Statistics,

    /// Percentage of zero gradients
    pub zero_ratio: f64,

    /// Percentage of NaN gradients
    pub nan_ratio: f64,

    /// Percentage of infinite gradients
    pub inf_ratio: f64,

    /// Gradient-to-parameter ratio
    pub grad_to_param_ratio: f64,

    /// Flow efficiency (0.0 to 1.0)
    pub flow_efficiency: f64,

    /// Number of parameters
    pub parameter_count: usize,
}

/// Gradient anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnomaly {
    /// Anomaly type
    pub anomaly_type: GradientAnomalyType,

    /// Affected layer
    pub layer_name: String,

    /// Severity (0.0 to 1.0, higher is more severe)
    pub severity: f64,

    /// Description
    pub description: String,

    /// Suggested fix
    pub suggestion: String,
}

/// Type of gradient anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientAnomalyType {
    /// Vanishing gradients (too small)
    Vanishing,

    /// Exploding gradients (too large)
    Exploding,

    /// Dead neurons (zero gradients)
    Dead,

    /// Unstable gradients (high variance)
    Unstable,

    /// NaN gradients
    NaN,

    /// Infinite gradients
    Infinity,

    /// Imbalanced flow
    Imbalanced,
}

/// Gradient flow analyzer
pub struct GradientFlowAnalyzer {
    /// Analysis history
    history: Arc<Mutex<VecDeque<GradientFlowAnalysis>>>,

    /// Configuration
    config: AnalysisConfig,

    /// Maximum history size
    max_history: usize,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Threshold for vanishing gradients
    pub vanishing_threshold: f64,

    /// Threshold for exploding gradients
    pub exploding_threshold: f64,

    /// Threshold for dead neuron detection (zero ratio)
    pub dead_threshold: f64,

    /// Threshold for high variance (unstable)
    pub variance_threshold: f64,

    /// Minimum flow efficiency
    pub min_flow_efficiency: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 100.0,
            dead_threshold: 0.9, // 90% zeros
            variance_threshold: 10.0,
            min_flow_efficiency: 0.3,
        }
    }
}

impl GradientFlowAnalyzer {
    /// Create a new gradient flow analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            history: Arc::new(Mutex::new(VecDeque::new())),
            config,
            max_history: 1000,
        }
    }

    /// Analyze gradient flow for a training step
    pub fn analyze(
        &self,
        step: usize,
        layer_gradients: Vec<LayerGradients>,
    ) -> GradientFlowAnalysis {
        let mut layer_statistics = Vec::new();
        let mut anomalies = Vec::new();
        let mut critical_paths = Vec::new();

        for (depth, layer) in layer_gradients.iter().enumerate() {
            let stats = self.compute_layer_statistics(layer, depth);

            // Detect anomalies
            let layer_anomalies = self.detect_anomalies(&stats);
            if !layer_anomalies.is_empty() {
                critical_paths.push(layer.layer_name.clone());
                anomalies.extend(layer_anomalies);
            }

            layer_statistics.push(stats);
        }

        // Calculate overall health score
        let health_score = self.calculate_health_score(&layer_statistics);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&layer_statistics, &anomalies);

        let analysis = GradientFlowAnalysis {
            timestamp: Utc::now(),
            step,
            layer_statistics,
            anomalies,
            health_score,
            critical_paths,
            recommendations,
        };

        // Add to history
        let mut history = self.history.lock();
        history.push_back(analysis.clone());

        while history.len() > self.max_history {
            history.pop_front();
        }

        analysis
    }

    /// Compute statistics for a single layer
    fn compute_layer_statistics(&self, layer: &LayerGradients, depth: usize) -> LayerGradientStats {
        let gradients = &layer.gradients;

        // Calculate magnitude statistics
        let magnitudes: Vec<f64> = gradients.iter().map(|g| g.abs()).collect();
        let magnitude = calculate_statistics(&magnitudes);

        // Calculate ratios
        let total = gradients.len() as f64;
        let zero_count = gradients.iter().filter(|&&g| g == 0.0).count() as f64;
        let nan_count = gradients.iter().filter(|&&g| g.is_nan()).count() as f64;
        let inf_count = gradients.iter().filter(|&&g| g.is_infinite()).count() as f64;

        let zero_ratio = zero_count / total;
        let nan_ratio = nan_count / total;
        let inf_ratio = inf_count / total;

        // Calculate grad-to-param ratio
        let grad_to_param_ratio = if !layer.parameters.is_empty() {
            let param_mean = mean(&layer.parameters);
            if param_mean != 0.0 {
                magnitude.mean / param_mean.abs()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate flow efficiency
        let flow_efficiency =
            self.calculate_flow_efficiency(magnitude.mean, zero_ratio, magnitude.std);

        LayerGradientStats {
            layer_name: layer.layer_name.clone(),
            layer_type: layer.layer_type.clone(),
            depth,
            magnitude,
            zero_ratio,
            nan_ratio,
            inf_ratio,
            grad_to_param_ratio,
            flow_efficiency,
            parameter_count: layer.parameters.len(),
        }
    }

    /// Detect gradient anomalies in a layer
    fn detect_anomalies(&self, stats: &LayerGradientStats) -> Vec<GradientAnomaly> {
        let mut anomalies = Vec::new();

        // Check for vanishing gradients
        if stats.magnitude.mean < self.config.vanishing_threshold {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::Vanishing,
                layer_name: stats.layer_name.clone(),
                severity: 1.0 - (stats.magnitude.mean / self.config.vanishing_threshold).min(1.0),
                description: format!(
                    "Vanishing gradients detected (mean: {:.2e})",
                    stats.magnitude.mean
                ),
                suggestion: "Consider: 1) Using ReLU/LeakyReLU instead of sigmoid/tanh, 2) Batch normalization, 3) Residual connections, 4) Gradient clipping".to_string(),
            });
        }

        // Check for exploding gradients
        if stats.magnitude.mean > self.config.exploding_threshold {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::Exploding,
                layer_name: stats.layer_name.clone(),
                severity: (stats.magnitude.mean / self.config.exploding_threshold - 1.0).min(1.0),
                description: format!(
                    "Exploding gradients detected (mean: {:.2e})",
                    stats.magnitude.mean
                ),
                suggestion: "Consider: 1) Gradient clipping, 2) Lower learning rate, 3) Better weight initialization, 4) Batch normalization".to_string(),
            });
        }

        // Check for dead neurons
        if stats.zero_ratio > self.config.dead_threshold {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::Dead,
                layer_name: stats.layer_name.clone(),
                severity: stats.zero_ratio,
                description: format!(
                    "Dead neurons detected ({:.1}% zero gradients)",
                    stats.zero_ratio * 100.0
                ),
                suggestion: "Consider: 1) Using LeakyReLU/PReLU instead of ReLU, 2) Lower learning rate, 3) Better weight initialization".to_string(),
            });
        }

        // Check for NaN
        if stats.nan_ratio > 0.0 {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::NaN,
                layer_name: stats.layer_name.clone(),
                severity: 1.0,
                description: format!(
                    "NaN gradients detected ({:.1}% NaN)",
                    stats.nan_ratio * 100.0
                ),
                suggestion: "Critical issue! Check for: 1) Division by zero, 2) Log of negative numbers, 3) Numerical overflow, 4) Invalid operations".to_string(),
            });
        }

        // Check for infinite gradients
        if stats.inf_ratio > 0.0 {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::Infinity,
                layer_name: stats.layer_name.clone(),
                severity: 1.0,
                description: format!(
                    "Infinite gradients detected ({:.1}% inf)",
                    stats.inf_ratio * 100.0
                ),
                suggestion: "Critical issue! Check for: 1) Numerical overflow, 2) Extreme weight values, 3) Invalid operations".to_string(),
            });
        }

        // Check for unstable gradients (high variance)
        if stats.magnitude.std > self.config.variance_threshold * stats.magnitude.mean {
            anomalies.push(GradientAnomaly {
                anomaly_type: GradientAnomalyType::Unstable,
                layer_name: stats.layer_name.clone(),
                severity: (stats.magnitude.std
                    / (self.config.variance_threshold * stats.magnitude.mean))
                    .min(1.0),
                description: "Unstable gradients (high variance)".to_string(),
                suggestion:
                    "Consider: 1) Batch normalization, 2) Gradient clipping, 3) Lower learning rate"
                        .to_string(),
            });
        }

        anomalies
    }

    /// Calculate flow efficiency for a layer
    fn calculate_flow_efficiency(&self, mean_magnitude: f64, zero_ratio: f64, std_dev: f64) -> f64 {
        // Flow efficiency based on:
        // 1. Non-zero gradients (1 - zero_ratio)
        // 2. Magnitude in reasonable range (not too small or large)
        // 3. Low variance (stable flow)

        let non_zero_factor = 1.0 - zero_ratio;

        let magnitude_factor = if mean_magnitude < self.config.vanishing_threshold {
            mean_magnitude / self.config.vanishing_threshold
        } else if mean_magnitude > self.config.exploding_threshold {
            self.config.exploding_threshold / mean_magnitude
        } else {
            1.0
        };

        let stability_factor = if mean_magnitude > 0.0 {
            1.0 / (1.0 + std_dev / mean_magnitude)
        } else {
            0.0
        };

        (non_zero_factor * 0.4 + magnitude_factor * 0.4 + stability_factor * 0.2).clamp(0.0, 1.0)
    }

    /// Calculate overall health score
    fn calculate_health_score(&self, layer_stats: &[LayerGradientStats]) -> f64 {
        if layer_stats.is_empty() {
            return 0.0;
        }

        let avg_efficiency: f64 =
            layer_stats.iter().map(|s| s.flow_efficiency).sum::<f64>() / layer_stats.len() as f64;

        let has_nan = layer_stats.iter().any(|s| s.nan_ratio > 0.0);
        let has_inf = layer_stats.iter().any(|s| s.inf_ratio > 0.0);

        let critical_penalty = if has_nan || has_inf { 0.5 } else { 0.0 };

        (avg_efficiency - critical_penalty).clamp(0.0, 1.0)
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        &self,
        layer_stats: &[LayerGradientStats],
        anomalies: &[GradientAnomaly],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for systematic issues
        let vanishing_count = anomalies
            .iter()
            .filter(|a| a.anomaly_type == GradientAnomalyType::Vanishing)
            .count();

        let exploding_count = anomalies
            .iter()
            .filter(|a| a.anomaly_type == GradientAnomalyType::Exploding)
            .count();

        let dead_count = anomalies
            .iter()
            .filter(|a| a.anomaly_type == GradientAnomalyType::Dead)
            .count();

        if vanishing_count > layer_stats.len() / 2 {
            recommendations.push(
                "Widespread vanishing gradients detected. Consider using residual connections or reducing network depth.".to_string()
            );
        }

        if exploding_count > 0 {
            recommendations.push(
                "Exploding gradients detected. Implement gradient clipping or reduce learning rate.".to_string()
            );
        }

        if dead_count > layer_stats.len() / 3 {
            recommendations.push(
                "Many dead neurons detected. Consider using LeakyReLU or better weight initialization.".to_string()
            );
        }

        // Check for critical issues
        if anomalies
            .iter()
            .any(|a| a.anomaly_type == GradientAnomalyType::NaN)
        {
            recommendations.push(
                "CRITICAL: NaN gradients detected. Training may be unstable. Check for invalid operations.".to_string()
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Gradient flow looks healthy!".to_string());
        }

        recommendations
    }

    /// Get analysis history
    pub fn history(&self) -> Vec<GradientFlowAnalysis> {
        self.history.lock().iter().cloned().collect()
    }

    /// Generate flow report
    pub fn generate_report(&self, analysis: &GradientFlowAnalysis) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== Gradient Flow Analysis Report ===\n"));
        report.push_str(&format!("Step: {}\n", analysis.step));
        report.push_str(&format!("Timestamp: {}\n", analysis.timestamp));
        report.push_str(&format!(
            "Health Score: {:.2}%\n\n",
            analysis.health_score * 100.0
        ));

        report.push_str("Layer Statistics:\n");
        for (i, layer) in analysis.layer_statistics.iter().enumerate() {
            report.push_str(&format!(
                "  [{}] {} ({})\n",
                i, layer.layer_name, layer.layer_type
            ));
            report.push_str(&format!(
                "      Magnitude: mean={:.2e}, std={:.2e}, min={:.2e}, max={:.2e}\n",
                layer.magnitude.mean, layer.magnitude.std, layer.magnitude.min, layer.magnitude.max
            ));
            report.push_str(&format!(
                "      Zero ratio: {:.1}%, Flow efficiency: {:.1}%\n",
                layer.zero_ratio * 100.0,
                layer.flow_efficiency * 100.0
            ));
        }

        if !analysis.anomalies.is_empty() {
            report.push_str(&format!(
                "\nDetected Anomalies ({}):\n",
                analysis.anomalies.len()
            ));
            for anomaly in &analysis.anomalies {
                report.push_str(&format!(
                    "  [{:?}] {} (severity: {:.2})\n",
                    anomaly.anomaly_type, anomaly.layer_name, anomaly.severity
                ));
                report.push_str(&format!("    {}\n", anomaly.description));
                report.push_str(&format!("    Suggestion: {}\n", anomaly.suggestion));
            }
        }

        if !analysis.recommendations.is_empty() {
            report.push_str("\nRecommendations:\n");
            for (i, rec) in analysis.recommendations.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        report
    }
}

/// Layer gradients for analysis
#[derive(Debug, Clone)]
pub struct LayerGradients {
    /// Layer name
    pub layer_name: String,

    /// Layer type
    pub layer_type: String,

    /// Gradient values
    pub gradients: Vec<f64>,

    /// Parameter values (for comparison)
    pub parameters: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_gradients() -> Vec<LayerGradients> {
        vec![
            LayerGradients {
                layer_name: "layer1".to_string(),
                layer_type: "linear".to_string(),
                gradients: vec![0.1, 0.2, 0.15, 0.18, 0.12],
                parameters: vec![1.0, 1.5, 1.2, 1.3, 1.1],
            },
            LayerGradients {
                layer_name: "layer2".to_string(),
                layer_type: "linear".to_string(),
                gradients: vec![0.05, 0.06, 0.04, 0.055, 0.045],
                parameters: vec![0.8, 0.9, 0.85, 0.82, 0.88],
            },
        ]
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());
        assert_eq!(analyzer.history().len(), 0);
    }

    #[test]
    fn test_gradient_analysis() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());
        let gradients = create_test_gradients();

        let analysis = analyzer.analyze(0, gradients);

        assert_eq!(analysis.step, 0);
        assert_eq!(analysis.layer_statistics.len(), 2);
        assert!(analysis.health_score >= 0.0 && analysis.health_score <= 1.0);
    }

    #[test]
    fn test_vanishing_gradient_detection() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());

        let gradients = vec![LayerGradients {
            layer_name: "vanishing_layer".to_string(),
            layer_type: "linear".to_string(),
            gradients: vec![1e-10, 1e-10, 1e-10], // Very small gradients
            parameters: vec![1.0, 1.0, 1.0],
        }];

        let analysis = analyzer.analyze(0, gradients);

        assert!(analysis
            .anomalies
            .iter()
            .any(|a| a.anomaly_type == GradientAnomalyType::Vanishing));
    }

    #[test]
    fn test_exploding_gradient_detection() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());

        let gradients = vec![LayerGradients {
            layer_name: "exploding_layer".to_string(),
            layer_type: "linear".to_string(),
            gradients: vec![1000.0, 1500.0, 1200.0], // Very large gradients
            parameters: vec![1.0, 1.0, 1.0],
        }];

        let analysis = analyzer.analyze(0, gradients);

        assert!(analysis
            .anomalies
            .iter()
            .any(|a| a.anomaly_type == GradientAnomalyType::Exploding));
    }

    #[test]
    fn test_dead_neuron_detection() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());

        let gradients = vec![LayerGradients {
            layer_name: "dead_layer".to_string(),
            layer_type: "linear".to_string(),
            gradients: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], // >90% zeros
            parameters: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }];

        let analysis = analyzer.analyze(0, gradients);

        assert!(analysis
            .anomalies
            .iter()
            .any(|a| a.anomaly_type == GradientAnomalyType::Dead));
    }

    #[test]
    fn test_report_generation() {
        let analyzer = GradientFlowAnalyzer::new(AnalysisConfig::default());
        let gradients = create_test_gradients();

        let analysis = analyzer.analyze(0, gradients);
        let report = analyzer.generate_report(&analysis);

        assert!(report.contains("Gradient Flow Analysis Report"));
        assert!(report.contains("Step: 0"));
        assert!(report.contains("Health Score"));
    }
}
