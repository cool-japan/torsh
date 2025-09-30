//! Debugging and analysis tools for optimizers
//!
//! This module provides tools for analyzing optimizer behavior, convergence diagnostics,
//! gradient flow analysis, and optimizer state visualization.

use crate::{OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Optimizer state analyzer for debugging and diagnostics
pub struct OptimizerAnalyzer {
    /// History of optimization steps
    step_history: Vec<OptimizationStep>,
    /// Gradient statistics
    gradient_stats: GradientStatistics,
    /// Parameter statistics
    parameter_stats: ParameterStatistics,
    /// Convergence tracker
    convergence_tracker: ConvergenceTracker,
    /// Configuration
    config: AnalyzerConfig,
}

/// Configuration for optimizer analyzer
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Maximum number of steps to keep in history
    pub max_history_size: usize,
    /// Whether to track gradient norms
    pub track_gradient_norms: bool,
    /// Whether to track parameter norms
    pub track_parameter_norms: bool,
    /// Whether to track gradient flow
    pub track_gradient_flow: bool,
    /// Window size for moving averages
    pub moving_average_window: usize,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            track_gradient_norms: true,
            track_parameter_norms: true,
            track_gradient_flow: true,
            moving_average_window: 100,
        }
    }
}

/// Information about a single optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Step number
    pub step: usize,
    /// Learning rates at this step
    pub learning_rates: Vec<f32>,
    /// Gradient norms for each parameter group
    pub gradient_norms: Vec<f32>,
    /// Parameter norms for each parameter group
    pub parameter_norms: Vec<f32>,
    /// Update norms for each parameter group
    pub update_norms: Vec<f32>,
    /// Loss value if available
    pub loss: Option<f32>,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Gradient flow statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Gradient norm history
    pub norm_history: VecDeque<f32>,
    /// Average gradient norm
    pub average_norm: f32,
    /// Maximum gradient norm seen
    pub max_norm: f32,
    /// Minimum gradient norm seen
    pub min_norm: f32,
    /// Gradient explosion detection
    pub explosion_count: usize,
    /// Gradient vanishing detection
    pub vanishing_count: usize,
    /// Recent gradient variance
    pub recent_variance: f32,
}

/// Parameter statistics
#[derive(Debug, Clone)]
pub struct ParameterStatistics {
    /// Parameter norm history
    pub norm_history: VecDeque<f32>,
    /// Parameter update ratio (update_norm / param_norm)
    pub update_ratios: VecDeque<f32>,
    /// Average update ratio
    pub average_update_ratio: f32,
    /// Parameter change velocity
    pub velocity: f32,
    /// Parameter stability indicator
    pub stability_score: f32,
}

/// Convergence tracking and analysis
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    /// Loss history
    pub loss_history: VecDeque<f32>,
    /// Moving average of loss
    pub loss_moving_average: f32,
    /// Convergence rate estimate
    pub convergence_rate: f32,
    /// Whether converged
    pub is_converged: bool,
    /// Steps since improvement
    pub steps_since_improvement: usize,
    /// Best loss seen
    pub best_loss: f32,
    /// Plateau detection
    pub plateau_length: usize,
}

impl OptimizerAnalyzer {
    /// Create a new optimizer analyzer
    pub fn new(config: Option<AnalyzerConfig>) -> Self {
        let config = config.unwrap_or_default();
        let max_size = config.moving_average_window;

        Self {
            step_history: Vec::new(),
            gradient_stats: GradientStatistics {
                norm_history: VecDeque::with_capacity(max_size),
                average_norm: 0.0,
                max_norm: 0.0,
                min_norm: f32::INFINITY,
                explosion_count: 0,
                vanishing_count: 0,
                recent_variance: 0.0,
            },
            parameter_stats: ParameterStatistics {
                norm_history: VecDeque::with_capacity(max_size),
                update_ratios: VecDeque::with_capacity(max_size),
                average_update_ratio: 0.0,
                velocity: 0.0,
                stability_score: 1.0,
            },
            convergence_tracker: ConvergenceTracker {
                loss_history: VecDeque::with_capacity(max_size),
                loss_moving_average: 0.0,
                convergence_rate: 0.0,
                is_converged: false,
                steps_since_improvement: 0,
                best_loss: f32::INFINITY,
                plateau_length: 0,
            },
            config,
        }
    }

    /// Analyze a single optimization step
    pub fn analyze_step(
        &mut self,
        step: usize,
        params: &[Arc<RwLock<Tensor>>],
        learning_rates: &[f32],
        loss: Option<f32>,
    ) -> Result<()> {
        let timestamp = std::time::Instant::now();

        // Calculate gradient and parameter norms
        let mut gradient_norms = Vec::new();
        let mut parameter_norms = Vec::new();
        let mut update_norms = Vec::new();

        for param in params {
            let param_tensor = param.read();

            // Parameter norm
            let param_norm = param_tensor.norm()?;
            parameter_norms.push(param_norm.item()?);

            // Gradient norm
            if let Some(grad) = param_tensor.grad() {
                let grad_norm = grad.norm()?;
                gradient_norms.push(grad_norm.item()?);

                // Update norm (approximated as lr * grad_norm)
                let lr = learning_rates.get(0).copied().unwrap_or(0.001);
                update_norms.push(lr * grad_norm.item()?);
            } else {
                gradient_norms.push(0.0);
                update_norms.push(0.0);
            }
        }

        // Create optimization step record
        let opt_step = OptimizationStep {
            step,
            learning_rates: learning_rates.to_vec(),
            gradient_norms: gradient_norms.clone(),
            parameter_norms: parameter_norms.clone(),
            update_norms: update_norms.clone(),
            loss,
            timestamp,
        };

        // Update histories
        self.step_history.push(opt_step);
        if self.step_history.len() > self.config.max_history_size {
            self.step_history.remove(0);
        }

        // Update gradient statistics
        if self.config.track_gradient_norms {
            self.update_gradient_stats(&gradient_norms)?;
        }

        // Update parameter statistics
        if self.config.track_parameter_norms {
            self.update_parameter_stats(&parameter_norms, &update_norms)?;
        }

        // Update convergence tracking
        if let Some(loss_val) = loss {
            self.update_convergence_tracking(loss_val)?;
        }

        Ok(())
    }

    /// Update gradient statistics
    fn update_gradient_stats(&mut self, gradient_norms: &[f32]) -> Result<()> {
        for &norm in gradient_norms {
            // Add to history
            self.gradient_stats.norm_history.push_back(norm);
            if self.gradient_stats.norm_history.len() > self.config.moving_average_window {
                self.gradient_stats.norm_history.pop_front();
            }

            // Update statistics
            self.gradient_stats.max_norm = self.gradient_stats.max_norm.max(norm);
            self.gradient_stats.min_norm = self.gradient_stats.min_norm.min(norm);

            // Detect gradient explosion (norm > 10.0)
            if norm > 10.0 {
                self.gradient_stats.explosion_count += 1;
            }

            // Detect gradient vanishing (norm < 1e-7)
            if norm < 1e-7 {
                self.gradient_stats.vanishing_count += 1;
            }
        }

        // Update average and variance
        if !self.gradient_stats.norm_history.is_empty() {
            let sum: f32 = self.gradient_stats.norm_history.iter().sum();
            self.gradient_stats.average_norm = sum / self.gradient_stats.norm_history.len() as f32;

            // Calculate variance
            let mean = self.gradient_stats.average_norm;
            let variance: f32 = self
                .gradient_stats
                .norm_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / self.gradient_stats.norm_history.len() as f32;
            self.gradient_stats.recent_variance = variance;
        }

        Ok(())
    }

    /// Update parameter statistics
    fn update_parameter_stats(
        &mut self,
        parameter_norms: &[f32],
        update_norms: &[f32],
    ) -> Result<()> {
        for (param_norm, update_norm) in parameter_norms.iter().zip(update_norms.iter()) {
            // Add parameter norm to history
            self.parameter_stats.norm_history.push_back(*param_norm);
            if self.parameter_stats.norm_history.len() > self.config.moving_average_window {
                self.parameter_stats.norm_history.pop_front();
            }

            // Calculate update ratio
            let ratio = if *param_norm > 1e-8 {
                update_norm / param_norm
            } else {
                0.0
            };

            self.parameter_stats.update_ratios.push_back(ratio);
            if self.parameter_stats.update_ratios.len() > self.config.moving_average_window {
                self.parameter_stats.update_ratios.pop_front();
            }
        }

        // Update average update ratio
        if !self.parameter_stats.update_ratios.is_empty() {
            let sum: f32 = self.parameter_stats.update_ratios.iter().sum();
            self.parameter_stats.average_update_ratio =
                sum / self.parameter_stats.update_ratios.len() as f32;
        }

        // Calculate velocity (rate of parameter change)
        if self.parameter_stats.norm_history.len() >= 2 {
            let current = self.parameter_stats.norm_history.back().unwrap();
            let previous = self
                .parameter_stats
                .norm_history
                .get(self.parameter_stats.norm_history.len() - 2)
                .unwrap();
            self.parameter_stats.velocity = (current - previous).abs();
        }

        // Calculate stability score (inverse of update ratio variance)
        if self.parameter_stats.update_ratios.len() > 1 {
            let mean = self.parameter_stats.average_update_ratio;
            let variance: f32 = self
                .parameter_stats
                .update_ratios
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / self.parameter_stats.update_ratios.len() as f32;
            self.parameter_stats.stability_score = 1.0 / (1.0 + variance);
        }

        Ok(())
    }

    /// Update convergence tracking
    fn update_convergence_tracking(&mut self, loss: f32) -> Result<()> {
        // Add to loss history
        self.convergence_tracker.loss_history.push_back(loss);
        if self.convergence_tracker.loss_history.len() > self.config.moving_average_window {
            self.convergence_tracker.loss_history.pop_front();
        }

        // Update moving average
        if !self.convergence_tracker.loss_history.is_empty() {
            let sum: f32 = self.convergence_tracker.loss_history.iter().sum();
            self.convergence_tracker.loss_moving_average =
                sum / self.convergence_tracker.loss_history.len() as f32;
        }

        // Check for improvement
        if loss < self.convergence_tracker.best_loss {
            self.convergence_tracker.best_loss = loss;
            self.convergence_tracker.steps_since_improvement = 0;
            self.convergence_tracker.plateau_length = 0;
        } else {
            self.convergence_tracker.steps_since_improvement += 1;
            self.convergence_tracker.plateau_length += 1;
        }

        // Estimate convergence rate
        if self.convergence_tracker.loss_history.len() >= 10 {
            let recent_losses: Vec<f32> = self
                .convergence_tracker
                .loss_history
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect();

            // Simple linear regression to estimate convergence rate
            let n = recent_losses.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean = recent_losses.iter().sum::<f32>() / n;

            let numerator: f32 = recent_losses
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
                .sum();

            let denominator: f32 = (0..recent_losses.len())
                .map(|i| (i as f32 - x_mean).powi(2))
                .sum();

            if denominator > 1e-8 {
                self.convergence_tracker.convergence_rate = numerator / denominator;
            }
        }

        // Check convergence (plateau for too long or very small convergence rate)
        self.convergence_tracker.is_converged = self.convergence_tracker.plateau_length > 1000
            || (self.convergence_tracker.convergence_rate.abs() < 1e-6
                && self.convergence_tracker.loss_history.len() > 100);

        Ok(())
    }

    /// Generate a comprehensive analysis report
    pub fn generate_report(&self) -> AnalysisReport {
        AnalysisReport {
            total_steps: self.step_history.len(),
            gradient_stats: self.gradient_stats.clone(),
            parameter_stats: self.parameter_stats.clone(),
            convergence_tracker: self.convergence_tracker.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate optimization recommendations based on analysis
    fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check for gradient explosion
        if self.gradient_stats.explosion_count > 10 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::GradientNorms,
                severity: Severity::High,
                message: "Frequent gradient explosions detected. Consider gradient clipping or reducing learning rate.".to_string(),
                suggested_actions: vec![
                    "Add gradient clipping with max_norm=1.0".to_string(),
                    "Reduce learning rate by factor of 10".to_string(),
                    "Use adaptive optimizers like Adam".to_string(),
                ],
            });
        }

        // Check for gradient vanishing
        if self.gradient_stats.vanishing_count > 10 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::GradientNorms,
                severity: Severity::Medium,
                message: "Frequent gradient vanishing detected. Model may be too deep or have saturation issues.".to_string(),
                suggested_actions: vec![
                    "Check activation functions for saturation".to_string(),
                    "Consider batch normalization".to_string(),
                    "Use residual connections".to_string(),
                ],
            });
        }

        // Check learning rate
        if self.parameter_stats.average_update_ratio > 0.1 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::LearningRate,
                severity: Severity::Medium,
                message: "Update ratios are high. Learning rate might be too large.".to_string(),
                suggested_actions: vec![
                    "Reduce learning rate by factor of 2-5".to_string(),
                    "Use learning rate scheduling".to_string(),
                ],
            });
        } else if self.parameter_stats.average_update_ratio < 0.001 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::LearningRate,
                severity: Severity::Low,
                message: "Update ratios are very small. Learning rate might be too small."
                    .to_string(),
                suggested_actions: vec![
                    "Increase learning rate by factor of 2-10".to_string(),
                    "Consider warmup schedule".to_string(),
                ],
            });
        }

        // Check convergence
        if self.convergence_tracker.plateau_length > 500 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Convergence,
                severity: Severity::Medium,
                message: "Training has plateaued for many steps.".to_string(),
                suggested_actions: vec![
                    "Reduce learning rate".to_string(),
                    "Add regularization".to_string(),
                    "Consider early stopping".to_string(),
                ],
            });
        }

        recommendations
    }

    /// Get recent gradient flow visualization data
    pub fn get_gradient_flow_data(&self, num_steps: usize) -> Vec<GradientFlowPoint> {
        self.step_history
            .iter()
            .rev()
            .take(num_steps)
            .map(|step| GradientFlowPoint {
                step: step.step,
                gradient_norms: step.gradient_norms.clone(),
                parameter_norms: step.parameter_norms.clone(),
                update_norms: step.update_norms.clone(),
                loss: step.loss,
            })
            .collect()
    }
}

/// Complete analysis report
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    pub total_steps: usize,
    pub gradient_stats: GradientStatistics,
    pub parameter_stats: ParameterStatistics,
    pub convergence_tracker: ConvergenceTracker,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub severity: Severity,
    pub message: String,
    pub suggested_actions: Vec<String>,
}

/// Recommendation categories
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    GradientNorms,
    LearningRate,
    Convergence,
    Stability,
    Performance,
}

/// Severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Gradient flow visualization point
#[derive(Debug, Clone)]
pub struct GradientFlowPoint {
    pub step: usize,
    pub gradient_norms: Vec<f32>,
    pub parameter_norms: Vec<f32>,
    pub update_norms: Vec<f32>,
    pub loss: Option<f32>,
}

/// Hyperparameter sensitivity analyzer
pub struct HyperparameterSensitivity {
    /// Sensitivity analysis results
    sensitivity_data: HashMap<String, SensitivityResult>,
    /// Base configuration
    base_config: HashMap<String, f32>,
}

/// Sensitivity analysis result for a hyperparameter
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Hyperparameter name
    pub name: String,
    /// Test values used
    pub test_values: Vec<f32>,
    /// Resulting performance metrics
    pub performance_metrics: Vec<f32>,
    /// Sensitivity score (0-1, higher = more sensitive)
    pub sensitivity_score: f32,
    /// Optimal value found
    pub optimal_value: f32,
}

impl HyperparameterSensitivity {
    /// Create a new sensitivity analyzer
    pub fn new(base_config: HashMap<String, f32>) -> Self {
        Self {
            sensitivity_data: HashMap::new(),
            base_config,
        }
    }

    /// Analyze sensitivity for a specific hyperparameter
    pub fn analyze_parameter(
        &mut self,
        param_name: &str,
        test_values: Vec<f32>,
        performance_evaluator: impl Fn(f32) -> Result<f32>,
    ) -> Result<SensitivityResult> {
        let mut performance_metrics = Vec::new();

        // Test each value
        for &value in &test_values {
            let performance = performance_evaluator(value)?;
            performance_metrics.push(performance);
        }

        // Calculate sensitivity score
        let max_perf = performance_metrics
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_perf = performance_metrics
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let sensitivity_score = if max_perf != min_perf {
            (max_perf - min_perf) / max_perf.abs()
        } else {
            0.0
        };

        // Find optimal value
        let optimal_idx = performance_metrics
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_value = test_values[optimal_idx];

        let result = SensitivityResult {
            name: param_name.to_string(),
            test_values,
            performance_metrics,
            sensitivity_score,
            optimal_value,
        };

        self.sensitivity_data
            .insert(param_name.to_string(), result.clone());
        Ok(result)
    }

    /// Get sensitivity ranking (most sensitive first)
    pub fn get_sensitivity_ranking(&self) -> Vec<(&str, f32)> {
        let mut ranking: Vec<_> = self
            .sensitivity_data
            .iter()
            .map(|(name, result)| (name.as_str(), result.sensitivity_score))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranking
    }

    /// Generate sensitivity report
    pub fn generate_sensitivity_report(&self) -> SensitivityReport {
        let ranking = self.get_sensitivity_ranking();
        let most_sensitive = ranking
            .first()
            .map(|(name, score)| (name.to_string(), *score));
        let least_sensitive = ranking
            .last()
            .map(|(name, score)| (name.to_string(), *score));

        SensitivityReport {
            analyzed_parameters: self.sensitivity_data.keys().cloned().collect(),
            sensitivity_ranking: ranking
                .into_iter()
                .map(|(n, s)| (n.to_string(), s))
                .collect(),
            most_sensitive_parameter: most_sensitive,
            least_sensitive_parameter: least_sensitive,
            recommendations: self.generate_sensitivity_recommendations(),
        }
    }

    /// Generate recommendations based on sensitivity analysis
    fn generate_sensitivity_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let ranking = self.get_sensitivity_ranking();

        if let Some((most_sensitive, score)) = ranking.first() {
            if *score > 0.5 {
                recommendations.push(format!(
                    "Parameter '{}' is highly sensitive (score: {:.3}). Fine-tune carefully.",
                    most_sensitive, score
                ));
            }
        }

        if let Some((least_sensitive, score)) = ranking.last() {
            if *score < 0.1 {
                recommendations.push(format!(
                    "Parameter '{}' has low sensitivity (score: {:.3}). Consider using default values.",
                    least_sensitive, score
                ));
            }
        }

        recommendations
    }
}

/// Sensitivity analysis report
#[derive(Debug, Clone)]
pub struct SensitivityReport {
    pub analyzed_parameters: Vec<String>,
    pub sensitivity_ranking: Vec<(String, f32)>,
    pub most_sensitive_parameter: Option<(String, f32)>,
    pub least_sensitive_parameter: Option<(String, f32)>,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_optimizer_analyzer_creation() {
        let analyzer = OptimizerAnalyzer::new(None);
        assert_eq!(analyzer.step_history.len(), 0);
    }

    #[test]
    fn test_analyzer_step() {
        let mut analyzer = OptimizerAnalyzer::new(None);
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];

        let result = analyzer.analyze_step(1, &params, &[0.01], Some(0.5));
        assert!(result.is_ok());
        assert_eq!(analyzer.step_history.len(), 1);
    }

    #[test]
    fn test_sensitivity_analyzer() -> OptimizerResult<()> {
        let base_config = [("lr".to_string(), 0.01)].iter().cloned().collect();
        let mut sensitivity = HyperparameterSensitivity::new(base_config);

        let test_values = vec![0.001, 0.01, 0.1];
        let evaluator = |lr: f32| Ok(1.0 / lr); // Simple inverse relationship

        let _result = sensitivity.analyze_parameter("lr", test_values, evaluator)?;

        let ranking = sensitivity.get_sensitivity_ranking();
        assert_eq!(ranking.len(), 1);
        assert_eq!(ranking[0].0, "lr");
        Ok(())
    }
}
