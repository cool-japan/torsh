//! Robustness features for optimizers
//!
//! This module provides tools for making optimizers more robust to various
//! forms of adversarial perturbations, noisy gradients, and training instabilities.
//! It includes implementations of robust optimization techniques and defensive
//! training strategies.

use crate::{OptimizerError, OptimizerResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ops::Add;
use torsh_tensor::Tensor;

/// Configuration for robust optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessConfig {
    /// Enable gradient clipping for stability
    pub gradient_clipping: bool,
    /// Maximum gradient norm for clipping
    pub max_gradient_norm: f32,
    /// Enable outlier detection and filtering
    pub outlier_detection: bool,
    /// Z-score threshold for outlier detection
    pub outlier_threshold: f32,
    /// Enable smooth gradient aggregation
    pub smooth_aggregation: bool,
    /// Smoothing factor for gradient aggregation
    pub smoothing_factor: f32,
    /// Enable adversarial training support
    pub adversarial_training: bool,
    /// Perturbation budget for adversarial examples
    pub perturbation_budget: f32,
    /// Number of adversarial steps
    pub adversarial_steps: usize,
}

impl Default for RobustnessConfig {
    fn default() -> Self {
        Self {
            gradient_clipping: true,
            max_gradient_norm: 1.0,
            outlier_detection: true,
            outlier_threshold: 3.0,
            smooth_aggregation: true,
            smoothing_factor: 0.1,
            adversarial_training: false,
            perturbation_budget: 0.01,
            adversarial_steps: 1,
        }
    }
}

/// Statistics for robustness monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessStats {
    /// Number of gradients processed
    pub total_gradients: usize,
    /// Number of gradients clipped
    pub clipped_gradients: usize,
    /// Number of outliers detected
    pub outliers_detected: usize,
    /// Average gradient norm
    pub avg_gradient_norm: f32,
    /// Maximum gradient norm observed
    pub max_gradient_norm: f32,
    /// Gradient variance over time
    pub gradient_variance: f32,
    /// Stability score (0-1, higher is more stable)
    pub stability_score: f32,
}

impl RobustnessStats {
    pub fn new() -> Self {
        Self {
            total_gradients: 0,
            clipped_gradients: 0,
            outliers_detected: 0,
            avg_gradient_norm: 0.0,
            max_gradient_norm: 0.0,
            gradient_variance: 0.0,
            stability_score: 1.0,
        }
    }

    pub fn clipping_rate(&self) -> f32 {
        if self.total_gradients == 0 {
            0.0
        } else {
            self.clipped_gradients as f32 / self.total_gradients as f32
        }
    }

    pub fn outlier_rate(&self) -> f32 {
        if self.total_gradients == 0 {
            0.0
        } else {
            self.outliers_detected as f32 / self.total_gradients as f32
        }
    }
}

/// Gradient history for analysis
#[derive(Debug, Clone)]
struct GradientHistory {
    norms: VecDeque<f32>,
    window_size: usize,
}

impl GradientHistory {
    fn new(window_size: usize) -> Self {
        Self {
            norms: VecDeque::new(),
            window_size,
        }
    }

    fn add(&mut self, norm: f32) {
        self.norms.push_back(norm);
        if self.norms.len() > self.window_size {
            self.norms.pop_front();
        }
    }

    fn mean(&self) -> f32 {
        if self.norms.is_empty() {
            0.0
        } else {
            self.norms.iter().sum::<f32>() / self.norms.len() as f32
        }
    }

    fn variance(&self) -> f32 {
        if self.norms.len() < 2 {
            0.0
        } else {
            let mean = self.mean();
            let sum_sq_diff: f32 = self.norms.iter().map(|x| (x - mean).powi(2)).sum();
            sum_sq_diff / (self.norms.len() - 1) as f32
        }
    }

    fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }
}

/// Robustness manager for optimizers
pub struct RobustnessManager {
    config: RobustnessConfig,
    stats: RobustnessStats,
    gradient_history: HashMap<String, GradientHistory>,
    smoothed_gradients: HashMap<String, Tensor>,
}

impl RobustnessManager {
    pub fn new(config: RobustnessConfig) -> Self {
        Self {
            config,
            stats: RobustnessStats::new(),
            gradient_history: HashMap::new(),
            smoothed_gradients: HashMap::new(),
        }
    }

    /// Apply robustness transformations to gradients
    pub fn process_gradients(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        let mut processed_gradients = HashMap::new();

        for (param_name, gradient) in gradients {
            let mut processed_grad = gradient.clone();

            // Step 1: Outlier detection and filtering
            if self.config.outlier_detection {
                if self.is_outlier(param_name, &processed_grad)? {
                    self.stats.outliers_detected += 1;
                    processed_grad = self.filter_outlier(param_name, &processed_grad)?;
                }
            }

            // Step 2: Gradient clipping
            if self.config.gradient_clipping {
                processed_grad = self.clip_gradient(&processed_grad)?;
            }

            // Step 3: Smooth aggregation
            if self.config.smooth_aggregation {
                processed_grad = self.smooth_gradient(param_name, &processed_grad)?;
            }

            // Update statistics
            self.update_statistics(param_name, &processed_grad)?;
            processed_gradients.insert(param_name.clone(), processed_grad);
        }

        Ok(processed_gradients)
    }

    /// Check if gradient is an outlier
    fn is_outlier(&mut self, param_name: &str, gradient: &Tensor) -> OptimizerResult<bool> {
        let grad_norm = gradient.norm()?.item()?;

        let history = self
            .gradient_history
            .entry(param_name.to_string())
            .or_insert_with(|| GradientHistory::new(100));

        if history.norms.len() < 10 {
            // Not enough history, assume not an outlier
            history.add(grad_norm);
            return Ok(false);
        }

        let mean = history.mean();
        let std_dev = history.std_dev();

        if std_dev < 1e-8 {
            // Gradients are essentially constant, not an outlier
            history.add(grad_norm);
            return Ok(false);
        }

        let z_score = (grad_norm - mean).abs() / std_dev;
        let is_outlier = z_score > self.config.outlier_threshold;

        if !is_outlier {
            history.add(grad_norm);
        }

        Ok(is_outlier)
    }

    /// Filter outlier gradients
    fn filter_outlier(&self, param_name: &str, gradient: &Tensor) -> OptimizerResult<Tensor> {
        if let Some(history) = self.gradient_history.get(param_name) {
            let mean_norm = history.mean();
            let current_norm = gradient.norm()?.item()?;

            if current_norm > 1e-8 {
                // Scale down to mean norm
                let scale_factor = mean_norm / current_norm;
                return Ok(gradient.mul_scalar(scale_factor)?);
            }
        }

        // Fallback: return zero gradient
        Ok(gradient.zeros_like()?)
    }

    /// Clip gradients to maximum norm
    fn clip_gradient(&mut self, gradient: &Tensor) -> OptimizerResult<Tensor> {
        let grad_norm = gradient.norm()?.item()?;

        if grad_norm > self.config.max_gradient_norm {
            self.stats.clipped_gradients += 1;
            let scale_factor = self.config.max_gradient_norm / grad_norm;
            Ok(gradient.mul_scalar(scale_factor)?)
        } else {
            Ok(gradient.clone())
        }
    }

    /// Apply smooth aggregation to gradients
    fn smooth_gradient(&mut self, param_name: &str, gradient: &Tensor) -> OptimizerResult<Tensor> {
        let alpha = self.config.smoothing_factor;

        match self.smoothed_gradients.get(param_name) {
            Some(prev_smoothed) => {
                // Exponential moving average: smoothed = alpha * new + (1 - alpha) * prev
                let new_contribution = gradient.mul_scalar(alpha)?;
                let prev_contribution = prev_smoothed.mul_scalar(1.0 - alpha)?;
                let smoothed = new_contribution.add(&prev_contribution)?;

                self.smoothed_gradients
                    .insert(param_name.to_string(), smoothed.clone());
                Ok(smoothed)
            }
            None => {
                // First gradient, use as-is
                self.smoothed_gradients
                    .insert(param_name.to_string(), gradient.clone());
                Ok(gradient.clone())
            }
        }
    }

    /// Update robustness statistics
    fn update_statistics(&mut self, param_name: &str, gradient: &Tensor) -> OptimizerResult<()> {
        let grad_norm = gradient.norm()?.item()?;

        self.stats.total_gradients += 1;
        self.stats.max_gradient_norm = self.stats.max_gradient_norm.max(grad_norm);

        // Update average gradient norm
        let prev_avg = self.stats.avg_gradient_norm;
        let n = self.stats.total_gradients as f32;
        self.stats.avg_gradient_norm = (prev_avg * (n - 1.0) + grad_norm) / n;

        // Update gradient variance
        if let Some(history) = self.gradient_history.get(param_name) {
            self.stats.gradient_variance = history.variance();
        }

        // Update stability score (inverse of coefficient of variation)
        if self.stats.avg_gradient_norm > 1e-8 {
            let cv = self.stats.gradient_variance.sqrt() / self.stats.avg_gradient_norm;
            self.stats.stability_score = 1.0 / (1.0 + cv);
        }

        Ok(())
    }

    /// Generate adversarial perturbations for robust training
    pub fn generate_adversarial_perturbation(
        &self,
        input: &Tensor,
        gradient: &Tensor,
    ) -> OptimizerResult<Tensor> {
        if !self.config.adversarial_training {
            return Ok(input.zeros_like()?);
        }

        let grad_norm = gradient.norm()?.item()?;
        if grad_norm < 1e-8 {
            return Ok(input.zeros_like()?);
        }

        // Generate perturbation in direction of gradient (FGSM-style)
        let perturbation_direction = gradient.div_scalar(grad_norm)?;
        let perturbation = perturbation_direction.mul_scalar(self.config.perturbation_budget)?;

        Ok(perturbation)
    }

    /// Get current robustness statistics
    pub fn get_stats(&self) -> &RobustnessStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = RobustnessStats::new();
        self.gradient_history.clear();
        self.smoothed_gradients.clear();
    }

    /// Check if training appears stable
    pub fn is_training_stable(&self) -> bool {
        self.stats.stability_score > 0.5
            && self.stats.clipping_rate() < 0.3
            && self.stats.outlier_rate() < 0.1
    }

    /// Get recommendations for improving robustness
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.stats.clipping_rate() > 0.5 {
            recommendations
                .push("Consider reducing learning rate - high gradient clipping rate".to_string());
        }

        if self.stats.outlier_rate() > 0.2 {
            recommendations.push(
                "High outlier rate detected - consider data cleaning or regularization".to_string(),
            );
        }

        if self.stats.stability_score < 0.3 {
            recommendations
                .push("Low stability score - consider increasing smoothing factor".to_string());
        }

        if self.stats.gradient_variance > self.stats.avg_gradient_norm.powi(2) {
            recommendations.push(
                "High gradient variance - consider batch normalization or different architecture"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Training appears stable and robust".to_string());
        }

        recommendations
    }
}

/// Trait for robust optimizers
pub trait RobustOptimizer {
    /// Enable robustness features
    fn enable_robustness(&mut self, config: RobustnessConfig);

    /// Get robustness statistics
    fn robustness_stats(&self) -> Option<&RobustnessStats>;

    /// Check if training is stable
    fn is_stable(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_robustness_config_default() {
        let config = RobustnessConfig::default();
        assert!(config.gradient_clipping);
        assert_eq!(config.max_gradient_norm, 1.0);
    }

    #[test]
    fn test_robustness_manager_creation() {
        let config = RobustnessConfig::default();
        let manager = RobustnessManager::new(config);
        assert_eq!(manager.stats.total_gradients, 0);
    }

    #[test]
    fn test_gradient_clipping() -> OptimizerResult<()> {
        let config = RobustnessConfig {
            max_gradient_norm: 1.0,
            gradient_clipping: true,
            ..Default::default()
        };
        let mut manager = RobustnessManager::new(config);

        let large_gradient = randn::<f32>(&[2, 2]).unwrap().mul_scalar(10.0).unwrap();
        let clipped = manager.clip_gradient(&large_gradient).unwrap();

        let clipped_norm = clipped.norm().unwrap().to_vec()?[0];
        assert!(clipped_norm <= 1.0 + 1e-6);

        Ok(())
    }

    #[test]
    fn test_gradient_smoothing() -> OptimizerResult<()> {
        let config = RobustnessConfig {
            smooth_aggregation: true,
            smoothing_factor: 0.5,
            ..Default::default()
        };
        let mut manager = RobustnessManager::new(config);

        let grad1 = randn::<f32>(&[2, 2]).unwrap();
        let grad2 = randn::<f32>(&[2, 2]).unwrap();

        let smoothed1 = manager.smooth_gradient("param1", &grad1).unwrap();
        let smoothed2 = manager.smooth_gradient("param1", &grad2).unwrap();

        // First gradient should be unchanged
        assert_eq!(smoothed1.data()?, grad1.data()?);

        // Second should be blend
        let expected = grad2
            .mul_scalar(0.5)
            .unwrap()
            .add(&grad1.mul_scalar(0.5).unwrap())
            .unwrap();
        let diff = smoothed2.sub(&expected).unwrap().norm().unwrap().item()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn test_statistics_tracking() {
        let config = RobustnessConfig::default();
        let mut manager = RobustnessManager::new(config);

        let gradient = randn::<f32>(&[2, 2]).unwrap();
        manager.update_statistics("param1", &gradient).unwrap();

        assert_eq!(manager.stats.total_gradients, 1);
        assert!(manager.stats.avg_gradient_norm > 0.0);
    }

    #[test]
    fn test_adversarial_perturbation() -> OptimizerResult<()> {
        let config = RobustnessConfig {
            adversarial_training: true,
            perturbation_budget: 0.1,
            ..Default::default()
        };
        let manager = RobustnessManager::new(config);

        let input = randn::<f32>(&[2, 2]).unwrap();
        let gradient = randn::<f32>(&[2, 2]).unwrap();

        let perturbation = manager
            .generate_adversarial_perturbation(&input, &gradient)
            .unwrap();
        let pert_norm = perturbation.norm().unwrap().to_vec()?[0];

        assert!(pert_norm <= 0.1 + 1e-6);

        Ok(())
    }

    #[test]
    fn test_stability_assessment() {
        let mut stats = RobustnessStats::new();
        stats.stability_score = 0.8;
        stats.total_gradients = 100;
        stats.clipped_gradients = 10;
        stats.outliers_detected = 5;

        let manager = RobustnessManager {
            config: RobustnessConfig::default(),
            stats,
            gradient_history: HashMap::new(),
            smoothed_gradients: HashMap::new(),
        };

        assert!(manager.is_training_stable());
    }
}
