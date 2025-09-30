//! Differential Privacy support for optimizers
//!
//! This module provides differential privacy (DP) mechanisms for training machine learning
//! models with privacy guarantees. It implements both Differentially Private SGD (DP-SGD)
//! and other DP-aware optimization algorithms.
//!
//! Reference: "Deep Learning with Differential Privacy" by Abadi et al.
//! https://arxiv.org/abs/1607.00133

use crate::{OptimizerError, OptimizerResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Add;
use torsh_tensor::Tensor;

// SciRS2 policy compliance - use scirs2_core for random generation
use scirs2_core::random::prelude::*;

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPConfig {
    /// Target epsilon for privacy budget
    pub target_epsilon: f64,
    /// Target delta for privacy budget  
    pub target_delta: f64,
    /// Noise multiplier for gradient perturbation
    pub noise_multiplier: f64,
    /// L2 norm clipping bound for gradients
    pub l2_norm_clip: f32,
    /// Number of training steps
    pub num_training_steps: usize,
    /// Batch sampling rate (lot_size / dataset_size)
    pub sampling_rate: f64,
    /// Whether to use adaptive clipping
    pub adaptive_clipping: bool,
    /// Quantile for adaptive clipping
    pub clipping_quantile: f64,
}

impl DPConfig {
    pub fn new(target_epsilon: f64, target_delta: f64, num_training_steps: usize) -> Self {
        Self {
            target_epsilon,
            target_delta,
            noise_multiplier: 1.0,
            l2_norm_clip: 1.0,
            num_training_steps,
            sampling_rate: 0.01,
            adaptive_clipping: false,
            clipping_quantile: 0.5,
        }
    }

    /// Calculate noise multiplier for given privacy parameters
    pub fn calculate_noise_multiplier(&self) -> f64 {
        // Simplified calculation - in practice would use more sophisticated methods
        // like the accountant from TensorFlow Privacy
        let steps = self.num_training_steps as f64;
        let q = self.sampling_rate;

        // Approximate formula for noise multiplier
        let sigma = (q * steps * (self.target_epsilon.exp() - 1.0) / self.target_delta).sqrt();
        sigma.max(0.1) // Minimum noise for stability
    }

    /// Get privacy budget spent so far
    pub fn privacy_spent(&self, step: usize) -> (f64, f64) {
        let steps_ratio = step as f64 / self.num_training_steps as f64;
        let epsilon_spent = self.target_epsilon * steps_ratio;
        let delta_spent = self.target_delta * steps_ratio;
        (epsilon_spent, delta_spent)
    }
}

/// Differential privacy state tracking
#[derive(Debug, Clone)]
pub struct DPState {
    /// Current step in training
    pub current_step: usize,
    /// Cumulative privacy budget spent
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    /// Gradient norms for adaptive clipping
    pub gradient_norms: Vec<f32>,
    /// Clipping statistics
    pub clipping_stats: ClippingStats,
}

impl Default for DPState {
    fn default() -> Self {
        Self {
            current_step: 0,
            epsilon_spent: 0.0,
            delta_spent: 0.0,
            gradient_norms: Vec::new(),
            clipping_stats: ClippingStats::default(),
        }
    }
}

impl DPState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_step(&mut self, config: &DPConfig) {
        self.current_step += 1;
        let (eps, delta) = config.privacy_spent(self.current_step);
        self.epsilon_spent = eps;
        self.delta_spent = delta;
    }
}

/// Statistics for gradient clipping
#[derive(Debug, Clone)]
pub struct ClippingStats {
    pub total_gradients: usize,
    pub clipped_gradients: usize,
    pub average_norm_before: f32,
    pub average_norm_after: f32,
    pub max_norm_observed: f32,
}

impl Default for ClippingStats {
    fn default() -> Self {
        Self {
            total_gradients: 0,
            clipped_gradients: 0,
            average_norm_before: 0.0,
            average_norm_after: 0.0,
            max_norm_observed: 0.0,
        }
    }
}

impl ClippingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clipping_rate(&self) -> f32 {
        if self.total_gradients == 0 {
            0.0
        } else {
            self.clipped_gradients as f32 / self.total_gradients as f32
        }
    }
}

/// Differential privacy manager for optimizers
pub struct DPManager {
    config: DPConfig,
    state: DPState,
    rng: Random,
}

impl DPManager {
    pub fn new(config: DPConfig) -> Self {
        Self {
            config,
            state: DPState::new(),
            rng: Random::default(),
        }
    }

    /// Apply differential privacy to gradients
    pub fn privatize_gradients(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        let mut private_gradients = HashMap::new();

        for (param_name, gradient) in gradients {
            let private_grad = self.apply_dp_to_gradient(gradient)?;
            private_gradients.insert(param_name.clone(), private_grad);
        }

        self.state.update_step(&self.config);
        Ok(private_gradients)
    }

    /// Apply DP mechanism to a single gradient tensor
    fn apply_dp_to_gradient(&mut self, gradient: &Tensor) -> OptimizerResult<Tensor> {
        // Step 1: Clip gradient to bound sensitivity
        let clipped_grad = self.clip_gradient(gradient)?;

        // Step 2: Add calibrated noise
        let noisy_grad = self.add_noise(&clipped_grad)?;

        Ok(noisy_grad)
    }

    /// Clip gradient to L2 norm bound
    fn clip_gradient(&mut self, gradient: &Tensor) -> OptimizerResult<Tensor> {
        let grad_norm = gradient.norm()?.item()?;
        self.state.clipping_stats.total_gradients += 1;
        self.state.clipping_stats.max_norm_observed =
            self.state.clipping_stats.max_norm_observed.max(grad_norm);

        let clip_bound = if self.config.adaptive_clipping {
            self.get_adaptive_clip_bound()
        } else {
            self.config.l2_norm_clip
        };

        self.state.clipping_stats.average_norm_before =
            (self.state.clipping_stats.average_norm_before
                * (self.state.clipping_stats.total_gradients - 1) as f32
                + grad_norm)
                / self.state.clipping_stats.total_gradients as f32;

        let clipped_grad = if grad_norm > clip_bound {
            self.state.clipping_stats.clipped_gradients += 1;
            gradient.mul_scalar(clip_bound / grad_norm)?
        } else {
            gradient.clone()
        };

        let clipped_norm = if grad_norm > clip_bound {
            clip_bound
        } else {
            grad_norm
        };
        self.state.clipping_stats.average_norm_after =
            (self.state.clipping_stats.average_norm_after
                * (self.state.clipping_stats.total_gradients - 1) as f32
                + clipped_norm)
                / self.state.clipping_stats.total_gradients as f32;

        Ok(clipped_grad)
    }

    /// Add calibrated Gaussian noise to gradient
    fn add_noise(&mut self, gradient: &Tensor) -> OptimizerResult<Tensor> {
        let noise_scale = self.config.noise_multiplier * self.config.l2_norm_clip as f64;
        let noise_tensor =
            self.generate_gaussian_noise(gradient.shape().dims(), noise_scale as f32)?;

        Ok(gradient.add(&noise_tensor)?)
    }

    /// Generate Gaussian noise tensor
    fn generate_gaussian_noise(
        &mut self,
        shape: &[usize],
        std_dev: f32,
    ) -> OptimizerResult<Tensor> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random generation
        let total_elements: usize = shape.iter().product();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            OptimizerError::InvalidParameter(format!("Invalid noise parameters: {e}"))
        })?;
        let noise_data: Vec<f32> = (0..total_elements).map(|_| rng.sample(normal)).collect();

        Tensor::from_data(
            noise_data,
            shape.to_vec(),
            torsh_core::device::DeviceType::Cpu,
        )
        .map_err(OptimizerError::TensorError)
    }

    /// Get adaptive clipping bound based on gradient norm quantiles
    fn get_adaptive_clip_bound(&self) -> f32 {
        if self.state.gradient_norms.is_empty() {
            return self.config.l2_norm_clip;
        }

        let mut sorted_norms = self.state.gradient_norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let quantile_idx = (sorted_norms.len() as f64 * self.config.clipping_quantile) as usize;
        sorted_norms[quantile_idx.min(sorted_norms.len() - 1)]
    }

    /// Check if privacy budget is exhausted
    pub fn is_privacy_exhausted(&self) -> bool {
        self.state.epsilon_spent >= self.config.target_epsilon
            || self.state.delta_spent >= self.config.target_delta
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> (f64, f64) {
        let remaining_epsilon = (self.config.target_epsilon - self.state.epsilon_spent).max(0.0);
        let remaining_delta = (self.config.target_delta - self.state.delta_spent).max(0.0);
        (remaining_epsilon, remaining_delta)
    }

    /// Get privacy accounting summary
    pub fn get_privacy_summary(&self) -> PrivacySummary {
        PrivacySummary {
            epsilon_spent: self.state.epsilon_spent,
            delta_spent: self.state.delta_spent,
            epsilon_remaining: self.config.target_epsilon - self.state.epsilon_spent,
            delta_remaining: self.config.target_delta - self.state.delta_spent,
            steps_completed: self.state.current_step,
            clipping_rate: self.state.clipping_stats.clipping_rate(),
            noise_multiplier: self.config.noise_multiplier,
        }
    }

    /// Update gradient norm history for adaptive clipping
    pub fn update_gradient_norms(&mut self, norms: &[f32]) {
        self.state.gradient_norms.extend_from_slice(norms);

        // Keep only recent norms to prevent unbounded growth
        const MAX_HISTORY: usize = 1000;
        if self.state.gradient_norms.len() > MAX_HISTORY {
            let start = self.state.gradient_norms.len() - MAX_HISTORY;
            self.state.gradient_norms = self.state.gradient_norms[start..].to_vec();
        }
    }
}

/// Privacy accounting summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySummary {
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub epsilon_remaining: f64,
    pub delta_remaining: f64,
    pub steps_completed: usize,
    pub clipping_rate: f32,
    pub noise_multiplier: f64,
}

/// Wrapper trait for DP-aware optimizers
pub trait DifferentiallyPrivateOptimizer {
    /// Apply differential privacy to the next optimization step
    fn dp_step(&mut self, dp_manager: &mut DPManager) -> OptimizerResult<()>;

    /// Get the current privacy budget status
    fn privacy_status(&self, dp_manager: &DPManager) -> PrivacySummary;

    /// Check if training should stop due to privacy budget exhaustion
    fn should_stop_for_privacy(&self, dp_manager: &DPManager) -> bool {
        dp_manager.is_privacy_exhausted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_dp_config_creation() {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        assert_eq!(config.target_epsilon, 1.0);
        assert_eq!(config.target_delta, 1e-5);
        assert_eq!(config.num_training_steps, 1000);
    }

    #[test]
    fn test_noise_multiplier_calculation() {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        let noise_multiplier = config.calculate_noise_multiplier();
        assert!(noise_multiplier > 0.0);
    }

    #[test]
    fn test_privacy_budget_tracking() {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        let (eps, delta) = config.privacy_spent(500);
        assert!((eps - 0.5).abs() < 1e-6);
        assert!((delta - 5e-6).abs() < 1e-9);
    }

    #[test]
    fn test_dp_manager_creation() {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        let manager = DPManager::new(config);
        assert_eq!(manager.state.current_step, 0);
        assert_eq!(manager.state.epsilon_spent, 0.0);
    }

    #[test]
    fn test_gradient_clipping() -> OptimizerResult<()> {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        let mut manager = DPManager::new(config);

        let gradient = randn::<f32>(&[2, 2])?.mul_scalar(10.0)?; // Large gradient
        let clipped = manager.clip_gradient(&gradient)?;

        let clipped_norm = clipped.norm()?.item()?;
        assert!(clipped_norm <= manager.config.l2_norm_clip + 1e-6);
        Ok(())
    }

    #[test]
    fn test_noise_generation() {
        let config = DPConfig::new(1.0, 1e-5, 1000);
        let mut manager = DPManager::new(config);

        let noise = manager.generate_gaussian_noise(&[2, 2], 1.0).unwrap();
        assert_eq!(noise.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_privacy_exhaustion() {
        let config = DPConfig::new(1.0, 1e-5, 10);
        let mut state = DPState::new();

        // Simulate many steps
        for _ in 0..20 {
            state.update_step(&config);
        }

        let manager = DPManager {
            config,
            state,
            rng: Random::default(),
        };
        assert!(manager.is_privacy_exhausted());
    }

    #[test]
    fn test_clipping_statistics() {
        let mut stats = ClippingStats::new();
        assert_eq!(stats.clipping_rate(), 0.0);

        stats.total_gradients = 10;
        stats.clipped_gradients = 3;
        assert!((stats.clipping_rate() - 0.3).abs() < 1e-6);
    }
}
