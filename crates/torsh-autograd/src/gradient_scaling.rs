//! Gradient scaling strategies for different optimizers
//!
//! This module provides adaptive gradient scaling techniques optimized for
//! different optimizer types, including SGD, Adam, AdamW, and specialized
//! scaling for mixed precision training and large batch scenarios.

use std::collections::VecDeque;
use std::f32;
use torsh_core::error::Result;
use tracing::{debug, warn};

/// Different optimizer types that require specific scaling strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer
    Adam,
    /// AdamW optimizer (Adam with weight decay)
    AdamW,
    /// AdaGrad optimizer
    AdaGrad,
    /// RMSprop optimizer
    RMSprop,
    /// LAMB optimizer (for large batch training)
    LAMB,
    /// AdaFactor optimizer
    AdaFactor,
}

/// Gradient scaling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingStrategy {
    /// Fixed scaling factor
    Fixed(f32),
    /// Dynamic scaling based on gradient statistics
    Dynamic,
    /// Adaptive scaling based on optimizer state
    Adaptive,
    /// Loss-based scaling for mixed precision
    LossBased,
    /// Percentile-based scaling
    Percentile(f32),
    /// Learning rate dependent scaling
    LearningRateDependent,
    /// Batch size dependent scaling
    BatchSizeDependent,
}

/// Configuration for gradient scaling
#[derive(Debug, Clone)]
pub struct GradientScalingConfig {
    /// Primary scaling strategy
    pub strategy: ScalingStrategy,
    /// Optimizer type this config is for
    pub optimizer_type: OptimizerType,
    /// Initial scaling factor
    pub initial_scale: f32,
    /// Minimum scaling factor
    pub min_scale: f32,
    /// Maximum scaling factor
    pub max_scale: f32,
    /// Update frequency for dynamic strategies
    pub update_frequency: usize,
    /// Growth factor for dynamic scaling
    pub growth_factor: f32,
    /// Backoff factor when overflow detected
    pub backoff_factor: f32,
    /// Enable overflow detection
    pub detect_overflow: bool,
    /// Enable underflow detection
    pub detect_underflow: bool,
    /// History window size for statistics
    pub history_window: usize,
}

impl Default for GradientScalingConfig {
    fn default() -> Self {
        Self {
            strategy: ScalingStrategy::Dynamic,
            optimizer_type: OptimizerType::Adam,
            initial_scale: 2.0f32.powi(16), // 65536
            min_scale: 2.0f32.powi(10),     // 1024
            max_scale: 2.0f32.powi(24),     // 16M
            update_frequency: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            detect_overflow: true,
            detect_underflow: true,
            history_window: 1000,
        }
    }
}

/// Gradient statistics for scaling decisions
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Mean gradient magnitude
    pub mean_magnitude: f32,
    /// Standard deviation of gradients
    pub std_deviation: f32,
    /// Maximum gradient magnitude
    pub max_magnitude: f32,
    /// Minimum gradient magnitude
    pub min_magnitude: f32,
    /// Number of zero gradients
    pub zero_count: usize,
    /// Number of infinite/NaN gradients
    pub invalid_count: usize,
    /// Total number of gradients
    pub total_count: usize,
}

/// Scaling decision made by the scaler
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    /// New scaling factor to apply
    pub scale_factor: f32,
    /// Whether scaling factor changed
    pub scale_changed: bool,
    /// Reason for the scaling decision
    pub reason: ScalingReason,
    /// Whether overflow was detected
    pub overflow_detected: bool,
    /// Whether underflow was detected
    pub underflow_detected: bool,
}

/// Reason for scaling decision
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingReason {
    /// Initialization
    Initialization,
    /// Overflow detected, scaling down
    OverflowDetected,
    /// Underflow detected, scaling up
    UnderflowDetected,
    /// Adaptive adjustment based on gradient statistics
    AdaptiveAdjustment,
    /// Scheduled update
    ScheduledUpdate,
    /// Learning rate change
    LearningRateChange,
    /// Batch size change
    BatchSizeChange,
    /// Manual override
    Manual,
}

/// Gradient scaler for different optimizers
pub struct GradientScaler {
    config: GradientScalingConfig,
    current_scale: f32,
    step_count: usize,
    growth_tracker: usize,
    gradient_history: VecDeque<GradientStats>,
    overflow_history: VecDeque<bool>,
    scaling_history: Vec<ScalingDecision>,
    optimizer_state: Option<OptimizerState>,
}

/// Optimizer-specific state for scaling decisions
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Current learning rate
    pub learning_rate: f32,
    /// Current batch size
    pub batch_size: usize,
    /// Momentum values (for SGD with momentum)
    pub momentum: Option<f32>,
    /// Beta values (for Adam-family optimizers)
    pub betas: Option<(f32, f32)>,
    /// Weight decay factor
    pub weight_decay: Option<f32>,
    /// Current optimizer step
    pub step: usize,
}

impl GradientScaler {
    /// Create a new gradient scaler
    pub fn new(config: GradientScalingConfig) -> Self {
        Self {
            current_scale: config.initial_scale,
            config,
            step_count: 0,
            growth_tracker: 0,
            gradient_history: VecDeque::new(),
            overflow_history: VecDeque::new(),
            scaling_history: Vec::new(),
            optimizer_state: None,
        }
    }

    /// Create a scaler optimized for a specific optimizer type
    pub fn for_optimizer(optimizer_type: OptimizerType) -> Self {
        let mut config = GradientScalingConfig::default();
        config.optimizer_type = optimizer_type;

        // Customize config based on optimizer type
        match optimizer_type {
            OptimizerType::SGD => {
                config.strategy = ScalingStrategy::Fixed(1.0);
                config.initial_scale = 1.0;
            }
            OptimizerType::SGDMomentum => {
                config.strategy = ScalingStrategy::Dynamic;
                config.initial_scale = 2.0f32.powi(8); // 256
            }
            OptimizerType::Adam | OptimizerType::AdamW => {
                config.strategy = ScalingStrategy::Adaptive;
                config.initial_scale = 2.0f32.powi(16); // 65536
                config.update_frequency = 2000;
            }
            OptimizerType::AdaGrad => {
                config.strategy = ScalingStrategy::LearningRateDependent;
                config.initial_scale = 2.0f32.powi(12); // 4096
            }
            OptimizerType::RMSprop => {
                config.strategy = ScalingStrategy::Dynamic;
                config.initial_scale = 2.0f32.powi(14); // 16384
            }
            OptimizerType::LAMB => {
                config.strategy = ScalingStrategy::BatchSizeDependent;
                config.initial_scale = 2.0f32.powi(10); // 1024
            }
            OptimizerType::AdaFactor => {
                config.strategy = ScalingStrategy::Adaptive;
                config.initial_scale = 2.0f32.powi(8); // 256
            }
        }

        Self::new(config)
    }

    /// Update optimizer state
    pub fn update_optimizer_state(&mut self, state: OptimizerState) {
        self.optimizer_state = Some(state);
    }

    /// Scale gradients and return scaling decision
    pub fn scale_gradients(&mut self, gradients: &mut [f32]) -> Result<ScalingDecision> {
        self.step_count += 1;

        // Compute gradient statistics
        let stats = self.compute_gradient_stats(gradients);

        // Check for overflow/underflow
        let overflow_detected = self.detect_overflow(&stats);
        let underflow_detected = self.detect_underflow(&stats);

        // Determine scaling decision
        let decision = self.make_scaling_decision(&stats, overflow_detected, underflow_detected)?;

        // Apply scaling
        let actual_scale_factor = match self.config.strategy {
            ScalingStrategy::Fixed(scale) => scale,
            _ => decision.scale_factor,
        };

        if actual_scale_factor != 1.0 {
            for grad in gradients.iter_mut() {
                *grad *= actual_scale_factor;
            }
        }

        // Update internal state
        self.update_internal_state(&stats, &decision);

        // Store history
        self.gradient_history.push_back(stats);
        self.overflow_history.push_back(overflow_detected);
        self.scaling_history.push(decision.clone());

        // Maintain history window
        self.maintain_history_window();

        Ok(decision)
    }

    /// Make scaling decision based on current state
    fn make_scaling_decision(
        &mut self,
        stats: &GradientStats,
        overflow_detected: bool,
        underflow_detected: bool,
    ) -> Result<ScalingDecision> {
        let old_scale = self.current_scale;
        let (new_scale, reason) = if overflow_detected {
            // Scale down on overflow
            let scale = (old_scale * self.config.backoff_factor).max(self.config.min_scale);
            self.growth_tracker = 0; // Reset growth tracker

            warn!(
                "Overflow detected, scaling down from {:.2} to {:.2}",
                old_scale, scale
            );
            (scale, ScalingReason::OverflowDetected)
        } else if underflow_detected && self.config.detect_underflow {
            // Scale up on underflow
            let scale = (old_scale * self.config.growth_factor).min(self.config.max_scale);

            debug!(
                "Underflow detected, scaling up from {:.2} to {:.2}",
                old_scale, scale
            );
            (scale, ScalingReason::UnderflowDetected)
        } else {
            // Apply strategy-specific scaling
            match self.config.strategy {
                ScalingStrategy::Fixed(scale) => (scale, ScalingReason::Initialization),
                ScalingStrategy::Dynamic => (
                    self.dynamic_scaling(stats)?,
                    ScalingReason::AdaptiveAdjustment,
                ),
                ScalingStrategy::Adaptive => (
                    self.adaptive_scaling(stats)?,
                    ScalingReason::AdaptiveAdjustment,
                ),
                ScalingStrategy::LossBased => (
                    self.loss_based_scaling(stats)?,
                    ScalingReason::AdaptiveAdjustment,
                ),
                ScalingStrategy::Percentile(p) => (
                    self.percentile_scaling(stats, p)?,
                    ScalingReason::AdaptiveAdjustment,
                ),
                ScalingStrategy::LearningRateDependent => (
                    self.learning_rate_dependent_scaling()?,
                    ScalingReason::LearningRateChange,
                ),
                ScalingStrategy::BatchSizeDependent => (
                    self.batch_size_dependent_scaling()?,
                    ScalingReason::BatchSizeChange,
                ),
            }
        };

        // Clamp to bounds
        let new_scale = new_scale.clamp(self.config.min_scale, self.config.max_scale);

        self.current_scale = new_scale;

        let scale_factor = match self.config.strategy {
            ScalingStrategy::Fixed(scale) => scale,
            _ => new_scale,
        };

        Ok(ScalingDecision {
            scale_factor,
            scale_changed: (new_scale - old_scale).abs() > f32::EPSILON,
            reason,
            overflow_detected,
            underflow_detected,
        })
    }

    /// Dynamic scaling based on gradient statistics
    fn dynamic_scaling(&self, _stats: &GradientStats) -> Result<f32> {
        if self.step_count % self.config.update_frequency == 0 && self.step_count > 0 {
            // Check if we've been overflow-free for a while
            let recent_overflows = self
                .overflow_history
                .iter()
                .rev()
                .take(self.config.update_frequency)
                .any(|&overflow| overflow);

            if !recent_overflows {
                // Safe to grow
                Ok((self.current_scale * self.config.growth_factor).min(self.config.max_scale))
            } else {
                Ok(self.current_scale)
            }
        } else {
            Ok(self.current_scale)
        }
    }

    /// Adaptive scaling based on gradient statistics and optimizer type
    fn adaptive_scaling(&self, stats: &GradientStats) -> Result<f32> {
        match self.config.optimizer_type {
            OptimizerType::Adam | OptimizerType::AdamW => self.adam_adaptive_scaling(stats),
            OptimizerType::SGDMomentum => self.sgd_momentum_adaptive_scaling(stats),
            OptimizerType::AdaGrad => self.adagrad_adaptive_scaling(stats),
            OptimizerType::RMSprop => self.rmsprop_adaptive_scaling(stats),
            OptimizerType::LAMB => self.lamb_adaptive_scaling(stats),
            _ => Ok(self.current_scale),
        }
    }

    /// Adam-specific adaptive scaling
    fn adam_adaptive_scaling(&self, stats: &GradientStats) -> Result<f32> {
        // For Adam, we want to balance the adaptive moments
        // Scale based on the ratio of gradient magnitude to expected range

        let target_magnitude = 1e-4; // Target gradient magnitude for Adam
        let current_magnitude = stats.mean_magnitude;

        if current_magnitude > 0.0 {
            let ratio = target_magnitude / current_magnitude;
            let adjustment = ratio.log2().clamp(-2.0, 2.0); // Limit to 4x change
            Ok(self.current_scale * 2.0f32.powf(adjustment))
        } else {
            Ok(self.current_scale)
        }
    }

    /// SGD with momentum adaptive scaling
    fn sgd_momentum_adaptive_scaling(&self, stats: &GradientStats) -> Result<f32> {
        // For SGD with momentum, we want consistent gradient magnitudes
        let target_magnitude = 1e-3;
        let current_magnitude = stats.mean_magnitude;

        if current_magnitude > 0.0 && current_magnitude < target_magnitude * 0.1 {
            // Scale up if gradients are too small
            Ok((self.current_scale * 1.5).min(self.config.max_scale))
        } else if current_magnitude > target_magnitude * 10.0 {
            // Scale down if gradients are too large
            Ok((self.current_scale * 0.75).max(self.config.min_scale))
        } else {
            Ok(self.current_scale)
        }
    }

    /// AdaGrad adaptive scaling
    fn adagrad_adaptive_scaling(&self, _stats: &GradientStats) -> Result<f32> {
        // AdaGrad accumulates squared gradients, so we need to be more conservative
        if self.step_count % (self.config.update_frequency * 2) == 0 {
            Ok((self.current_scale * 1.1).min(self.config.max_scale))
        } else {
            Ok(self.current_scale)
        }
    }

    /// RMSprop adaptive scaling
    fn rmsprop_adaptive_scaling(&self, stats: &GradientStats) -> Result<f32> {
        // RMSprop uses exponential moving average of squared gradients
        let variance = stats.std_deviation * stats.std_deviation;

        if variance < 1e-8 {
            // Very small variance, safe to scale up
            Ok((self.current_scale * 1.2).min(self.config.max_scale))
        } else if variance > 1e-2 {
            // High variance, scale down for stability
            Ok((self.current_scale * 0.9).max(self.config.min_scale))
        } else {
            Ok(self.current_scale)
        }
    }

    /// LAMB adaptive scaling (for large batch training)
    fn lamb_adaptive_scaling(&self, _stats: &GradientStats) -> Result<f32> {
        // LAMB is designed for large batches, so we need to account for batch size
        if let Some(ref optimizer_state) = self.optimizer_state {
            let batch_scale_factor = (optimizer_state.batch_size as f32 / 256.0).sqrt();
            Ok(self.current_scale * batch_scale_factor)
        } else {
            Ok(self.current_scale)
        }
    }

    /// Loss-based scaling
    fn loss_based_scaling(&self, _stats: &GradientStats) -> Result<f32> {
        // This would require loss information, simplified for now
        Ok(self.current_scale)
    }

    /// Percentile-based scaling
    fn percentile_scaling(&self, stats: &GradientStats, _percentile: f32) -> Result<f32> {
        // Scale based on gradient magnitude percentiles
        let target_magnitude = 1e-4;
        let scale_factor = target_magnitude / stats.mean_magnitude.max(1e-8);
        Ok((self.current_scale * scale_factor).clamp(self.config.min_scale, self.config.max_scale))
    }

    /// Learning rate dependent scaling
    fn learning_rate_dependent_scaling(&self) -> Result<f32> {
        if let Some(ref optimizer_state) = self.optimizer_state {
            // Scale inversely with learning rate
            let lr_scale = (0.001 / optimizer_state.learning_rate).sqrt();
            Ok((self.current_scale * lr_scale).clamp(self.config.min_scale, self.config.max_scale))
        } else {
            Ok(self.current_scale)
        }
    }

    /// Batch size dependent scaling
    fn batch_size_dependent_scaling(&self) -> Result<f32> {
        if let Some(ref optimizer_state) = self.optimizer_state {
            // Scale with square root of batch size (common in large batch training)
            let batch_scale = (optimizer_state.batch_size as f32 / 32.0).sqrt();
            Ok((self.current_scale * batch_scale)
                .clamp(self.config.min_scale, self.config.max_scale))
        } else {
            Ok(self.current_scale)
        }
    }

    /// Compute gradient statistics
    fn compute_gradient_stats(&self, gradients: &[f32]) -> GradientStats {
        if gradients.is_empty() {
            return GradientStats {
                mean_magnitude: 0.0,
                std_deviation: 0.0,
                max_magnitude: 0.0,
                min_magnitude: 0.0,
                zero_count: 0,
                invalid_count: 0,
                total_count: 0,
            };
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut max_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;
        let mut zero_count = 0;
        let mut invalid_count = 0;

        for &grad in gradients {
            let abs_grad = grad.abs();

            if grad.is_finite() {
                sum += abs_grad;
                sum_sq += abs_grad * abs_grad;
                max_val = max_val.max(abs_grad);
                min_val = min_val.min(abs_grad);

                if abs_grad == 0.0 {
                    zero_count += 1;
                }
            } else {
                invalid_count += 1;
            }
        }

        let valid_count = gradients.len() - invalid_count;
        let mean = if valid_count > 0 {
            sum / valid_count as f32
        } else {
            0.0
        };
        let variance = if valid_count > 0 {
            (sum_sq / valid_count as f32) - (mean * mean)
        } else {
            0.0
        };

        GradientStats {
            mean_magnitude: mean,
            std_deviation: variance.sqrt(),
            max_magnitude: if max_val.is_finite() { max_val } else { 0.0 },
            min_magnitude: if min_val.is_finite() { min_val } else { 0.0 },
            zero_count,
            invalid_count,
            total_count: gradients.len(),
        }
    }

    /// Detect gradient overflow
    fn detect_overflow(&self, stats: &GradientStats) -> bool {
        if !self.config.detect_overflow {
            return false;
        }

        // Check for invalid gradients (inf/nan)
        if stats.invalid_count > 0 {
            return true;
        }

        // Check if gradients are too large
        let overflow_threshold = 1e4; // Adjust based on precision
        stats.max_magnitude > overflow_threshold
    }

    /// Detect gradient underflow
    fn detect_underflow(&self, stats: &GradientStats) -> bool {
        if !self.config.detect_underflow {
            return false;
        }

        // Check if gradients are too small
        let underflow_threshold = 1e-8;
        stats.mean_magnitude < underflow_threshold && stats.zero_count < stats.total_count / 2
    }

    /// Update internal state after scaling decision
    fn update_internal_state(&mut self, _stats: &GradientStats, decision: &ScalingDecision) {
        if decision.overflow_detected {
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
        }
    }

    /// Maintain history window size
    fn maintain_history_window(&mut self) {
        while self.gradient_history.len() > self.config.history_window {
            self.gradient_history.pop_front();
        }

        while self.overflow_history.len() > self.config.history_window {
            self.overflow_history.pop_front();
        }

        while self.scaling_history.len() > self.config.history_window {
            self.scaling_history.remove(0);
        }
    }

    /// Get current scaling factor
    pub fn get_scale(&self) -> f32 {
        self.current_scale
    }

    /// Get scaling statistics
    pub fn get_stats(&self) -> ScalingStats {
        let total_overflows = self
            .overflow_history
            .iter()
            .filter(|&&overflow| overflow)
            .count();
        let recent_overflows = self
            .overflow_history
            .iter()
            .rev()
            .take(100)
            .filter(|&&overflow| overflow)
            .count();

        let avg_scale = if !self.scaling_history.is_empty() {
            self.scaling_history
                .iter()
                .map(|d| d.scale_factor)
                .sum::<f32>()
                / self.scaling_history.len() as f32
        } else {
            self.current_scale
        };

        ScalingStats {
            current_scale: self.current_scale,
            step_count: self.step_count,
            total_overflows,
            recent_overflows,
            growth_tracker: self.growth_tracker,
            avg_scale,
            min_scale_seen: self
                .scaling_history
                .iter()
                .map(|d| d.scale_factor)
                .fold(f32::INFINITY, f32::min),
            max_scale_seen: self
                .scaling_history
                .iter()
                .map(|d| d.scale_factor)
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }

    /// Reset scaler state
    pub fn reset(&mut self) {
        self.current_scale = self.config.initial_scale;
        self.step_count = 0;
        self.growth_tracker = 0;
        self.gradient_history.clear();
        self.overflow_history.clear();
        self.scaling_history.clear();
    }
}

/// Statistics about gradient scaling performance
#[derive(Debug, Clone)]
pub struct ScalingStats {
    pub current_scale: f32,
    pub step_count: usize,
    pub total_overflows: usize,
    pub recent_overflows: usize,
    pub growth_tracker: usize,
    pub avg_scale: f32,
    pub min_scale_seen: f32,
    pub max_scale_seen: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_scaler_creation() {
        let config = GradientScalingConfig::default();
        let scaler = GradientScaler::new(config);

        assert_eq!(scaler.current_scale, 2.0f32.powi(16));
        assert_eq!(scaler.step_count, 0);
    }

    #[test]
    fn test_optimizer_specific_scalers() {
        let sgd_scaler = GradientScaler::for_optimizer(OptimizerType::SGD);
        let adam_scaler = GradientScaler::for_optimizer(OptimizerType::Adam);

        assert_eq!(sgd_scaler.current_scale, 1.0);
        assert_eq!(adam_scaler.current_scale, 2.0f32.powi(16));
    }

    #[test]
    fn test_gradient_stats_computation() {
        let config = GradientScalingConfig::default();
        let scaler = GradientScaler::new(config);

        let gradients = vec![1.0, -2.0, 3.0, 0.0, -1.0];
        let stats = scaler.compute_gradient_stats(&gradients);

        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.zero_count, 1);
        assert_eq!(stats.invalid_count, 0);
        assert_eq!(stats.max_magnitude, 3.0);
    }

    #[test]
    fn test_overflow_detection() {
        let config = GradientScalingConfig::default();
        let scaler = GradientScaler::new(config);

        let stats = GradientStats {
            mean_magnitude: 1e5,
            std_deviation: 0.0,
            max_magnitude: 1e5,
            min_magnitude: 1e5,
            zero_count: 0,
            invalid_count: 0,
            total_count: 1,
        };

        assert!(scaler.detect_overflow(&stats));
    }

    #[test]
    fn test_scaling_decision() {
        let mut config = GradientScalingConfig::default();
        config.strategy = ScalingStrategy::Fixed(2.0);
        let mut scaler = GradientScaler::new(config);

        let mut gradients = vec![0.1, -0.2, 0.3];
        let decision = scaler.scale_gradients(&mut gradients).unwrap();

        assert_eq!(decision.scale_factor, 2.0);
        assert_eq!(gradients, vec![0.2, -0.4, 0.6]);
    }
}
