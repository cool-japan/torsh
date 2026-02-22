//! Enhanced learning rate schedulers with advanced features
//!
//! This module provides sophisticated learning rate scheduling strategies including
//! polynomial decay with warmup, adaptive scheduling, and enhanced versions of
//! standard schedulers with additional features.

use crate::{
    lr_scheduler::{BaseScheduler, LRScheduler, SchedulerState},
    Optimizer, OptimizerError, OptimizerResult,
};
use std::f32::consts::PI;

/// Polynomial decay learning rate scheduler with warmup
///
/// This scheduler combines a warmup phase with polynomial decay, which is commonly
/// used in transformer training and other modern deep learning architectures.
/// During warmup, the learning rate increases linearly or polynomially from 0 to the base LR.
/// After warmup, the learning rate decays polynomially.
pub struct PolynomialDecayWithWarmup<O: Optimizer> {
    base: BaseScheduler<O>,
    /// Number of warmup steps
    warmup_steps: i32,
    /// Total number of training steps
    total_steps: i32,
    /// Power for polynomial decay (1.0 = linear, 2.0 = quadratic, etc.)
    power: f32,
    /// Minimum learning rate multiplier (relative to base_lr)
    end_lr_factor: f32,
    /// Warmup strategy: "linear" or "polynomial"
    warmup_strategy: WarmupStrategy,
    /// Power for warmup (only used if warmup_strategy is Polynomial)
    warmup_power: f32,
}

/// Strategy for warmup phase
#[derive(Debug, Clone)]
pub enum WarmupStrategy {
    /// Linear warmup: lr = base_lr * (step / warmup_steps)
    Linear,
    /// Polynomial warmup: lr = base_lr * (step / warmup_steps)^power
    Polynomial,
}

impl<O: Optimizer> PolynomialDecayWithWarmup<O> {
    /// Create a new polynomial decay with warmup scheduler
    pub fn new(
        optimizer: O,
        warmup_steps: i32,
        total_steps: i32,
        power: Option<f32>,
        end_lr_factor: Option<f32>,
        warmup_strategy: Option<WarmupStrategy>,
        warmup_power: Option<f32>,
    ) -> Self {
        let power = power.unwrap_or(1.0);
        let end_lr_factor = end_lr_factor.unwrap_or(0.0);
        let warmup_strategy = warmup_strategy.unwrap_or(WarmupStrategy::Linear);
        let warmup_power = warmup_power.unwrap_or(1.0);

        Self {
            base: BaseScheduler::new(optimizer),
            warmup_steps,
            total_steps,
            power,
            end_lr_factor,
            warmup_strategy,
            warmup_power,
        }
    }

    /// Create with linear warmup and quadratic decay (common configuration)
    pub fn linear_warmup_quadratic_decay(
        optimizer: O,
        warmup_steps: i32,
        total_steps: i32,
        end_lr_factor: Option<f32>,
    ) -> Self {
        Self::new(
            optimizer,
            warmup_steps,
            total_steps,
            Some(2.0), // Quadratic decay
            end_lr_factor,
            Some(WarmupStrategy::Linear),
            None,
        )
    }

    /// Create with linear warmup and linear decay
    pub fn linear_warmup_linear_decay(
        optimizer: O,
        warmup_steps: i32,
        total_steps: i32,
        end_lr_factor: Option<f32>,
    ) -> Self {
        Self::new(
            optimizer,
            warmup_steps,
            total_steps,
            Some(1.0), // Linear decay
            end_lr_factor,
            Some(WarmupStrategy::Linear),
            None,
        )
    }

    /// Compute learning rate for current step
    fn compute_lr(&self, step: i32, base_lr: f32) -> f32 {
        if step < self.warmup_steps {
            // Warmup phase
            let warmup_factor = match self.warmup_strategy {
                WarmupStrategy::Linear => step as f32 / self.warmup_steps as f32,
                WarmupStrategy::Polynomial => {
                    (step as f32 / self.warmup_steps as f32).powf(self.warmup_power)
                }
            };
            base_lr * warmup_factor
        } else if step < self.total_steps {
            // Decay phase
            let decay_steps = self.total_steps - self.warmup_steps;
            let remaining_steps = self.total_steps - step;
            let decay_factor = (remaining_steps as f32 / decay_steps as f32).powf(self.power);

            // Interpolate between end_lr_factor and 1.0
            let lr_factor = self.end_lr_factor + (1.0 - self.end_lr_factor) * decay_factor;
            base_lr * lr_factor
        } else {
            // After total steps, use minimum learning rate
            base_lr * self.end_lr_factor
        }
    }
}

impl<O: Optimizer> LRScheduler for PolynomialDecayWithWarmup<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| self.compute_lr(self.base.last_epoch, base_lr))
            .collect();

        // Update optimizer with new learning rate
        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
        Ok(())
    }

    fn get_last_lr(&self) -> &[f32] {
        &self.base.last_lr
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
    }

    fn get_last_epoch(&self) -> i32 {
        self.base.last_epoch
    }

    fn reset(&mut self) {
        self.base.last_epoch = 0;
        self.base.last_lr = self.base.base_lrs.clone();
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("PolynomialDecayWithWarmup".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state
            .state
            .insert("warmup_steps".to_string(), self.warmup_steps as f32);
        state
            .state
            .insert("total_steps".to_string(), self.total_steps as f32);
        state.state.insert("power".to_string(), self.power);
        state
            .state
            .insert("end_lr_factor".to_string(), self.end_lr_factor);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&warmup_steps) = state.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as i32;
        }
        if let Some(&total_steps) = state.state.get("total_steps") {
            self.total_steps = total_steps as i32;
        }
        if let Some(&power) = state.state.get("power") {
            self.power = power;
        }
        if let Some(&end_lr_factor) = state.state.get("end_lr_factor") {
            self.end_lr_factor = end_lr_factor;
        }
        Ok(())
    }
}

/// Adaptive learning rate scheduler that adjusts based on metrics
///
/// This scheduler monitors training metrics and adapts the learning rate dynamically
/// based on performance. It can increase LR when training is progressing well and
/// decrease it when training stagnates.
pub struct AdaptiveLRScheduler<O: Optimizer> {
    base: BaseScheduler<O>,
    /// Current metric value (loss, accuracy, etc.)
    current_metric: Option<f32>,
    /// History of metric values for trend analysis
    metric_history: Vec<f32>,
    /// Maximum history length to keep
    max_history: usize,
    /// Whether higher metric values are better (true for accuracy, false for loss)
    higher_is_better: bool,
    /// Factor to multiply LR when increasing
    increase_factor: f32,
    /// Factor to multiply LR when decreasing
    decrease_factor: f32,
    /// Minimum learning rate
    min_lr: f32,
    /// Maximum learning rate
    max_lr: f32,
    /// Number of steps to look back for trend analysis
    patience: usize,
    /// Threshold for detecting improvement/degradation
    improvement_threshold: f32,
    /// Current strategy being applied
    current_strategy: AdaptiveStrategy,
    /// Steps since last strategy change
    steps_since_change: i32,
    /// Minimum steps between strategy changes
    min_steps_between_changes: i32,
    /// Best metric value seen so far
    best_metric: Option<f32>,
    /// Counter for patience-based adjustments
    patience_counter: usize,
    /// Factor for learning rate adjustments
    factor: f32,
    /// Threshold for metric improvement
    threshold: f32,
    /// Cooldown period between adjustments
    cooldown: usize,
    /// Minimum learning rate factor
    min_lr_factor: f32,
}

/// Strategy for adaptive learning rate adjustment
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveStrategy {
    /// Maintain current learning rate
    Maintain,
    /// Increase learning rate (training is progressing well)
    Increase,
    /// Decrease learning rate (training is stagnating)
    Decrease,
    /// Oscillate learning rate (explore different ranges)
    Oscillate,
}

impl<O: Optimizer> AdaptiveLRScheduler<O> {
    /// Create a new adaptive learning rate scheduler
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        optimizer: O,
        higher_is_better: bool,
        increase_factor: Option<f32>,
        decrease_factor: Option<f32>,
        min_lr: Option<f32>,
        max_lr: Option<f32>,
        patience: Option<usize>,
        improvement_threshold: Option<f32>,
        min_steps_between_changes: Option<i32>,
    ) -> Self {
        let increase_factor = increase_factor.unwrap_or(1.05);
        let decrease_factor = decrease_factor.unwrap_or(0.9);
        let min_lr = min_lr.unwrap_or(1e-8);
        let max_lr = max_lr.unwrap_or(1.0);
        let patience = patience.unwrap_or(10);
        let improvement_threshold = improvement_threshold.unwrap_or(0.01);
        let min_steps_between_changes = min_steps_between_changes.unwrap_or(5);

        Self {
            base: BaseScheduler::new(optimizer),
            current_metric: None,
            metric_history: Vec::new(),
            max_history: 100,
            higher_is_better,
            increase_factor,
            decrease_factor,
            min_lr,
            max_lr,
            patience,
            improvement_threshold,
            current_strategy: AdaptiveStrategy::Maintain,
            steps_since_change: 0,
            min_steps_between_changes,
            best_metric: None,
            patience_counter: 0,
            factor: decrease_factor, // Default to decrease factor
            threshold: improvement_threshold,
            cooldown: 0,
            min_lr_factor: 0.01,
        }
    }

    /// Update the scheduler with a new metric value
    pub fn step_with_metric(&mut self, metric: f32) {
        self.current_metric = Some(metric);
        self.metric_history.push(metric);

        // Limit history size
        if self.metric_history.len() > self.max_history {
            self.metric_history.remove(0);
        }

        // Analyze trend and adjust strategy
        let new_strategy = self.analyze_trend();

        // Only change strategy if enough steps have passed
        if self.steps_since_change >= self.min_steps_between_changes {
            if new_strategy != self.current_strategy {
                self.current_strategy = new_strategy;
                self.steps_since_change = 0;
            }
        }

        self.steps_since_change += 1;

        // Apply the current strategy
        self.apply_strategy();

        self.base.last_epoch += 1;
    }

    /// Analyze metric trend to determine strategy
    fn analyze_trend(&self) -> AdaptiveStrategy {
        if self.metric_history.len() < self.patience {
            return AdaptiveStrategy::Maintain;
        }

        let recent_metrics = &self.metric_history[self.metric_history.len() - self.patience..];
        let older_metrics = if self.metric_history.len() >= 2 * self.patience {
            &self.metric_history[self.metric_history.len() - 2 * self.patience
                ..self.metric_history.len() - self.patience]
        } else {
            &self.metric_history[0..self.metric_history.len() - self.patience]
        };

        let recent_avg = recent_metrics.iter().sum::<f32>() / recent_metrics.len() as f32;
        let older_avg = older_metrics.iter().sum::<f32>() / older_metrics.len() as f32;

        let improvement = if self.higher_is_better {
            recent_avg - older_avg
        } else {
            older_avg - recent_avg
        };

        let relative_improvement = improvement / older_avg.abs().max(1e-8);

        if relative_improvement > self.improvement_threshold {
            AdaptiveStrategy::Increase
        } else if relative_improvement < -self.improvement_threshold {
            AdaptiveStrategy::Decrease
        } else {
            // Check for stagnation
            let variance = recent_metrics
                .iter()
                .map(|&x| (x - recent_avg).powi(2))
                .sum::<f32>()
                / recent_metrics.len() as f32;

            if variance < 1e-6 {
                AdaptiveStrategy::Oscillate
            } else {
                AdaptiveStrategy::Maintain
            }
        }
    }

    /// Apply the current strategy to adjust learning rate
    fn apply_strategy(&mut self) {
        let current_lrs = self.base.last_lr.clone();
        let mut new_lrs = Vec::new();

        for &current_lr in &current_lrs {
            let new_lr = match self.current_strategy {
                AdaptiveStrategy::Maintain => current_lr,
                AdaptiveStrategy::Increase => (current_lr * self.increase_factor).min(self.max_lr),
                AdaptiveStrategy::Decrease => (current_lr * self.decrease_factor).max(self.min_lr),
                AdaptiveStrategy::Oscillate => {
                    // Oscillate between current LR and slightly higher/lower values
                    let oscillation = (self.base.last_epoch as f32 * 0.1).sin() * 0.1 + 1.0;
                    (current_lr * oscillation).clamp(self.min_lr, self.max_lr)
                }
            };
            new_lrs.push(new_lr);
        }

        // Update optimizer with new learning rate
        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> &AdaptiveStrategy {
        &self.current_strategy
    }

    /// Get metric history
    pub fn metric_history(&self) -> &[f32] {
        &self.metric_history
    }

    /// Get statistics about the scheduler
    pub fn stats(&self) -> AdaptiveSchedulerStats {
        let avg_metric = if !self.metric_history.is_empty() {
            self.metric_history.iter().sum::<f32>() / self.metric_history.len() as f32
        } else {
            0.0
        };

        let current_lr = if !self.base.last_lr.is_empty() {
            self.base.last_lr[0]
        } else {
            0.0
        };

        AdaptiveSchedulerStats {
            current_lr,
            current_metric: self.current_metric,
            average_metric: avg_metric,
            current_strategy: self.current_strategy.clone(),
            steps_since_change: self.steps_since_change,
            metric_history_length: self.metric_history.len(),
        }
    }
}

/// Statistics for the adaptive scheduler
#[derive(Debug, Clone)]
pub struct AdaptiveSchedulerStats {
    pub current_lr: f32,
    pub current_metric: Option<f32>,
    pub average_metric: f32,
    pub current_strategy: AdaptiveStrategy,
    pub steps_since_change: i32,
    pub metric_history_length: usize,
}

impl<O: Optimizer> LRScheduler for AdaptiveLRScheduler<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // For regular step without metric, just maintain current LR
        self.base.last_epoch += 1;
        self.steps_since_change += 1;
        Ok(())
    }

    fn get_last_lr(&self) -> &[f32] {
        &self.base.last_lr
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
    }

    fn get_last_epoch(&self) -> i32 {
        self.base.last_epoch
    }

    fn reset(&mut self) {
        self.base.last_epoch = 0;
        self.base.last_lr = self.base.base_lrs.clone();
        self.steps_since_change = 0;
        self.best_metric = None;
        self.patience_counter = 0;
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("AdaptiveLRScheduler".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("factor".to_string(), self.factor);
        state
            .state
            .insert("patience".to_string(), self.patience as f32);
        state.state.insert("threshold".to_string(), self.threshold);
        state
            .state
            .insert("cooldown".to_string(), self.cooldown as f32);
        state
            .state
            .insert("min_lr_factor".to_string(), self.min_lr_factor);
        state.state.insert(
            "steps_since_change".to_string(),
            self.steps_since_change as f32,
        );
        state
            .state
            .insert("patience_counter".to_string(), self.patience_counter as f32);
        if let Some(best_metric) = self.best_metric {
            state.state.insert("best_metric".to_string(), best_metric);
        }
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&factor) = state.state.get("factor") {
            self.factor = factor;
        }
        if let Some(&patience) = state.state.get("patience") {
            self.patience = patience as usize;
        }
        if let Some(&threshold) = state.state.get("threshold") {
            self.threshold = threshold;
        }
        if let Some(&cooldown) = state.state.get("cooldown") {
            self.cooldown = cooldown as usize;
        }
        if let Some(&min_lr_factor) = state.state.get("min_lr_factor") {
            self.min_lr_factor = min_lr_factor;
        }
        if let Some(&steps_since_change) = state.state.get("steps_since_change") {
            self.steps_since_change = steps_since_change as i32;
        }
        if let Some(&patience_counter) = state.state.get("patience_counter") {
            self.patience_counter = patience_counter as usize;
        }
        if let Some(&best_metric) = state.state.get("best_metric") {
            self.best_metric = Some(best_metric);
        }
        Ok(())
    }
}

/// Cosine annealing with warm restarts and polynomial warmup
///
/// This scheduler combines polynomial warmup with cosine annealing and supports
/// warm restarts for improved convergence in long training runs.
pub struct CosineAnnealingWarmRestartsWithWarmup<O: Optimizer> {
    base: BaseScheduler<O>,
    /// Initial restart period
    t_0: i32,
    /// Factor to multiply restart period after each restart
    t_mult: i32,
    /// Number of warmup steps at the beginning
    warmup_steps: i32,
    /// Minimum learning rate multiplier
    eta_min_factor: f32,
    /// Current restart period
    current_t: i32,
    /// Steps since last restart
    steps_since_restart: i32,
    /// Number of completed restarts
    restart_count: i32,
    /// Warmup strategy
    warmup_strategy: WarmupStrategy,
    /// Power for polynomial warmup
    warmup_power: f32,
}

impl<O: Optimizer> CosineAnnealingWarmRestartsWithWarmup<O> {
    /// Create a new cosine annealing with warm restarts and warmup scheduler
    pub fn new(
        optimizer: O,
        t_0: i32,
        t_mult: Option<i32>,
        warmup_steps: Option<i32>,
        eta_min_factor: Option<f32>,
        warmup_strategy: Option<WarmupStrategy>,
        warmup_power: Option<f32>,
    ) -> Self {
        let t_mult = t_mult.unwrap_or(1);
        let warmup_steps = warmup_steps.unwrap_or(0);
        let eta_min_factor = eta_min_factor.unwrap_or(0.0);
        let warmup_strategy = warmup_strategy.unwrap_or(WarmupStrategy::Linear);
        let warmup_power = warmup_power.unwrap_or(1.0);

        Self {
            base: BaseScheduler::new(optimizer),
            t_0,
            t_mult,
            warmup_steps,
            eta_min_factor,
            current_t: t_0,
            steps_since_restart: 0,
            restart_count: 0,
            warmup_strategy,
            warmup_power,
        }
    }

    /// Compute learning rate for current step
    fn compute_lr(&self, step: i32, base_lr: f32) -> f32 {
        if step < self.warmup_steps {
            // Warmup phase
            let warmup_factor = match self.warmup_strategy {
                WarmupStrategy::Linear => step as f32 / self.warmup_steps as f32,
                WarmupStrategy::Polynomial => {
                    (step as f32 / self.warmup_steps as f32).powf(self.warmup_power)
                }
            };
            base_lr * warmup_factor
        } else {
            // Cosine annealing phase
            let adjusted_step = step - self.warmup_steps;
            let cycle_progress = (adjusted_step % self.current_t) as f32 / self.current_t as f32;
            let cosine_factor = 0.5 * (1.0 + (PI * cycle_progress).cos());

            base_lr * (self.eta_min_factor + (1.0 - self.eta_min_factor) * cosine_factor)
        }
    }
}

impl<O: Optimizer> LRScheduler for CosineAnnealingWarmRestartsWithWarmup<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        // Check for restart (only after warmup)
        if self.base.last_epoch >= self.warmup_steps {
            let adjusted_step = self.base.last_epoch - self.warmup_steps;
            if adjusted_step > 0 && adjusted_step % self.current_t == 0 {
                // Restart
                self.restart_count += 1;
                self.current_t *= self.t_mult;
                self.steps_since_restart = 0;
            } else {
                self.steps_since_restart += 1;
            }
        }

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| self.compute_lr(self.base.last_epoch, base_lr))
            .collect();

        // Update optimizer with new learning rate
        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
        Ok(())
    }

    fn get_last_lr(&self) -> &[f32] {
        &self.base.last_lr
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
    }

    fn get_last_epoch(&self) -> i32 {
        self.base.last_epoch
    }

    fn reset(&mut self) {
        self.base.last_epoch = 0;
        self.base.last_lr = self.base.base_lrs.clone();
        self.current_t = self.t_0;
        self.steps_since_restart = 0;
        self.restart_count = 0;
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("CosineAnnealingWarmRestartsWithWarmup".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("t_0".to_string(), self.t_0 as f32);
        state.state.insert("t_mult".to_string(), self.t_mult as f32);
        state
            .state
            .insert("warmup_steps".to_string(), self.warmup_steps as f32);
        state
            .state
            .insert("eta_min_factor".to_string(), self.eta_min_factor);
        state
            .state
            .insert("current_t".to_string(), self.current_t as f32);
        state.state.insert(
            "steps_since_restart".to_string(),
            self.steps_since_restart as f32,
        );
        state
            .state
            .insert("restart_count".to_string(), self.restart_count as f32);
        state
            .state
            .insert("warmup_power".to_string(), self.warmup_power);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&t_0) = state.state.get("t_0") {
            self.t_0 = t_0 as i32;
        }
        if let Some(&t_mult) = state.state.get("t_mult") {
            self.t_mult = t_mult as i32;
        }
        if let Some(&warmup_steps) = state.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as i32;
        }
        if let Some(&eta_min_factor) = state.state.get("eta_min_factor") {
            self.eta_min_factor = eta_min_factor;
        }
        if let Some(&current_t) = state.state.get("current_t") {
            self.current_t = current_t as i32;
        }
        if let Some(&steps_since_restart) = state.state.get("steps_since_restart") {
            self.steps_since_restart = steps_since_restart as i32;
        }
        if let Some(&restart_count) = state.state.get("restart_count") {
            self.restart_count = restart_count as i32;
        }
        if let Some(&warmup_power) = state.state.get("warmup_power") {
            self.warmup_power = warmup_power;
        }
        Ok(())
    }
}

/// Utility functions for creating common scheduler configurations
pub mod utils {
    use super::*;

    /// Create a polynomial decay with linear warmup (common transformer configuration)
    pub fn transformer_scheduler<O: Optimizer>(
        optimizer: O,
        warmup_steps: i32,
        total_steps: i32,
    ) -> PolynomialDecayWithWarmup<O> {
        PolynomialDecayWithWarmup::linear_warmup_linear_decay(
            optimizer,
            warmup_steps,
            total_steps,
            Some(0.0),
        )
    }

    /// Create an adaptive scheduler for loss monitoring
    pub fn adaptive_loss_scheduler<O: Optimizer>(
        optimizer: O,
        patience: Option<usize>,
    ) -> AdaptiveLRScheduler<O> {
        AdaptiveLRScheduler::new(
            optimizer,
            false,      // Lower loss is better
            Some(1.02), // Conservative increase
            Some(0.8),  // More aggressive decrease
            Some(1e-7),
            Some(0.1),
            patience,
            Some(0.005), // 0.5% improvement threshold
            Some(10),
        )
    }

    /// Create an adaptive scheduler for accuracy monitoring
    pub fn adaptive_accuracy_scheduler<O: Optimizer>(
        optimizer: O,
        patience: Option<usize>,
    ) -> AdaptiveLRScheduler<O> {
        AdaptiveLRScheduler::new(
            optimizer,
            true,       // Higher accuracy is better
            Some(1.05), // Moderate increase
            Some(0.9),  // Conservative decrease
            Some(1e-7),
            Some(0.1),
            patience,
            Some(0.01), // 1% improvement threshold
            Some(8),
        )
    }

    /// Create cosine annealing with warm restarts and linear warmup
    pub fn cosine_restart_with_warmup<O: Optimizer>(
        optimizer: O,
        restart_period: i32,
        warmup_steps: Option<i32>,
        t_mult: Option<i32>,
    ) -> CosineAnnealingWarmRestartsWithWarmup<O> {
        CosineAnnealingWarmRestartsWithWarmup::new(
            optimizer,
            restart_period,
            t_mult,
            warmup_steps,
            Some(0.0),
            Some(WarmupStrategy::Linear),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SGD;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_polynomial_decay_with_warmup() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let sgd = SGD::new(params, 0.1, None, None, None, false);
        let mut scheduler =
            PolynomialDecayWithWarmup::linear_warmup_quadratic_decay(sgd, 5, 20, Some(0.0));

        // Test warmup phase
        for step in 0..5 {
            let _ = scheduler.step();
            let lr = scheduler.get_last_lr()[0];
            let expected_lr = 0.1 * (step + 1) as f32 / 5.0;
            assert!(
                (lr - expected_lr).abs() < 1e-6,
                "Step {}: expected {}, got {}",
                step,
                expected_lr,
                lr
            );
        }

        // Test decay phase
        let _ = scheduler.step(); // Step 5
        let lr_step_5 = scheduler.get_last_lr()[0];
        assert!(lr_step_5 <= 0.1, "LR should start decaying after warmup");
    }

    #[test]
    fn test_adaptive_scheduler_with_improving_metric() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let sgd = SGD::new(params, 0.1, None, None, None, false);
        let mut scheduler = AdaptiveLRScheduler::new(
            sgd,
            false,
            Some(1.1),
            Some(0.9),
            Some(1e-4),
            Some(1.0),
            Some(3),
            Some(0.1),
            Some(2),
        );

        // Simulate improving loss (decreasing)
        let losses = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        for loss in losses {
            scheduler.step_with_metric(loss);
        }

        // Should increase learning rate due to improvement
        let final_lr = scheduler.get_last_lr()[0];
        assert!(final_lr >= 0.1, "LR should increase with improving metrics");
    }

    #[test]
    fn test_cosine_annealing_with_warmup() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let sgd = SGD::new(params, 0.1, None, None, None, false);
        let mut scheduler = CosineAnnealingWarmRestartsWithWarmup::new(
            sgd,
            10,
            Some(2),
            Some(5),
            Some(0.0),
            Some(WarmupStrategy::Linear),
            None,
        );

        // Test warmup phase
        for _ in 0..5 {
            let _ = scheduler.step();
        }
        let warmup_lr = scheduler.get_last_lr()[0];
        assert!(
            (warmup_lr - 0.1).abs() < 1e-6,
            "Should reach base LR at end of warmup"
        );

        // Test cosine annealing
        for _ in 5..15 {
            let _ = scheduler.step();
        }
        let final_lr = scheduler.get_last_lr()[0];
        assert!(
            final_lr >= 0.0 && final_lr <= 0.1,
            "LR should be within expected range"
        );
    }

    #[test]
    fn test_warmup_strategies() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];

        // Test linear warmup
        let sgd_linear = SGD::new(params.clone(), 0.1, None, None, None, false);
        let mut scheduler_linear = PolynomialDecayWithWarmup::new(
            sgd_linear,
            4,
            10,
            None,
            None,
            Some(WarmupStrategy::Linear),
            None,
        );

        // Test polynomial warmup
        let sgd_poly = SGD::new(params, 0.1, None, None, None, false);
        let mut scheduler_poly = PolynomialDecayWithWarmup::new(
            sgd_poly,
            4,
            10,
            None,
            None,
            Some(WarmupStrategy::Polynomial),
            Some(2.0),
        );

        // Compare warmup curves
        for step in 0..4 {
            let _ = scheduler_linear.step();
            let _ = scheduler_poly.step();

            let lr_linear = scheduler_linear.get_last_lr()[0];
            let lr_poly = scheduler_poly.get_last_lr()[0];

            if step < 3 {
                assert!(
                    lr_poly < lr_linear,
                    "Polynomial warmup should be slower initially"
                );
            }
        }
    }

    #[test]
    fn test_adaptive_strategy_detection() -> Result<(), Box<dyn std::error::Error>> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let sgd = SGD::new(params, 0.1, None, None, None, false);
        let mut scheduler = AdaptiveLRScheduler::new(
            sgd,
            false,
            Some(1.1),
            Some(0.9),
            Some(1e-4),
            Some(1.0),
            Some(3),
            Some(0.05),
            Some(1),
        );

        // Simulate stagnating loss
        for _ in 0..10 {
            scheduler.step_with_metric(1.0);
        }

        let stats = scheduler.stats();
        // Should detect stagnation and potentially oscillate
        assert!(matches!(
            stats.current_strategy,
            AdaptiveStrategy::Oscillate | AdaptiveStrategy::Decrease
        ));
        Ok(())
    }
}
