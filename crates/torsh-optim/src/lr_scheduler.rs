//! Learning rate schedulers

use crate::{Optimizer, OptimizerError, OptimizerResult};
use torsh_core::error::{Result, TorshError};

/// Base trait for learning rate schedulers
pub trait LRScheduler {
    /// Update learning rates based on current epoch/step
    fn step(&mut self) -> OptimizerResult<()>;

    /// Update learning rates with optional metrics (for ReduceLROnPlateau)
    fn step_with_metric(&mut self, metric: Option<f32>) -> OptimizerResult<()> {
        // Default implementation ignores the metric
        self.step()
    }

    /// Get current learning rates
    fn get_last_lr(&self) -> &[f32];

    /// Get base learning rates
    fn get_base_lrs(&self) -> &[f32];

    /// Get current epoch/step count
    fn get_last_epoch(&self) -> i32;

    /// Reset the scheduler state
    fn reset(&mut self);

    /// Get scheduler state for serialization
    fn state_dict(&self) -> SchedulerState;

    /// Load scheduler state from serialization
    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()>;
}

/// Macro to implement common LRScheduler methods for schedulers with a `base` field
#[macro_export]
macro_rules! impl_base_scheduler_methods {
    ($scheduler_type:ty, $scheduler_name:expr) => {
        impl<O: Optimizer> LRScheduler for $scheduler_type {
            fn step(&mut self) -> OptimizerResult<()> {
                // Default implementation - should be overridden
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
                let mut state = SchedulerState::new($scheduler_name.to_string());
                state.last_epoch = self.base.last_epoch;
                state.base_lrs = self.base.base_lrs.clone();
                state.last_lr = self.base.last_lr.clone();
                state
            }

            fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
                self.base.last_epoch = state.last_epoch;
                self.base.base_lrs = state.base_lrs;
                self.base.last_lr = state.last_lr;
                Ok(())
            }
        }
    };
}

/// Macro to implement common LRScheduler methods with custom state handling
#[macro_export]
macro_rules! impl_scheduler_with_state {
    ($scheduler_type:ty, $scheduler_name:expr, $state_fields:expr, $load_state_fields:expr) => {
        impl<O: Optimizer> LRScheduler for $scheduler_type {
            fn step(&mut self) -> OptimizerResult<()> {
                // Default implementation - should be overridden
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
                let mut state = SchedulerState::new($scheduler_name.to_string());
                state.last_epoch = self.base.last_epoch;
                state.base_lrs = self.base.base_lrs.clone();
                state.last_lr = self.base.last_lr.clone();

                // Add custom state fields
                $state_fields(&self, &mut state);

                state
            }

            fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
                self.base.last_epoch = state.last_epoch;
                self.base.base_lrs = state.base_lrs;
                self.base.last_lr = state.last_lr;

                // Load custom state fields
                $load_state_fields(self, &state)?;

                Ok(())
            }
        }
    };
}

/// Scheduler state for serialization
#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub scheduler_type: String,
    pub last_epoch: i32,
    pub base_lrs: Vec<f32>,
    pub last_lr: Vec<f32>,
    pub state: std::collections::HashMap<String, f32>,
}

impl SchedulerState {
    pub fn new(scheduler_type: String) -> Self {
        Self {
            scheduler_type,
            last_epoch: 0,
            base_lrs: Vec::new(),
            last_lr: Vec::new(),
            state: std::collections::HashMap::new(),
        }
    }
}

/// Base scheduler implementation
pub struct BaseScheduler<O: Optimizer> {
    pub optimizer: O,
    pub base_lrs: Vec<f32>,
    pub last_lr: Vec<f32>,
    pub last_epoch: i32,
}

impl<O: Optimizer> BaseScheduler<O> {
    pub fn new(optimizer: O) -> Self {
        let base_lrs = optimizer.get_lr();
        let last_lr = base_lrs.clone();

        Self {
            optimizer,
            base_lrs,
            last_lr,
            last_epoch: 0,
        }
    }

    /// Set the learning rates for all parameter groups
    pub fn set_learning_rates(&mut self, lrs: &[f32]) {
        if !lrs.is_empty() {
            if lrs.len() == 1 {
                // Single learning rate - apply to all groups
                self.optimizer.set_lr(lrs[0]);
            } else {
                // Multiple learning rates - need to implement per-group setting
                // For now, just use the first one
                self.optimizer.set_lr(lrs[0]);
            }
        }
        self.last_lr = lrs.to_vec();
    }

    /// Get a mutable reference to the optimizer
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get a reference to the optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Increment the epoch counter
    pub fn increment_epoch(&mut self) {
        self.last_epoch += 1;
    }

    /// Set the epoch counter
    pub fn set_epoch(&mut self, epoch: i32) {
        self.last_epoch = epoch;
    }
}

impl<O: Optimizer> LRScheduler for BaseScheduler<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.increment_epoch();
        // Base implementation does nothing - schedulers override this
        Ok(())
    }

    fn get_last_lr(&self) -> &[f32] {
        &self.last_lr
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base_lrs
    }

    fn get_last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn reset(&mut self) {
        self.last_epoch = 0;
        self.last_lr = self.base_lrs.clone();
        self.set_learning_rates(&self.base_lrs.clone());
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("BaseScheduler".to_string());
        state.last_epoch = self.last_epoch;
        state.base_lrs = self.base_lrs.clone();
        state.last_lr = self.last_lr.clone();
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.last_epoch = state.last_epoch;
        self.base_lrs = state.base_lrs;
        self.last_lr = state.last_lr.clone();
        self.set_learning_rates(&state.last_lr);
        Ok(())
    }
}

/// Step learning rate scheduler
pub struct StepLR<O: Optimizer> {
    base: BaseScheduler<O>,
    step_size: i32,
    gamma: f32,
}

impl<O: Optimizer> StepLR<O> {
    pub fn new(optimizer: O, step_size: i32, gamma: f32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            step_size,
            gamma,
        }
    }
}

impl<O: Optimizer> LRScheduler for StepLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.increment_epoch();

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| {
                let num_steps = self.base.last_epoch / self.step_size;
                base_lr * self.gamma.powi(num_steps)
            })
            .collect();

        self.base.set_learning_rates(&new_lrs);
        Ok(())
    }

    fn get_last_lr(&self) -> &[f32] {
        self.base.get_last_lr()
    }

    fn get_base_lrs(&self) -> &[f32] {
        self.base.get_base_lrs()
    }

    fn get_last_epoch(&self) -> i32 {
        self.base.get_last_epoch()
    }

    fn reset(&mut self) {
        self.base.reset()
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = self.base.state_dict();
        state.scheduler_type = "StepLR".to_string();
        state
            .state
            .insert("step_size".to_string(), self.step_size as f32);
        state.state.insert("gamma".to_string(), self.gamma);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.load_state_dict(state.clone())?;

        if let Some(&step_size) = state.state.get("step_size") {
            self.step_size = step_size as i32;
        }
        if let Some(&gamma) = state.state.get("gamma") {
            self.gamma = gamma;
        }

        Ok(())
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR<O: Optimizer> {
    base: BaseScheduler<O>,
    gamma: f32,
}

impl<O: Optimizer> ExponentialLR<O> {
    pub fn new(optimizer: O, gamma: f32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            gamma,
        }
    }
}

impl<O: Optimizer> LRScheduler for ExponentialLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| base_lr * self.gamma.powi(self.base.last_epoch))
            .collect();

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
        let mut state = SchedulerState::new("ExponentialLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("gamma".to_string(), self.gamma);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&gamma) = state.state.get("gamma") {
            self.gamma = gamma;
        }
        Ok(())
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR<O: Optimizer> {
    base: BaseScheduler<O>,
    t_max: i32,
    eta_min: f32,
}

impl<O: Optimizer> CosineAnnealingLR<O> {
    pub fn new(optimizer: O, t_max: i32, eta_min: f32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            t_max,
            eta_min,
        }
    }
}

impl<O: Optimizer> LRScheduler for CosineAnnealingLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| {
                self.eta_min
                    + (base_lr - self.eta_min)
                        * (1.0
                            + (std::f32::consts::PI * self.base.last_epoch as f32
                                / self.t_max as f32)
                                .cos())
                        / 2.0
            })
            .collect();

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
        let mut state = SchedulerState::new("CosineAnnealingLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("t_max".to_string(), self.t_max as f32);
        state.state.insert("eta_min".to_string(), self.eta_min);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&t_max) = state.state.get("t_max") {
            self.t_max = t_max as i32;
        }
        if let Some(&eta_min) = state.state.get("eta_min") {
            self.eta_min = eta_min;
        }
        Ok(())
    }
}

/// Reduce learning rate on plateau
pub struct ReduceLROnPlateau<O: Optimizer> {
    optimizer: O,
    mode: String,
    factor: f32,
    patience: i32,
    threshold: f32,
    threshold_mode: String,
    cooldown: i32,
    min_lr: f32,
    eps: f32,
    best: Option<f32>,
    num_bad_epochs: i32,
    cooldown_counter: i32,
}

impl<O: Optimizer> ReduceLROnPlateau<O> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        optimizer: O,
        mode: &str,
        factor: f32,
        patience: i32,
        threshold: f32,
        threshold_mode: &str,
        cooldown: i32,
        min_lr: f32,
        eps: f32,
    ) -> Result<Self> {
        if factor >= 1.0 {
            return Err(TorshError::Other("Factor should be < 1.0".to_string()));
        }

        Ok(Self {
            optimizer,
            mode: mode.to_string(),
            factor,
            patience,
            threshold,
            threshold_mode: threshold_mode.to_string(),
            cooldown,
            min_lr,
            eps,
            best: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        })
    }

    pub fn step(&mut self, metrics: f32) {
        let current = metrics;

        if self.best.is_none() {
            self.best = Some(current);
        } else {
            let best_value = self.best.expect("best should exist after is_none check");
            let is_better = match self.mode.as_str() {
                "min" => match self.threshold_mode.as_str() {
                    "rel" => current < best_value * (1.0 - self.threshold),
                    "abs" => current < best_value - self.threshold,
                    _ => false,
                },
                "max" => match self.threshold_mode.as_str() {
                    "rel" => current > best_value * (1.0 + self.threshold),
                    "abs" => current > best_value + self.threshold,
                    _ => false,
                },
                _ => false,
            };

            if is_better {
                self.best = Some(current);
                self.num_bad_epochs = 0;
            } else {
                self.num_bad_epochs += 1;
            }

            if self.cooldown_counter > 0 {
                self.cooldown_counter -= 1;
                self.num_bad_epochs = 0;
            }

            if self.num_bad_epochs > self.patience {
                self.reduce_lr();
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }

    fn reduce_lr(&mut self) {
        let old_lrs = self.optimizer.get_lr();
        let new_lrs: Vec<f32> = old_lrs
            .iter()
            .map(|&lr| (lr * self.factor).max(self.min_lr))
            .collect();

        // Only reduce if the change is significant
        for (old_lr, new_lr) in old_lrs.iter().zip(new_lrs.iter()) {
            if old_lr - new_lr > self.eps {
                self.optimizer.set_lr(*new_lr);
                println!("Reducing learning rate from {} to {}", old_lr, new_lr);
            }
        }
    }
}

/// One cycle learning rate scheduler
pub struct OneCycleLR<O: Optimizer> {
    base: BaseScheduler<O>,
    max_lr: Vec<f32>,
    total_steps: i32,
    pct_start: f32,
    anneal_strategy: String,
    #[allow(dead_code)]
    cycle_momentum: bool,
    #[allow(dead_code)]
    base_momentum: f32,
    #[allow(dead_code)]
    max_momentum: f32,
    #[allow(dead_code)]
    div_factor: f32,
    final_div_factor: f32,
    step_count: i32,
}

impl<O: Optimizer> OneCycleLR<O> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        optimizer: O,
        max_lr: Vec<f32>,
        total_steps: i32,
        pct_start: Option<f32>,
        anneal_strategy: Option<&str>,
        cycle_momentum: Option<bool>,
        base_momentum: Option<f32>,
        max_momentum: Option<f32>,
        div_factor: Option<f32>,
        final_div_factor: Option<f32>,
    ) -> Self {
        let pct_start = pct_start.unwrap_or(0.3);
        let anneal_strategy = anneal_strategy.unwrap_or("cos").to_string();
        let cycle_momentum = cycle_momentum.unwrap_or(true);
        let base_momentum = base_momentum.unwrap_or(0.85);
        let max_momentum = max_momentum.unwrap_or(0.95);
        let div_factor = div_factor.unwrap_or(25.0);
        let final_div_factor = final_div_factor.unwrap_or(10000.0);

        let mut base = BaseScheduler::new(optimizer);

        // Initialize base learning rates
        base.base_lrs = max_lr.iter().map(|&lr| lr / div_factor).collect();

        Self {
            base,
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy,
            cycle_momentum,
            base_momentum,
            max_momentum,
            div_factor,
            final_div_factor,
            step_count: 0,
        }
    }
}

impl<O: Optimizer> LRScheduler for OneCycleLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        let step_num = self.step_count as f32;
        let step_size_up = (self.pct_start * self.total_steps as f32).floor();
        let step_size_down = self.total_steps as f32 - step_size_up;

        let new_lrs: Vec<f32> = if step_num <= step_size_up {
            // Increase phase
            let computed_lr =
                |base_lr: f32, max_lr: f32| base_lr + (max_lr - base_lr) * step_num / step_size_up;

            self.base
                .base_lrs
                .iter()
                .zip(self.max_lr.iter())
                .map(|(&base, &max)| computed_lr(base, max))
                .collect()
        } else {
            // Decrease phase
            let down_step_num = step_num - step_size_up;
            match self.anneal_strategy.as_str() {
                "cos" => {
                    let computed_lr = |max_lr: f32, base_lr: f32| {
                        let min_lr = base_lr / self.final_div_factor;
                        min_lr
                            + (max_lr - min_lr)
                                * (1.0
                                    + (std::f32::consts::PI * down_step_num / step_size_down).cos())
                                / 2.0
                    };

                    self.max_lr
                        .iter()
                        .zip(self.base.base_lrs.iter())
                        .map(|(&max, &base)| computed_lr(max, base))
                        .collect()
                }
                "linear" => {
                    let computed_lr = |max_lr: f32, base_lr: f32| {
                        let min_lr = base_lr / self.final_div_factor;
                        max_lr - (max_lr - min_lr) * down_step_num / step_size_down
                    };

                    self.max_lr
                        .iter()
                        .zip(self.base.base_lrs.iter())
                        .map(|(&max, &base)| computed_lr(max, base))
                        .collect()
                }
                _ => {
                    return Err(OptimizerError::InvalidParameter(format!(
                        "Unknown anneal strategy: {}",
                        self.anneal_strategy
                    )))
                }
            }
        };

        // Update learning rates
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
        self.step_count
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.base.last_lr = self.base.base_lrs.clone();
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("OneCycleLR".to_string());
        state.last_epoch = self.step_count;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state
            .state
            .insert("total_steps".to_string(), self.total_steps as f32);
        state.state.insert("pct_start".to_string(), self.pct_start);
        state
            .state
            .insert("final_div_factor".to_string(), self.final_div_factor);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.step_count = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&total_steps) = state.state.get("total_steps") {
            self.total_steps = total_steps as i32;
        }
        if let Some(&pct_start) = state.state.get("pct_start") {
            self.pct_start = pct_start;
        }
        if let Some(&final_div_factor) = state.state.get("final_div_factor") {
            self.final_div_factor = final_div_factor;
        }
        Ok(())
    }
}
