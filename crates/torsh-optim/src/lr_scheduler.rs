//! Learning rate schedulers

use crate::Optimizer;
use torsh_core::error::{Result, TorshError};

/// Base trait for learning rate schedulers
pub trait LRScheduler {
    /// Update learning rates
    fn step(&mut self);

    /// Get current learning rates
    fn get_last_lr(&self) -> Vec<f32>;

    /// Get base learning rates
    fn get_base_lrs(&self) -> &[f32];
}

/// Base scheduler implementation
pub struct BaseScheduler<O: Optimizer> {
    optimizer: O,
    base_lrs: Vec<f32>,
    last_lr: Vec<f32>,
    last_epoch: i32,
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
    fn step(&mut self) {
        self.base.last_epoch += 1;

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| {
                let num_steps = self.base.last_epoch / self.step_size;
                base_lr * self.gamma.powi(num_steps)
            })
            .collect();

        // Update optimizer learning rates
        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
            // TODO: Set individual param group LRs when supported
        }

        self.base.last_lr = new_lrs;
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.base.last_lr.clone()
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
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
    fn step(&mut self) {
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
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.base.last_lr.clone()
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
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
    fn step(&mut self) {
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
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.base.last_lr.clone()
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
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
            let is_better = match self.mode.as_str() {
                "min" => match self.threshold_mode.as_str() {
                    "rel" => current < self.best.unwrap() * (1.0 - self.threshold),
                    "abs" => current < self.best.unwrap() - self.threshold,
                    _ => false,
                },
                "max" => match self.threshold_mode.as_str() {
                    "rel" => current > self.best.unwrap() * (1.0 + self.threshold),
                    "abs" => current > self.best.unwrap() + self.threshold,
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
    fn step(&mut self) {
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
                _ => panic!("Unknown anneal strategy: {}", self.anneal_strategy),
            }
        };

        // Update learning rates
        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
    }

    fn get_last_lr(&self) -> Vec<f32> {
        self.base.last_lr.clone()
    }

    fn get_base_lrs(&self) -> &[f32] {
        &self.base.base_lrs
    }
}
