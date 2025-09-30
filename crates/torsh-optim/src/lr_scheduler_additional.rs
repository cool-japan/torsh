//! Additional learning rate schedulers

use crate::{
    lr_scheduler::{BaseScheduler, LRScheduler, SchedulerState},
    Optimizer, OptimizerError, OptimizerResult,
};

/// Multi-step learning rate scheduler
pub struct MultiStepLR<O: Optimizer> {
    base: BaseScheduler<O>,
    milestones: Vec<i32>,
    gamma: f32,
}

impl<O: Optimizer> MultiStepLR<O> {
    pub fn new(optimizer: O, milestones: Vec<i32>, gamma: f32) -> Self {
        let mut milestones = milestones;
        milestones.sort_unstable();

        Self {
            base: BaseScheduler::new(optimizer),
            milestones,
            gamma,
        }
    }
}

impl<O: Optimizer> LRScheduler for MultiStepLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let num_milestones_passed = self
            .milestones
            .iter()
            .filter(|&&milestone| self.base.last_epoch >= milestone)
            .count() as i32;

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| base_lr * self.gamma.powi(num_milestones_passed))
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
        let mut state = SchedulerState::new("MultiStepLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("gamma".to_string(), self.gamma);
        for (i, &milestone) in self.milestones.iter().enumerate() {
            state
                .state
                .insert(format!("milestone_{}", i), milestone as f32);
        }
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&gamma) = state.state.get("gamma") {
            self.gamma = gamma;
        }
        // Note: milestones are immutable after construction
        Ok(())
    }
}

/// Cyclic learning rate scheduler
pub struct CyclicLR<O: Optimizer> {
    base: BaseScheduler<O>,
    base_lr: Vec<f32>,
    max_lr: Vec<f32>,
    step_size_up: i32,
    step_size_down: Option<i32>,
    mode: String,
    gamma: f32,
    scale_fn: Option<Box<dyn Fn(i32) -> f32>>,
    scale_mode: String,
    cycle: i32,
    step_in_cycle: i32,
}

impl<O: Optimizer> CyclicLR<O> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        optimizer: O,
        base_lr: Vec<f32>,
        max_lr: Vec<f32>,
        step_size_up: i32,
        step_size_down: Option<i32>,
        mode: Option<&str>,
        gamma: Option<f32>,
        scale_fn: Option<Box<dyn Fn(i32) -> f32>>,
        scale_mode: Option<&str>,
    ) -> Self {
        let mode = mode.unwrap_or("triangular").to_string();
        let gamma = gamma.unwrap_or(1.0);
        let scale_mode = scale_mode.unwrap_or("cycle").to_string();

        Self {
            base: BaseScheduler::new(optimizer),
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            gamma,
            scale_fn,
            scale_mode,
            cycle: 0,
            step_in_cycle: 0,
        }
    }

    fn get_cycle_length(&self) -> i32 {
        self.step_size_up + self.step_size_down.unwrap_or(self.step_size_up)
    }
}

impl<O: Optimizer> LRScheduler for CyclicLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_in_cycle += 1;

        if self.step_in_cycle >= self.get_cycle_length() {
            self.step_in_cycle = 0;
            self.cycle += 1;
        }

        let step_size_down = self.step_size_down.unwrap_or(self.step_size_up);

        let scale_factor = match self.mode.as_str() {
            "triangular" => 1.0,
            "triangular2" => 1.0 / (2.0_f32.powi(self.cycle)),
            "exp_range" => self.gamma.powi(self.base.last_epoch),
            _ => {
                if let Some(ref scale_fn) = self.scale_fn {
                    match self.scale_mode.as_str() {
                        "cycle" => scale_fn(self.cycle),
                        _ => scale_fn(self.base.last_epoch),
                    }
                } else {
                    1.0
                }
            }
        };

        let new_lrs: Vec<f32> = if self.step_in_cycle < self.step_size_up {
            // Ascending
            let pct = self.step_in_cycle as f32 / self.step_size_up as f32;
            self.base_lr
                .iter()
                .zip(self.max_lr.iter())
                .map(|(&base, &max)| base + (max - base) * pct * scale_factor)
                .collect()
        } else {
            // Descending
            let down_step = self.step_in_cycle - self.step_size_up;
            let pct = 1.0 - (down_step as f32 / step_size_down as f32);
            self.base_lr
                .iter()
                .zip(self.max_lr.iter())
                .map(|(&base, &max)| base + (max - base) * pct * scale_factor)
                .collect()
        };

        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
        self.base.last_epoch += 1;
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
        self.cycle = 0;
        self.step_in_cycle = 0;
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("CyclicLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state
            .state
            .insert("step_size_up".to_string(), self.step_size_up as f32);
        if let Some(step_size_down) = self.step_size_down {
            state
                .state
                .insert("step_size_down".to_string(), step_size_down as f32);
        }
        state.state.insert("gamma".to_string(), self.gamma);
        state.state.insert("cycle".to_string(), self.cycle as f32);
        state
            .state
            .insert("step_in_cycle".to_string(), self.step_in_cycle as f32);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&step_size_up) = state.state.get("step_size_up") {
            self.step_size_up = step_size_up as i32;
        }
        if let Some(&step_size_down) = state.state.get("step_size_down") {
            self.step_size_down = Some(step_size_down as i32);
        }
        if let Some(&gamma) = state.state.get("gamma") {
            self.gamma = gamma;
        }
        if let Some(&cycle) = state.state.get("cycle") {
            self.cycle = cycle as i32;
        }
        if let Some(&step_in_cycle) = state.state.get("step_in_cycle") {
            self.step_in_cycle = step_in_cycle as i32;
        }
        Ok(())
    }
}

/// Polynomial learning rate scheduler
pub struct PolynomialLR<O: Optimizer> {
    base: BaseScheduler<O>,
    total_iters: i32,
    power: f32,
}

impl<O: Optimizer> PolynomialLR<O> {
    pub fn new(optimizer: O, total_iters: i32, power: f32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            total_iters,
            power,
        }
    }
}

impl<O: Optimizer> LRScheduler for PolynomialLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let factor = if self.base.last_epoch > self.total_iters {
            0.0
        } else {
            (1.0 - self.base.last_epoch as f32 / self.total_iters as f32).powf(self.power)
        };

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| base_lr * factor)
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
        let mut state = SchedulerState::new("PolynomialLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state
            .state
            .insert("total_iters".to_string(), self.total_iters as f32);
        state.state.insert("power".to_string(), self.power);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&total_iters) = state.state.get("total_iters") {
            self.total_iters = total_iters as i32;
        }
        if let Some(&power) = state.state.get("power") {
            self.power = power;
        }
        Ok(())
    }
}

/// Linear learning rate scheduler
pub struct LinearLR<O: Optimizer> {
    base: BaseScheduler<O>,
    start_factor: f32,
    end_factor: f32,
    total_iters: i32,
}

impl<O: Optimizer> LinearLR<O> {
    pub fn new(optimizer: O, start_factor: f32, end_factor: f32, total_iters: i32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            start_factor,
            end_factor,
            total_iters,
        }
    }
}

impl<O: Optimizer> LRScheduler for LinearLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let factor = if self.base.last_epoch >= self.total_iters {
            self.end_factor
        } else {
            self.start_factor
                + (self.end_factor - self.start_factor)
                    * (self.base.last_epoch as f32 / self.total_iters as f32)
        };

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| base_lr * factor)
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
        let mut state = SchedulerState::new("LinearLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state
            .state
            .insert("start_factor".to_string(), self.start_factor);
        state
            .state
            .insert("end_factor".to_string(), self.end_factor);
        state
            .state
            .insert("total_iters".to_string(), self.total_iters as f32);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&start_factor) = state.state.get("start_factor") {
            self.start_factor = start_factor;
        }
        if let Some(&end_factor) = state.state.get("end_factor") {
            self.end_factor = end_factor;
        }
        if let Some(&total_iters) = state.state.get("total_iters") {
            self.total_iters = total_iters as i32;
        }
        Ok(())
    }
}

/// Constant learning rate scheduler
pub struct ConstantLR<O: Optimizer> {
    base: BaseScheduler<O>,
    factor: f32,
    total_iters: i32,
}

impl<O: Optimizer> ConstantLR<O> {
    pub fn new(optimizer: O, factor: f32, total_iters: i32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            factor,
            total_iters,
        }
    }
}

impl<O: Optimizer> LRScheduler for ConstantLR<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.base.last_epoch += 1;

        let factor = if self.base.last_epoch < self.total_iters {
            self.factor
        } else {
            1.0
        };

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| base_lr * factor)
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
        let mut state = SchedulerState::new("ConstantLR".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("factor".to_string(), self.factor);
        state
            .state
            .insert("total_iters".to_string(), self.total_iters as f32);
        state
    }

    fn load_state_dict(&mut self, state: SchedulerState) -> OptimizerResult<()> {
        self.base.last_epoch = state.last_epoch;
        self.base.base_lrs = state.base_lrs;
        self.base.last_lr = state.last_lr;
        if let Some(&factor) = state.state.get("factor") {
            self.factor = factor;
        }
        if let Some(&total_iters) = state.state.get("total_iters") {
            self.total_iters = total_iters as i32;
        }
        Ok(())
    }
}

/// Cosine annealing with warm restarts
pub struct CosineAnnealingWarmRestarts<O: Optimizer> {
    base: BaseScheduler<O>,
    t_0: i32,
    t_mult: i32,
    eta_min: f32,
    t_cur: i32,
}

impl<O: Optimizer> CosineAnnealingWarmRestarts<O> {
    pub fn new(optimizer: O, t_0: i32, t_mult: i32, eta_min: f32) -> Self {
        Self {
            base: BaseScheduler::new(optimizer),
            t_0,
            t_mult,
            eta_min,
            t_cur: -1,
        }
    }
}

impl<O: Optimizer> LRScheduler for CosineAnnealingWarmRestarts<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.t_cur += 1;

        if self.t_cur >= self.t_0 {
            self.t_cur = 0;
            self.t_0 *= self.t_mult;
        }

        let new_lrs: Vec<f32> = self
            .base
            .base_lrs
            .iter()
            .map(|&base_lr| {
                self.eta_min
                    + (base_lr - self.eta_min)
                        * (1.0 + (std::f32::consts::PI * self.t_cur as f32 / self.t_0 as f32).cos())
                        / 2.0
            })
            .collect();

        for (i, &lr) in new_lrs.iter().enumerate() {
            if i == 0 {
                self.base.optimizer.set_lr(lr);
            }
        }

        self.base.last_lr = new_lrs;
        self.base.last_epoch += 1;
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
        self.t_cur = -1;
    }

    fn state_dict(&self) -> SchedulerState {
        let mut state = SchedulerState::new("CosineAnnealingWarmRestarts".to_string());
        state.last_epoch = self.base.last_epoch;
        state.base_lrs = self.base.base_lrs.clone();
        state.last_lr = self.base.last_lr.clone();
        state.state.insert("t_0".to_string(), self.t_0 as f32);
        state.state.insert("t_mult".to_string(), self.t_mult as f32);
        state.state.insert("eta_min".to_string(), self.eta_min);
        state.state.insert("t_cur".to_string(), self.t_cur as f32);
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
        if let Some(&eta_min) = state.state.get("eta_min") {
            self.eta_min = eta_min;
        }
        if let Some(&t_cur) = state.state.get("t_cur") {
            self.t_cur = t_cur as i32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_multi_step_lr() {
        let param = Arc::new(RwLock::new(ones(&[10]).unwrap()));
        let optimizer = SGD::new(vec![param], 0.1, None, None, None, false);
        let mut scheduler = MultiStepLR::new(optimizer, vec![10, 20, 30], 0.5);

        // Initial LR
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);

        // Step through epochs
        for _ in 0..9 {
            let _ = scheduler.step();
        }
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);

        // After milestone 10
        let _ = scheduler.step();
        assert_eq!(scheduler.get_last_lr(), vec![0.05]);

        // After milestone 20
        for _ in 0..10 {
            let _ = scheduler.step();
        }
        assert_eq!(scheduler.get_last_lr(), vec![0.025]);

        // After milestone 30
        for _ in 0..10 {
            let _ = scheduler.step();
        }
        assert_eq!(scheduler.get_last_lr(), vec![0.0125]);
    }

    #[test]
    fn test_linear_lr() {
        let param = Arc::new(RwLock::new(ones(&[10]).unwrap()));
        let optimizer = SGD::new(vec![param], 0.1, None, None, None, false);
        let mut scheduler = LinearLR::new(optimizer, 0.1, 1.0, 10);

        // Initial LR should be base_lr (no step taken yet)
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);

        // After first step
        let _ = scheduler.step();
        // Now it should be base_lr * (start_factor + (end_factor - start_factor) * (1/10))
        let expected_step1 = 0.1 * (0.1 + (1.0 - 0.1) * (1.0 / 10.0));
        assert!((scheduler.get_last_lr()[0] - expected_step1).abs() < 1e-6);

        // Halfway through (4 more steps to reach step 5)
        for _ in 0..4 {
            let _ = scheduler.step();
        }
        let expected = 0.1 * (0.1 + (1.0 - 0.1) * (5.0 / 10.0));
        assert!((scheduler.get_last_lr()[0] - expected).abs() < 1e-6);

        // At the end (5 more steps to reach step 10)
        for _ in 0..5 {
            let _ = scheduler.step();
        }
        assert_eq!(scheduler.get_last_lr(), vec![0.1]);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts() {
        let param = Arc::new(RwLock::new(ones(&[10]).unwrap()));
        let optimizer = SGD::new(vec![param], 0.1, None, None, None, false);
        let mut scheduler = CosineAnnealingWarmRestarts::new(optimizer, 10, 2, 0.0);

        // Initial step
        let _ = scheduler.step();
        let lr0 = scheduler.get_last_lr()[0];

        // Should decrease then restart
        for _ in 1..10 {
            let _ = scheduler.step();
            let lr = scheduler.get_last_lr()[0];
            assert!(lr <= lr0);
        }

        // After restart, LR should be back to base
        let _ = scheduler.step();
        assert!((scheduler.get_last_lr()[0] - 0.1).abs() < 1e-5);
    }
}
