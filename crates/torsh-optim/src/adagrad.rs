//! AdaGrad optimizer

use crate::{optimizer::BaseOptimizer, Optimizer, OptimizerState, ParamGroup};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_autograd::prelude::*;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::zeros_like, Tensor};

/// AdaGrad optimizer
pub struct AdaGrad {
    base: BaseOptimizer,
    lr_decay: f32,
    weight_decay: f32,
    initial_accumulator_value: f32,
    eps: f32,
}

impl AdaGrad {
    /// Create a new AdaGrad optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        lr_decay: Option<f32>,
        weight_decay: Option<f32>,
        initial_accumulator_value: Option<f32>,
        eps: Option<f32>,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let lr_decay = lr_decay.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let initial_accumulator_value = initial_accumulator_value.unwrap_or(0.0);
        let eps = eps.unwrap_or(1e-10);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("lr_decay".to_string(), lr_decay);
        defaults.insert("weight_decay".to_string(), weight_decay);
        defaults.insert(
            "initial_accumulator_value".to_string(),
            initial_accumulator_value,
        );
        defaults.insert("eps".to_string(), eps);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "AdaGrad".to_string(),
            defaults,
        };

        Self {
            base,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps,
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self) -> Result<()> {
        // Temporarily disabled - would implement AdaGrad algorithm when tensor ops are ready
        // For now, return a placeholder error
        Err(TorshError::Other(
            "AdaGrad optimizer step not yet implemented - pending tensor operation integration"
                .to_string(),
        ))
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerState {
        self.base.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()> {
        self.base.load_state_dict(state)
    }
}

/// Builder for AdaGrad optimizer
pub struct AdaGradBuilder {
    lr: f32,
    lr_decay: f32,
    weight_decay: f32,
    initial_accumulator_value: f32,
    eps: f32,
}

impl AdaGradBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-2,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn lr_decay(mut self, lr_decay: f32) -> Self {
        self.lr_decay = lr_decay;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn initial_accumulator_value(mut self, value: f32) -> Self {
        self.initial_accumulator_value = value;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> AdaGrad {
        AdaGrad::new(
            params,
            Some(self.lr),
            Some(self.lr_decay),
            Some(self.weight_decay),
            Some(self.initial_accumulator_value),
            Some(self.eps),
        )
    }
}

impl Default for AdaGradBuilder {
    fn default() -> Self {
        Self::new()
    }
}
