//! RMSprop optimizer

use crate::{optimizer::BaseOptimizer, Optimizer, OptimizerState, ParamGroup};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_autograd::prelude::*;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::zeros_like, Tensor};

/// RMSprop optimizer
pub struct RMSprop {
    base: BaseOptimizer,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        alpha: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let alpha = alpha.unwrap_or(0.99);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("alpha".to_string(), alpha);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("centered".to_string(), if centered { 1.0 } else { 0.0 });

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "RMSprop".to_string(),
            defaults,
        };

        Self {
            base,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<()> {
        // Temporarily disabled - would implement RMSprop algorithm when tensor ops are ready
        // For now, return a placeholder error
        Err(TorshError::Other(
            "RMSprop optimizer step not yet implemented - pending tensor operation integration"
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

/// Builder for RMSprop optimizer
pub struct RMSpropBuilder {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSpropBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> RMSprop {
        RMSprop::new(
            params,
            Some(self.lr),
            Some(self.alpha),
            Some(self.eps),
            Some(self.weight_decay),
            Some(self.momentum),
            self.centered,
        )
    }
}

impl Default for RMSpropBuilder {
    fn default() -> Self {
        Self::new()
    }
}
