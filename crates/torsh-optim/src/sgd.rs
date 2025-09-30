//! Stochastic Gradient Descent optimizer

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
// Temporarily disable scirs2 integration
// use scirs2::optim::sgd::SGD as SciSGD;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// SGD optimizer with momentum and Nesterov acceleration
#[derive(Clone)]
pub struct SGD {
    base: BaseOptimizer,
    #[allow(dead_code)]
    momentum: f32,
    #[allow(dead_code)]
    dampening: f32,
    #[allow(dead_code)]
    weight_decay: f32,
    #[allow(dead_code)]
    nesterov: bool,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        momentum: Option<f32>,
        dampening: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: bool,
    ) -> Self {
        let momentum = momentum.unwrap_or(0.0);
        let dampening = dampening.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);

        if nesterov && (momentum <= 0.0 || dampening != 0.0) {
            panic!("Nesterov momentum requires a momentum and zero dampening");
        }

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("dampening".to_string(), dampening);
        defaults.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "SGD".to_string(),
            defaults,
        };

        Self {
            base,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> OptimizerResult<()> {
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());

                // Apply weight decay to gradient if specified
                let mut d_p = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    d_p = d_p
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                if self.momentum != 0.0 {
                    // Get or initialize momentum buffer
                    let needs_init = !self.base.state.contains_key(&param_id);
                    let state = self
                        .base
                        .state
                        .entry(param_id.clone())
                        .or_insert_with(HashMap::new);

                    if needs_init {
                        state.insert(
                            "momentum_buffer".to_string(),
                            torsh_tensor::creation::zeros_like(&param)?,
                        );
                    }

                    let mut buf = state.get("momentum_buffer").unwrap().clone();

                    // Update momentum buffer: buf = momentum * buf + (1 - dampening) * d_p
                    buf.mul_scalar_(self.momentum)
                        .map_err(OptimizerError::TensorError)?;
                    let grad_term = d_p
                        .mul_scalar(1.0 - self.dampening)
                        .map_err(OptimizerError::TensorError)?;
                    buf = buf.add(&grad_term).map_err(OptimizerError::TensorError)?;

                    if self.nesterov {
                        // Nesterov momentum: d_p = d_p + momentum * buf
                        let nesterov_term = buf
                            .mul_scalar(self.momentum)
                            .map_err(OptimizerError::TensorError)?;
                        d_p = d_p
                            .add(&nesterov_term)
                            .map_err(OptimizerError::TensorError)?;
                    } else {
                        // Standard momentum: d_p = buf
                        d_p = buf.clone();
                    }

                    // Update state
                    state.insert("momentum_buffer".to_string(), buf);
                }

                // Apply update: param = param - lr * d_p
                let update = d_p
                    .mul_scalar(group.lr)
                    .map_err(OptimizerError::TensorError)?;
                *param = param.sub(&update).map_err(OptimizerError::TensorError)?;
            }
        }

        Ok(())
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

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.base.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base.load_state_dict(state)
    }
}

/// Builder for SGD optimizer
pub struct SGDBuilder {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
}

impl SGDBuilder {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> SGD {
        SGD::new(
            params,
            self.lr,
            Some(self.momentum),
            Some(self.dampening),
            Some(self.weight_decay),
            self.nesterov,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_sgd_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let params = vec![param];

        let optimizer = SGD::new(params, 0.1, Some(0.9), None, None, false);
        assert_eq!(optimizer.base.get_lr(), vec![0.1]);
        Ok(())
    }

    #[test]
    fn test_sgd_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let params = vec![param];

        let optimizer = SGDBuilder::new(0.01)
            .momentum(0.9)
            .weight_decay(1e-4)
            .nesterov(true)
            .build(params);

        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.weight_decay, 1e-4);
        assert!(optimizer.nesterov);
        Ok(())
    }
}
