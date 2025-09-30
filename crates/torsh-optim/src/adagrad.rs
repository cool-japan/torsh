//! AdaGrad optimizer

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// AdaGrad optimizer
pub struct AdaGrad {
    base: BaseOptimizer,
    #[allow(dead_code)]
    lr_decay: f32,
    #[allow(dead_code)]
    weight_decay: f32,
    #[allow(dead_code)]
    initial_accumulator_value: f32,
    #[allow(dead_code)]
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
                let mut grad = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    grad = grad
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    // Initialize sum of squares with initial_accumulator_value
                    let mut sum_of_squares = torsh_tensor::creation::zeros_like(&param)?;
                    if self.initial_accumulator_value != 0.0 {
                        sum_of_squares
                            .add_scalar_(self.initial_accumulator_value)
                            .unwrap();
                    }
                    state.insert("sum_of_squares".to_string(), sum_of_squares);
                    state.insert(
                        "step".to_string(),
                        torsh_tensor::creation::zeros_like(&param)?,
                    );
                }

                let mut sum_of_squares = state.get("sum_of_squares").unwrap().clone();
                let mut step_tensor = state.get("step").unwrap().clone();

                // Increment step count
                step_tensor
                    .add_scalar_(1.0)
                    .map_err(OptimizerError::TensorError)?;
                let step = step_tensor.to_vec().map_err(OptimizerError::TensorError)?[0] as f32;

                // Update sum of squares: sum_of_squares = sum_of_squares + grad^2
                let grad_squared = grad.mul_op(&grad).map_err(OptimizerError::TensorError)?;
                sum_of_squares
                    .add(&grad_squared)
                    .map_err(OptimizerError::TensorError)?;

                // Compute learning rate with decay: clr = lr / (1 + (step - 1) * lr_decay)
                let clr = if self.lr_decay != 0.0 {
                    group.lr / (1.0 + (step - 1.0) * self.lr_decay)
                } else {
                    group.lr
                };

                // Compute standard deviation: std = sqrt(sum_of_squares) + eps
                let std = sum_of_squares
                    .sqrt()
                    .map_err(OptimizerError::TensorError)?
                    .add_scalar(self.eps)
                    .map_err(OptimizerError::TensorError)?;

                // Apply update: param = param - clr * grad / std
                let update = grad
                    .div(&std)
                    .map_err(OptimizerError::TensorError)?
                    .mul_scalar(clr)
                    .map_err(OptimizerError::TensorError)?;
                param.sub(&update).map_err(OptimizerError::TensorError)?;

                // Update state
                state.insert("sum_of_squares".to_string(), sum_of_squares);
                state.insert("step".to_string(), step_tensor);
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
