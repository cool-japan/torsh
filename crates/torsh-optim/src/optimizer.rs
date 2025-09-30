//! Base optimizer implementation utilities

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
// Temporarily disable scirs2 integration
// use scirs2::optim::{Optimizer as SciOptimizer, OptimizerConfig};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Base optimizer struct (simplified without scirs2 integration)
#[derive(Clone)]
pub struct BaseOptimizer {
    pub(crate) param_groups: Vec<ParamGroup>,
    pub(crate) state: HashMap<String, HashMap<String, Tensor>>,
    // Placeholder for optimizer-specific data
    #[allow(dead_code)]
    pub(crate) optimizer_type: String,
    pub(crate) defaults: HashMap<String, f32>,
}

impl BaseOptimizer {
    /// Apply weight decay if specified
    #[allow(dead_code)]
    pub(crate) fn apply_weight_decay(
        &self,
        param: &mut Tensor,
        weight_decay: f32,
    ) -> OptimizerResult<()> {
        if weight_decay != 0.0 {
            let decay = param
                .mul_scalar(weight_decay)
                .map_err(OptimizerError::TensorError)?;
            *param = param.sub(&decay).map_err(OptimizerError::TensorError)?;
        }
        Ok(())
    }

    /// Get parameter ID for state tracking
    #[allow(dead_code)]
    pub(crate) fn param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", param.as_ref())
    }

    /// Initialize state for a parameter if not exists
    #[allow(dead_code)]
    pub(crate) fn init_state(&mut self, param_id: String) {
        self.state.entry(param_id).or_default();
    }

    /// Get or create state tensor
    #[allow(dead_code)]
    pub(crate) fn get_or_create_state(
        &mut self,
        param_id: &str,
        state_name: &str,
        init_fn: impl FnOnce() -> Tensor,
    ) -> Tensor {
        self.state
            .get_mut(param_id)
            .unwrap()
            .entry(state_name.to_string())
            .or_insert_with(init_fn)
            .clone()
    }

    /// Update state tensor
    #[allow(dead_code)]
    pub(crate) fn update_state(&mut self, param_id: &str, state_name: &str, value: Tensor) {
        self.state
            .get_mut(param_id)
            .unwrap()
            .insert(state_name.to_string(), value);
    }

    /// Initialize state with zeros_like for common optimizer states
    #[allow(dead_code)]
    pub(crate) fn init_state_with_zeros(
        &mut self,
        param_id: String,
        param: &Tensor,
        state_names: &[&str],
    ) -> OptimizerResult<()> {
        let state = self.state.entry(param_id).or_default();
        for &name in state_names {
            if !state.contains_key(name) {
                let zeros = torsh_tensor::creation::zeros_like(param)
                    .map_err(OptimizerError::TensorError)?;
                state.insert(name.to_string(), zeros);
            }
        }
        Ok(())
    }

    /// Initialize common Adam-like optimizer state
    #[allow(dead_code)]
    pub(crate) fn init_adam_state(
        &mut self,
        param_id: String,
        param: &Tensor,
        amsgrad: bool,
    ) -> OptimizerResult<()> {
        let state_names = if amsgrad {
            vec!["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
        } else {
            vec!["step", "exp_avg", "exp_avg_sq"]
        };
        self.init_state_with_zeros(param_id, param, &state_names)
    }

    /// Initialize common SGD-like optimizer state
    #[allow(dead_code)]
    pub(crate) fn init_sgd_state(
        &mut self,
        param_id: String,
        param: &Tensor,
        momentum: bool,
    ) -> OptimizerResult<()> {
        let state_names = if momentum {
            vec!["momentum_buffer"]
        } else {
            vec![]
        };
        if !state_names.is_empty() {
            self.init_state_with_zeros(param_id, param, &state_names)
        } else {
            self.init_state(param_id);
            Ok(())
        }
    }

    /// Apply weight decay to gradients
    #[allow(dead_code)]
    pub(crate) fn apply_weight_decay_to_grad(
        &self,
        grad: &mut Tensor,
        param: &Tensor,
        weight_decay: f32,
    ) -> OptimizerResult<()> {
        if weight_decay != 0.0 {
            let weight_decay_term = param
                .mul_scalar(weight_decay)
                .map_err(OptimizerError::TensorError)?;
            *grad = grad
                .add_op(&weight_decay_term)
                .map_err(OptimizerError::TensorError)?;
        }
        Ok(())
    }

    /// Get step count from state, incrementing if requested
    #[allow(dead_code)]
    pub(crate) fn get_step_count(
        &mut self,
        param_id: &str,
        increment: bool,
    ) -> OptimizerResult<i32> {
        let state = self.state.get_mut(param_id).unwrap();
        let step_tensor = state.get_mut("step").unwrap();

        if increment {
            step_tensor
                .add_scalar_(1.0)
                .map_err(OptimizerError::TensorError)?;
        }

        let step = step_tensor.to_vec().map_err(OptimizerError::TensorError)?[0] as i32;
        Ok(step)
    }

    /// Compute bias correction terms for Adam-like optimizers
    #[allow(dead_code)]
    pub(crate) fn compute_bias_correction(&self, betas: (f32, f32), step: i32) -> (f32, f32) {
        let bias_correction1 = 1.0 - betas.0.powi(step);
        let bias_correction2 = 1.0 - betas.1.powi(step);
        (bias_correction1, bias_correction2)
    }

    /// Update exponential moving average
    #[allow(dead_code)]
    pub(crate) fn update_exp_avg(
        &self,
        exp_avg: &mut Tensor,
        grad: &Tensor,
        beta: f32,
    ) -> OptimizerResult<()> {
        exp_avg
            .mul_scalar_(beta)
            .map_err(OptimizerError::TensorError)?;
        let grad_term = grad
            .mul_scalar(1.0 - beta)
            .map_err(OptimizerError::TensorError)?;
        exp_avg
            .add(&grad_term)
            .map_err(OptimizerError::TensorError)?;
        Ok(())
    }

    /// Update exponential moving average of squared gradients
    #[allow(dead_code)]
    pub(crate) fn update_exp_avg_sq(
        &self,
        exp_avg_sq: &mut Tensor,
        grad: &Tensor,
        beta: f32,
    ) -> OptimizerResult<()> {
        exp_avg_sq
            .mul_scalar_(beta)
            .map_err(OptimizerError::TensorError)?;
        let grad_squared = grad.mul_op(grad).map_err(OptimizerError::TensorError)?;
        let grad_sq_term = grad_squared
            .mul_scalar(1.0 - beta)
            .map_err(OptimizerError::TensorError)?;
        exp_avg_sq
            .add(&grad_sq_term)
            .map_err(OptimizerError::TensorError)?;
        Ok(())
    }

    /// Apply gradient clipping to a gradient tensor
    #[allow(dead_code)]
    pub(crate) fn clip_gradient(&self, grad: &mut Tensor, max_norm: f32) -> OptimizerResult<f32> {
        let norm = grad.norm().map_err(OptimizerError::TensorError)?;
        let norm_value = norm.to_vec().map_err(OptimizerError::TensorError)?[0];

        if norm_value > max_norm {
            let scale = max_norm / norm_value;
            *grad = grad
                .mul_scalar(scale)
                .map_err(OptimizerError::TensorError)?;
        }

        Ok(norm_value)
    }

    /// Check if all parameters have gradients
    #[allow(dead_code)]
    pub(crate) fn validate_gradients(&self) -> bool {
        self.param_groups
            .iter()
            .all(|group| group.params.iter().all(|param| param.read().has_grad()))
    }
}

impl Optimizer for BaseOptimizer {
    fn step(&mut self) -> OptimizerResult<()> {
        // Temporarily disabled - would use scirs2's optimizer when integrated
        // For now, return a placeholder error
        Err(OptimizerError::TensorError(TorshError::Other(
            "Optimizer step not yet implemented - scirs2 integration pending".to_string(),
        )))
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        mut options: HashMap<String, f32>,
    ) {
        let lr = options
            .remove("lr")
            .unwrap_or_else(|| self.defaults.get("lr").copied().unwrap_or(1e-3));

        let mut group = ParamGroup::new(params, lr);
        group.options = options;
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.create_state_dict(None)
    }

    // This method is moved outside of the trait implementation block below

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Validate the incoming state
        state
            .validate()
            .map_err(|e| OptimizerError::StateError(e.to_string()))?;

        // Check compatibility
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::StateError(
                "Loaded state dict has different number of parameter groups".to_string(),
            ));
        }

        // Check parameter counts match
        for (i, (group, state_group)) in self
            .param_groups
            .iter()
            .zip(state.param_groups.iter())
            .enumerate()
        {
            if group.params.len() != state_group.param_count {
                return Err(OptimizerError::StateError(format!(
                    "Parameter count mismatch in group {}: expected {}, got {}",
                    i,
                    group.params.len(),
                    state_group.param_count
                )));
            }
        }

        // Update parameter groups
        for (group, state_group) in self.param_groups.iter_mut().zip(state.param_groups.iter()) {
            group.lr = state_group.lr;
            group.options = state_group.options.clone();
        }

        // Update optimizer state
        self.state = state.state;

        // Update defaults from global state
        for (key, value) in state.global_state {
            self.defaults.insert(key, value);
        }

        Ok(())
    }
}

impl BaseOptimizer {
    /// Create a standardized state dict with optional additional global state
    #[allow(dead_code)]
    pub(crate) fn create_state_dict(
        &self,
        additional_global_state: Option<HashMap<String, f32>>,
    ) -> OptimizerResult<OptimizerState> {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState::from_param_group(g))
            .collect();

        let mut optimizer_state = OptimizerState::new(self.optimizer_type.clone());
        optimizer_state.param_groups = param_groups;
        optimizer_state.state = self.state.clone();

        // Add any global state from defaults
        for (key, value) in &self.defaults {
            optimizer_state.global_state.insert(key.clone(), *value);
        }

        // Add additional global state if provided
        if let Some(additional) = additional_global_state {
            for (key, value) in additional {
                optimizer_state.global_state.insert(key, value);
            }
        }

        Ok(optimizer_state)
    }
}

/// Functional utilities for optimizers
pub mod functional {
    use super::*;

    /// Apply gradient clipping before optimizer step
    pub fn clip_grad_before_step<O: Optimizer>(
        _optimizer: &O,
        max_norm: Option<f32>,
        _norm_type: f32,
    ) -> f32 {
        if let Some(_max_norm) = max_norm {
            // Collect all parameters from optimizer
            // This would need access to parameters through the optimizer trait
            // For now, return 0.0 as placeholder
            0.0
        } else {
            0.0
        }
    }
}
