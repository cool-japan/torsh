//! Base optimizer implementation utilities

use crate::{Optimizer, OptimizerState, ParamGroup, ParamGroupState};
use torsh_autograd::prelude::*;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
// Temporarily disable scirs2 integration
// use scirs2::optim::{Optimizer as SciOptimizer, OptimizerConfig};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Base optimizer struct (simplified without scirs2 integration)
pub struct BaseOptimizer {
    pub(crate) param_groups: Vec<ParamGroup>,
    pub(crate) state: HashMap<String, HashMap<String, Tensor>>,
    // Placeholder for optimizer-specific data
    pub(crate) optimizer_type: String,
    pub(crate) defaults: HashMap<String, f32>,
}

impl BaseOptimizer {
    /// Apply weight decay if specified
    pub(crate) fn apply_weight_decay(&self, param: &mut Tensor, weight_decay: f32) -> Result<()> {
        if weight_decay != 0.0 {
            let decay = param.mul_scalar(weight_decay)?;
            *param = param.sub(&decay)?;
        }
        Ok(())
    }

    /// Get parameter ID for state tracking
    pub(crate) fn param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", param.as_ref())
    }

    /// Initialize state for a parameter if not exists
    pub(crate) fn init_state(&mut self, param_id: String) {
        self.state.entry(param_id).or_insert_with(HashMap::new);
    }

    /// Get or create state tensor
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
    pub(crate) fn update_state(&mut self, param_id: &str, state_name: &str, value: Tensor) {
        self.state
            .get_mut(param_id)
            .unwrap()
            .insert(state_name.to_string(), value);
    }
}

impl Optimizer for BaseOptimizer {
    fn step(&mut self) -> Result<()> {
        // Temporarily disabled - would use scirs2's optimizer when integrated
        // For now, return a placeholder error
        Err(TorshError::Other(
            "Optimizer step not yet implemented - scirs2 integration pending".to_string(),
        ))
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

    fn state_dict(&self) -> OptimizerState {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
            })
            .collect();

        OptimizerState {
            param_groups,
            state: self.state.clone(),
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(TorshError::Other(
                "Loaded state dict has different number of parameter groups".to_string(),
            ));
        }

        for (group, state_group) in self.param_groups.iter_mut().zip(state.param_groups.iter()) {
            group.lr = state_group.lr;
            group.options = state_group.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Functional utilities for optimizers
pub mod functional {
    use super::*;

    /// Apply gradient clipping before optimizer step
    pub fn clip_grad_before_step<O: Optimizer>(
        optimizer: &O,
        max_norm: Option<f32>,
        norm_type: f32,
    ) -> f32 {
        if let Some(max_norm) = max_norm {
            // Collect all parameters from optimizer
            // This would need access to parameters through the optimizer trait
            // For now, return 0.0 as placeholder
            0.0
        } else {
            0.0
        }
    }
}
