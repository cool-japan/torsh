//! Optimization algorithms for ToRSh
//!
//! This crate provides PyTorch-compatible optimizers built on top of scirs2-optim.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod lr_scheduler;
pub mod nadam;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Re-export scirs2 optimizer functionality
use scirs2::optim as sci_optim;

/// Base optimizer trait
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self) -> Result<()>;

    /// Zero all gradients
    fn zero_grad(&mut self);

    /// Get the current learning rate
    fn get_lr(&self) -> Vec<f32>;

    /// Set the learning rate
    fn set_lr(&mut self, lr: f32);

    /// Add a parameter group
    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>);

    /// Get state dict for serialization
    fn state_dict(&self) -> OptimizerState;

    /// Load state dict
    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()>;
}

/// Optimizer state for serialization
#[derive(Clone)]
pub struct OptimizerState {
    pub param_groups: Vec<ParamGroupState>,
    pub state: HashMap<String, HashMap<String, Tensor>>,
}

/// Parameter group state
#[derive(Clone)]
pub struct ParamGroupState {
    pub lr: f32,
    pub options: HashMap<String, f32>,
}

/// Parameter group
pub struct ParamGroup {
    pub params: Vec<Arc<RwLock<Tensor>>>,
    pub lr: f32,
    pub options: HashMap<String, f32>,
}

impl ParamGroup {
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self {
            params,
            lr,
            options: HashMap::new(),
        }
    }

    pub fn with_options(mut self, options: HashMap<String, f32>) -> Self {
        self.options = options;
        self
    }
}

/// Common optimizer options
pub struct OptimizerOptions {
    pub lr: f32,
    pub weight_decay: f32,
    pub eps: f32,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 0.0,
            eps: 1e-8,
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adagrad::AdaGrad;
    pub use crate::adam::{Adam, AdamW};
    pub use crate::adamax::AdaMax;
    pub use crate::lr_scheduler::{CosineAnnealingLR, ExponentialLR, LRScheduler, StepLR};
    pub use crate::nadam::NAdam;
    pub use crate::rmsprop::RMSprop;
    pub use crate::sgd::SGD;
    pub use crate::{Optimizer, OptimizerOptions, OptimizerState, ParamGroup};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_group() {
        let params = vec![];
        let group = ParamGroup::new(params, 0.01);
        assert_eq!(group.lr, 0.01);
    }
}
