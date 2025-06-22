//! Stochastic Gradient Descent optimizer

use crate::{Optimizer, OptimizerState, ParamGroup, optimizer::BaseOptimizer};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
use torsh_autograd::prelude::*;
// Temporarily disable scirs2 integration
// use scirs2::optim::sgd::SGD as SciSGD;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// SGD optimizer with momentum and Nesterov acceleration
pub struct SGD {
    base: BaseOptimizer,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
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
    fn step(&mut self) -> Result<()> {
        // Temporarily disabled - would implement SGD algorithm when tensor ops are ready
        // For now, return a placeholder error
        Err(TorshError::Other(
            "SGD optimizer step not yet implemented - pending tensor operation integration".to_string()
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
    fn test_sgd_creation() {
        let param = Arc::new(RwLock::new(randn(&[10, 10])));
        let params = vec![param];
        
        let optimizer = SGD::new(params, 0.1, Some(0.9), None, None, false);
        assert_eq!(optimizer.base.get_lr(), vec![0.1]);
    }
    
    #[test]
    fn test_sgd_builder() {
        let param = Arc::new(RwLock::new(randn(&[10, 10])));
        let params = vec![param];
        
        let optimizer = SGDBuilder::new(0.01)
            .momentum(0.9)
            .weight_decay(1e-4)
            .nesterov(true)
            .build(params);
        
        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.weight_decay, 1e-4);
        assert!(optimizer.nesterov);
    }
}