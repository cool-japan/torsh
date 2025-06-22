//! Adam and AdamW optimizers

use crate::{Optimizer, OptimizerState, ParamGroup, optimizer::BaseOptimizer};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{Tensor, creation::zeros_like};
use torsh_autograd::prelude::*;
// Temporarily disable scirs2 integration
// use scirs2::optim::adam::{Adam as SciAdam, AdamW as SciAdamW};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Adam optimizer
pub struct Adam {
    base: BaseOptimizer,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-3);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        
        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("beta1".to_string(), betas.0);
        defaults.insert("beta2".to_string(), betas.1);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);
        
        let param_group = ParamGroup::new(params, lr);
        
        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "Adam".to_string(),
            defaults,
        };
        
        Self {
            base,
            betas,
            eps,
            weight_decay,
            amsgrad,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        // Increment step count
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();
                
                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }
                
                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());
                
                // Get or initialize optimizer state
                let state = self.base.state.entry(param_id.clone()).or_insert_with(|| {
                    let mut state = HashMap::new();
                    state.insert("step".to_string(), zeros_like(&param));
                    state.insert("exp_avg".to_string(), zeros_like(&param)); 
                    state.insert("exp_avg_sq".to_string(), zeros_like(&param));
                    if self.amsgrad {
                        state.insert("max_exp_avg_sq".to_string(), zeros_like(&param));
                    }
                    state
                });
                
                let mut step_tensor = state.get("step").unwrap().clone();
                let mut exp_avg = state.get("exp_avg").unwrap().clone();
                let mut exp_avg_sq = state.get("exp_avg_sq").unwrap().clone();
                
                // Increment step count
                step_tensor.add_scalar_(1.0)?;
                let step = step_tensor.to_vec()[0] as i32;
                
                // Apply weight decay
                let mut grad = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param.mul_scalar(self.weight_decay)?;
                    grad = grad.add(&weight_decay_term)?;
                }
                
                // Update biased first moment estimate
                exp_avg.mul_scalar_(self.betas.0)?;
                let grad_term = grad.mul_scalar(1.0 - self.betas.0)?;
                exp_avg.add_(&grad_term)?;
                
                // Update biased second raw moment estimate
                exp_avg_sq.mul_scalar_(self.betas.1)?;
                let grad_squared = grad.mul(&grad)?;
                let grad_sq_term = grad_squared.mul_scalar(1.0 - self.betas.1)?;
                exp_avg_sq.add_(&grad_sq_term)?;
                
                let denom = if self.amsgrad {
                    // Update max of exp_avg_sq
                    let mut max_exp_avg_sq = state.get("max_exp_avg_sq").unwrap().clone();
                    max_exp_avg_sq = max_exp_avg_sq.maximum(&exp_avg_sq)?;
                    state.insert("max_exp_avg_sq".to_string(), max_exp_avg_sq.clone());
                    
                    // Use max for denominator
                    let sqrt_max = max_exp_avg_sq.sqrt()?;
                    sqrt_max.add_scalar(self.eps)?
                } else {
                    // Bias correction
                    let bias_correction1 = 1.0 - self.betas.0.powi(step);
                    let bias_correction2 = 1.0 - self.betas.1.powi(step);
                    
                    let corrected_exp_avg = exp_avg.div_scalar(bias_correction1)?;
                    let corrected_exp_avg_sq = exp_avg_sq.div_scalar(bias_correction2)?;
                    
                    let sqrt_corrected = corrected_exp_avg_sq.sqrt()?;
                    sqrt_corrected.add_scalar(self.eps)?
                };
                
                // Compute step
                let step_size = group.lr;
                let bias_correction1 = 1.0 - self.betas.0.powi(step);
                let corrected_exp_avg = exp_avg.div_scalar(bias_correction1)?;
                
                let update = corrected_exp_avg.div(&denom)?.mul_scalar(step_size)?;
                param.sub_(&update)?;
                
                // Update state
                state.insert("step".to_string(), step_tensor);
                state.insert("exp_avg".to_string(), exp_avg);
                state.insert("exp_avg_sq".to_string(), exp_avg_sq);
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
    
    fn state_dict(&self) -> OptimizerState {
        self.base.state_dict()
    }
    
    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()> {
        self.base.load_state_dict(state)
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    base: BaseOptimizer,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-3);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.01);
        
        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("beta1".to_string(), betas.0);
        defaults.insert("beta2".to_string(), betas.1);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);
        
        let param_group = ParamGroup::new(params, lr);
        
        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "AdamW".to_string(),
            defaults,
        };
        
        Self {
            base,
            betas,
            eps,
            weight_decay,
            amsgrad,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        // AdamW with decoupled weight decay
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();
                
                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }
                
                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());
                
                // Get or initialize optimizer state
                let state = self.base.state.entry(param_id.clone()).or_insert_with(|| {
                    let mut state = HashMap::new();
                    state.insert("step".to_string(), zeros_like(&param));
                    state.insert("exp_avg".to_string(), zeros_like(&param)); 
                    state.insert("exp_avg_sq".to_string(), zeros_like(&param));
                    if self.amsgrad {
                        state.insert("max_exp_avg_sq".to_string(), zeros_like(&param));
                    }
                    state
                });
                
                let mut step_tensor = state.get("step").unwrap().clone();
                let mut exp_avg = state.get("exp_avg").unwrap().clone();
                let mut exp_avg_sq = state.get("exp_avg_sq").unwrap().clone();
                
                // Increment step count
                step_tensor.add_scalar_(1.0)?;
                let step = step_tensor.to_vec()[0] as i32;
                
                // Apply weight decay directly to the parameter (decoupled)
                if self.weight_decay != 0.0 {
                    let weight_decay_update = param.mul_scalar(group.lr * self.weight_decay)?;
                    param.sub_(&weight_decay_update)?;
                }
                
                // Update biased first moment estimate
                exp_avg.mul_scalar_(self.betas.0)?;
                let grad_term = grad.mul_scalar(1.0 - self.betas.0)?;
                exp_avg.add_(&grad_term)?;
                
                // Update biased second raw moment estimate
                exp_avg_sq.mul_scalar_(self.betas.1)?;
                let grad_squared = grad.mul(&grad)?;
                let grad_sq_term = grad_squared.mul_scalar(1.0 - self.betas.1)?;
                exp_avg_sq.add_(&grad_sq_term)?;
                
                // Bias correction
                let bias_correction1 = 1.0 - self.betas.0.powi(step);
                let bias_correction2 = 1.0 - self.betas.1.powi(step);
                
                let corrected_exp_avg = exp_avg.div_scalar(bias_correction1)?;
                let corrected_exp_avg_sq = exp_avg_sq.div_scalar(bias_correction2)?;
                
                let denom = if self.amsgrad {
                    // Update max of exp_avg_sq
                    let mut max_exp_avg_sq = state.get("max_exp_avg_sq").unwrap().clone();
                    max_exp_avg_sq = max_exp_avg_sq.maximum(&corrected_exp_avg_sq)?;
                    state.insert("max_exp_avg_sq".to_string(), max_exp_avg_sq.clone());
                    
                    // Use max for denominator
                    let sqrt_max = max_exp_avg_sq.sqrt()?;
                    sqrt_max.add_scalar(self.eps)?
                } else {
                    let sqrt_corrected = corrected_exp_avg_sq.sqrt()?;
                    sqrt_corrected.add_scalar(self.eps)?
                };
                
                // Compute step
                let step_size = group.lr;
                let update = corrected_exp_avg.div(&denom)?.mul_scalar(step_size)?;
                param.sub_(&update)?;
                
                // Update state
                state.insert("step".to_string(), step_tensor);
                state.insert("exp_avg".to_string(), exp_avg);
                state.insert("exp_avg_sq".to_string(), exp_avg_sq);
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
    
    fn state_dict(&self) -> OptimizerState {
        self.base.state_dict()
    }
    
    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()> {
        self.base.load_state_dict(state)
    }
}

/// Builder for Adam optimizer
pub struct AdamBuilder {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl AdamBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
    
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }
    
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
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
    
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
    
    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> Adam {
        Adam::new(
            params,
            Some(self.lr),
            Some(self.betas),
            Some(self.eps),
            Some(self.weight_decay),
            self.amsgrad,
        )
    }
    
    pub fn build_adamw(self, params: Vec<Arc<RwLock<Tensor>>>) -> AdamW {
        AdamW::new(
            params,
            Some(self.lr),
            Some(self.betas),
            Some(self.eps),
            Some(self.weight_decay),
            self.amsgrad,
        )
    }
}

impl Default for AdamBuilder {
    fn default() -> Self {
        Self::new()
    }
}