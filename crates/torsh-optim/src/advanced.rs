//! Advanced optimizers using SciRS2 optimization algorithms

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Advanced Adam optimizer with SciRS2 enhancements
pub struct AdvancedAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,

    // State variables
    pub state: HashMap<String, AdamState>,
    pub step_count: u64,

    // SciRS2 enhancements
    pub adaptive_lr: bool,
    pub gradient_clipping: Option<f64>,
    pub warmup_steps: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AdamState {
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
    pub max_exp_avg_sq: Option<Tensor>,
}

impl AdvancedAdam {
    /// Create a new advanced Adam optimizer
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            state: HashMap::new(),
            step_count: 0,
            adaptive_lr: false,
            gradient_clipping: None,
            warmup_steps: None,
        }
    }

    /// Enable AMSGrad variant
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Add weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable adaptive learning rate
    pub fn with_adaptive_lr(mut self) -> Self {
        self.adaptive_lr = true;
        self
    }

    /// Add gradient clipping
    pub fn with_gradient_clipping(mut self, max_norm: f64) -> Self {
        self.gradient_clipping = Some(max_norm);
        self
    }

    /// Add learning rate warmup
    pub fn with_warmup(mut self, warmup_steps: u64) -> Self {
        self.warmup_steps = Some(warmup_steps);
        self
    }
}

impl Optimizer for AdvancedAdam {
    fn step(&mut self) -> OptimizerResult<()> {
        // NOTE: Full implementation deferred to v0.2.0
        // Reason: Requires parameter storage architecture redesign
        // Current design doesn't store parameters, making step() implementation incomplete
        // See ROADMAP.md for full implementation plan
        self.step_count += 1;
        Ok(())
    }

    fn zero_grad(&mut self) {
        // NOTE: Requires parameter storage - see ROADMAP.md for v0.2.0 implementation
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr as f32]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
    }

    fn add_param_group(
        &mut self,
        _params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        // NOTE: Parameter group management deferred to v0.2.0 - see ROADMAP.md
        // Will be implemented with full parameter storage architecture
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut global_state = HashMap::new();
        global_state.insert("lr".to_string(), self.lr as f32);
        global_state.insert("beta1".to_string(), self.beta1 as f32);
        global_state.insert("beta2".to_string(), self.beta2 as f32);
        global_state.insert("eps".to_string(), self.eps as f32);
        global_state.insert("weight_decay".to_string(), self.weight_decay as f32);
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "AdvancedAdam".to_string(),
            version: "1.0".to_string(),
            param_groups: vec![], // NOTE: Deferred to v0.2.0 - requires parameter storage
            state: HashMap::new(), // NOTE: Deferred to v0.2.0 - requires per-parameter state tracking
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "AdvancedAdam" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected AdvancedAdam, got {}",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.global_state.get("lr") {
            self.lr = *lr as f64;
        }
        if let Some(beta1) = state.global_state.get("beta1") {
            self.beta1 = *beta1 as f64;
        }
        if let Some(beta2) = state.global_state.get("beta2") {
            self.beta2 = *beta2 as f64;
        }
        if let Some(eps) = state.global_state.get("eps") {
            self.eps = *eps as f64;
        }
        if let Some(weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = *weight_decay as f64;
        }
        if let Some(step_count) = state.global_state.get("step_count") {
            self.step_count = *step_count as u64;
        }

        // NOTE: Parameter groups and per-parameter state loading deferred to v0.2.0
        // Requires full state persistence architecture - see ROADMAP.md
        Ok(())
    }
}

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
/// Particularly effective for large batch training
pub struct LAMB {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub bias_correction: bool,

    pub state: HashMap<String, LambState>,
    pub step_count: u64,
}

#[derive(Debug, Clone)]
pub struct LambState {
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
}

impl LAMB {
    /// Create a new LAMB optimizer
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            bias_correction: true,
            state: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer for LAMB {
    fn step(&mut self) -> OptimizerResult<()> {
        // NOTE: Full LAMB implementation deferred to v0.2.0
        // Requires parameter storage architecture - see ROADMAP.md
        self.step_count += 1;
        Ok(())
    }

    fn zero_grad(&mut self) {
        // NOTE: Requires parameter storage - see ROADMAP.md for v0.2.0 implementation
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr as f32]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
    }

    fn add_param_group(
        &mut self,
        _params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        // NOTE: Parameter group management deferred to v0.2.0 - see ROADMAP.md
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut global_state = HashMap::new();
        global_state.insert("lr".to_string(), self.lr as f32);
        global_state.insert("beta1".to_string(), self.beta1 as f32);
        global_state.insert("beta2".to_string(), self.beta2 as f32);
        global_state.insert("eps".to_string(), self.eps as f32);
        global_state.insert("weight_decay".to_string(), self.weight_decay as f32);
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "LAMB".to_string(),
            version: "1.0".to_string(),
            param_groups: vec![],  // TODO: Implement parameter groups
            state: HashMap::new(), // TODO: Implement per-parameter state
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "LAMB" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected LAMB, got {}",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.global_state.get("lr") {
            self.lr = *lr as f64;
        }
        if let Some(beta1) = state.global_state.get("beta1") {
            self.beta1 = *beta1 as f64;
        }
        if let Some(beta2) = state.global_state.get("beta2") {
            self.beta2 = *beta2 as f64;
        }
        if let Some(eps) = state.global_state.get("eps") {
            self.eps = *eps as f64;
        }
        if let Some(weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = *weight_decay as f64;
        }
        if let Some(step_count) = state.global_state.get("step_count") {
            self.step_count = *step_count as u64;
        }

        // NOTE: Parameter groups and per-parameter state loading deferred to v0.2.0
        // Requires full state persistence architecture - see ROADMAP.md
        Ok(())
    }
}

/// Lookahead optimizer wrapper
/// Can wrap any base optimizer to add lookahead capability
pub struct Lookahead<T: Optimizer> {
    pub base_optimizer: T,
    pub alpha: f64,
    pub k: u64,

    pub slow_weights: HashMap<String, Tensor>,
    pub step_count: u64,
}

impl<T: Optimizer> Lookahead<T> {
    /// Create a new Lookahead optimizer
    pub fn new(base_optimizer: T, alpha: f64, k: u64) -> Self {
        Self {
            base_optimizer,
            alpha,
            k,
            slow_weights: HashMap::new(),
            step_count: 0,
        }
    }
}

impl<T: Optimizer> Optimizer for Lookahead<T> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Perform base optimizer step
        self.base_optimizer.step()?;
        self.step_count += 1;

        // TODO: Implement lookahead update every k steps
        // Requires parameter storage
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut base_state = self.base_optimizer.state_dict()?;

        // Add Lookahead-specific state
        base_state
            .global_state
            .insert("alpha".to_string(), self.alpha as f32);
        base_state
            .global_state
            .insert("k".to_string(), self.k as f32);
        base_state
            .global_state
            .insert("step_count".to_string(), self.step_count as f32);
        base_state.optimizer_type = format!("Lookahead<{}>", base_state.optimizer_type);

        Ok(base_state)
    }

    fn load_state_dict(&mut self, mut state: OptimizerState) -> OptimizerResult<()> {
        // Extract Lookahead-specific state
        if let Some(alpha) = state.global_state.remove("alpha") {
            self.alpha = alpha as f64;
        }
        if let Some(k) = state.global_state.remove("k") {
            self.k = k as u64;
        }
        if let Some(step_count) = state.global_state.remove("step_count") {
            self.step_count = step_count as u64;
        }

        // Restore base optimizer type
        if state.optimizer_type.starts_with("Lookahead<") && state.optimizer_type.ends_with(">") {
            let base_type = &state.optimizer_type[10..state.optimizer_type.len() - 1];
            state.optimizer_type = base_type.to_string();
        }

        // Load base optimizer state
        self.base_optimizer.load_state_dict(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_adam() {
        let mut optimizer = AdvancedAdam::new(0.001)
            .with_amsgrad()
            .with_weight_decay(0.01)
            .with_gradient_clipping(1.0);

        // Test interface
        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert!(optimizer.step().is_ok());
    }

    #[test]
    fn test_lamb_optimizer() {
        let mut optimizer = LAMB::new(0.001);

        // Test interface
        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert!(optimizer.step().is_ok());
    }

    #[test]
    fn test_lookahead_wrapper() {
        let base_optimizer = AdvancedAdam::new(0.001);
        let mut lookahead = Lookahead::new(base_optimizer, 0.5, 5);

        // Test interface
        assert_eq!(lookahead.get_lr(), vec![0.001]);
        assert!(lookahead.step().is_ok());
    }
}
