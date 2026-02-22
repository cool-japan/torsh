//! Schedule-Free Optimizers
//!
//! Schedule-Free optimizers eliminate the need for learning rate schedules by maintaining
//! two sequences of iterates: a fast sequence and a slow (averaged) sequence. This approach
//! achieves the benefits of learning rate scheduling without manual tuning.
//!
//! ## Key Innovation
//!
//! Traditional optimization requires carefully tuned learning rate schedules (warmup, decay, etc.).
//! Schedule-Free methods maintain:
//! - **Fast sequence (z)**: Used for gradient computation
//! - **Slow sequence (x)**: Exponential moving average, used for evaluation
//!
//! This simple change achieves:
//! - No need for learning rate schedules
//! - Better generalization than Adam with constant LR
//! - Comparable to Adam with optimal schedule
//! - Simpler hyperparameter tuning
//!
//! ## Algorithm (Schedule-Free AdamW)
//!
//! ```text
//! // Initialize
//! x_0 = θ_0  // slow weights (for evaluation)
//! z_0 = θ_0  // fast weights (for training)
//! y_0 = θ_0  // momentum state
//!
//! // Training step
//! g_t = ∇L(z_t)                                    // Gradient at fast sequence
//! y_t = (1-β₁)*g_t + β₁*y_{t-1}                   // Momentum update
//! v_t = (1-β₂)*g_t² + β₂*v_{t-1}                  // Second moment
//! z_t = z_{t-1} - γ * (y_t / (√v_t + ε) + λ*z_{t-1})  // Fast update
//! x_t = (1-c)*x_{t-1} + c*z_t                     // Slow update (EMA)
//!
//! // Evaluation: use x_t instead of z_t
//! ```
//!
//! Where:
//! - `z_t` is the fast sequence (used for gradient computation)
//! - `x_t` is the slow sequence (used for evaluation/inference)
//! - `y_t` is the momentum state
//! - `v_t` is the second moment (AdamW only)
//! - `β₁, β₂` are momentum coefficients
//! - `γ` is the learning rate
//! - `c` is the averaging coefficient (typically 0.01-0.1)
//! - `λ` is weight decay
//!
//! ## Typical Hyperparameters
//!
//! ### Schedule-Free AdamW:
//! - Learning rate: 1e-3 (constant, no schedule needed)
//! - β₁ (beta1): 0.9
//! - β₂ (beta2): 0.999
//! - Averaging coefficient (c): 0.05
//! - Weight decay: 0.01
//!
//! ### Schedule-Free SGD:
//! - Learning rate: 1.0 (constant, no schedule needed)
//! - β (beta): 0.9
//! - Averaging coefficient (c): 0.05
//! - Weight decay: 0.0
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::prelude::{ScheduleFreeAdamW, Optimizer};
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! let param = Arc::new(RwLock::new(randn::<f32>(&[768, 768])?));
//! let params = vec![param.clone()];
//!
//! // No learning rate schedule needed!
//! let mut optimizer = ScheduleFreeAdamW::new(
//!     params,
//!     1e-3,   // lr (constant)
//!     0.9,    // beta1
//!     0.999,  // beta2
//!     0.05,   // averaging coefficient
//!     0.01,   // weight_decay
//! );
//!
//! // Training mode
//! optimizer.train();
//! for _step in 0..10000 {
//!     // ... compute gradients at fast sequence ...
//!     optimizer.step()?;
//!     optimizer.zero_grad();
//! }
//!
//! // Evaluation mode (use slow sequence)
//! optimizer.eval();
//! // ... evaluate model ...
//! # Ok(())
//! # }
//! ```
//!
//! ## Reference
//!
//! Defazio, A., Mishchenko, K., & Stern, M. (2024).
//! "The Road Less Scheduled: Schedule-Free Learning Rate Optimization".
//! arXiv preprint arXiv:2405.15682.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Schedule-Free AdamW optimizer
///
/// Eliminates the need for learning rate schedules by maintaining fast and slow sequences.
pub struct ScheduleFreeAdamW {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,
    /// Learning rate (constant, no schedule needed)
    lr: f32,
    /// Beta1 (momentum coefficient)
    beta1: f32,
    /// Beta2 (second moment coefficient)
    beta2: f32,
    /// Averaging coefficient (typically 0.01-0.1)
    c: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Epsilon for numerical stability
    eps: f32,
    /// Fast sequence (z) - used for gradient computation
    z_params: HashMap<String, Tensor>,
    /// Slow sequence (x) - used for evaluation
    x_params: HashMap<String, Tensor>,
    /// Momentum state (y)
    momentum: HashMap<String, Tensor>,
    /// Second moment (v)
    second_moment: HashMap<String, Tensor>,
    /// Current step count
    step_count: usize,
    /// Training mode flag
    is_training: bool,
}

impl ScheduleFreeAdamW {
    /// Create a new Schedule-Free AdamW optimizer
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (constant, typically 1e-3)
    /// * `beta1` - Momentum coefficient (default: 0.9)
    /// * `beta2` - Second moment coefficient (default: 0.999)
    /// * `c` - Averaging coefficient (default: 0.05)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        c: f32,
        weight_decay: f32,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr);
        Self {
            param_groups: vec![param_group],
            lr,
            beta1,
            beta2,
            c,
            weight_decay,
            eps: 1e-8,
            z_params: HashMap::new(),
            x_params: HashMap::new(),
            momentum: HashMap::new(),
            second_moment: HashMap::new(),
            step_count: 0,
            is_training: true,
        }
    }

    /// Builder for Schedule-Free AdamW
    pub fn builder() -> ScheduleFreeAdamWBuilder {
        ScheduleFreeAdamWBuilder::default()
    }

    /// Switch to training mode (use fast sequence for gradients)
    pub fn train(&mut self) {
        if !self.is_training {
            self.is_training = true;
            // Copy slow sequence to parameters
            for group in &self.param_groups {
                for (idx, param) in group.params.iter().enumerate() {
                    let param_key = format!("param_{}", idx);
                    if let Some(z) = self.z_params.get(&param_key) {
                        *param.write() = z.clone();
                    }
                }
            }
        }
    }

    /// Switch to evaluation mode (use slow sequence)
    pub fn eval(&mut self) {
        if self.is_training {
            self.is_training = false;
            // Copy slow sequence to parameters
            for group in &self.param_groups {
                for (idx, param) in group.params.iter().enumerate() {
                    let param_key = format!("param_{}", idx);
                    if let Some(x) = self.x_params.get(&param_key) {
                        *param.write() = x.clone();
                    }
                }
            }
        }
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.is_training
    }
}

impl Optimizer for ScheduleFreeAdamW {
    fn step(&mut self) -> OptimizerResult<()> {
        if !self.is_training {
            return Err(OptimizerError::InvalidInput(
                "Cannot call step() in eval mode. Call train() first.".to_string(),
            ));
        }

        self.step_count += 1;

        for group in &self.param_groups {
            let lr = group.lr;
            let beta1 = self.beta1;
            let beta2 = self.beta2;
            let c = self.c;
            let weight_decay = self.weight_decay;
            let eps = self.eps;

            for (idx, param) in group.params.iter().enumerate() {
                let mut param_guard = param.write();

                // Skip parameters without gradients
                if !param_guard.has_grad() {
                    continue;
                }

                let grad = param_guard
                    .grad()
                    .ok_or_else(|| OptimizerError::InvalidInput("No gradient found".to_string()))?;

                let param_key = format!("param_{}", idx);

                // Initialize z (fast), x (slow), y (momentum), v (second moment)
                let z_entry = self
                    .z_params
                    .entry(param_key.clone())
                    .or_insert_with(|| param_guard.clone());
                let x_entry = self
                    .x_params
                    .entry(param_key.clone())
                    .or_insert_with(|| param_guard.clone());
                let y_entry = self.momentum.entry(param_key.clone()).or_insert_with(|| {
                    grad.zeros_like().expect("Failed to create momentum buffer")
                });
                let v_entry = self
                    .second_moment
                    .entry(param_key.clone())
                    .or_insert_with(|| {
                        grad.zeros_like()
                            .expect("Failed to create second moment buffer")
                    });

                // Update momentum: y = (1-β₁)*g + β₁*y
                let new_y = grad
                    .mul_scalar(1.0 - beta1)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &y_entry
                            .mul_scalar(beta1)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Update second moment: v = (1-β₂)*g² + β₂*v
                let grad_squared = grad
                    .mul(&grad)
                    .map_err(|e| OptimizerError::TensorError(e))?;
                let new_v = grad_squared
                    .mul_scalar(1.0 - beta2)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &v_entry
                            .mul_scalar(beta2)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Compute preconditioned update: y / (√v + ε)
                let v_sqrt = new_v
                    .sqrt()
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add_scalar(eps)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                let preconditioned = new_y
                    .div(&v_sqrt)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Update fast sequence: z = z - γ * (preconditioned + λ*z)
                let update = if weight_decay > 0.0 {
                    let decay_term = z_entry
                        .mul_scalar(weight_decay)
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    preconditioned
                        .add(&decay_term)
                        .map_err(|e| OptimizerError::TensorError(e))?
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                } else {
                    preconditioned
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                };

                let new_z = z_entry
                    .sub(&update)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Update slow sequence: x = (1-c)*x + c*z
                let new_x = x_entry
                    .mul_scalar(1.0 - c)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &new_z
                            .mul_scalar(c)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Store updated states
                *y_entry = new_y;
                *v_entry = new_v;
                *z_entry = new_z.clone();
                *x_entry = new_x;

                // Update parameter to z (fast sequence for training)
                *param_guard = new_z;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            group.zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(self.lr);
        let group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group_states = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState::from_param_group(g))
            .collect();

        let mut state = HashMap::new();
        for (key, _) in &self.z_params {
            let mut param_state = HashMap::new();
            if let Some(z) = self.z_params.get(key) {
                param_state.insert("z".to_string(), z.clone());
            }
            if let Some(x) = self.x_params.get(key) {
                param_state.insert("x".to_string(), x.clone());
            }
            if let Some(y) = self.momentum.get(key) {
                param_state.insert("momentum".to_string(), y.clone());
            }
            if let Some(v) = self.second_moment.get(key) {
                param_state.insert("second_moment".to_string(), v.clone());
            }
            state.insert(key.clone(), param_state);
        }

        let mut global_state = HashMap::new();
        global_state.insert("beta1".to_string(), self.beta1);
        global_state.insert("beta2".to_string(), self.beta2);
        global_state.insert("c".to_string(), self.c);
        global_state.insert("weight_decay".to_string(), self.weight_decay);
        global_state.insert(
            "is_training".to_string(),
            if self.is_training { 1.0 } else { 0.0 },
        );

        Ok(OptimizerState {
            optimizer_type: "ScheduleFreeAdamW".to_string(),
            version: "1.0".to_string(),
            param_groups: param_group_states,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "ScheduleFreeAdamW" {
            return Err(OptimizerError::InvalidInput(format!(
                "Expected ScheduleFreeAdamW state dict, got {}",
                state.optimizer_type
            )));
        }

        // Restore hyperparameters
        if let Some(&beta1) = state.global_state.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state.global_state.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&c) = state.global_state.get("c") {
            self.c = c;
        }
        if let Some(&weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&is_training) = state.global_state.get("is_training") {
            self.is_training = is_training > 0.5;
        }

        // Restore optimizer state
        self.z_params.clear();
        self.x_params.clear();
        self.momentum.clear();
        self.second_moment.clear();

        for (key, param_state) in state.state {
            if let Some(z) = param_state.get("z") {
                self.z_params.insert(key.clone(), z.clone());
            }
            if let Some(x) = param_state.get("x") {
                self.x_params.insert(key.clone(), x.clone());
            }
            if let Some(y) = param_state.get("momentum") {
                self.momentum.insert(key.clone(), y.clone());
            }
            if let Some(v) = param_state.get("second_moment") {
                self.second_moment.insert(key.clone(), v.clone());
            }
        }

        Ok(())
    }
}

/// Builder for Schedule-Free AdamW
#[derive(Debug, Clone)]
pub struct ScheduleFreeAdamWBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    c: f32,
    weight_decay: f32,
}

impl Default for ScheduleFreeAdamWBuilder {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            c: 0.05,
            weight_decay: 0.01,
        }
    }
}

impl ScheduleFreeAdamWBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameters
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    /// Set learning rate (constant, no schedule needed)
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set averaging coefficient
    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the optimizer
    pub fn build(self) -> ScheduleFreeAdamW {
        ScheduleFreeAdamW::new(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.c,
            self.weight_decay,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_schedule_free_adamw_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[64, 64])?));
        let params = vec![param];

        let optimizer = ScheduleFreeAdamW::new(params, 1e-3, 0.9, 0.999, 0.05, 0.01);
        assert_eq!(optimizer.lr, 1e-3);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.c, 0.05);
        assert!(optimizer.is_training());

        Ok(())
    }

    #[test]
    fn test_schedule_free_train_eval_mode() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[32, 32])?));
        let params = vec![param];

        let mut optimizer = ScheduleFreeAdamW::new(params, 1e-3, 0.9, 0.999, 0.05, 0.01);

        assert!(optimizer.is_training());

        optimizer.eval();
        assert!(!optimizer.is_training());

        optimizer.train();
        assert!(optimizer.is_training());

        Ok(())
    }

    #[test]
    fn test_schedule_free_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = ScheduleFreeAdamW::new(params, 1e-3, 0.9, 0.999, 0.05, 0.01);

        // Set gradient
        let grad = randn::<f32>(&[16, 16])?;
        param.write().set_grad(Some(grad));

        let param_before = param.read().clone();

        // Perform optimization step
        optimizer.step()?;

        let param_after = param.read().clone();

        // Parameters should change
        let diff = param_before.sub(&param_after)?;
        let diff_norm = diff.norm()?.to_vec()?[0];
        assert!(diff_norm > 0.0, "Parameters should have changed");

        Ok(())
    }

    #[test]
    fn test_schedule_free_cannot_step_in_eval() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[8, 8])?));
        let params = vec![param.clone()];

        let mut optimizer = ScheduleFreeAdamW::new(params, 1e-3, 0.9, 0.999, 0.05, 0.01);

        optimizer.eval();

        let grad = randn::<f32>(&[8, 8])?;
        param.write().set_grad(Some(grad));

        // Should fail in eval mode
        let result = optimizer.step();
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_schedule_free_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = ScheduleFreeAdamW::new(params, 1e-3, 0.9, 0.999, 0.05, 0.01);

        // Perform steps
        for _ in 0..5 {
            let grad = randn::<f32>(&[16, 16])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();
        }

        let state = optimizer.state_dict()?;
        assert_eq!(state.optimizer_type, "ScheduleFreeAdamW");
        assert!(state.global_state.contains_key("c"));

        Ok(())
    }
}
