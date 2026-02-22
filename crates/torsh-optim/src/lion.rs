//! Lion (Evolved Sign Momentum) optimizer
//!
//! Lion is a simple and memory-efficient optimizer discovered through program search by Google Research.
//! It achieves better performance than Adam/AdamW on many tasks while requiring less memory.
//!
//! ## Key Features
//!
//! - **Memory Efficient**: Only stores momentum (no second moment like Adam)
//! - **Simple Updates**: Uses sign of interpolated gradients
//! - **Strong Performance**: Often matches or exceeds Adam/AdamW
//! - **Robust**: Works well across vision and language tasks
//!
//! ## Algorithm
//!
//! ```text
//! c_t = β₁ * m_{t-1} + (1 - β₁) * g_t           // Interpolate gradient and momentum
//! m_t = β₂ * m_{t-1} + (1 - β₂) * g_t           // Update momentum
//! θ_t = θ_{t-1} - α * (sign(c_t) + λ * θ_{t-1}) // Update parameters with weight decay
//! ```
//!
//! Where:
//! - `g_t` is the gradient at step t
//! - `m_t` is the momentum
//! - `β₁, β₂` are interpolation coefficients (typically 0.9, 0.99)
//! - `α` is the learning rate (typically 10x smaller than Adam)
//! - `λ` is the weight decay coefficient
//!
//! ## Typical Hyperparameters
//!
//! - Learning rate: 1e-4 (10x smaller than Adam's typical 1e-3)
//! - β₁ (beta1): 0.9
//! - β₂ (beta2): 0.99
//! - Weight decay: 1e-2 to 1e-1 (typically larger than Adam)
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::prelude::{Lion, Optimizer};
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! let param = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param];
//!
//! // Lion typically uses 10x smaller learning rate than Adam
//! let mut optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.01);
//!
//! // Training loop
//! for _step in 0..1000 {
//!     // ... compute gradients ...
//!     optimizer.step()?;
//!     optimizer.zero_grad();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Reference
//!
//! Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., ... & Le, Q. V. (2023).
//! "Symbolic Discovery of Optimization Algorithms". arXiv preprint arXiv:2302.06675.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Lion optimizer configuration
#[derive(Debug, Clone)]
pub struct LionConfig {
    /// Learning rate (typically 1e-4, 10x smaller than Adam)
    pub lr: f32,
    /// Beta1 for momentum interpolation (typically 0.9)
    pub beta1: f32,
    /// Beta2 for momentum update (typically 0.99)
    pub beta2: f32,
    /// Weight decay coefficient (typically 0.01-0.1, larger than Adam)
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
        }
    }
}

/// Lion optimizer (Evolved Sign Momentum)
///
/// Lion is a simple and memory-efficient optimizer that achieves strong performance
/// across computer vision and natural language processing tasks. It was discovered
/// through program search and requires less memory than Adam while often achieving
/// better performance.
pub struct Lion {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,
    /// Learning rate
    lr: f32,
    /// Beta1 (momentum interpolation coefficient)
    beta1: f32,
    /// Beta2 (momentum update coefficient)
    beta2: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Momentum buffers for each parameter
    momentum: HashMap<String, Tensor>,
    /// Current step count
    step_count: usize,
}

impl Lion {
    /// Create a new Lion optimizer
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 1e-4, typically 10x smaller than Adam)
    /// * `beta1` - Momentum interpolation coefficient (default: 0.9)
    /// * `beta2` - Momentum update coefficient (default: 0.99)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use torsh_tensor::creation::randn;
    /// # use torsh_core::error::Result;
    /// # fn main() -> Result<()> {
    /// use torsh_optim::prelude::Lion;
    /// use parking_lot::RwLock;
    /// use std::sync::Arc;
    ///
    /// let param = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
    /// let params = vec![param];
    ///
    /// // Create Lion optimizer with default hyperparameters
    /// let optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.01);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr);
        Self {
            param_groups: vec![param_group],
            lr,
            beta1,
            beta2,
            weight_decay,
            momentum: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create a new Lion optimizer from configuration
    pub fn from_config(params: Vec<Arc<RwLock<Tensor>>>, config: LionConfig) -> Self {
        Self::new(
            params,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
        )
    }

    /// Builder for Lion optimizer
    pub fn builder() -> LionBuilder {
        LionBuilder::default()
    }

    /// Set learning rate
    pub fn set_lr_value(&mut self, lr: f32) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    /// Set beta1
    pub fn set_beta1(&mut self, beta1: f32) {
        self.beta1 = beta1;
    }

    /// Set beta2
    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    /// Set weight decay
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }

    /// Get current step count
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }
}

impl Optimizer for Lion {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        for group in &self.param_groups {
            let lr = group.lr;
            let beta1 = self.beta1;
            let beta2 = self.beta2;
            let weight_decay = self.weight_decay;

            for (idx, param) in group.params.iter().enumerate() {
                let mut param_guard = param.write();

                // Skip parameters without gradients
                if !param_guard.has_grad() {
                    continue;
                }

                let grad = param_guard
                    .grad()
                    .ok_or_else(|| OptimizerError::InvalidInput("No gradient found".to_string()))?;

                // Get parameter key for state lookup
                let param_key = format!("param_{}", idx);

                // Get or initialize momentum
                let momentum_tensor = self.momentum.entry(param_key.clone()).or_insert_with(|| {
                    grad.zeros_like().expect("Failed to create momentum buffer")
                });

                // Compute interpolated gradient: c = beta1 * m + (1 - beta1) * g
                let interpolated = momentum_tensor
                    .mul_scalar(beta1)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &grad
                            .mul_scalar(1.0 - beta1)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Compute sign of interpolated gradient
                let update_direction = interpolated
                    .sign()
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Update momentum: m = beta2 * m + (1 - beta2) * g
                let new_momentum = momentum_tensor
                    .mul_scalar(beta2)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &grad
                            .mul_scalar(1.0 - beta2)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                *momentum_tensor = new_momentum;

                // Apply weight decay if specified
                let param_data = param_guard.clone();
                let update = if weight_decay > 0.0 {
                    // Update with weight decay: θ = θ - lr * (sign(c) + λ * θ)
                    let decay_term = param_data
                        .mul_scalar(weight_decay)
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    update_direction
                        .add(&decay_term)
                        .map_err(|e| OptimizerError::TensorError(e))?
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                } else {
                    // Update without weight decay: θ = θ - lr * sign(c)
                    update_direction
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                };

                // Apply update
                let new_param = param_data
                    .sub(&update)
                    .map_err(|e| OptimizerError::TensorError(e))?;
                *param_guard = new_param;
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
        self.set_lr_value(lr);
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
        for (key, momentum) in &self.momentum {
            let mut param_state = HashMap::new();
            param_state.insert("momentum".to_string(), momentum.clone());
            param_state.insert(
                "step".to_string(),
                Tensor::scalar(self.step_count as f32)
                    .map_err(|e| OptimizerError::TensorError(e))?,
            );
            state.insert(key.clone(), param_state);
        }

        let mut global_state = HashMap::new();
        global_state.insert("beta1".to_string(), self.beta1);
        global_state.insert("beta2".to_string(), self.beta2);
        global_state.insert("weight_decay".to_string(), self.weight_decay);

        Ok(OptimizerState {
            optimizer_type: "Lion".to_string(),
            version: "1.0".to_string(),
            param_groups: param_group_states,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "Lion" {
            return Err(OptimizerError::InvalidInput(format!(
                "Expected Lion state dict, got {}",
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
        if let Some(&weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }

        // Restore momentum state
        self.momentum.clear();
        for (key, param_state) in state.state {
            if let Some(momentum) = param_state.get("momentum") {
                self.momentum.insert(key.clone(), momentum.clone());
            }
            if let Some(step_tensor) = param_state.get("step") {
                self.step_count = step_tensor
                    .to_vec()
                    .map_err(|e| OptimizerError::TensorError(e))?[0]
                    as usize;
            }
        }

        Ok(())
    }
}

/// Builder for Lion optimizer
#[derive(Debug, Clone)]
pub struct LionBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

impl Default for LionBuilder {
    fn default() -> Self {
        let config = LionConfig::default();
        Self {
            params: Vec::new(),
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            weight_decay: config.weight_decay,
        }
    }
}

impl LionBuilder {
    /// Create a new Lion builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameters to optimize
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    /// Set learning rate (typically 1e-4, 10x smaller than Adam)
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (momentum interpolation coefficient)
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (momentum update coefficient)
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set weight decay coefficient (typically 0.01-0.1, larger than Adam)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the Lion optimizer
    pub fn build(self) -> Lion {
        Lion::new(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.weight_decay,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_lion_creation() {
        let param = Arc::new(RwLock::new(
            randn::<f32>(&[2, 3]).expect("Failed to create tensor"),
        ));
        let params = vec![param];

        let optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.01);
        assert_eq!(optimizer.lr, 1e-4);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.99);
        assert_eq!(optimizer.weight_decay, 0.01);
    }

    #[test]
    fn test_lion_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param];

        let optimizer = Lion::builder()
            .params(params)
            .lr(2e-4)
            .beta1(0.95)
            .beta2(0.999)
            .weight_decay(0.05)
            .build();

        assert_eq!(optimizer.lr, 2e-4);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.weight_decay, 0.05);

        Ok(())
    }

    #[test]
    fn test_lion_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param.clone()];

        let mut optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.0);

        // Set gradient
        let grad = randn::<f32>(&[2, 3])?;
        param.write().set_grad(Some(grad.clone()));

        let param_before = param.read().clone();

        // Perform optimization step
        optimizer.step()?;

        let param_after = param.read().clone();

        // Check that parameters changed
        let diff = param_before.sub(&param_after)?;
        let diff_norm = diff.norm()?.to_vec()?[0];
        assert!(diff_norm > 0.0, "Parameters should have changed");

        Ok(())
    }

    #[test]
    fn test_lion_weight_decay() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param.clone()];

        // Optimizer with weight decay
        let mut optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.1);

        // Set gradient
        let grad = randn::<f32>(&[2, 3])?;
        param.write().set_grad(Some(grad));

        let param_before = param.read().clone();

        // Perform optimization step
        optimizer.step()?;

        let param_after = param.read().clone();

        // Parameters should have decayed (moved toward zero)
        let param_norm_before = param_before.norm()?.to_vec()?[0];
        let param_norm_after = param_after.norm()?.to_vec()?[0];

        // With weight decay, the parameter norm should generally decrease
        // (though gradient updates could increase it)
        assert!(
            (param_norm_before - param_norm_after).abs() > 0.0,
            "Weight decay should affect parameters"
        );

        Ok(())
    }

    #[test]
    fn test_lion_zero_grad() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param.clone()];

        let mut optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.01);

        // Set gradient
        let grad = randn::<f32>(&[2, 3])?;
        param.write().set_grad(Some(grad));

        assert!(param.read().has_grad());

        // Zero gradients
        optimizer.zero_grad();

        assert!(!param.read().has_grad());

        Ok(())
    }

    #[test]
    fn test_lion_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param.clone()];

        let mut optimizer = Lion::new(params, 1e-4, 0.9, 0.99, 0.01);

        // Perform a few steps to populate state
        for _ in 0..3 {
            let grad = randn::<f32>(&[2, 3])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();
        }

        // Get state dict
        let state = optimizer.state_dict()?;

        assert_eq!(state.optimizer_type, "Lion");
        assert_eq!(state.param_groups.len(), 1);
        assert!(state.global_state.contains_key("beta1"));
        assert!(state.global_state.contains_key("beta2"));

        Ok(())
    }

    #[test]
    fn test_lion_load_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 3])?));
        let params = vec![param.clone()];

        let mut optimizer1 = Lion::new(params.clone(), 1e-4, 0.9, 0.99, 0.01);

        // Perform a few steps
        for _ in 0..3 {
            let grad = randn::<f32>(&[2, 3])?;
            param.write().set_grad(Some(grad));
            optimizer1.step()?;
            optimizer1.zero_grad();
        }

        // Save state
        let state = optimizer1.state_dict()?;

        // Create new optimizer and load state
        let mut optimizer2 = Lion::new(params, 2e-4, 0.8, 0.98, 0.02);
        optimizer2.load_state_dict(state)?;

        // Hyperparameters should match original optimizer
        assert_relative_eq!(optimizer2.beta1, 0.9, epsilon = 1e-6);
        assert_relative_eq!(optimizer2.beta2, 0.99, epsilon = 1e-6);

        Ok(())
    }
}
