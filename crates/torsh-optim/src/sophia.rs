//! Sophia (Second-order Clipped Stochastic Optimization) optimizer
//!
//! Sophia is a scalable second-order optimizer designed specifically for language model
//! pre-training. It estimates diagonal Hessian information and uses it to pre-condition
//! gradient updates, achieving faster convergence than Adam with similar memory overhead.
//!
//! ## Key Features
//!
//! - **Second-Order Information**: Uses lightweight Hessian diagonal estimation
//! - **Clipped Updates**: Clips updates based on Hessian information for stability
//! - **LM-Optimized**: Specifically designed for transformer language models
//! - **Efficient**: Similar memory overhead to Adam
//! - **2-3x Speedup**: Achieves comparable perplexity in 50% fewer steps than AdamW
//!
//! ## Algorithm
//!
//! ```text
//! // Momentum update (like Adam)
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//!
//! // Hessian diagonal estimate (updated periodically, e.g., every k steps)
//! if t % k == 0:
//!     h_t = E[∇²L] ≈ E[g ⊙ g]  // element-wise square of gradients
//!
//! // EMA of Hessian diagonal
//! h_t = β₂ * h_{t-1} + (1 - β₂) * h_t
//!
//! // Clipped update
//! θ_t = θ_{t-1} - α * clip(m̂_t / (√ĥ_t + ε), -γ, γ)
//! ```
//!
//! Where:
//! - `g_t` is the gradient at step t
//! - `m_t` is the momentum
//! - `h_t` is the Hessian diagonal estimate
//! - `β₁, β₂` are exponential decay rates (typically 0.96, 0.99)
//! - `α` is the learning rate
//! - `γ` is the clipping threshold (typically 1.0)
//! - `ε` is a small constant for numerical stability
//!
//! ## Typical Hyperparameters
//!
//! - Learning rate: 2e-4 to 1e-3
//! - β₁ (beta1): 0.96
//! - β₂ (beta2): 0.99
//! - Clipping threshold (gamma): 1.0
//! - Hessian update interval: 10 steps
//! - Weight decay: 0.1
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::prelude::{Sophia, Optimizer};
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! let param = Arc::new(RwLock::new(randn::<f32>(&[768, 768])?)); // Transformer layer
//! let params = vec![param];
//!
//! // Sophia optimized for transformer training
//! let mut optimizer = Sophia::new(
//!     params,
//!     5e-4,    // lr
//!     0.96,    // beta1
//!     0.99,    // beta2
//!     1.0,     // gamma (clipping)
//!     10,      // hessian_update_interval
//!     0.1,     // weight_decay
//! );
//!
//! // Training loop
//! for _step in 0..10000 {
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
//! Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. (2023).
//! "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training".
//! arXiv preprint arXiv:2305.14342.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Sophia optimizer configuration
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Learning rate (typically 2e-4 to 1e-3)
    pub lr: f32,
    /// Beta1 for momentum (typically 0.96)
    pub beta1: f32,
    /// Beta2 for Hessian EMA (typically 0.99)
    pub beta2: f32,
    /// Clipping threshold (typically 1.0)
    pub gamma: f32,
    /// Steps between Hessian updates (typically 10)
    pub hessian_update_interval: usize,
    /// Weight decay coefficient (typically 0.1)
    pub weight_decay: f32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: 5e-4,
            beta1: 0.96,
            beta2: 0.99,
            gamma: 1.0,
            hessian_update_interval: 10,
            weight_decay: 0.1,
            eps: 1e-12,
        }
    }
}

/// Sophia optimizer (Second-order Clipped Stochastic Optimization)
///
/// Sophia is a scalable second-order optimizer that uses lightweight Hessian
/// diagonal estimation to achieve 2-3x speedup over AdamW for language model training.
pub struct Sophia {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,
    /// Learning rate
    lr: f32,
    /// Beta1 (momentum coefficient)
    beta1: f32,
    /// Beta2 (Hessian EMA coefficient)
    beta2: f32,
    /// Clipping threshold
    gamma: f32,
    /// Steps between Hessian updates
    hessian_update_interval: usize,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Epsilon for numerical stability
    eps: f32,
    /// Momentum buffers for each parameter
    momentum: HashMap<String, Tensor>,
    /// Hessian diagonal estimates for each parameter
    hessian: HashMap<String, Tensor>,
    /// Current step count
    step_count: usize,
}

impl Sophia {
    /// Create a new Sophia optimizer
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 5e-4)
    /// * `beta1` - Momentum coefficient (default: 0.96)
    /// * `beta2` - Hessian EMA coefficient (default: 0.99)
    /// * `gamma` - Clipping threshold (default: 1.0)
    /// * `hessian_update_interval` - Steps between Hessian updates (default: 10)
    /// * `weight_decay` - Weight decay coefficient (default: 0.1)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        gamma: f32,
        hessian_update_interval: usize,
        weight_decay: f32,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr);
        Self {
            param_groups: vec![param_group],
            lr,
            beta1,
            beta2,
            gamma,
            hessian_update_interval,
            weight_decay,
            eps: 1e-12,
            momentum: HashMap::new(),
            hessian: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create a new Sophia optimizer from configuration
    pub fn from_config(params: Vec<Arc<RwLock<Tensor>>>, config: SophiaConfig) -> Self {
        let mut optimizer = Self::new(
            params,
            config.lr,
            config.beta1,
            config.beta2,
            config.gamma,
            config.hessian_update_interval,
            config.weight_decay,
        );
        optimizer.eps = config.eps;
        optimizer
    }

    /// Builder for Sophia optimizer
    pub fn builder() -> SophiaBuilder {
        SophiaBuilder::default()
    }

    /// Set learning rate
    pub fn set_lr_value(&mut self, lr: f32) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    /// Set clipping threshold
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    /// Get current step count
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }

    /// Update Hessian diagonal estimate for a parameter
    fn update_hessian(&mut self, param_key: &str, grad: &Tensor) -> OptimizerResult<()> {
        // Compute Hessian diagonal estimate: h ≈ g ⊙ g (element-wise square)
        let grad_squared = grad.mul(grad).map_err(|e| OptimizerError::TensorError(e))?;

        // Get or initialize Hessian
        let hessian_entry = self
            .hessian
            .entry(param_key.to_string())
            .or_insert_with(|| {
                grad_squared
                    .zeros_like()
                    .expect("Failed to create Hessian buffer")
            });

        // EMA update: h = beta2 * h + (1 - beta2) * g²
        let new_hessian = hessian_entry
            .mul_scalar(self.beta2)
            .map_err(|e| OptimizerError::TensorError(e))?
            .add(
                &grad_squared
                    .mul_scalar(1.0 - self.beta2)
                    .map_err(|e| OptimizerError::TensorError(e))?,
            )
            .map_err(|e| OptimizerError::TensorError(e))?;

        *hessian_entry = new_hessian;

        Ok(())
    }
}

impl Optimizer for Sophia {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let should_update_hessian = self.step_count % self.hessian_update_interval == 0;

        for group in &self.param_groups {
            let lr = group.lr;
            let beta1 = self.beta1;
            let gamma = self.gamma;
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

                // Get parameter key for state lookup
                let param_key = format!("param_{}", idx);

                // Update Hessian if it's time (inline to avoid borrowing issues)
                if should_update_hessian {
                    // Compute Hessian diagonal estimate: h ≈ g ⊙ g (element-wise square)
                    let grad_squared_hess = grad
                        .mul(&grad)
                        .map_err(|e| OptimizerError::TensorError(e))?;

                    // Get or initialize Hessian for update
                    let hessian_for_update =
                        self.hessian.entry(param_key.clone()).or_insert_with(|| {
                            grad_squared_hess
                                .zeros_like()
                                .expect("Failed to create Hessian buffer")
                        });

                    // EMA update: h = beta2 * h + (1 - beta2) * g²
                    let new_hessian_update = hessian_for_update
                        .mul_scalar(self.beta2)
                        .map_err(|e| OptimizerError::TensorError(e))?
                        .add(
                            &grad_squared_hess
                                .mul_scalar(1.0 - self.beta2)
                                .map_err(|e| OptimizerError::TensorError(e))?,
                        )
                        .map_err(|e| OptimizerError::TensorError(e))?;

                    *hessian_for_update = new_hessian_update;
                }

                // Get or initialize momentum
                let momentum_tensor = self.momentum.entry(param_key.clone()).or_insert_with(|| {
                    grad.zeros_like().expect("Failed to create momentum buffer")
                });

                // Update momentum: m = beta1 * m + (1 - beta1) * g
                let new_momentum = momentum_tensor
                    .mul_scalar(beta1)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &grad
                            .mul_scalar(1.0 - beta1)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                *momentum_tensor = new_momentum.clone();

                // Get Hessian (initialize if not present)
                let hessian_tensor = self.hessian.entry(param_key.clone()).or_insert_with(|| {
                    // Initialize with ones (safe default)
                    grad.ones_like().expect("Failed to create Hessian buffer")
                });

                // Compute bias-corrected estimates
                let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
                let m_hat = new_momentum
                    .mul_scalar(1.0 / bias_correction1)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Compute preconditioned update: m / (√h + ε)
                let h_sqrt = hessian_tensor
                    .sqrt()
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add_scalar(eps)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                let preconditioned_update = m_hat
                    .div(&h_sqrt)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Clip the update: clip(update, -gamma, gamma)
                let clipped_update = preconditioned_update
                    .clamp(-gamma, gamma)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Apply weight decay if specified (AdamW-style)
                let param_data = param_guard.clone();
                let final_update = if weight_decay > 0.0 {
                    let decay_term = param_data
                        .mul_scalar(weight_decay * lr)
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    clipped_update
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                        .add(&decay_term)
                        .map_err(|e| OptimizerError::TensorError(e))?
                } else {
                    clipped_update
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorError(e))?
                };

                // Apply update
                let new_param = param_data
                    .sub(&final_update)
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
        for (key, _) in &self.momentum {
            let mut param_state = HashMap::new();
            if let Some(momentum) = self.momentum.get(key) {
                param_state.insert("momentum".to_string(), momentum.clone());
            }
            if let Some(hessian) = self.hessian.get(key) {
                param_state.insert("hessian".to_string(), hessian.clone());
            }
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
        global_state.insert("gamma".to_string(), self.gamma);
        global_state.insert("weight_decay".to_string(), self.weight_decay);
        global_state.insert("eps".to_string(), self.eps);

        Ok(OptimizerState {
            optimizer_type: "Sophia".to_string(),
            version: "1.0".to_string(),
            param_groups: param_group_states,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "Sophia" {
            return Err(OptimizerError::InvalidInput(format!(
                "Expected Sophia state dict, got {}",
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
        if let Some(&gamma) = state.global_state.get("gamma") {
            self.gamma = gamma;
        }
        if let Some(&weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&eps) = state.global_state.get("eps") {
            self.eps = eps;
        }

        // Restore optimizer state
        self.momentum.clear();
        self.hessian.clear();
        for (key, param_state) in state.state {
            if let Some(momentum) = param_state.get("momentum") {
                self.momentum.insert(key.clone(), momentum.clone());
            }
            if let Some(hessian) = param_state.get("hessian") {
                self.hessian.insert(key.clone(), hessian.clone());
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

/// Builder for Sophia optimizer
#[derive(Debug, Clone)]
pub struct SophiaBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    gamma: f32,
    hessian_update_interval: usize,
    weight_decay: f32,
    eps: f32,
}

impl Default for SophiaBuilder {
    fn default() -> Self {
        let config = SophiaConfig::default();
        Self {
            params: Vec::new(),
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            gamma: config.gamma,
            hessian_update_interval: config.hessian_update_interval,
            weight_decay: config.weight_decay,
            eps: config.eps,
        }
    }
}

impl SophiaBuilder {
    /// Create a new Sophia builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameters to optimize
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    /// Set learning rate
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (momentum coefficient)
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (Hessian EMA coefficient)
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set gamma (clipping threshold)
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set Hessian update interval
    pub fn hessian_update_interval(mut self, interval: usize) -> Self {
        self.hessian_update_interval = interval;
        self
    }

    /// Set weight decay coefficient
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon for numerical stability
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Build the Sophia optimizer
    pub fn build(self) -> Sophia {
        let mut optimizer = Sophia::new(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.gamma,
            self.hessian_update_interval,
            self.weight_decay,
        );
        optimizer.eps = self.eps;
        optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_sophia_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[128, 128])?));
        let params = vec![param];

        let optimizer = Sophia::new(params, 5e-4, 0.96, 0.99, 1.0, 10, 0.1);
        assert_eq!(optimizer.lr, 5e-4);
        assert_eq!(optimizer.beta1, 0.96);
        assert_eq!(optimizer.beta2, 0.99);
        assert_eq!(optimizer.gamma, 1.0);
        assert_eq!(optimizer.hessian_update_interval, 10);

        Ok(())
    }

    #[test]
    fn test_sophia_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[64, 64])?));
        let params = vec![param];

        let optimizer = Sophia::builder()
            .params(params)
            .lr(1e-3)
            .beta1(0.95)
            .beta2(0.98)
            .gamma(2.0)
            .hessian_update_interval(20)
            .weight_decay(0.05)
            .build();

        assert_eq!(optimizer.lr, 1e-3);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.gamma, 2.0);

        Ok(())
    }

    #[test]
    fn test_sophia_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[32, 32])?));
        let params = vec![param.clone()];

        let mut optimizer = Sophia::new(params, 5e-4, 0.96, 0.99, 1.0, 10, 0.0);

        // Set gradient
        let grad = randn::<f32>(&[32, 32])?;
        param.write().set_grad(Some(grad));

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
    fn test_sophia_hessian_update() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = Sophia::new(params, 5e-4, 0.96, 0.99, 1.0, 5, 0.0);

        // Perform several steps
        for step in 0..12 {
            let grad = randn::<f32>(&[16, 16])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;

            // Hessian should be updated at steps 5 and 10
            if step == 5 || step == 10 {
                assert!(
                    !optimizer.hessian.is_empty(),
                    "Hessian should be updated at step {}",
                    step
                );
            }

            optimizer.zero_grad();
        }

        Ok(())
    }

    #[test]
    fn test_sophia_clipping() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[8, 8])?));
        let params = vec![param.clone()];

        // Use small gamma for aggressive clipping
        let mut optimizer = Sophia::new(params, 1e-3, 0.96, 0.99, 0.1, 10, 0.0);

        // Use large gradient to test clipping
        let large_grad = randn::<f32>(&[8, 8])?.mul_scalar(100.0)?;
        param.write().set_grad(Some(large_grad));

        let param_before = param.read().clone();
        optimizer.step()?;
        let param_after = param.read().clone();

        // With clipping, the update should be bounded
        let diff = param_before.sub(&param_after)?;
        let diff_abs = diff.abs()?;
        let diff_max = diff_abs.max(None, false)?.to_vec()?[0];

        // The maximum change should not exceed lr * gamma
        assert!(
            diff_max <= optimizer.lr * optimizer.gamma * 1.5,
            "Update should be clipped"
        );

        Ok(())
    }

    #[test]
    fn test_sophia_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = Sophia::new(params, 5e-4, 0.96, 0.99, 1.0, 10, 0.1);

        // Perform a few steps to populate state
        for _ in 0..15 {
            let grad = randn::<f32>(&[16, 16])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();
        }

        // Get state dict
        let state = optimizer.state_dict()?;

        assert_eq!(state.optimizer_type, "Sophia");
        assert_eq!(state.param_groups.len(), 1);
        assert!(state.global_state.contains_key("beta1"));
        assert!(state.global_state.contains_key("gamma"));
        assert!(!state.state.is_empty());

        Ok(())
    }

    #[test]
    fn test_sophia_load_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer1 = Sophia::new(params.clone(), 5e-4, 0.96, 0.99, 1.0, 10, 0.1);

        // Perform a few steps
        for _ in 0..15 {
            let grad = randn::<f32>(&[16, 16])?;
            param.write().set_grad(Some(grad));
            optimizer1.step()?;
            optimizer1.zero_grad();
        }

        // Save state
        let state = optimizer1.state_dict()?;

        // Create new optimizer and load state
        let mut optimizer2 = Sophia::new(params, 1e-3, 0.9, 0.98, 2.0, 20, 0.05);
        optimizer2.load_state_dict(state)?;

        // Hyperparameters should match original optimizer
        assert_relative_eq!(optimizer2.beta1, 0.96, epsilon = 1e-6);
        assert_relative_eq!(optimizer2.gamma, 1.0, epsilon = 1e-6);

        Ok(())
    }
}
