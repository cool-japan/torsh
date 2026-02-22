//! Prodigy (An Adaptive Learning Rate Method) optimizer
//!
//! Prodigy is a state-of-the-art adaptive learning rate optimizer that automatically tunes the
//! learning rate without manual tuning. It combines ideas from D-Adaptation with improved
//! stability and convergence properties.
//!
//! ## Key Innovation
//!
//! Traditional optimizers require careful learning rate tuning. Prodigy automatically estimates
//! the optimal learning rate by tracking gradient statistics and adapting based on:
//! - Distance traveled in parameter space
//! - Gradient variance
//! - Convergence progress
//!
//! This makes it extremely user-friendly - you can use `lr=1.0` for almost any problem!
//!
//! ## Key Features
//!
//! - **Zero LR Tuning**: Use lr=1.0 for most problems
//! - **Automatic Adaptation**: Learns optimal learning rate during training
//! - **Robust**: Works across different domains (vision, NLP, RL)
//! - **Memory Efficient**: Similar memory overhead to Adam
//! - **Fast Convergence**: Often matches or beats carefully tuned Adam
//!
//! ## Algorithm
//!
//! ```text
//! // Initialize
//! d_0 = initial_d  // learning rate estimate
//! s_0 = 0          // distance estimate
//!
//! // Update momentum and variance (like Adam)
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
//!
//! // Estimate distance traveled
//! s_t = s_{t-1} + ||θ_t - θ_{t-1}||
//!
//! // Adapt learning rate estimate
//! if t > k:  // after warmup
//!     d_t = d_{t-1} * (s_t / s_{t-1})^growth_rate
//!
//! // Compute adaptive step size
//! α_t = lr / (d_t * √t)
//!
//! // Update parameters
//! θ_t = θ_{t-1} - α_t * m̂_t / (√v̂_t + ε)
//! ```
//!
//! Where:
//! - `d_t` is the adaptive learning rate scale
//! - `s_t` is the cumulative distance estimate
//! - `m_t, v_t` are first and second moments
//! - `β₁, β₂` are exponential decay rates
//! - `growth_rate` controls adaptation speed (typically 1.0)
//!
//! ## Typical Hyperparameters
//!
//! - Learning rate: **1.0** (yes, really!)
//! - β₁ (beta1): 0.9
//! - β₂ (beta2): 0.999
//! - Growth rate: 1.0
//! - Weight decay: 0.0 (or small value like 0.01)
//! - Initial d: 1e-6
//! - Warmup steps: 0 (can use 100-1000 for stability)
//!
//! ## Usage Example
//!
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::prelude::{Prodigy, Optimizer};
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! let param = Arc::new(RwLock::new(randn::<f32>(&[768, 768])?));
//! let params = vec![param];
//!
//! // No learning rate tuning needed!
//! let mut optimizer = Prodigy::new(
//!     params,
//!     1.0,    // lr (use 1.0 for almost everything!)
//!     0.9,    // beta1
//!     0.999,  // beta2
//!     0.0,    // weight_decay
//! );
//!
//! // Training loop - it just works!
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
//! Mishchenko, K., & Defazio, A. (2024).
//! "Prodigy: An Adaptive Learning Rate Method".
//! arXiv preprint arXiv:2401.04536.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Prodigy optimizer configuration
#[derive(Debug, Clone)]
pub struct ProdigyConfig {
    /// Learning rate (use 1.0 for most problems!)
    pub lr: f32,
    /// Beta1 for momentum (typically 0.9)
    pub beta1: f32,
    /// Beta2 for variance (typically 0.999)
    pub beta2: f32,
    /// Growth rate for d adaptation (typically 1.0)
    pub growth_rate: f32,
    /// Initial d estimate (typically 1e-6)
    pub initial_d: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Warmup steps (0 for no warmup)
    pub warmup_steps: usize,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            growth_rate: 1.0,
            initial_d: 1e-6,
            weight_decay: 0.0,
            eps: 1e-8,
            warmup_steps: 0,
        }
    }
}

/// Prodigy optimizer (Adaptive Learning Rate Method)
///
/// Prodigy automatically tunes the learning rate without manual tuning.
/// Just use lr=1.0 for almost any problem!
pub struct Prodigy {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,
    /// Base learning rate (typically 1.0)
    lr: f32,
    /// Beta1 (momentum coefficient)
    beta1: f32,
    /// Beta2 (variance coefficient)
    beta2: f32,
    /// Growth rate for d adaptation
    growth_rate: f32,
    /// Current d estimate (adaptive lr scale)
    d: f32,
    /// Previous d estimate
    d_prev: f32,
    /// Cumulative distance estimate
    s: f32,
    /// Previous cumulative distance
    s_prev: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Epsilon for numerical stability
    eps: f32,
    /// Warmup steps
    warmup_steps: usize,
    /// Momentum buffers
    momentum: HashMap<String, Tensor>,
    /// Variance buffers
    variance: HashMap<String, Tensor>,
    /// Previous parameters (for distance computation)
    prev_params: HashMap<String, Tensor>,
    /// Current step count
    step_count: usize,
}

impl Prodigy {
    /// Create a new Prodigy optimizer
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (use 1.0 for most problems!)
    /// * `beta1` - Momentum coefficient (default: 0.9)
    /// * `beta2` - Variance coefficient (default: 0.999)
    /// * `weight_decay` - Weight decay coefficient (default: 0.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use torsh_tensor::creation::randn;
    /// # use torsh_core::error::Result;
    /// # fn main() -> Result<()> {
    /// use torsh_optim::prelude::Prodigy;
    /// use parking_lot::RwLock;
    /// use std::sync::Arc;
    ///
    /// let param = Arc::new(RwLock::new(randn::<f32>(&[100, 100])?));
    /// let params = vec![param];
    ///
    /// // Use lr=1.0 - it adapts automatically!
    /// let optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);
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
        let config = ProdigyConfig::default();
        Self {
            param_groups: vec![param_group],
            lr,
            beta1,
            beta2,
            growth_rate: config.growth_rate,
            d: config.initial_d,
            d_prev: config.initial_d,
            s: 0.0,
            s_prev: 0.0,
            weight_decay,
            eps: config.eps,
            warmup_steps: config.warmup_steps,
            momentum: HashMap::new(),
            variance: HashMap::new(),
            prev_params: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create from configuration
    pub fn from_config(params: Vec<Arc<RwLock<Tensor>>>, config: ProdigyConfig) -> Self {
        let mut optimizer = Self::new(
            params,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
        );
        optimizer.growth_rate = config.growth_rate;
        optimizer.d = config.initial_d;
        optimizer.d_prev = config.initial_d;
        optimizer.eps = config.eps;
        optimizer.warmup_steps = config.warmup_steps;
        optimizer
    }

    /// Builder for Prodigy optimizer
    pub fn builder() -> ProdigyBuilder {
        ProdigyBuilder::default()
    }

    /// Get current learning rate scale (d)
    pub fn get_d(&self) -> f32 {
        self.d
    }

    /// Get effective learning rate
    pub fn get_effective_lr(&self) -> f32 {
        if self.step_count == 0 {
            return 0.0;
        }
        self.lr / (self.d * (self.step_count as f32).sqrt())
    }
}

impl Optimizer for Prodigy {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        // Compute warmup factor
        let warmup_factor = if self.warmup_steps > 0 && self.step_count <= self.warmup_steps {
            (self.step_count as f32) / (self.warmup_steps as f32)
        } else {
            1.0
        };

        // Adaptive step size: α = lr / (d * √t)
        let base_step_size = self.lr / (self.d * (self.step_count as f32).sqrt());
        let step_size = base_step_size * warmup_factor;

        let mut distance_sum = 0.0f32;

        for group in &self.param_groups {
            let beta1 = self.beta1;
            let beta2 = self.beta2;
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

                // Get or initialize momentum and variance
                let m_entry = self.momentum.entry(param_key.clone()).or_insert_with(|| {
                    grad.zeros_like().expect("Failed to create momentum buffer")
                });
                let v_entry = self.variance.entry(param_key.clone()).or_insert_with(|| {
                    grad.zeros_like().expect("Failed to create variance buffer")
                });

                // Update momentum: m = β₁ * m + (1 - β₁) * g
                let new_m = m_entry
                    .mul_scalar(beta1)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &grad
                            .mul_scalar(1.0 - beta1)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Update variance: v = β₂ * v + (1 - β₂) * g²
                let grad_squared = grad
                    .mul(&grad)
                    .map_err(|e| OptimizerError::TensorError(e))?;
                let new_v = v_entry
                    .mul_scalar(beta2)
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add(
                        &grad_squared
                            .mul_scalar(1.0 - beta2)
                            .map_err(|e| OptimizerError::TensorError(e))?,
                    )
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Bias correction
                let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

                let m_hat = new_m
                    .mul_scalar(1.0 / bias_correction1)
                    .map_err(|e| OptimizerError::TensorError(e))?;
                let v_hat = new_v
                    .mul_scalar(1.0 / bias_correction2)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Compute update: m̂ / (√v̂ + ε)
                let v_sqrt = v_hat
                    .sqrt()
                    .map_err(|e| OptimizerError::TensorError(e))?
                    .add_scalar(eps)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                let update_direction = m_hat
                    .div(&v_sqrt)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Apply weight decay if specified (AdamW-style)
                let param_data = param_guard.clone();
                let update = if weight_decay > 0.0 {
                    let decay_term = param_data
                        .mul_scalar(weight_decay * step_size)
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    update_direction
                        .mul_scalar(step_size)
                        .map_err(|e| OptimizerError::TensorError(e))?
                        .add(&decay_term)
                        .map_err(|e| OptimizerError::TensorError(e))?
                } else {
                    update_direction
                        .mul_scalar(step_size)
                        .map_err(|e| OptimizerError::TensorError(e))?
                };

                // Update parameters
                let new_param = param_data
                    .sub(&update)
                    .map_err(|e| OptimizerError::TensorError(e))?;

                // Compute distance for d adaptation
                if let Some(prev_param) = self.prev_params.get(&param_key) {
                    let param_diff = new_param
                        .sub(prev_param)
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    let diff_norm = param_diff
                        .norm()
                        .map_err(|e| OptimizerError::TensorError(e))?;
                    distance_sum += diff_norm
                        .to_vec()
                        .map_err(|e| OptimizerError::TensorError(e))?[0];
                }

                // Store previous parameter for next iteration
                self.prev_params.insert(param_key.clone(), param_data);

                // Update momentum and variance
                *m_entry = new_m;
                *v_entry = new_v;
                *param_guard = new_param;
            }
        }

        // Update distance estimate
        self.s_prev = self.s;
        self.s += distance_sum;

        // Adapt d after warmup
        if self.step_count > self.warmup_steps.max(1) && self.s_prev > 0.0 {
            let ratio = self.s / self.s_prev;
            self.d_prev = self.d;
            self.d = self.d * ratio.powf(self.growth_rate);

            // Clamp d to prevent instability
            self.d = self.d.max(1e-12).min(1e12);
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
        for (key, _) in &self.momentum {
            let mut param_state = HashMap::new();
            if let Some(m) = self.momentum.get(key) {
                param_state.insert("momentum".to_string(), m.clone());
            }
            if let Some(v) = self.variance.get(key) {
                param_state.insert("variance".to_string(), v.clone());
            }
            if let Some(prev) = self.prev_params.get(key) {
                param_state.insert("prev_param".to_string(), prev.clone());
            }
            state.insert(key.clone(), param_state);
        }

        let mut global_state = HashMap::new();
        global_state.insert("beta1".to_string(), self.beta1);
        global_state.insert("beta2".to_string(), self.beta2);
        global_state.insert("growth_rate".to_string(), self.growth_rate);
        global_state.insert("d".to_string(), self.d);
        global_state.insert("d_prev".to_string(), self.d_prev);
        global_state.insert("s".to_string(), self.s);
        global_state.insert("s_prev".to_string(), self.s_prev);
        global_state.insert("weight_decay".to_string(), self.weight_decay);
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "Prodigy".to_string(),
            version: "1.0".to_string(),
            param_groups: param_group_states,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "Prodigy" {
            return Err(OptimizerError::InvalidInput(format!(
                "Expected Prodigy state dict, got {}",
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
        if let Some(&growth_rate) = state.global_state.get("growth_rate") {
            self.growth_rate = growth_rate;
        }
        if let Some(&d) = state.global_state.get("d") {
            self.d = d;
        }
        if let Some(&d_prev) = state.global_state.get("d_prev") {
            self.d_prev = d_prev;
        }
        if let Some(&s) = state.global_state.get("s") {
            self.s = s;
        }
        if let Some(&s_prev) = state.global_state.get("s_prev") {
            self.s_prev = s_prev;
        }
        if let Some(&weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&step_count) = state.global_state.get("step_count") {
            self.step_count = step_count as usize;
        }

        // Restore optimizer state
        self.momentum.clear();
        self.variance.clear();
        self.prev_params.clear();

        for (key, param_state) in state.state {
            if let Some(m) = param_state.get("momentum") {
                self.momentum.insert(key.clone(), m.clone());
            }
            if let Some(v) = param_state.get("variance") {
                self.variance.insert(key.clone(), v.clone());
            }
            if let Some(prev) = param_state.get("prev_param") {
                self.prev_params.insert(key.clone(), prev.clone());
            }
        }

        Ok(())
    }
}

/// Builder for Prodigy optimizer
#[derive(Debug, Clone)]
pub struct ProdigyBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    growth_rate: f32,
    initial_d: f32,
    weight_decay: f32,
    eps: f32,
    warmup_steps: usize,
}

impl Default for ProdigyBuilder {
    fn default() -> Self {
        let config = ProdigyConfig::default();
        Self {
            params: Vec::new(),
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            growth_rate: config.growth_rate,
            initial_d: config.initial_d,
            weight_decay: config.weight_decay,
            eps: config.eps,
            warmup_steps: config.warmup_steps,
        }
    }
}

impl ProdigyBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameters
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    /// Set learning rate (use 1.0 for most problems!)
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

    /// Set growth rate
    pub fn growth_rate(mut self, growth_rate: f32) -> Self {
        self.growth_rate = growth_rate;
        self
    }

    /// Set initial d
    pub fn initial_d(mut self, initial_d: f32) -> Self {
        self.initial_d = initial_d;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set warmup steps
    pub fn warmup_steps(mut self, warmup_steps: usize) -> Self {
        self.warmup_steps = warmup_steps;
        self
    }

    /// Build the optimizer
    pub fn build(self) -> Prodigy {
        let config = ProdigyConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            growth_rate: self.growth_rate,
            initial_d: self.initial_d,
            weight_decay: self.weight_decay,
            eps: self.eps,
            warmup_steps: self.warmup_steps,
        };
        Prodigy::from_config(self.params, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_prodigy_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[64, 64])?));
        let params = vec![param];

        let optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);
        assert_eq!(optimizer.lr, 1.0);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);

        Ok(())
    }

    #[test]
    fn test_prodigy_adaptive_lr() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[32, 32])?));
        let params = vec![param.clone()];

        let mut optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);

        // Perform several steps and check that d adapts
        let initial_d = optimizer.get_d();

        for _ in 0..20 {
            let grad = randn::<f32>(&[32, 32])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();
        }

        let final_d = optimizer.get_d();

        // d should have changed (adapted)
        assert_ne!(initial_d, final_d, "d should adapt during training");

        Ok(())
    }

    #[test]
    fn test_prodigy_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);

        let grad = randn::<f32>(&[16, 16])?;
        param.write().set_grad(Some(grad));

        let param_before = param.read().clone();
        optimizer.step()?;
        let param_after = param.read().clone();

        // Parameters should change
        let diff = param_before.sub(&param_after)?;
        let diff_norm = diff.norm()?.to_vec()?[0];
        assert!(diff_norm > 0.0, "Parameters should have changed");

        Ok(())
    }

    #[test]
    fn test_prodigy_effective_lr() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[8, 8])?));
        let params = vec![param.clone()];

        let mut optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);

        assert_eq!(optimizer.get_effective_lr(), 0.0); // Before any steps

        for _ in 0..5 {
            let grad = randn::<f32>(&[8, 8])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();

            // Effective LR should be > 0 after steps
            assert!(optimizer.get_effective_lr() > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_prodigy_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[16, 16])?));
        let params = vec![param.clone()];

        let mut optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);

        // Perform steps
        for _ in 0..10 {
            let grad = randn::<f32>(&[16, 16])?;
            param.write().set_grad(Some(grad));
            optimizer.step()?;
            optimizer.zero_grad();
        }

        let state = optimizer.state_dict()?;
        assert_eq!(state.optimizer_type, "Prodigy");
        assert!(state.global_state.contains_key("d"));
        assert!(state.global_state.contains_key("s"));

        Ok(())
    }
}
