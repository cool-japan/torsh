//! YellowFin optimizer
//!
//! YellowFin is an automatic momentum tuning optimizer that adjusts both
//! momentum and learning rate based on local quadratic approximation.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// YellowFin optimizer configuration
#[derive(Clone)]
pub struct YellowFinConfig {
    /// Learning rate (will be automatically tuned)
    pub lr: f32,
    /// Initial momentum value
    pub beta: f32,
    /// Momentum bounds
    pub beta_min: f32,
    pub beta_max: f32,
    /// Learning rate bounds
    pub lr_min: f32,
    pub lr_max: f32,
    /// Curv_win_width for curvature estimation
    pub curv_win_width: usize,
    /// Zero bias correction for moving averages
    pub zero_debias: bool,
    /// Weight decay factor
    pub weight_decay: f32,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Window size for gradient variance estimation
    pub variance_window: usize,
    /// Sparsity averaging parameter
    pub sparsity_average: f32,
}

impl Default for YellowFinConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            beta: 0.999,
            beta_min: 0.0,
            beta_max: 0.999,
            lr_min: 1e-8,
            lr_max: 10.0,
            curv_win_width: 20,
            zero_debias: true,
            weight_decay: 0.0,
            eps: 1e-8,
            variance_window: 10,
            sparsity_average: 0.999,
        }
    }
}

/// YellowFin optimizer implementation
///
/// YellowFin automatically tunes momentum and learning rate based on local
/// quadratic approximation and curvature information.
pub struct YellowFin {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    config: YellowFinConfig,

    // Global state for tuning
    global_state: HashMap<String, f32>,
    gradient_history: Vec<f32>, // For curvature estimation
    variance_history: Vec<f32>, // For gradient variance
}

impl YellowFin {
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        beta: Option<f32>,
        weight_decay: Option<f32>,
        config: Option<YellowFinConfig>,
    ) -> Self {
        let mut config = config.unwrap_or_default();
        if let Some(lr) = lr {
            config.lr = lr;
        }
        if let Some(beta) = beta {
            config.beta = beta;
        }
        if let Some(wd) = weight_decay {
            config.weight_decay = wd;
        }

        let param_group = ParamGroup::new(params, config.lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            config,
            global_state: HashMap::new(),
            gradient_history: Vec::new(),
            variance_history: Vec::new(),
        }
    }

    pub fn builder() -> YellowFinBuilder {
        YellowFinBuilder::new()
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }

    /// Estimate local quadratic properties for automatic tuning
    fn estimate_curvature(&mut self, grad_norm: f32) -> (f32, f32) {
        // Add current gradient norm to history
        self.gradient_history.push(grad_norm);

        // Keep only recent history
        if self.gradient_history.len() > self.config.curv_win_width {
            self.gradient_history.remove(0);
        }

        // Estimate curvature from gradient norm changes
        let mut curvature = 1.0;
        let mut variance = 1.0;

        if self.gradient_history.len() >= 2 {
            let recent_grads =
                &self.gradient_history[self.gradient_history.len().saturating_sub(5)..];

            // Simple curvature estimation based on gradient norm variance
            let mean = recent_grads.iter().sum::<f32>() / recent_grads.len() as f32;
            let var = recent_grads
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / recent_grads.len() as f32;

            curvature = (var + self.config.eps).sqrt();
            variance = var + self.config.eps;
        }

        (curvature, variance)
    }

    /// Compute optimal momentum based on curvature
    fn compute_optimal_momentum(&self, curvature: f32, lr: f32) -> f32 {
        // YellowFin momentum formula based on quadratic approximation
        // β* = (√κ - 1) / (√κ + 1) where κ is condition number approximation
        let kappa = (curvature / lr).max(1.0);
        let sqrt_kappa = kappa.sqrt();
        let optimal_momentum = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0);

        optimal_momentum.clamp(self.config.beta_min, self.config.beta_max)
    }

    /// Compute optimal learning rate based on curvature and variance
    fn compute_optimal_lr(&self, curvature: f32, variance: f32) -> f32 {
        // YellowFin learning rate formula
        // lr* = 2 / (μ + L) where μ and L are strong convexity and smoothness parameters
        let mu = variance.sqrt();
        let l = curvature;
        let optimal_lr = 2.0 / (mu + l);

        optimal_lr.clamp(self.config.lr_min, self.config.lr_max)
    }

    /// Update global tuning parameters
    fn update_global_parameters(&mut self, grad_norm: f32) -> (f32, f32) {
        let (curvature, variance) = self.estimate_curvature(grad_norm);

        // Get current tuned parameters from global state
        let current_lr = self
            .global_state
            .get("tuned_lr")
            .copied()
            .unwrap_or(self.config.lr);
        let current_momentum = self
            .global_state
            .get("tuned_momentum")
            .copied()
            .unwrap_or(self.config.beta);

        // Compute optimal values
        let optimal_lr = self.compute_optimal_lr(curvature, variance);
        let optimal_momentum = self.compute_optimal_momentum(curvature, current_lr);

        // Smooth updates using exponential moving average
        let lr_smooth_factor = 0.1;
        let momentum_smooth_factor = 0.1;

        let new_lr = current_lr * (1.0 - lr_smooth_factor) + optimal_lr * lr_smooth_factor;
        let new_momentum = current_momentum * (1.0 - momentum_smooth_factor)
            + optimal_momentum * momentum_smooth_factor;

        // Store updated values
        self.global_state.insert("tuned_lr".to_string(), new_lr);
        self.global_state
            .insert("tuned_momentum".to_string(), new_momentum);
        self.global_state.insert("curvature".to_string(), curvature);
        self.global_state.insert("variance".to_string(), variance);

        (new_lr, new_momentum)
    }

    /// Compute bias correction factors
    fn bias_correction(&self, beta: f32, step: usize) -> f32 {
        if self.config.zero_debias {
            1.0 - beta.powi(step as i32)
        } else {
            1.0
        }
    }

    /// YellowFin step for a single parameter
    fn yellowfin_step(
        &mut self,
        param: &Arc<RwLock<Tensor>>,
        tuned_lr: f32,
        tuned_momentum: f32,
    ) -> Result<()> {
        let param_id = Self::get_param_id(param);
        let param_state = self.state.entry(param_id.clone()).or_default();

        let mut param_write = param.write();
        let grad = param_write.grad().ok_or_else(|| {
            TorshError::invalid_argument_with_context("Parameter has no gradient", "yellowfin_step")
        })?;

        // Apply weight decay if specified
        let effective_grad = if self.config.weight_decay > 0.0 {
            grad.add(&param_write.mul_scalar(self.config.weight_decay)?)?
        } else {
            grad.clone()
        };

        // Get or initialize momentum buffer
        let momentum_key = "momentum".to_string();
        let momentum = if let Some(m) = param_state.get(&momentum_key) {
            m.clone()
        } else {
            let m = Tensor::zeros(param_write.shape().dims(), param_write.device())?;
            param_state.insert(momentum_key.clone(), m.clone());
            m
        };

        // Update momentum buffer
        // m_t = β * m_{t-1} + (1 - β) * g_t
        let new_momentum = momentum
            .mul_scalar(tuned_momentum)?
            .add(&effective_grad.mul_scalar(1.0 - tuned_momentum)?)?;

        // Calculate bias correction before using param_state again
        let step_count = self.step_count;

        // Drop param_state borrow temporarily to call bias_correction
        let _ = param_state;
        let bias_correction = self.bias_correction(tuned_momentum, step_count);
        let param_state = self.state.entry(param_id).or_default();

        let corrected_momentum = new_momentum.div_scalar(bias_correction)?;

        // Update parameters
        // θ_t = θ_{t-1} - lr * m̂_t
        *param_write = param_write.sub(&corrected_momentum.mul_scalar(tuned_lr)?)?;

        // Store updated momentum
        param_state.insert(momentum_key, new_momentum);

        Ok(())
    }

    /// Get current tuning statistics
    pub fn get_tuning_stats(&self) -> HashMap<String, f32> {
        let mut stats = self.global_state.clone();
        stats.insert("step_count".to_string(), self.step_count as f32);
        stats.insert(
            "history_length".to_string(),
            self.gradient_history.len() as f32,
        );
        stats
    }

    /// Reset tuning history (useful for transfer learning)
    pub fn reset_tuning(&mut self) {
        self.gradient_history.clear();
        self.variance_history.clear();
        self.global_state.clear();
    }
}

impl Optimizer for YellowFin {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        // Compute global gradient norm for curvature estimation
        let mut total_grad_norm_sq = 0.0;
        let mut param_count = 0;

        for group in &self.param_groups {
            for param in &group.params {
                let param_read = param.read();
                if let Some(grad) = param_read.grad() {
                    let grad_norm_sq = grad.norm()?.item()?.powi(2);
                    total_grad_norm_sq += grad_norm_sq;
                    param_count += 1;
                }
            }
        }

        let global_grad_norm = if param_count > 0 {
            (total_grad_norm_sq / param_count as f32).sqrt()
        } else {
            0.0
        };

        // Update global tuning parameters
        let (tuned_lr, tuned_momentum) = self.update_global_parameters(global_grad_norm);

        // Apply YellowFin updates to all parameters
        // Collect parameters first to avoid borrowing issues
        let params: Vec<Arc<RwLock<Tensor>>> = self
            .param_groups
            .iter()
            .flat_map(|group| group.params.iter().cloned())
            .collect();

        for param in params {
            self.yellowfin_step(&param, tuned_lr, tuned_momentum)?;
        }

        // Update learning rates in parameter groups (for compatibility)
        for group in &mut self.param_groups {
            group.lr = tuned_lr;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        // Return tuned learning rate if available
        let tuned_lr = self
            .global_state
            .get("tuned_lr")
            .copied()
            .unwrap_or(self.config.lr);
        vec![tuned_lr; self.param_groups.len()]
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
        self.global_state.insert("tuned_lr".to_string(), lr);
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(self.config.lr);
        let group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
                param_count: g.params.len(),
            })
            .collect();

        let mut state = self.state.clone();

        // Add global state as tensors
        for (key, &value) in &self.global_state {
            let global_key = format!("global_{}", key);
            state
                .entry("global".to_string())
                .or_default()
                .insert(global_key, Tensor::scalar(value)?);
        }

        Ok(OptimizerState {
            optimizer_type: "YellowFin".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state,
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::TensorError(TorshError::InvalidArgument(
                "Parameter group count mismatch".to_string(),
            )));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        // Extract global state
        self.global_state.clear();
        if let Some(global_state) = state.state.get("global") {
            for (key, tensor) in global_state {
                if let Some(clean_key) = key.strip_prefix("global_") {
                    self.global_state
                        .insert(clean_key.to_string(), tensor.item()?);
                }
            }
        }

        // Extract parameter state (excluding global)
        self.state = state
            .state
            .into_iter()
            .filter(|(k, _)| k != "global")
            .collect();

        Ok(())
    }
}

/// Builder for YellowFin optimizer
pub struct YellowFinBuilder {
    config: YellowFinConfig,
}

impl YellowFinBuilder {
    pub fn new() -> Self {
        Self {
            config: YellowFinConfig::default(),
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.config.lr = lr;
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.config.beta = beta;
        self
    }

    pub fn beta_range(mut self, min: f32, max: f32) -> Self {
        self.config.beta_min = min;
        self.config.beta_max = max;
        self
    }

    pub fn lr_range(mut self, min: f32, max: f32) -> Self {
        self.config.lr_min = min;
        self.config.lr_max = max;
        self
    }

    pub fn curv_win_width(mut self, width: usize) -> Self {
        self.config.curv_win_width = width;
        self
    }

    pub fn zero_debias(mut self, debias: bool) -> Self {
        self.config.zero_debias = debias;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.config.weight_decay = wd;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.config.eps = eps;
        self
    }

    pub fn variance_window(mut self, window: usize) -> Self {
        self.config.variance_window = window;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> YellowFin {
        YellowFin::new(params, None, None, None, Some(self.config))
    }
}

impl Default for YellowFinBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerResult;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_yellowfin_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = YellowFin::new(params, Some(0.1), None, None, None);
        assert_eq!(optimizer.get_lr()[0], 0.1);
        Ok(())
    }

    #[test]
    fn test_yellowfin_builder() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = YellowFin::builder()
            .lr(0.01)
            .beta(0.9)
            .beta_range(0.1, 0.99)
            .curv_win_width(15)
            .zero_debias(true)
            .build(params);

        assert_eq!(optimizer.config.lr, 0.01);
        assert_eq!(optimizer.config.beta, 0.9);
        assert_eq!(optimizer.config.beta_min, 0.1);
        assert_eq!(optimizer.config.beta_max, 0.99);
        assert_eq!(optimizer.config.curv_win_width, 15);
        assert!(optimizer.config.zero_debias);
        Ok(())
    }

    #[test]
    fn test_yellowfin_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = YellowFin::new(params, Some(0.1), None, None, None);

        let result = optimizer.step();
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_curvature_estimation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let mut optimizer = YellowFin::new(params, Some(0.1), None, None, None);

        let (curvature, variance) = optimizer.estimate_curvature(1.0);
        assert!(curvature > 0.0);
        assert!(variance > 0.0);
        Ok(())
    }

    #[test]
    fn test_optimal_momentum_computation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = YellowFin::new(params, Some(0.1), None, None, None);

        let momentum = optimizer.compute_optimal_momentum(2.0, 0.1);
        assert!(momentum >= optimizer.config.beta_min);
        assert!(momentum <= optimizer.config.beta_max);
        Ok(())
    }

    #[test]
    fn test_tuning_stats() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = YellowFin::new(params, Some(0.1), None, None, None);

        let stats = optimizer.get_tuning_stats();
        assert!(stats.contains_key("step_count"));
        assert!(stats.contains_key("history_length"));
        Ok(())
    }

    #[test]
    fn test_reset_tuning() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let mut optimizer = YellowFin::new(params, Some(0.1), None, None, None);

        // Add some history
        optimizer.gradient_history.push(1.0);
        optimizer.global_state.insert("test".to_string(), 2.0);

        optimizer.reset_tuning();
        assert!(optimizer.gradient_history.is_empty());
        assert!(optimizer.global_state.is_empty());
        Ok(())
    }
}
