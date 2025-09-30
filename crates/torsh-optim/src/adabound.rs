//! AdaBound optimizer implementation
//!
//! AdaBound is an optimizer that employs dynamic bounds on learning rates to
//! achieve a gradual and smooth transition from adaptive methods to SGD.
//! It bridges the gap between adaptive methods and SGD by reducing the
//! learning rate bounds as training progresses.
//!
//! Reference: "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
//! https://arxiv.org/abs/1902.09843

use crate::{Optimizer, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
// use torsh_core::error::Result;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// AdaBound optimizer
///
/// Combines the benefits of adaptive learning rates with the generalization
/// of SGD by applying dynamic bounds on the learning rates.
pub struct AdaBound {
    param_groups: Vec<ParamGroup>,
    /// First moment estimates (momentum)
    exp_avg: HashMap<String, Tensor>,
    /// Second moment estimates (RMSprop-like)
    exp_avg_sq: HashMap<String, Tensor>,
    /// Step count for bias correction and bounds computation
    step_count: usize,
    /// Base learning rate
    lr: f32,
    /// Exponential decay rate for first moment estimates
    beta1: f32,
    /// Exponential decay rate for second moment estimates
    beta2: f32,
    /// Term added to denominator for numerical stability
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Lower bound for learning rate
    lr_lower_bound: f32,
    /// Upper bound for learning rate
    lr_upper_bound: f32,
    /// Whether to use AMSBound variant
    amsbound: bool,
    /// Maximum second moment estimates (for AMSBound)
    max_exp_avg_sq: HashMap<String, Tensor>,
}

impl AdaBound {
    /// Create a new AdaBound optimizer
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        let param_group = ParamGroup::new(params, lr);
        Self {
            param_groups: vec![param_group],
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            step_count: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            lr_lower_bound: 0.1,
            lr_upper_bound: 10.0,
            amsbound: false,
            max_exp_avg_sq: HashMap::new(),
        }
    }

    /// Create AdaBound with custom parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        lr_lower_bound: f32,
        lr_upper_bound: f32,
        amsbound: bool,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr);
        Self {
            param_groups: vec![param_group],
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            step_count: 0,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            lr_lower_bound,
            lr_upper_bound,
            amsbound,
            max_exp_avg_sq: HashMap::new(),
        }
    }

    /// Get parameter key for state storage
    fn get_param_key(param: &Tensor) -> Result<String> {
        Ok(format!("param_{:p}", param.data()?.as_ptr()))
    }

    /// Compute dynamic learning rate bounds
    #[allow(dead_code)]
    fn compute_lr_bounds(&self, step: usize) -> (f32, f32) {
        let step_ratio = step as f32 / (step as f32 + 1.0);

        // Dynamic bounds that converge to final_lr / gamma and final_lr * gamma
        let lower_bound = self.lr_lower_bound * step_ratio;
        let upper_bound = self.lr_upper_bound * step_ratio;

        (lower_bound, upper_bound)
    }

    /// Clip learning rate to dynamic bounds
    #[allow(dead_code)]
    fn clip_lr(&self, lr: f32, step: usize) -> f32 {
        let (lower_bound, upper_bound) = self.compute_lr_bounds(step);
        lr.clamp(lower_bound, upper_bound)
    }

    /// Static version of clip_lr to avoid borrow checker issues
    fn clip_lr_static(lr: f32, step: usize, lr_lower_bound: f32, lr_upper_bound: f32) -> f32 {
        let step_ratio = step as f32 / (step as f32 + 1.0);
        let lower_bound = lr_lower_bound * step_ratio;
        let upper_bound = lr_upper_bound * step_ratio;
        lr.clamp(lower_bound, upper_bound)
    }
}

impl Optimizer for AdaBound {
    fn step(&mut self) -> OptimizerResult<()> {
        if self.param_groups.is_empty() {
            return Ok(());
        }

        self.step_count += 1;

        for group in &mut self.param_groups {
            let lr = group.lr;
            let weight_decay = group
                .options
                .get("weight_decay")
                .copied()
                .unwrap_or(self.weight_decay);

            for param_ref in &group.params {
                let param = param_ref.read();

                // Skip parameters without gradients
                let grad = match param.grad() {
                    Some(g) => g,
                    None => continue,
                };

                let param_key = Self::get_param_key(&param)?;

                // Initialize state if needed
                if !self.exp_avg.contains_key(&param_key) {
                    self.exp_avg.insert(
                        param_key.clone(),
                        torsh_tensor::creation::zeros_like(&param)?,
                    );
                    self.exp_avg_sq.insert(
                        param_key.clone(),
                        torsh_tensor::creation::zeros_like(&param)?,
                    );
                    if self.amsbound {
                        self.max_exp_avg_sq.insert(
                            param_key.clone(),
                            torsh_tensor::creation::zeros_like(&param)?,
                        );
                    }
                }

                // Get momentum states
                let exp_avg = self.exp_avg.get_mut(&param_key).unwrap();
                let exp_avg_sq = self.exp_avg_sq.get_mut(&param_key).unwrap();

                // Add weight decay to gradient
                let grad_with_decay = if weight_decay != 0.0 {
                    grad.add(&param.mul_scalar(weight_decay)?)?
                } else {
                    grad.clone()
                };

                // Update first moment estimate (exponential moving average of gradient)
                *exp_avg = exp_avg
                    .mul_scalar(self.beta1)?
                    .add(&grad_with_decay.mul_scalar(1.0 - self.beta1)?)?;

                // Update second moment estimate (exponential moving average of squared gradient)
                let grad_sq = grad_with_decay.mul_op(&grad_with_decay)?;
                *exp_avg_sq = exp_avg_sq
                    .mul_scalar(self.beta2)?
                    .add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;

                // Bias correction
                let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

                let corrected_exp_avg = exp_avg.div_scalar(bias_correction1)?;

                let corrected_exp_avg_sq = if self.amsbound {
                    // AMSBound: use maximum of current and previous second moments
                    let max_exp_avg_sq = self.max_exp_avg_sq.get_mut(&param_key).unwrap();
                    let current_exp_avg_sq = exp_avg_sq.div_scalar(bias_correction2)?;

                    // Element-wise maximum
                    let current_data = current_exp_avg_sq.to_vec()?;
                    let max_data = max_exp_avg_sq.to_vec()?;
                    let new_max_data: Vec<f32> = current_data
                        .into_iter()
                        .zip(max_data.into_iter())
                        .map(|(curr, max)| curr.max(max))
                        .collect();

                    *max_exp_avg_sq = Tensor::from_data(
                        new_max_data,
                        max_exp_avg_sq.shape().dims().to_vec(),
                        max_exp_avg_sq.device(),
                    )?;

                    max_exp_avg_sq.clone()
                } else {
                    exp_avg_sq.div_scalar(bias_correction2)?
                };

                // Compute step size with bounds
                let sqrt_v = corrected_exp_avg_sq.sqrt()?;
                let step_size_data = corrected_exp_avg.to_vec()?;
                let denom_data = sqrt_v.add_scalar(self.eps)?.to_vec()?;

                // Apply dynamic bounds element-wise
                let step_count = self.step_count;
                let lr_lower_bound = self.lr_lower_bound;
                let lr_upper_bound = self.lr_upper_bound;
                let bounded_step_data: Vec<f32> = step_size_data
                    .into_iter()
                    .zip(denom_data.into_iter())
                    .map(|(numerator, denominator)| {
                        let raw_step_size = lr * numerator / denominator;
                        Self::clip_lr_static(
                            raw_step_size,
                            step_count,
                            lr_lower_bound,
                            lr_upper_bound,
                        )
                    })
                    .collect();

                let update = Tensor::from_data(
                    bounded_step_data,
                    param.shape().dims().to_vec(),
                    param.device(),
                )?;

                // Update parameters (in-place)
                drop(param); // Release read lock
                let mut param_mut = param_ref.write();
                *param_mut = param_mut.sub(&update)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param_ref in &group.params {
                let mut param = param_ref.write();
                param.zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|group| group.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(self.lr);
        let param_group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(param_group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_groups: Vec<ParamGroupState> = self
            .param_groups
            .iter()
            .map(|group| ParamGroupState {
                lr: group.lr,
                options: group.options.clone(),
                param_count: group.params.len(),
            })
            .collect();

        let mut state = HashMap::new();

        // Save momentum states
        for (key, tensor) in &self.exp_avg {
            let mut param_state = HashMap::new();
            param_state.insert("exp_avg".to_string(), tensor.clone());
            state.insert(format!("{key}_exp_avg"), param_state);
        }

        for (key, tensor) in &self.exp_avg_sq {
            let mut param_state = HashMap::new();
            param_state.insert("exp_avg_sq".to_string(), tensor.clone());
            state.insert(format!("{key}_exp_avg_sq"), param_state);
        }

        if self.amsbound {
            for (key, tensor) in &self.max_exp_avg_sq {
                let mut param_state = HashMap::new();
                param_state.insert("max_exp_avg_sq".to_string(), tensor.clone());
                state.insert(format!("{key}_max_exp_avg_sq"), param_state);
            }
        }

        let mut global_state = HashMap::new();
        global_state.insert("beta1".to_string(), self.beta1);
        global_state.insert("beta2".to_string(), self.beta2);
        global_state.insert("eps".to_string(), self.eps);
        global_state.insert("weight_decay".to_string(), self.weight_decay);
        global_state.insert("lr_lower_bound".to_string(), self.lr_lower_bound);
        global_state.insert("lr_upper_bound".to_string(), self.lr_upper_bound);
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "AdaBound".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Restore parameter groups
        if state.param_groups.len() != self.param_groups.len() {
            return Err(crate::OptimizerError::InvalidParameter(
                "Mismatch in number of parameter groups".to_string(),
            ));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        // Restore optimizer state
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        self.max_exp_avg_sq.clear();

        for (key, param_state) in state.state {
            if key.ends_with("_exp_avg") {
                if let Some(tensor) = param_state.get("exp_avg") {
                    let param_key = key.strip_suffix("_exp_avg").unwrap().to_string();
                    self.exp_avg.insert(param_key, tensor.clone());
                }
            } else if key.ends_with("_exp_avg_sq") {
                if let Some(tensor) = param_state.get("exp_avg_sq") {
                    let param_key = key.strip_suffix("_exp_avg_sq").unwrap().to_string();
                    self.exp_avg_sq.insert(param_key, tensor.clone());
                }
            } else if key.ends_with("_max_exp_avg_sq") {
                if let Some(tensor) = param_state.get("max_exp_avg_sq") {
                    let param_key = key.strip_suffix("_max_exp_avg_sq").unwrap().to_string();
                    self.max_exp_avg_sq.insert(param_key, tensor.clone());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_adabound_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = AdaBound::new(vec![param], 0.001);

        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.eps, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.0);
        assert_eq!(optimizer.lr_lower_bound, 0.1);
        assert_eq!(optimizer.lr_upper_bound, 10.0);
        assert!(!optimizer.amsbound);
        Ok(())
    }

    #[test]
    fn test_adabound_with_params() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = AdaBound::with_params(
            vec![param],
            0.002,
            0.95,
            0.9999,
            1e-7,
            0.01,
            0.05,
            5.0,
            true,
        );

        assert_eq!(optimizer.lr, 0.002);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.beta2, 0.9999);
        assert_eq!(optimizer.eps, 1e-7);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.lr_lower_bound, 0.05);
        assert_eq!(optimizer.lr_upper_bound, 5.0);
        assert!(optimizer.amsbound);
        Ok(())
    }

    #[test]
    fn test_adabound_lr_bounds() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));
        let optimizer =
            AdaBound::with_params(vec![param], 1.0, 0.9, 0.999, 1e-8, 0.0, 0.1, 10.0, false);

        // Test bounds computation
        let (lower, upper) = optimizer.compute_lr_bounds(1);
        assert_eq!(lower, 0.1 * 1.0 / 2.0); // 0.05
        assert_eq!(upper, 10.0 * 1.0 / 2.0); // 5.0

        // Test clipping
        let clipped_low = optimizer.clip_lr(0.01, 1);
        let clipped_high = optimizer.clip_lr(20.0, 1);
        let clipped_normal = optimizer.clip_lr(1.0, 1);

        assert_eq!(clipped_low, 0.05);
        assert_eq!(clipped_high, 5.0);
        assert_eq!(clipped_normal, 1.0);
        Ok(())
    }

    #[test]
    fn test_adabound_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?.mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = AdaBound::new(vec![param.clone()], 0.01);

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);
        Ok(())
    }

    #[test]
    fn test_adabound_amsbound_variant() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?.mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = AdaBound::with_params(
            vec![param.clone()],
            0.01,
            0.9,
            0.999,
            1e-8,
            0.0,
            0.1,
            10.0,
            true, // Enable AMSBound
        );

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);

        // Check that max_exp_avg_sq is populated for AMSBound
        assert!(!optimizer.max_exp_avg_sq.is_empty());

        Ok(())
    }

    #[test]
    fn test_adabound_zero_grad() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?;
            p.set_grad(Some(grad));
            assert!(p.grad().is_some());
        }

        let mut optimizer = AdaBound::new(vec![param.clone()], 0.01);
        optimizer.zero_grad();

        // Check gradient is cleared
        let p = param.read();
        assert!(p.grad().is_none());

        Ok(())
    }

    #[test]
    fn test_adabound_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));
        let optimizer = AdaBound::new(vec![param], 0.01);

        let state = optimizer.state_dict()?;
        assert_eq!(state.param_groups.len(), 1);
        assert_eq!(state.param_groups[0].lr, 0.01);

        Ok(())
    }
}
