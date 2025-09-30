//! LAMB (Large Batch Optimization for Deep Learning) optimizer implementation
//!
//! LAMB is an optimization algorithm designed for training deep neural networks
//! with large batch sizes. It adapts the learning rate both layer-wise and
//! element-wise, combining the benefits of LARS and Adam.
//!
//! Reference: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
//! https://arxiv.org/abs/1904.00962

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// LAMB optimizer
///
/// Combines layer-wise adaptation (LARS) with element-wise adaptation (Adam)
/// for training with very large batch sizes.
pub struct LAMB {
    param_groups: Vec<ParamGroup>,
    /// First moment estimates (momentum)
    exp_avg: HashMap<String, Tensor>,
    /// Second moment estimates (RMSprop-like)
    exp_avg_sq: HashMap<String, Tensor>,
    /// Step count for bias correction
    step_count: usize,
    /// Learning rate
    lr: f32,
    /// Exponential decay rate for first moment estimates
    beta1: f32,
    /// Exponential decay rate for second moment estimates  
    beta2: f32,
    /// Term added to denominator for numerical stability
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Whether to apply bias correction
    bias_correction: bool,
    /// Trust ratio threshold
    trust_ratio_threshold: f32,
}

impl LAMB {
    /// Create a new LAMB optimizer
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
            eps: 1e-6,
            weight_decay: 0.01,
            bias_correction: true,
            trust_ratio_threshold: 1.0,
        }
    }

    /// Create LAMB with custom parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction: bool,
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
            bias_correction,
            trust_ratio_threshold: 1.0,
        }
    }

    /// Set trust ratio threshold
    pub fn set_trust_ratio_threshold(mut self, threshold: f32) -> Self {
        self.trust_ratio_threshold = threshold;
        self
    }

    /// Get parameter key for state storage
    fn get_param_key(param: &Tensor) -> Result<String> {
        Ok(format!("param_{:p}", param.data()?.as_ptr()))
    }

    /// Compute layer-wise learning rate adaptation
    #[allow(dead_code)]
    fn compute_trust_ratio(&self, param: &Tensor, update: &Tensor) -> Result<f32> {
        Self::compute_trust_ratio_static(param, update, self.eps, self.trust_ratio_threshold)
    }

    /// Static version of compute_trust_ratio to avoid borrow checker issues
    fn compute_trust_ratio_static(
        param: &Tensor,
        update: &Tensor,
        eps: f32,
        trust_ratio_threshold: f32,
    ) -> Result<f32> {
        // Compute parameter norm
        let param_data = param.to_vec()?;
        let param_norm: f32 = param_data.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Compute update norm
        let update_data = update.to_vec()?;
        let update_norm: f32 = update_data.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Compute trust ratio
        let trust_ratio = if update_norm > eps && param_norm > eps {
            param_norm / update_norm
        } else {
            1.0
        };

        // Apply threshold
        Ok(trust_ratio.min(trust_ratio_threshold))
    }
}

impl Optimizer for LAMB {
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
                let (corrected_exp_avg, corrected_exp_avg_sq) = if self.bias_correction {
                    let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                    let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
                    (
                        exp_avg.div_scalar(bias_correction1)?,
                        exp_avg_sq.div_scalar(bias_correction2)?,
                    )
                } else {
                    (exp_avg.clone(), exp_avg_sq.clone())
                };

                // Compute Adam-like update
                let sqrt_v = corrected_exp_avg_sq.sqrt()?;
                let adam_update = corrected_exp_avg.div(&sqrt_v.add_scalar(self.eps)?)?;

                // Compute trust ratio (layer-wise adaptation)
                let trust_ratio = Self::compute_trust_ratio_static(
                    &param,
                    &adam_update,
                    self.eps,
                    self.trust_ratio_threshold,
                )?;

                // Apply LAMB update with trust ratio
                let lamb_lr = lr * trust_ratio;
                let update = adam_update.mul_scalar(lamb_lr)?;

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
            state.insert(format!("{}_exp_avg", key), param_state);
        }

        for (key, tensor) in &self.exp_avg_sq {
            let mut param_state = HashMap::new();
            param_state.insert("exp_avg_sq".to_string(), tensor.clone());
            state.insert(format!("{}_exp_avg_sq", key), param_state);
        }

        let mut global_state = HashMap::new();
        global_state.insert("beta1".to_string(), self.beta1);
        global_state.insert("beta2".to_string(), self.beta2);
        global_state.insert("eps".to_string(), self.eps);
        global_state.insert("weight_decay".to_string(), self.weight_decay);
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "LAMB".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Restore parameter groups
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::TensorError(TorshError::InvalidArgument(
                "Mismatch in number of parameter groups".to_string(),
            )));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        // Restore optimizer state
        self.exp_avg.clear();
        self.exp_avg_sq.clear();

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
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_lamb_creation() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = LAMB::new(vec![param], 0.001);

        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.eps, 1e-6);
        assert_eq!(optimizer.weight_decay, 0.01);
    }

    #[test]
    fn test_lamb_with_params() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = LAMB::with_params(vec![param], 0.002, 0.95, 0.9999, 1e-7, 0.02, false);

        assert_eq!(optimizer.lr, 0.002);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.beta2, 0.9999);
        assert_eq!(optimizer.eps, 1e-7);
        assert_eq!(optimizer.weight_decay, 0.02);
        assert!(!optimizer.bias_correction);
    }

    #[test]
    fn test_lamb_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap().mul_scalar(0.1).unwrap();
            p.set_grad(Some(grad));
        }

        let mut optimizer = LAMB::new(vec![param.clone()], 0.01);

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);

        Ok(())
    }

    #[test]
    fn test_lamb_zero_grad() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap();
            p.set_grad(Some(grad));
            assert!(p.grad().is_some());
        }

        let mut optimizer = LAMB::new(vec![param.clone()], 0.01);
        optimizer.zero_grad();

        // Check gradient is cleared
        let p = param.read();
        assert!(p.grad().is_none());
    }

    #[test]
    fn test_lamb_lr_operations() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let mut optimizer = LAMB::new(vec![param], 0.01);

        assert_eq!(optimizer.get_lr(), vec![0.01]);

        optimizer.set_lr(0.02);
        assert_eq!(optimizer.get_lr(), vec![0.02]);
        assert_eq!(optimizer.lr, 0.02);
    }

    #[test]
    fn test_lamb_trust_ratio() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let optimizer = LAMB::new(vec![param.clone()], 0.01);

        let param_tensor = param.read();
        let update = zeros(&[2, 2]).unwrap();

        let trust_ratio = optimizer
            .compute_trust_ratio(&param_tensor, &update)
            .unwrap();
        assert!(trust_ratio >= 0.0);

        // When update norm is 0, trust ratio should be 1.0
        assert_eq!(trust_ratio, 1.0);
    }

    #[test]
    fn test_lamb_param_group() {
        let param1 = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let param2 = Arc::new(RwLock::new(ones(&[3, 3]).unwrap()));

        let mut optimizer = LAMB::new(vec![param1], 0.01);

        let mut options = HashMap::new();
        options.insert("lr".to_string(), 0.02);
        options.insert("weight_decay".to_string(), 0.05);

        optimizer.add_param_group(vec![param2], options);

        assert_eq!(optimizer.param_groups.len(), 2);
        assert_eq!(optimizer.param_groups[1].lr, 0.02);
        assert_eq!(
            optimizer.param_groups[1].options.get("weight_decay"),
            Some(&0.05)
        );
    }

    #[test]
    fn test_lamb_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let optimizer = LAMB::new(vec![param], 0.01);

        let state = optimizer.state_dict()?;
        assert_eq!(state.param_groups.len(), 1);
        assert_eq!(state.param_groups[0].lr, 0.01);

        Ok(())
    }
}
