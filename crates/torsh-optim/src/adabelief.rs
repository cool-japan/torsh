//! AdaBelief optimizer implementation
//!
//! AdaBelief adapts the step size according to the "belief" in the gradient
//! direction, combining the benefits of adaptive learning rates with fast
//! convergence. It adapts the step size based on how much the gradient
//! magnitude varies from the exponential moving average.
//!
//! Reference: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
//! https://arxiv.org/abs/2010.07468

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// AdaBelief optimizer
///
/// Adapts step sizes by the belief in the gradient direction, providing
/// better convergence properties than traditional adaptive methods.
pub struct AdaBelief {
    param_groups: Vec<ParamGroup>,
    /// First moment estimates (exponential moving average of gradients)
    exp_avg: HashMap<String, Tensor>,
    /// Second moment estimates (variance of gradients from exp_avg)
    exp_avg_sq: HashMap<String, Tensor>,
    /// Step count for bias correction
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
    /// Whether to apply AMSGrad-style maximum
    amsgrad: bool,
    /// Maximum second moment estimates (for AMSGrad variant)
    max_exp_avg_sq: HashMap<String, Tensor>,
    /// Whether to debias the first and second moments
    weight_decouple: bool,
    /// Whether to apply rectification as in RAdam
    rectify: bool,
}

impl AdaBelief {
    /// Create a new AdaBelief optimizer
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
            eps: 1e-16,
            weight_decay: 0.0,
            amsgrad: false,
            max_exp_avg_sq: HashMap::new(),
            weight_decouple: true,
            rectify: true,
        }
    }

    /// Create AdaBelief with custom parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
        weight_decouple: bool,
        rectify: bool,
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
            amsgrad,
            max_exp_avg_sq: HashMap::new(),
            weight_decouple,
            rectify,
        }
    }

    /// Get parameter key for state storage
    fn get_param_key(param: &Tensor) -> Result<String> {
        // Bind data to a variable to extend lifetime before taking pointer
        let data = param.data()?;
        Ok(format!("param_{:p}", data.as_ptr()))
    }

    /// Compute rectification term for RAdam-style adaptation
    #[allow(dead_code)]
    fn compute_rectification(&self, step: usize) -> Option<f32> {
        Self::compute_rectification_static(step, self.beta2, self.rectify)
    }

    /// Static version of compute_rectification to avoid borrow checker issues
    fn compute_rectification_static(step: usize, beta2: f32, rectify: bool) -> Option<f32> {
        if !rectify {
            return None;
        }

        let beta2_t = beta2.powi(step as i32);
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * step as f32 * beta2_t / (1.0 - beta2_t);

        if rho_t >= 5.0 {
            let variance_rectification = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt();
            Some(variance_rectification)
        } else {
            None
        }
    }
}

impl Optimizer for AdaBelief {
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
                    let exp_avg_init = torsh_tensor::creation::zeros_like(&param)
                        .map_err(OptimizerError::TensorError)?;
                    let exp_avg_sq_init = torsh_tensor::creation::zeros_like(&param)
                        .map_err(OptimizerError::TensorError)?;
                    self.exp_avg.insert(param_key.clone(), exp_avg_init);
                    self.exp_avg_sq.insert(param_key.clone(), exp_avg_sq_init);
                    if self.amsgrad {
                        let max_exp_avg_sq_init = torsh_tensor::creation::zeros_like(&param)
                            .map_err(OptimizerError::TensorError)?;
                        self.max_exp_avg_sq
                            .insert(param_key.clone(), max_exp_avg_sq_init);
                    }
                }

                // Get momentum states
                let exp_avg = self.exp_avg.get_mut(&param_key).unwrap();
                let exp_avg_sq = self.exp_avg_sq.get_mut(&param_key).unwrap();

                // Apply weight decay
                let (param_to_update, grad_to_use) = if self.weight_decouple && weight_decay != 0.0
                {
                    // Decouple weight decay (L2 regularization applied directly to parameters)
                    drop(param); // Release read lock
                    let mut param_mut = param_ref.write();
                    *param_mut = param_mut.mul_scalar(1.0 - lr * weight_decay)?;
                    let _param_updated = param_mut.clone();
                    drop(param_mut); // Release write lock
                    let param_read = param_ref.read();
                    (param_read, grad.clone())
                } else {
                    // Traditional weight decay (add to gradient)
                    let grad_with_decay = if weight_decay != 0.0 {
                        grad.add(&param.mul_scalar(weight_decay)?)?
                    } else {
                        grad.clone()
                    };
                    (param, grad_with_decay)
                };

                // Update first moment estimate (exponential moving average of gradient)
                *exp_avg = exp_avg
                    .mul_scalar(self.beta1)?
                    .add(&grad_to_use.mul_scalar(1.0 - self.beta1)?)?;

                // AdaBelief's key innovation: compute variance of gradient from exp_avg
                let grad_residual = grad_to_use.sub(exp_avg)?;
                let grad_residual_sq = grad_residual.mul_op(&grad_residual)?;

                // Update second moment estimate (variance estimate)
                *exp_avg_sq = exp_avg_sq
                    .mul_scalar(self.beta2)?
                    .add(&grad_residual_sq.mul_scalar(1.0 - self.beta2)?)?;

                // Bias correction
                let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

                let corrected_exp_avg = exp_avg.div_scalar(bias_correction1)?;

                let corrected_exp_avg_sq = if self.amsgrad {
                    // AMSGrad: use maximum of current and previous second moments
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

                // Compute update
                let step_count = self.step_count;
                let rectify = self.rectify;
                let update = if let Some(rect_coeff) =
                    Self::compute_rectification_static(step_count, self.beta2, rectify)
                {
                    // Apply rectification (RAdam-style)
                    let sqrt_v = corrected_exp_avg_sq.sqrt()?;
                    let denom = sqrt_v.add_scalar(self.eps)?;
                    corrected_exp_avg.div(&denom)?.mul_scalar(lr * rect_coeff)?
                } else if rectify {
                    // Skip update when rectification is not applicable
                    torsh_tensor::creation::zeros_like(&param_to_update)
                        .map_err(OptimizerError::TensorError)?
                } else {
                    // Standard AdaBelief update
                    let sqrt_v = corrected_exp_avg_sq.sqrt()?;
                    let denom = sqrt_v.add_scalar(self.eps)?;
                    corrected_exp_avg.div(&denom)?.mul_scalar(lr)?
                };

                // Update parameters (in-place)
                drop(param_to_update); // Release read lock
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

        if self.amsgrad {
            for (key, tensor) in &self.max_exp_avg_sq {
                let mut param_state = HashMap::new();
                param_state.insert("max_exp_avg_sq".to_string(), tensor.clone());
                state.insert(format!("{key}_max_exp_avg_sq"), param_state);
            }
        }

        Ok(OptimizerState {
            optimizer_type: "AdaBelief".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state,
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Restore parameter groups
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::InvalidParameter(
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
    fn test_adabelief_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = AdaBelief::new(vec![param], 0.001);

        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.eps, 1e-16);
        assert_eq!(optimizer.weight_decay, 0.0);
        assert!(!optimizer.amsgrad);
        assert!(optimizer.weight_decouple);
        assert!(optimizer.rectify);
        Ok(())
    }

    #[test]
    fn test_adabelief_with_params() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = AdaBelief::with_params(
            vec![param],
            0.002,
            0.95,
            0.9999,
            1e-8,
            0.01,
            true,
            false,
            false,
        );

        assert_eq!(optimizer.lr, 0.002);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.beta2, 0.9999);
        assert_eq!(optimizer.eps, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert!(optimizer.amsgrad);
        assert!(!optimizer.weight_decouple);
        assert!(!optimizer.rectify);
        Ok(())
    }

    #[test]
    fn test_adabelief_rectification() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));
        let optimizer = AdaBelief::with_params(
            vec![param],
            0.01,
            0.9,
            0.999,
            1e-16,
            0.0,
            false,
            true,
            true, // Enable rectification
        );

        // Test early steps (should return None due to insufficient warmup)
        let rect1 = optimizer.compute_rectification(1);
        assert!(rect1.is_none());

        // Test later steps (should return Some value)
        let rect100 = optimizer.compute_rectification(100);
        assert!(rect100.is_some());
        assert!(rect100.unwrap() > 0.0);
        Ok(())
    }

    #[test]
    fn test_adabelief_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?.mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = AdaBelief::new(vec![param.clone()], 0.01);

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);
        Ok(())
    }

    #[test]
    fn test_adabelief_amsgrad_variant() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?.mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = AdaBelief::with_params(
            vec![param.clone()],
            0.01,
            0.9,
            0.999,
            1e-16,
            0.0,
            true, // Enable AMSGrad
            true,
            true,
        );

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);

        // Check that max_exp_avg_sq is populated for AMSGrad
        assert!(!optimizer.max_exp_avg_sq.is_empty());
        Ok(())
    }

    #[test]
    fn test_adabelief_weight_decouple() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?.mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = AdaBelief::with_params(
            vec![param.clone()],
            0.01,
            0.9,
            0.999,
            1e-16,
            0.1, // Non-zero weight decay
            false,
            true, // Enable weight decouple
            false,
        );

        optimizer.step()?;
        assert_eq!(optimizer.step_count, 1);
        Ok(())
    }

    #[test]
    fn test_adabelief_zero_grad() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2])?;
            p.set_grad(Some(grad));
            assert!(p.grad().is_some());
        }

        let mut optimizer = AdaBelief::new(vec![param.clone()], 0.01);
        optimizer.zero_grad();

        // Check gradient is cleared
        let p = param.read();
        assert!(p.grad().is_none());
        Ok(())
    }

    #[test]
    fn test_adabelief_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2])?));
        let optimizer = AdaBelief::new(vec![param], 0.01);

        let state = optimizer.state_dict()?;
        assert_eq!(state.param_groups.len(), 1);
        assert_eq!(state.param_groups[0].lr, 0.01);

        Ok(())
    }
}
