//! NAdam optimizer implementation

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::zeros_like, Tensor};

/// NAdam optimizer
///
/// Nesterov-accelerated Adam optimizer.
///
/// References:
/// - Dozat, T. (2016). Incorporating nesterov momentum into adam.
pub struct NAdam {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, NAdamState>,
    step_count: usize,
}

#[derive(Clone)]
struct NAdamState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

/// NAdam optimizer builder
pub struct NAdamBuilder {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    momentum_decay: f32,
}

impl Default for NAdamBuilder {
    fn default() -> Self {
        Self {
            lr: 2e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 4e-3,
        }
    }
}

impl NAdamBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn betas(mut self, betas: (f32, f32)) -> Self {
        self.betas = betas;
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

    pub fn momentum_decay(mut self, momentum_decay: f32) -> Self {
        self.momentum_decay = momentum_decay;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> NAdam {
        let mut options = HashMap::new();
        options.insert("beta1".to_string(), self.betas.0);
        options.insert("beta2".to_string(), self.betas.1);
        options.insert("eps".to_string(), self.eps);
        options.insert("weight_decay".to_string(), self.weight_decay);
        options.insert("momentum_decay".to_string(), self.momentum_decay);

        let param_group = ParamGroup::new(params, self.lr).with_options(options);

        NAdam {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
        }
    }
}

impl NAdam {
    /// Create a new NAdam optimizer
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        NAdamBuilder::new().lr(lr).build(params)
    }

    /// Create a builder for NAdam optimizer
    pub fn builder() -> NAdamBuilder {
        NAdamBuilder::new()
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }
}

impl Optimizer for NAdam {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        for group in &self.param_groups {
            let beta1 = group.options.get("beta1").copied().unwrap_or(0.9);
            let beta2 = group.options.get("beta2").copied().unwrap_or(0.999);
            let eps = group.options.get("eps").copied().unwrap_or(1e-8);
            let weight_decay = group.options.get("weight_decay").copied().unwrap_or(0.0);
            let _momentum_decay = group.options.get("momentum_decay").copied().unwrap_or(4e-3);

            let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

            let mu_t = beta1 * (1.0 - 0.5 * 0.96_f32.powi(self.step_count as i32));
            let mu_t_next = beta1 * (1.0 - 0.5 * 0.96_f32.powi((self.step_count + 1) as i32));

            for param_ref in &group.params {
                let param_id = Self::get_param_id(param_ref);
                let mut param = param_ref.write();

                let grad = param
                    .grad()
                    .ok_or_else(|| TorshError::Other("Parameter has no gradient".to_string()))
                    .map_err(OptimizerError::TensorError)?;

                // Get or initialize state
                let state = if let Some(state) = self.state.get_mut(&param_id) {
                    state
                } else {
                    let new_state = NAdamState {
                        exp_avg: zeros_like(&param)?,
                        exp_avg_sq: zeros_like(&param)?,
                    };
                    self.state.insert(param_id.clone(), new_state);
                    self.state.get_mut(&param_id).unwrap()
                };

                // Apply weight decay
                let mut grad_with_decay = grad.clone();
                if weight_decay != 0.0 {
                    let weight_decay_term = param.mul_scalar(weight_decay)?;
                    grad_with_decay = grad_with_decay.add(&weight_decay_term)?;
                }

                // Update biased first and second moment estimates
                let grad_scaled = grad_with_decay.mul_scalar(1.0 - beta1)?;
                state.exp_avg = state.exp_avg.mul_scalar(beta1)?.add(&grad_scaled)?;
                let grad_sq = grad_with_decay.pow_scalar(2.0)?;
                let grad_sq_scaled = grad_sq.mul_scalar(1.0 - beta2)?;
                state.exp_avg_sq = state.exp_avg_sq.mul_scalar(beta2)?.add(&grad_sq_scaled)?;

                // Bias correction
                let exp_avg_corrected = state.exp_avg.div_scalar(bias_correction1)?;
                let exp_avg_sq_corrected = state.exp_avg_sq.div_scalar(bias_correction2)?;

                // NAdam update (with Nesterov momentum)
                let grad_term = grad_with_decay.mul_scalar((1.0 - mu_t) / (1.0 - mu_t_next))?;
                let m_hat = exp_avg_corrected
                    .mul_scalar(mu_t_next / (1.0 - mu_t_next))?
                    .add(&grad_term)?;

                let denom = exp_avg_sq_corrected.sqrt()?.add_scalar(eps)?;
                let update = m_hat.div(&denom)?.mul_scalar(group.lr)?;

                *param = param.sub(&update)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param_ref in &group.params {
                let mut param = param_ref.write();
                param.zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(2e-3);
        let param_group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(param_group);
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

        // Convert state to serializable format
        let mut state_map = HashMap::new();
        for (param_id, state) in &self.state {
            let mut param_state = HashMap::new();
            param_state.insert("exp_avg".to_string(), state.exp_avg.clone());
            param_state.insert("exp_avg_sq".to_string(), state.exp_avg_sq.clone());
            state_map.insert(param_id.clone(), param_state);
        }

        let mut global_state = HashMap::new();
        // Get values from first param group (assuming all groups have same config)
        if let Some(first_group) = self.param_groups.first() {
            global_state.insert(
                "beta1".to_string(),
                first_group.options.get("beta1").copied().unwrap_or(0.9),
            );
            global_state.insert(
                "beta2".to_string(),
                first_group.options.get("beta2").copied().unwrap_or(0.999),
            );
            global_state.insert(
                "eps".to_string(),
                first_group.options.get("eps").copied().unwrap_or(1e-8),
            );
            global_state.insert(
                "weight_decay".to_string(),
                first_group
                    .options
                    .get("weight_decay")
                    .copied()
                    .unwrap_or(0.0),
            );
            global_state.insert(
                "momentum_decay".to_string(),
                first_group
                    .options
                    .get("momentum_decay")
                    .copied()
                    .unwrap_or(0.004),
            );
        }
        global_state.insert("step_count".to_string(), self.step_count as f32);

        Ok(OptimizerState {
            optimizer_type: "NAdam".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state: state_map,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Update parameter groups
        for (i, group_state) in state.param_groups.iter().enumerate() {
            if i < self.param_groups.len() {
                self.param_groups[i].lr = group_state.lr;
                self.param_groups[i].options = group_state.options.clone();
            }
        }

        // Load optimizer state
        self.state.clear();
        for (param_id, param_state) in state.state {
            if let (Some(exp_avg), Some(exp_avg_sq)) =
                (param_state.get("exp_avg"), param_state.get("exp_avg_sq"))
            {
                self.state.insert(
                    param_id,
                    NAdamState {
                        exp_avg: exp_avg.clone(),
                        exp_avg_sq: exp_avg_sq.clone(),
                    },
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_nadam_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = NAdam::new(vec![param], 0.002);

        assert_eq!(optimizer.get_lr(), vec![0.002]);
        Ok(())
    }

    #[test]
    fn test_nadam_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let optimizer = NAdam::builder()
            .lr(0.001)
            .betas((0.8, 0.99))
            .eps(1e-6)
            .weight_decay(0.01)
            .momentum_decay(5e-3)
            .build(vec![param]);

        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert_eq!(optimizer.param_groups[0].options["beta1"], 0.8);
        assert_eq!(optimizer.param_groups[0].options["beta2"], 0.99);
        assert_eq!(optimizer.param_groups[0].options["eps"], 1e-6);
        assert_eq!(optimizer.param_groups[0].options["weight_decay"], 0.01);
        assert_eq!(optimizer.param_groups[0].options["momentum_decay"], 5e-3);
        Ok(())
    }

    #[test]
    fn test_nadam_zero_grad() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let mut optimizer = NAdam::new(vec![param], 0.002);

        optimizer.zero_grad();
        // Test passes if no panic occurs
        Ok(())
    }

    #[test]
    fn test_nadam_set_lr() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 3])?));
        let mut optimizer = NAdam::new(vec![param], 0.002);

        optimizer.set_lr(0.005);
        assert_eq!(optimizer.get_lr(), vec![0.005]);
        Ok(())
    }
}
