//! Averaged Stochastic Gradient Descent optimizer

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Averaged Stochastic Gradient Descent (ASGD) optimizer
///
/// This implements the Averaged SGD algorithm which maintains a running average
/// of parameters during training. It can achieve better convergence properties
/// than standard SGD in certain scenarios.
pub struct ASGD {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    alpha: f32,
    t0: f32,
    lambd: f32,
}

impl ASGD {
    /// Create a new ASGD optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 1e-2)
    /// * `alpha` - Power for computing average (default: 0.75)
    /// * `t0` - Point at which to start averaging (default: 1e6)
    /// * `lambd` - Decay term (default: 1e-4)
    /// * `weight_decay` - Weight decay (L2 penalty) (default: 0.0)
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        alpha: Option<f32>,
        t0: Option<f32>,
        lambd: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let alpha = alpha.unwrap_or(0.75);
        let t0 = t0.unwrap_or(1e6);
        let lambd = lambd.unwrap_or(1e-4);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let mut options = HashMap::new();
        options.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr).with_options(options);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            alpha,
            t0,
            lambd,
        }
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }
}

impl Optimizer for ASGD {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        for group in &self.param_groups {
            let lr = group.lr;
            let weight_decay = group.options.get("weight_decay").copied().unwrap_or(0.0);

            for param in &group.params {
                let param_id = Self::get_param_id(param);
                let param_read = param.read();

                let grad = param_read.grad().ok_or_else(|| {
                    TorshError::invalid_argument_with_context(
                        "Parameter has no gradient",
                        "asgd_step",
                    )
                })?;

                // Get or initialize state
                let param_state = self.state.entry(param_id.clone()).or_default();

                let mut eta = if !param_state.contains_key("eta") {
                    let eta = lr;
                    param_state.insert("eta".to_string(), Tensor::scalar(eta)?);
                    eta
                } else {
                    param_state.get("eta").unwrap().item()?
                };

                let ax = if !param_state.contains_key("ax") {
                    let ax = param_read.clone();
                    param_state.insert("ax".to_string(), ax.clone());
                    ax
                } else {
                    param_state.get("ax").unwrap().clone()
                };

                let mu = if !param_state.contains_key("mu") {
                    let mu = 1.0;
                    param_state.insert("mu".to_string(), Tensor::scalar(mu)?);
                    mu
                } else {
                    param_state.get("mu").unwrap().item()?
                };

                // Apply weight decay
                let mut grad_to_use = grad.clone();
                if weight_decay != 0.0 {
                    grad_to_use = grad_to_use.add(&param_read.mul_scalar(weight_decay)?)?;
                }

                // Update eta
                if self.step_count > 1 {
                    eta = lr / (1.0 + (self.step_count as f32 - 1.0) * self.lambd).powf(self.alpha);
                    param_state.insert("eta".to_string(), Tensor::scalar(eta)?);
                }

                // Update parameter
                drop(param_read);
                let mut param_write = param.write();
                *param_write = param_write.sub(&grad_to_use.mul_scalar(eta)?)?;

                // Update averaged parameter
                if self.step_count as f32 >= self.t0 {
                    let new_mu = mu / (mu + 1.0);
                    param_state.insert("mu".to_string(), Tensor::scalar(new_mu)?);

                    let new_ax = ax
                        .mul_scalar(new_mu)?
                        .add(&param_write.mul_scalar(1.0 - new_mu)?)?;
                    param_state.insert("ax".to_string(), new_ax);
                } else {
                    let new_mu = 1.0 / self.step_count as f32;
                    param_state.insert("mu".to_string(), Tensor::scalar(new_mu)?);

                    let new_ax = ax
                        .mul_scalar(1.0 - new_mu)?
                        .add(&param_write.mul_scalar(new_mu)?)?;
                    param_state.insert("ax".to_string(), new_ax);
                }
            }
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
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(1e-2);
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

        Ok(OptimizerState {
            optimizer_type: "ASGD".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::InvalidParameter(
                "Parameter group count mismatch".to_string(),
            ));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_asgd_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()))];
        let optimizer = ASGD::new(params, None, None, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 1e-2);
    }

    #[test]
    fn test_asgd_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = ASGD::new(params, Some(0.1), None, None, None, None);

        optimizer.step()?;
        Ok(())
    }
}
