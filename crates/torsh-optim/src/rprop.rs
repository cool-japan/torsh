//! Resilient Backpropagation optimizer

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Resilient Backpropagation (Rprop) optimizer
///
/// Rprop adapts the learning rate individually for each parameter based on
/// the sign of the gradient, making it robust to the choice of learning rate.
pub struct Rprop {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    eta_minus: f32,
    eta_plus: f32,
    step_size_min: f32,
    step_size_max: f32,
}

impl Rprop {
    /// Create a new Rprop optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Initial learning rate (default: 1e-2)
    /// * `eta_minus` - Multiplicative decrease factor (default: 0.5)
    /// * `eta_plus` - Multiplicative increase factor (default: 1.2)
    /// * `step_size_min` - Minimum step size (default: 1e-6)
    /// * `step_size_max` - Maximum step size (default: 50.0)
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        eta_minus: Option<f32>,
        eta_plus: Option<f32>,
        step_size_min: Option<f32>,
        step_size_max: Option<f32>,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let eta_minus = eta_minus.unwrap_or(0.5);
        let eta_plus = eta_plus.unwrap_or(1.2);
        let step_size_min = step_size_min.unwrap_or(1e-6);
        let step_size_max = step_size_max.unwrap_or(50.0);

        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            eta_minus,
            eta_plus,
            step_size_min,
            step_size_max,
        }
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }
}

impl Optimizer for Rprop {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        for group in &self.param_groups {
            for param in &group.params {
                let param_id = Self::get_param_id(param);
                let param_read = param.read();

                let grad = param_read.grad().ok_or_else(|| {
                    TorshError::invalid_argument_with_context(
                        "Parameter has no gradient",
                        "rprop_step",
                    )
                })?;

                // Get or initialize state
                let param_state = self.state.entry(param_id.clone()).or_default();

                let prev_grad = if !param_state.contains_key("prev_grad") {
                    let prev_grad = Tensor::zeros_like(&param_read)?;
                    param_state.insert("prev_grad".to_string(), prev_grad.clone());
                    prev_grad
                } else {
                    param_state
                        .get("prev_grad")
                        .expect("prev_grad state should exist")
                        .clone()
                };

                let step_size = if !param_state.contains_key("step_size") {
                    // Create tensor filled with initial learning rate
                    let mut step_size = Tensor::zeros_like(&param_read)?;
                    step_size = step_size.add_scalar(group.lr)?;
                    param_state.insert("step_size".to_string(), step_size.clone());
                    step_size
                } else {
                    param_state
                        .get("step_size")
                        .expect("step_size state should exist")
                        .clone()
                };

                // For simplicity in initial implementation, we'll use element-wise operations
                // In a full implementation, this would be vectorized
                let grad_data = grad.data()?;
                let prev_grad_data = prev_grad.data()?;
                let step_size_data = step_size.data()?;

                let mut new_step_size_data = Vec::with_capacity(step_size_data.len());
                let mut update_data = Vec::with_capacity(grad_data.len());

                for i in 0..grad_data.len() {
                    let sign_change = prev_grad_data[i] * grad_data[i];
                    let mut new_step = step_size_data[i];

                    if sign_change > 0.0 {
                        // Gradient signs agree, increase step size
                        new_step *= self.eta_plus;
                    } else if sign_change < 0.0 {
                        // Gradient signs disagree, decrease step size
                        new_step *= self.eta_minus;
                    }
                    // If sign_change == 0, keep current step size

                    // Clamp step size to valid range
                    new_step = new_step.max(self.step_size_min).min(self.step_size_max);
                    new_step_size_data.push(new_step);

                    // Compute update based on gradient sign
                    let grad_sign = if grad_data[i] > 0.0 {
                        1.0
                    } else if grad_data[i] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    update_data.push(grad_sign * new_step);
                }

                // Create tensors from computed data
                let new_step_size = Tensor::from_data(
                    new_step_size_data,
                    param_read.shape().dims().to_vec(),
                    param_read.device(),
                )?;
                let update = Tensor::from_data(
                    update_data,
                    param_read.shape().dims().to_vec(),
                    param_read.device(),
                )?;

                param_state.insert("step_size".to_string(), new_step_size);

                // Store current gradient as previous gradient for next iteration
                // For positions where gradient changed sign, don't update previous gradient
                let mut effective_grad_data = Vec::with_capacity(grad_data.len());
                for i in 0..grad_data.len() {
                    let sign_change = prev_grad_data[i] * grad_data[i];
                    if sign_change < 0.0 {
                        // Gradient changed sign, use zero
                        effective_grad_data.push(0.0);
                    } else {
                        effective_grad_data.push(grad_data[i]);
                    }
                }

                let effective_grad = Tensor::from_data(
                    effective_grad_data,
                    param_read.shape().dims().to_vec(),
                    param_read.device(),
                )?;
                param_state.insert("prev_grad".to_string(), effective_grad);

                // Apply update
                drop(param_read);
                let mut param_write = param.write();
                *param_write = param_write.sub(&update)?;
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
            optimizer_type: "Rprop".to_string(),
            version: "0.1.0".to_string(),
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
    fn test_rprop_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()))];
        let optimizer = Rprop::new(params, None, None, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 1e-2);
    }

    #[test]
    fn test_rprop_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2]).unwrap()));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = Rprop::new(params, Some(0.1), None, None, None, None);

        optimizer.step()?;
        Ok(())
    }
}
