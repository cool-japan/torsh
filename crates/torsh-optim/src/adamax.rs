//! AdaMax optimizer implementation

use crate::{Optimizer, OptimizerState, ParamGroup, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{
    creation::zeros_like,
    Tensor,
};

/// AdaMax optimizer
///
/// A variant of Adam based on the infinity norm.
///
/// References:
/// - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
pub struct AdaMax {
    param_groups: Vec<ParamGroup>,
    state: HashMap<*const Tensor, AdaMaxState>,
    step_count: usize,
}

#[derive(Clone)]
struct AdaMaxState {
    exp_avg: Tensor,
    exp_inf: Tensor,
}

/// AdaMax optimizer builder
pub struct AdaMaxBuilder {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
}

impl Default for AdaMaxBuilder {
    fn default() -> Self {
        Self {
            lr: 2e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdaMaxBuilder {
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

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> AdaMax {
        let mut options = HashMap::new();
        options.insert("beta1".to_string(), self.betas.0);
        options.insert("beta2".to_string(), self.betas.1);
        options.insert("eps".to_string(), self.eps);
        options.insert("weight_decay".to_string(), self.weight_decay);

        let param_group = ParamGroup::new(params, self.lr).with_options(options);

        AdaMax {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
        }
    }
}

impl AdaMax {
    /// Create a new AdaMax optimizer
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        AdaMaxBuilder::new().lr(lr).build(params)
    }

    /// Create a builder for AdaMax optimizer
    pub fn builder() -> AdaMaxBuilder {
        AdaMaxBuilder::new()
    }
}

impl Optimizer for AdaMax {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;

        for group in &self.param_groups {
            let beta1 = group.options.get("beta1").copied().unwrap_or(0.9);
            let beta2 = group.options.get("beta2").copied().unwrap_or(0.999);
            let eps = group.options.get("eps").copied().unwrap_or(1e-8);
            let weight_decay = group.options.get("weight_decay").copied().unwrap_or(0.0);

            let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);

            for param_ref in &group.params {
                let mut param = param_ref.write();

                let grad = param
                    .grad()
                    .ok_or_else(|| TorshError::Other("Parameter has no gradient".to_string()))?;

                // Get or initialize state
                let param_ptr = &*param as *const Tensor;
                let state = self.state.entry(param_ptr).or_insert_with(|| AdaMaxState {
                    exp_avg: zeros_like(&param),
                    exp_inf: zeros_like(&param),
                });

                // Apply weight decay
                let mut grad_with_decay = grad.clone();
                if weight_decay != 0.0 {
                    let weight_decay_term = param.mul_scalar(weight_decay)?;
                    grad_with_decay = grad_with_decay.add(&weight_decay_term)?;
                }

                // Update biased first moment estimate
                let grad_scaled = grad_with_decay.mul_scalar(1.0 - beta1)?;
                state.exp_avg = state.exp_avg.mul_scalar(beta1)?.add(&grad_scaled)?;

                // Update the exponentially weighted infinity norm
                let grad_abs = grad_with_decay.abs()?;
                let scaled_exp_inf = state.exp_inf.mul_scalar(beta2)?;
                state.exp_inf = scaled_exp_inf.maximum(&grad_abs)?;

                // Compute step size
                let step_size = group.lr / bias_correction1;

                // Update parameters
                let update = state.exp_avg.div(&state.exp_inf.add_scalar(eps)?)?;
                *param = param.sub(&update.mul_scalar(step_size)?)?;
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

    fn state_dict(&self) -> OptimizerState {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
            })
            .collect();

        // Convert state to serializable format
        let mut state_map = HashMap::new();
        for (ptr, state) in &self.state {
            let mut param_state = HashMap::new();
            param_state.insert("exp_avg".to_string(), state.exp_avg.clone());
            param_state.insert("exp_inf".to_string(), state.exp_inf.clone());
            state_map.insert(format!("{:p}", ptr), param_state);
        }

        OptimizerState {
            param_groups,
            state: state_map,
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> Result<()> {
        // Update parameter groups
        for (i, group_state) in state.param_groups.iter().enumerate() {
            if i < self.param_groups.len() {
                self.param_groups[i].lr = group_state.lr;
                self.param_groups[i].options = group_state.options.clone();
            }
        }

        // Load optimizer state
        self.state.clear();
        for (_ptr_str, param_state) in state.state {
            // This is a simplified implementation - in practice, we'd need a way to map
            // back to the actual parameter pointers
            if let (Some(exp_avg), Some(exp_inf)) =
                (param_state.get("exp_avg"), param_state.get("exp_inf"))
            {
                // Would need proper pointer reconstruction here
                // For now, just store with a dummy pointer
                let dummy_ptr = std::ptr::null::<Tensor>();
                self.state.insert(
                    dummy_ptr,
                    AdaMaxState {
                        exp_avg: exp_avg.clone(),
                        exp_inf: exp_inf.clone(),
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
    fn test_adamax_creation() {
        let param = Arc::new(RwLock::new(ones(&[2, 3])));
        let optimizer = AdaMax::new(vec![param], 0.002);

        assert_eq!(optimizer.get_lr(), vec![0.002]);
    }

    #[test]
    fn test_adamax_builder() {
        let param = Arc::new(RwLock::new(ones(&[2, 3])));
        let optimizer = AdaMax::builder()
            .lr(0.001)
            .betas((0.8, 0.99))
            .eps(1e-6)
            .weight_decay(0.01)
            .build(vec![param]);

        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert_eq!(optimizer.param_groups[0].options["beta1"], 0.8);
        assert_eq!(optimizer.param_groups[0].options["beta2"], 0.99);
        assert_eq!(optimizer.param_groups[0].options["eps"], 1e-6);
        assert_eq!(optimizer.param_groups[0].options["weight_decay"], 0.01);
    }

    #[test]
    fn test_adamax_zero_grad() {
        let param = Arc::new(RwLock::new(ones(&[2, 3])));
        let mut optimizer = AdaMax::new(vec![param], 0.002);

        optimizer.zero_grad();
        // Test passes if no panic occurs
    }

    #[test]
    fn test_adamax_set_lr() {
        let param = Arc::new(RwLock::new(ones(&[2, 3])));
        let mut optimizer = AdaMax::new(vec![param], 0.002);

        optimizer.set_lr(0.005);
        assert_eq!(optimizer.get_lr(), vec![0.005]);
    }
}
