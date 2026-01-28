//! AdaDelta optimizer
//!
//! AdaDelta is an adaptive learning rate optimizer that doesn't require manually setting a learning rate.
//! It uses exponential moving averages of squared gradients and squared parameter updates.
//!
//! Reference: "ADADELTA: An Adaptive Learning Rate Method" by Matthew D. Zeiler
//! Paper: <https://arxiv.org/abs/1212.5701>

use crate::{optimizer::BaseOptimizer, Optimizer, OptimizerResult, OptimizerState, ParamGroup};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// AdaDelta optimizer
///
/// AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
/// monotonically decreasing learning rate. Instead of accumulating all past squared gradients,
/// AdaDelta restricts the window of accumulated past gradients to some fixed size.
pub struct AdaDelta {
    base: BaseOptimizer,
    rho: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdaDelta {
    /// Create a new AdaDelta optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `rho` - Coefficient used for computing running averages of squared gradients and squared updates (default: 0.9)
    /// * `eps` - Term added to the denominator to improve numerical stability (default: 1e-6)
    /// * `weight_decay` - Weight decay (L2 penalty) coefficient (default: 0.0)
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        rho: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        let rho = rho.unwrap_or(0.9);
        let eps = eps.unwrap_or(1e-6);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let mut defaults = HashMap::new();
        defaults.insert("rho".to_string(), rho);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);

        // Note: AdaDelta doesn't use a fixed learning rate, so we set it to 1.0
        let param_group = ParamGroup::new(params, 1.0);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "AdaDelta".to_string(),
            defaults,
        };

        Self {
            base,
            rho,
            eps,
            weight_decay,
        }
    }

    /// Builder pattern constructor
    pub fn builder() -> AdaDeltaBuilder {
        AdaDeltaBuilder::new()
    }
}

impl Optimizer for AdaDelta {
    fn step(&mut self) -> OptimizerResult<()> {
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param
                    .grad()
                    .expect("gradient should exist after has_grad check");
                let param_id = format!("{:p}", param_arc.as_ref());

                // Apply weight decay to gradient if specified
                let mut grad = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param.mul_scalar(self.weight_decay)?;
                    grad = grad.add(&weight_decay_term)?;
                }

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    // Initialize exponential moving average of squared gradients
                    state.insert(
                        "square_avg".to_string(),
                        torsh_tensor::creation::zeros_like(&param)?,
                    );
                    // Initialize exponential moving average of squared parameter updates
                    state.insert(
                        "acc_delta".to_string(),
                        torsh_tensor::creation::zeros_like(&param)?,
                    );
                }

                let mut square_avg = state
                    .get("square_avg")
                    .expect("square_avg state should exist")
                    .clone();
                let mut acc_delta = state
                    .get("acc_delta")
                    .expect("acc_delta state should exist")
                    .clone();

                // Update exponential moving average of squared gradients
                // square_avg = rho * square_avg + (1 - rho) * grad^2
                let grad_squared = grad.mul_op(&grad)?;
                square_avg.mul_scalar_(self.rho)?;
                let grad_term = grad_squared.mul_scalar(1.0 - self.rho)?;
                square_avg.add(&grad_term)?;

                // Compute RMS of gradients
                // std = sqrt(square_avg + eps)
                let std = square_avg.add_scalar(self.eps)?.sqrt()?;

                // Compute RMS of accumulated deltas
                // delta_std = sqrt(acc_delta + eps)
                let delta_std = acc_delta.add_scalar(self.eps)?.sqrt()?;

                // Compute parameter update
                // delta = -(delta_std / std) * grad
                let delta = grad.mul_op(&delta_std)?.div(&std)?.mul_scalar(-1.0)?;

                // Update exponential moving average of squared parameter updates
                // acc_delta = rho * acc_delta + (1 - rho) * delta^2
                let delta_squared = delta.mul_op(&delta)?;
                acc_delta.mul_scalar_(self.rho)?;
                let delta_term = delta_squared.mul_scalar(1.0 - self.rho)?;
                acc_delta.add(&delta_term)?;

                // Apply update to parameter
                *param = param.add(&delta)?;

                // Update state
                state.insert("square_avg".to_string(), square_avg);
                state.insert("acc_delta".to_string(), acc_delta);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.base.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base.load_state_dict(state)
    }
}

/// Builder for AdaDelta optimizer
pub struct AdaDeltaBuilder {
    rho: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdaDeltaBuilder {
    pub fn new() -> Self {
        Self {
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
        }
    }

    /// Set the coefficient for computing running averages of squared gradients and updates
    pub fn rho(mut self, rho: f32) -> Self {
        self.rho = rho;
        self
    }

    /// Set the term added to the denominator to improve numerical stability
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set the weight decay (L2 penalty) coefficient
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the AdaDelta optimizer
    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> AdaDelta {
        AdaDelta::new(
            params,
            Some(self.rho),
            Some(self.eps),
            Some(self.weight_decay),
        )
    }
}

impl Default for AdaDeltaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerResult;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation;

    #[test]
    fn test_adadelta_creation() -> OptimizerResult<()> {
        let param1 = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3])?));
        let param2 = Arc::new(RwLock::new(creation::randn::<f32>(&[3, 4])?));

        let optimizer = AdaDelta::new(vec![param1, param2], None, None, None);
        assert_eq!(optimizer.rho, 0.9);
        assert_eq!(optimizer.eps, 1e-6);
        assert_eq!(optimizer.weight_decay, 0.0);
        Ok(())
    }

    #[test]
    fn test_adadelta_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3])?));

        let optimizer = AdaDelta::builder()
            .rho(0.95)
            .eps(1e-8)
            .weight_decay(0.01)
            .build(vec![param]);

        assert_eq!(optimizer.rho, 0.95);
        assert_eq!(optimizer.eps, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.01);
        Ok(())
    }

    #[test]
    fn test_adadelta_step() -> OptimizerResult<()> {
        let mut param = creation::ones(&[2, 2])?.requires_grad_(true);
        let original_values = param.to_vec()?;

        // Simulate a simple gradient
        let grad = creation::ones(&[2, 2])?;
        param.set_grad(Some(grad));

        let param_arc = Arc::new(RwLock::new(param));
        let mut optimizer = AdaDelta::new(vec![param_arc.clone()], Some(0.9), Some(1e-6), None);

        // Should not panic
        optimizer.step()?;

        // Parameter should have been updated
        let updated_param = param_arc.read();
        let param_values = updated_param.to_vec()?;

        // Check that parameters have changed (any change indicates the optimizer is working)
        let has_changed = param_values
            .iter()
            .zip(&original_values)
            .any(|(&new, &old)| (new - old).abs() > 1e-10);
        assert!(
            has_changed,
            "Parameters should change after optimization step"
        );

        Ok(())
    }

    #[test]
    fn test_adadelta_zero_grad() -> OptimizerResult<()> {
        let mut param = creation::ones(&[2, 2])?.requires_grad_(true);
        param.set_grad(Some(creation::ones(&[2, 2])?));

        let param_arc = Arc::new(RwLock::new(param));
        let mut optimizer = AdaDelta::new(vec![param_arc.clone()], None, None, None);

        optimizer.zero_grad();

        // Gradient should be None after zero_grad
        assert!(!param_arc.read().has_grad());
        Ok(())
    }

    #[test]
    fn test_adadelta_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3])?));
        let optimizer = AdaDelta::new(vec![param], Some(0.95), Some(1e-8), Some(0.01));

        let state_dict = optimizer.state_dict()?;
        assert_eq!(state_dict.param_groups.len(), 1);
        assert_eq!(state_dict.param_groups[0].lr, 1.0); // AdaDelta uses lr=1.0 internally
        Ok(())
    }
}
