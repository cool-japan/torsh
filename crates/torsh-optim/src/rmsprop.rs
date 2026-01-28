//! RMSprop optimizer

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// RMSprop optimizer
#[derive(Clone)]
pub struct RMSprop {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 1e-2)
    /// * `alpha` - Smoothing constant (default: 0.99)
    /// * `eps` - Term added to the denominator to improve numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay (L2 penalty) (default: 0.0)
    /// * `momentum` - Momentum factor (default: 0.0)
    /// * `centered` - If True, compute the centered RMSprop
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        alpha: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let alpha = alpha.unwrap_or(0.99);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);

        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
        }
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> OptimizerResult<()> {
        for group in &self.param_groups {
            for param_arc in &group.params {
                let param_id = Self::get_param_id(param_arc);
                let param_read = param_arc.read();

                let grad = param_read.grad().ok_or_else(|| {
                    OptimizerError::TensorError(TorshError::invalid_argument_with_context(
                        "Parameter has no gradient",
                        "rmsprop_step",
                    ))
                })?;

                // Apply weight decay to gradient if specified
                let mut grad_to_use = grad.clone();
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param_read
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    grad_to_use = grad_to_use
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                // Get or initialize optimizer state
                let param_state = self.state.entry(param_id.clone()).or_default();

                let square_avg = if !param_state.contains_key("square_avg") {
                    let square_avg =
                        Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                    param_state.insert("square_avg".to_string(), square_avg.clone());
                    square_avg
                } else {
                    param_state
                        .get("square_avg")
                        .expect("square_avg state should exist")
                        .clone()
                };

                // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
                let grad_squared = grad_to_use
                    .mul_op(&grad_to_use)
                    .map_err(OptimizerError::TensorError)?;
                let new_square_avg = square_avg
                    .mul_scalar(self.alpha)
                    .map_err(OptimizerError::TensorError)?
                    .add(
                        &grad_squared
                            .mul_scalar(1.0 - self.alpha)
                            .map_err(OptimizerError::TensorError)?,
                    )
                    .map_err(OptimizerError::TensorError)?;
                param_state.insert("square_avg".to_string(), new_square_avg.clone());

                let avg = if self.centered {
                    // Centered RMSprop: use variance instead of second moment
                    let grad_avg = if !param_state.contains_key("grad_avg") {
                        let grad_avg =
                            Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                        param_state.insert("grad_avg".to_string(), grad_avg.clone());
                        grad_avg
                    } else {
                        param_state
                            .get("grad_avg")
                            .expect("grad_avg state should exist")
                            .clone()
                    };

                    // Update gradient average: grad_avg = alpha * grad_avg + (1 - alpha) * grad
                    let new_grad_avg = grad_avg
                        .mul_scalar(self.alpha)
                        .map_err(OptimizerError::TensorError)?
                        .add(
                            &grad_to_use
                                .mul_scalar(1.0 - self.alpha)
                                .map_err(OptimizerError::TensorError)?,
                        )
                        .map_err(OptimizerError::TensorError)?;
                    param_state.insert("grad_avg".to_string(), new_grad_avg.clone());

                    // Compute variance: square_avg - grad_avg^2
                    let grad_avg_squared = new_grad_avg
                        .mul_op(&new_grad_avg)
                        .map_err(OptimizerError::TensorError)?;
                    let variance = new_square_avg
                        .sub(&grad_avg_squared)
                        .map_err(OptimizerError::TensorError)?;
                    variance
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                } else {
                    // Standard RMSprop: avg = sqrt(square_avg) + eps
                    new_square_avg
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                };

                let mut update = grad_to_use.div(&avg).map_err(OptimizerError::TensorError)?;

                if self.momentum != 0.0 {
                    // Apply momentum to the update
                    let momentum_buffer = if !param_state.contains_key("momentum_buffer") {
                        let buf =
                            Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                        param_state.insert("momentum_buffer".to_string(), buf.clone());
                        buf
                    } else {
                        param_state
                            .get("momentum_buffer")
                            .expect("momentum_buffer state should exist")
                            .clone()
                    };

                    // Update momentum buffer: buf = momentum * buf + update
                    let new_buf = momentum_buffer
                        .mul_scalar(self.momentum)
                        .map_err(OptimizerError::TensorError)?
                        .add(&update)
                        .map_err(OptimizerError::TensorError)?;
                    param_state.insert("momentum_buffer".to_string(), new_buf.clone());
                    update = new_buf;
                }

                // Apply update: param = param - lr * update
                drop(param_read);
                let mut param_write = param_arc.write();
                let step_update = update
                    .mul_scalar(group.lr)
                    .map_err(OptimizerError::TensorError)?;
                *param_write = param_write
                    .sub(&step_update)
                    .map_err(OptimizerError::TensorError)?;
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
            optimizer_type: "RMSProp".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(
                TorshError::InvalidArgument("Parameter group count mismatch".to_string()).into(),
            );
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Builder for RMSprop optimizer
pub struct RMSpropBuilder {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSpropBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
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

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> RMSprop {
        RMSprop::new(
            params,
            Some(self.lr),
            Some(self.alpha),
            Some(self.eps),
            Some(self.weight_decay),
            Some(self.momentum),
            self.centered,
        )
    }
}

impl Default for RMSpropBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::error::TorshError;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_rmsprop_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = RMSprop::new(params, None, None, None, None, None, false);
        assert_eq!(optimizer.get_lr()[0], 1e-2);
        Ok(())
    }

    #[test]
    fn test_rmsprop_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, None, false);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_centered() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, None, true);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_with_momentum() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, Some(0.9), false);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_builder() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = RMSpropBuilder::new()
            .lr(0.01)
            .alpha(0.95)
            .eps(1e-7)
            .weight_decay(0.01)
            .momentum(0.8)
            .centered(true)
            .build(params);

        assert_eq!(optimizer.get_lr()[0], 0.01);
        assert_eq!(optimizer.alpha, 0.95);
        assert_eq!(optimizer.eps, 1e-7);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.momentum, 0.8);
        assert!(optimizer.centered);

        Ok(())
    }
}
