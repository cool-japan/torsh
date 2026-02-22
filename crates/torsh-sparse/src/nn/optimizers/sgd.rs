//! Sparse SGD optimizer implementation

use super::super::common::traits::SparseOptimizer;
use crate::{CsrTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Sparse Stochastic Gradient Descent optimizer
///
/// This optimizer implements SGD with momentum for sparse tensors,
/// maintaining sparse momentum buffers to preserve memory efficiency.
pub struct SparseSGD {
    /// Learning rate
    lr: f32,
    /// Momentum coefficient (0.0 = no momentum)
    momentum: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Nesterov momentum flag
    nesterov: bool,
    /// Momentum buffers for each parameter (sparse format)
    momentum_buffers: Vec<Option<CsrTensor>>,
    /// Step counter for debugging
    step_count: usize,
}

impl SparseSGD {
    /// Create a new sparse SGD optimizer
    pub fn new(lr: f32, momentum: f32, weight_decay: f32, nesterov: bool) -> TorshResult<Self> {
        if lr <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Learning rate must be positive".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&momentum) {
            return Err(TorshError::InvalidArgument(
                "Momentum must be between 0.0 and 1.0".to_string(),
            ));
        }

        if weight_decay < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Weight decay must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            lr,
            momentum,
            weight_decay,
            nesterov,
            momentum_buffers: Vec::new(),
            step_count: 0,
        })
    }

    /// Create SGD optimizer with default settings
    pub fn default(lr: f32) -> TorshResult<Self> {
        Self::new(lr, 0.0, 0.0, false)
    }

    /// Create SGD optimizer with momentum
    pub fn with_momentum(lr: f32, momentum: f32) -> TorshResult<Self> {
        Self::new(lr, momentum, 0.0, false)
    }

    /// Create SGD optimizer with weight decay
    pub fn with_weight_decay(lr: f32, weight_decay: f32) -> TorshResult<Self> {
        Self::new(lr, 0.0, weight_decay, false)
    }

    /// Create Nesterov SGD optimizer
    pub fn nesterov(lr: f32, momentum: f32) -> TorshResult<Self> {
        Self::new(lr, momentum, 0.0, true)
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get momentum coefficient
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get weight decay coefficient
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Check if using Nesterov momentum
    pub fn is_nesterov(&self) -> bool {
        self.nesterov
    }

    /// Apply weight decay to gradients
    fn apply_weight_decay(&self, param: &CsrTensor, grad: &CsrTensor) -> TorshResult<CsrTensor> {
        if self.weight_decay == 0.0 {
            return Ok(grad.clone());
        }

        // For sparse tensors, we need to combine gradients with weight decay
        // grad_effective = grad + weight_decay * param
        let param_dense = param.to_dense()?;
        let grad_dense = grad.to_dense()?;

        // Apply weight decay
        let decay_term = param_dense.mul_scalar(self.weight_decay)?;
        let effective_grad_dense = grad_dense.add(&decay_term)?;

        // Convert back to sparse format
        super::super::common::utils::SparseWeightGenerator::dense_to_sparse(&effective_grad_dense)
    }

    /// Update momentum buffer and get velocity
    fn update_momentum(&mut self, param_idx: usize, grad: &CsrTensor) -> TorshResult<CsrTensor> {
        if self.momentum == 0.0 {
            return Ok(grad.clone());
        }

        // Ensure momentum buffer exists
        while self.momentum_buffers.len() <= param_idx {
            self.momentum_buffers.push(None);
        }

        let velocity = match &mut self.momentum_buffers[param_idx] {
            None => {
                // First step: velocity = gradient
                let velocity = grad.clone();
                self.momentum_buffers[param_idx] = Some(velocity.clone());
                velocity
            }
            Some(momentum_buffer) => {
                // Update momentum: velocity = momentum * velocity + gradient
                let momentum_dense = momentum_buffer.to_dense()?;
                let grad_dense = grad.to_dense()?;

                let velocity_dense = momentum_dense.mul_scalar(self.momentum)?.add(&grad_dense)?;

                let velocity = super::super::common::utils::SparseWeightGenerator::dense_to_sparse(
                    &velocity_dense,
                )?;
                *momentum_buffer = velocity.clone();
                velocity
            }
        };

        Ok(velocity)
    }

    /// Apply Nesterov momentum correction
    fn apply_nesterov(
        &self,
        momentum_buffer: &CsrTensor,
        grad: &CsrTensor,
    ) -> TorshResult<CsrTensor> {
        if !self.nesterov || self.momentum == 0.0 {
            return Ok(momentum_buffer.clone());
        }

        // Nesterov: velocity = momentum * velocity + gradient
        let momentum_dense = momentum_buffer.to_dense()?;
        let grad_dense = grad.to_dense()?;

        let nesterov_velocity_dense = momentum_dense.mul_scalar(self.momentum)?.add(&grad_dense)?;

        super::super::common::utils::SparseWeightGenerator::dense_to_sparse(
            &nesterov_velocity_dense,
        )
    }
}

impl SparseOptimizer for SparseSGD {
    fn step(
        &mut self,
        parameters: &mut [&mut CsrTensor],
        gradients: &[&CsrTensor],
    ) -> TorshResult<()> {
        if parameters.len() != gradients.len() {
            return Err(TorshError::InvalidArgument(
                "Number of parameters and gradients must match".to_string(),
            ));
        }

        self.step_count += 1;

        for (param_idx, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay to gradient
            let effective_grad = self.apply_weight_decay(param, grad)?;

            // Update momentum and get velocity
            let velocity = self.update_momentum(param_idx, &effective_grad)?;

            // Apply Nesterov correction if enabled
            let update = if self.nesterov && self.momentum > 0.0 {
                self.apply_nesterov(&velocity, &effective_grad)?
            } else {
                velocity
            };

            // Update parameters: param = param - lr * update
            let param_dense = param.to_dense()?;
            let update_dense = update.to_dense()?;

            let updated_param_dense = param_dense.sub(&update_dense.mul_scalar(self.lr)?)?;

            // Convert back to sparse format and update parameter
            **param = super::super::common::utils::SparseWeightGenerator::dense_to_sparse(
                &updated_param_dense,
            )?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // SGD doesn't accumulate gradients, so nothing to do
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        // Export momentum buffers if they exist
        for (i, buffer) in self.momentum_buffers.iter().enumerate() {
            if let Some(momentum_buffer) = buffer {
                if let Ok(dense_buffer) = momentum_buffer.to_dense() {
                    state.insert(format!("momentum_buffer_{}", i), dense_buffer);
                }
            }
        }

        if let Ok(step_tensor) = torsh_tensor::Tensor::scalar(self.step_count as f32) {
            state.insert("step_count".to_string(), step_tensor);
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> TorshResult<()> {
        // Load step count
        if let Some(step_tensor) = state.get("step_count") {
            self.step_count = step_tensor.get_item(&[])? as usize;
        }

        // Load momentum buffers
        self.momentum_buffers.clear();
        let mut buffer_idx = 0;

        loop {
            let buffer_key = format!("momentum_buffer_{}", buffer_idx);
            if let Some(buffer_tensor) = state.get(&buffer_key) {
                let sparse_buffer =
                    super::super::common::utils::SparseWeightGenerator::dense_to_sparse(
                        buffer_tensor,
                    )?;
                self.momentum_buffers.push(Some(sparse_buffer));
                buffer_idx += 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        if self.nesterov {
            "SparseSGD_Nesterov"
        } else if self.momentum > 0.0 {
            "SparseSGD_Momentum"
        } else {
            "SparseSGD"
        }
    }

    fn hyperparameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("lr".to_string(), self.lr);
        params.insert("momentum".to_string(), self.momentum);
        params.insert("weight_decay".to_string(), self.weight_decay);
        params.insert(
            "nesterov".to_string(),
            if self.nesterov { 1.0 } else { 0.0 },
        );
        params
    }
}

/// Builder for SparseSGD optimizer
pub struct SparseSGDBuilder {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
}

impl SparseSGDBuilder {
    /// Create a new SGD builder with required learning rate
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Set momentum coefficient
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay coefficient
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable Nesterov momentum
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Build the optimizer
    pub fn build(self) -> TorshResult<SparseSGD> {
        SparseSGD::new(self.lr, self.momentum, self.weight_decay, self.nesterov)
    }
}

impl SparseSGD {
    /// Create a builder for SparseSGD
    pub fn builder(lr: f32) -> SparseSGDBuilder {
        SparseSGDBuilder::new(lr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_sgd_creation() {
        let optimizer = SparseSGD::new(0.01, 0.9, 0.0001, false);
        assert!(optimizer.is_ok());

        let opt = optimizer.unwrap();
        assert_eq!(opt.lr(), 0.01);
        assert_eq!(opt.momentum(), 0.9);
        assert_eq!(opt.weight_decay(), 0.0001);
        assert!(!opt.is_nesterov());
    }

    #[test]
    fn test_sparse_sgd_default() {
        let optimizer = SparseSGD::default(0.01);
        assert!(optimizer.is_ok());

        let opt = optimizer.unwrap();
        assert_eq!(opt.lr(), 0.01);
        assert_eq!(opt.momentum(), 0.0);
        assert_eq!(opt.weight_decay(), 0.0);
    }

    #[test]
    fn test_sparse_sgd_builder() {
        let optimizer = SparseSGD::builder(0.01)
            .momentum(0.9)
            .weight_decay(0.0001)
            .nesterov(true)
            .build();

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.lr(), 0.01);
        assert_eq!(opt.momentum(), 0.9);
        assert_eq!(opt.weight_decay(), 0.0001);
        assert!(opt.is_nesterov());
    }

    #[test]
    fn test_hyperparameters() {
        let opt = SparseSGD::new(0.01, 0.9, 0.0001, true).unwrap();
        let params = opt.hyperparameters();

        assert_eq!(params["lr"], 0.01);
        assert_eq!(params["momentum"], 0.9);
        assert_eq!(params["weight_decay"], 0.0001);
        assert_eq!(params["nesterov"], 1.0);
    }

    #[test]
    fn test_invalid_learning_rate() {
        let result = SparseSGD::new(-0.01, 0.0, 0.0, false);
        assert!(result.is_err());

        let result = SparseSGD::new(0.0, 0.0, 0.0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_momentum() {
        let result = SparseSGD::new(0.01, -0.1, 0.0, false);
        assert!(result.is_err());

        let result = SparseSGD::new(0.01, 1.1, 0.0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimizer_name() {
        let sgd = SparseSGD::default(0.01).unwrap();
        assert_eq!(sgd.name(), "SparseSGD");

        let sgd_momentum = SparseSGD::with_momentum(0.01, 0.9).unwrap();
        assert_eq!(sgd_momentum.name(), "SparseSGD_Momentum");

        let sgd_nesterov = SparseSGD::nesterov(0.01, 0.9).unwrap();
        assert_eq!(sgd_nesterov.name(), "SparseSGD_Nesterov");
    }

    #[test]
    fn test_learning_rate_update() {
        let mut opt = SparseSGD::default(0.01).unwrap();
        assert_eq!(opt.lr(), 0.01);

        opt.set_lr(0.001);
        assert_eq!(opt.lr(), 0.001);
    }
}
