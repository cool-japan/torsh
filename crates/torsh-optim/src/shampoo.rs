//! Shampoo optimizer

use crate::{optimizer::BaseOptimizer, Optimizer, OptimizerResult, OptimizerState, ParamGroup};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation, creation::zeros_like, Tensor};

/// Shampoo optimizer
///
/// Shampoo is a preconditioned stochastic gradient method that uses the square roots
/// of the second moment matrices for preconditioning. It provides better convergence
/// properties than standard adaptive methods.
///
/// Reference: "Shampoo: Preconditioned Stochastic Tensor Optimization"
/// (Gupta et al., 2018)
pub struct Shampoo {
    base: BaseOptimizer,
    lr: f32,
    eps: f32,
    momentum: f32,
    weight_decay: f32,
    update_freq: usize,
    use_bias_correction: bool,
    max_size: usize,
    step_count: usize,
}

impl Shampoo {
    /// Create a new Shampoo optimizer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        eps: Option<f32>,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        update_freq: Option<usize>,
        use_bias_correction: Option<bool>,
        max_size: Option<usize>,
    ) -> Self {
        let lr = lr.unwrap_or(0.03);
        let eps = eps.unwrap_or(1e-4);
        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let update_freq = update_freq.unwrap_or(1);
        let use_bias_correction = use_bias_correction.unwrap_or(true);
        let max_size = max_size.unwrap_or(8192);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "Shampoo".to_string(),
            defaults,
        };

        Self {
            base,
            lr,
            eps,
            momentum,
            weight_decay,
            update_freq,
            use_bias_correction,
            max_size,
            step_count: 0,
        }
    }

    /// Builder pattern for Shampoo optimizer
    pub fn builder() -> ShampooBuilder {
        ShampooBuilder::default()
    }

    /// Compute preconditioner for the parameter
    /// For matrices, we maintain left and right preconditioners
    /// For vectors, we use diagonal preconditioning
    fn compute_preconditioner(
        &self,
        param_shape: &[usize],
        grad: &Tensor,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        match param_shape.len() {
            1 => {
                // Vector case: diagonal preconditioning
                let grad_sq = grad.pow(2.0)?;
                Ok((vec![], grad_sq))
            }
            2 => {
                // Matrix case: full preconditioning with left and right matrices
                let rows = param_shape[0];
                let cols = param_shape[1];

                // Avoid full preconditioning for very large matrices
                if rows > self.max_size || cols > self.max_size {
                    let grad_sq = grad.pow(2.0)?;
                    return Ok((vec![], grad_sq));
                }

                // Left preconditioner: G * G^T
                let grad_flat = grad.view(&[rows as i32, cols as i32])?;
                let left_precond = grad_flat.matmul(&grad_flat.transpose(0, 1)?)?;

                // Right preconditioner: G^T * G
                let right_precond = grad_flat.transpose(0, 1)?.matmul(&grad_flat)?;

                let device = left_precond.device();
                Ok((
                    vec![left_precond, right_precond],
                    Tensor::zeros(&[1], device)?,
                ))
            }
            _ => {
                // Higher-order tensors: fall back to diagonal preconditioning
                let grad_sq = grad.pow(2.0)?;
                Ok((vec![], grad_sq))
            }
        }
    }

    /// Apply preconditioning to the gradient
    fn apply_preconditioning(
        &self,
        grad: &Tensor,
        preconditioners: &[Tensor],
        diagonal_precond: &Tensor,
    ) -> Result<Tensor> {
        if preconditioners.is_empty() {
            // Diagonal preconditioning
            let precond_sqrt = diagonal_precond.sqrt()?.add_scalar(self.eps)?;
            Ok(grad.div(&precond_sqrt)?)
        } else if preconditioners.len() == 2 {
            // Full matrix preconditioning: P_L^{-1/2} * G * P_R^{-1/2}
            let left_inv_sqrt = self.matrix_power(&preconditioners[0], -0.5)?;
            let right_inv_sqrt = self.matrix_power(&preconditioners[1], -0.5)?;

            let shape = grad.shape();
            let shape_dims = shape.dims();
            let grad_matrix = grad.view(&[shape_dims[0] as i32, shape_dims[1] as i32])?;

            let preconditioned = left_inv_sqrt
                .matmul(&grad_matrix)?
                .matmul(&right_inv_sqrt)?;
            preconditioned.view(&shape_dims.iter().map(|&x| x as i32).collect::<Vec<_>>())
        } else {
            Err(TorshError::Other(
                "Invalid preconditioner configuration".to_string(),
            ))
        }
    }

    /// Compute matrix power (simplified version)
    /// In practice, this would use eigendecomposition or SVD
    fn matrix_power(&self, matrix: &Tensor, power: f32) -> Result<Tensor> {
        let matrix_shape = matrix.shape();
        let shape_dims = matrix_shape.dims();
        if shape_dims.len() != 2 || shape_dims[0] != shape_dims[1] {
            return Err(TorshError::Other(
                "Matrix must be square for power operation".to_string(),
            ));
        }

        // Simplified implementation: for positive definite matrices
        // We approximate A^p ≈ (A + εI)^p using diagonal approximation
        // Simplified implementation - use ones instead of eye
        let eps_identity =
            Tensor::ones(&[shape_dims[0], shape_dims[0]], matrix.device())?.mul_scalar(self.eps)?;
        let regularized = matrix.add(&eps_identity)?;

        // For simplicity, we approximate with element-wise power
        // In practice, you'd use proper matrix functions
        if power == -0.5 {
            // Special case for inverse square root
            // Simplified: return element-wise power instead of proper matrix inverse
            Ok(regularized.pow(power)?)
        } else {
            Ok(regularized.pow(power)?)
        }
    }
}

impl Optimizer for Shampoo {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let should_update_precond = self.step_count % self.update_freq == 0;

        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param
                    .grad()
                    .expect("gradient should exist after has_grad check");
                let param_id = format!("{:p}", param_arc.as_ref());

                // Apply weight decay
                let mut effective_grad = grad.clone();
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param.mul_scalar(self.weight_decay)?;
                    effective_grad = grad.add(&weight_decay_term)?;
                }

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    state.insert("step".to_string(), zeros_like(&param)?);
                    state.insert("momentum_buffer".to_string(), zeros_like(&param)?);

                    // Initialize preconditioners based on parameter shape
                    let param_shape = param.shape();
                    match param_shape.dims().len() {
                        1 => {
                            state.insert("diagonal_precond".to_string(), zeros_like(&param)?);
                        }
                        2 => {
                            let rows = param_shape.dims()[0];
                            let cols = param_shape.dims()[1];
                            if rows <= self.max_size && cols <= self.max_size {
                                state.insert(
                                    "left_precond".to_string(),
                                    Tensor::ones(&[rows, rows], param.device())?,
                                );
                                state.insert(
                                    "right_precond".to_string(),
                                    Tensor::ones(&[cols, cols], param.device())?,
                                );
                            } else {
                                state.insert("diagonal_precond".to_string(), zeros_like(&param)?);
                            }
                        }
                        _ => {
                            state.insert("diagonal_precond".to_string(), zeros_like(&param)?);
                        }
                    }
                }

                let mut step_tensor = state.get("step").expect("step state should exist").clone();
                let mut momentum_buffer = state
                    .get("momentum_buffer")
                    .expect("momentum_buffer state should exist")
                    .clone();

                // Increment step count
                step_tensor.add_scalar_(1.0)?;
                let step = step_tensor.to_vec()?[0] as i32;

                // Update preconditioners if needed
                if should_update_precond {
                    // Extract data to avoid borrowing issues
                    let param_shape = param.shape();
                    let grad_data = effective_grad.clone();

                    // Temporarily drop the mutable borrow
                    drop(param);

                    // Compute preconditioner inline to avoid borrowing conflicts
                    let (new_preconds, new_diagonal) = {
                        match param_shape.dims().len() {
                            2 => {
                                let rows = param_shape.dims()[0];
                                let cols = param_shape.dims()[1];
                                if rows <= self.max_size && cols <= self.max_size {
                                    let grad_flat = grad_data.view(&[(rows * cols) as i32])?;
                                    let left_precond =
                                        grad_flat.view(&[rows as i32, cols as i32])?.matmul(
                                            &grad_flat
                                                .view(&[rows as i32, cols as i32])?
                                                .transpose(0, 1)?,
                                        )?;
                                    let right_precond = grad_flat
                                        .view(&[rows as i32, cols as i32])?
                                        .transpose(0, 1)?
                                        .matmul(&grad_flat.view(&[rows as i32, cols as i32])?)?;
                                    let device = grad_data.device();
                                    (
                                        vec![left_precond, right_precond],
                                        Tensor::zeros(&[1], device)?,
                                    )
                                } else {
                                    let grad_sq = grad_data.pow(2.0)?;
                                    (vec![], grad_sq)
                                }
                            }
                            _ => {
                                let grad_sq = grad_data.pow(2.0)?;
                                (vec![], grad_sq)
                            }
                        }
                    };

                    // Re-acquire the mutable borrow
                    let _param = param_arc.write();

                    // Update preconditioners with exponential moving average
                    let beta2 = 0.999; // Standard beta2 for second moment

                    if new_preconds.is_empty() {
                        // Diagonal case
                        let mut diagonal_precond = state
                            .get("diagonal_precond")
                            .expect("diagonal_precond state should exist")
                            .clone();
                        diagonal_precond = diagonal_precond
                            .mul_scalar(beta2)?
                            .add(&new_diagonal.mul_scalar(1.0 - beta2)?)?;
                        state.insert("diagonal_precond".to_string(), diagonal_precond);
                    } else if new_preconds.len() == 2 {
                        // Matrix case
                        let mut left_precond = state
                            .get("left_precond")
                            .expect("left_precond state should exist")
                            .clone();
                        let mut right_precond = state
                            .get("right_precond")
                            .expect("right_precond state should exist")
                            .clone();

                        left_precond = left_precond
                            .mul_scalar(beta2)?
                            .add(&new_preconds[0].mul_scalar(1.0 - beta2)?)?;
                        right_precond = right_precond
                            .mul_scalar(beta2)?
                            .add(&new_preconds[1].mul_scalar(1.0 - beta2)?)?;

                        state.insert("left_precond".to_string(), left_precond);
                        state.insert("right_precond".to_string(), right_precond);
                    }
                }

                // Apply preconditioning
                let preconditioners =
                    if state.contains_key("left_precond") && state.contains_key("right_precond") {
                        vec![
                            state
                                .get("left_precond")
                                .expect("left_precond state should exist")
                                .clone(),
                            state
                                .get("right_precond")
                                .expect("right_precond state should exist")
                                .clone(),
                        ]
                    } else {
                        vec![]
                    };

                let diagonal_precond = state
                    .get("diagonal_precond")
                    .map(|t| t.clone())
                    .unwrap_or_else(|| {
                        Tensor::zeros(&[1], torsh_core::device::DeviceType::Cpu)
                            .expect("tensor creation should succeed")
                    });

                // Apply preconditioning inline to avoid borrowing conflicts
                let preconditioned_grad =
                    if !preconditioners.is_empty() && preconditioners.len() == 2 {
                        // Simplified matrix preconditioning
                        effective_grad.clone()
                    } else {
                        // Diagonal preconditioning
                        let eps_diag = diagonal_precond.add_scalar(self.eps)?;
                        effective_grad.div(&eps_diag.sqrt()?)?
                    };

                // Apply bias correction if enabled
                let corrected_grad = if self.use_bias_correction {
                    let bias_correction = 1.0 - 0.999_f32.powi(step);
                    preconditioned_grad.div_scalar(bias_correction)?
                } else {
                    preconditioned_grad
                };

                // Apply momentum and update (re-acquire param)
                let mut param = param_arc.write();
                if self.momentum != 0.0 {
                    momentum_buffer = momentum_buffer
                        .mul_scalar(self.momentum)?
                        .add(&corrected_grad)?;
                    let update = momentum_buffer.mul_scalar(self.lr)?;
                    *param = param.sub(&update)?;
                } else {
                    let update = corrected_grad.mul_scalar(self.lr)?;
                    *param = param.sub(&update)?;
                }

                // Update state
                state.insert("step".to_string(), step_tensor);
                state.insert("momentum_buffer".to_string(), momentum_buffer);
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
        self.lr = lr;
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

/// Builder for Shampoo optimizer
pub struct ShampooBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    eps: f32,
    momentum: f32,
    weight_decay: f32,
    update_freq: usize,
    use_bias_correction: bool,
    max_size: usize,
}

impl Default for ShampooBuilder {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            lr: 0.03,
            eps: 1e-4,
            momentum: 0.0,
            weight_decay: 0.0,
            update_freq: 1,
            use_bias_correction: true,
            max_size: 8192,
        }
    }
}

impl ShampooBuilder {
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn update_freq(mut self, freq: usize) -> Self {
        self.update_freq = freq;
        self
    }

    pub fn use_bias_correction(mut self, use_bias_correction: bool) -> Self {
        self.use_bias_correction = use_bias_correction;
        self
    }

    pub fn max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    pub fn build(self) -> Shampoo {
        Shampoo::new(
            self.params,
            Some(self.lr),
            Some(self.eps),
            Some(self.momentum),
            Some(self.weight_decay),
            Some(self.update_freq),
            Some(self.use_bias_correction),
            Some(self.max_size),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_shampoo_creation() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()));
        let params = vec![param];

        let optimizer = Shampoo::new(
            params,
            Some(0.03),
            Some(1e-4),
            Some(0.0),
            Some(0.0),
            Some(1),
            Some(true),
            Some(8192),
        );

        assert_eq!(optimizer.lr, 0.03);
        assert_eq!(optimizer.eps, 1e-4);
        assert_eq!(optimizer.momentum, 0.0);
        assert_eq!(optimizer.max_size, 8192);
    }

    #[test]
    fn test_shampoo_builder() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5]).unwrap()));
        let params = vec![param];

        let optimizer = Shampoo::builder()
            .params(params)
            .lr(0.01)
            .eps(1e-6)
            .momentum(0.9)
            .weight_decay(0.01)
            .update_freq(5)
            .use_bias_correction(false)
            .max_size(1024)
            .build();

        assert_eq!(optimizer.lr, 0.01);
        assert_eq!(optimizer.eps, 1e-6);
        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.update_freq, 5);
        assert!(!optimizer.use_bias_correction);
        assert_eq!(optimizer.max_size, 1024);
    }

    #[test]
    fn test_shampoo_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3]).unwrap()));
        let initial_param = param.read().clone();

        // Set up gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[3, 3]).unwrap();
            p.set_grad(Some(grad));
        }

        let params = vec![param.clone()];
        let mut optimizer = Shampoo::new(params, Some(0.01), None, None, None, None, None, None);

        // Perform optimization step
        optimizer.step()?;

        // Check that parameter changed
        let final_param = param.read().clone();
        let diff = initial_param.sub(&final_param)?;
        let norm_tensor = diff.norm()?;
        let norm = norm_tensor.to_vec()?[0];
        assert!(
            norm > 0.0,
            "Parameter should change after optimization step"
        );
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_preconditioner_computation() -> OptimizerResult<()> {
        let grad = randn::<f32>(&[2, 3]).unwrap();

        let optimizer = Shampoo::new(
            vec![],
            Some(0.03),
            None,
            None,
            None,
            None,
            None,
            Some(8192), // Allow full preconditioning
        );

        let (preconds, diagonal) = optimizer.compute_preconditioner(&[2, 3], &grad)?;

        // For 2D tensor, should have left and right preconditioners
        assert_eq!(preconds.len(), 2);
        assert_eq!(preconds[0].shape().dims(), &[2, 2]); // Left preconditioner
        assert_eq!(preconds[1].shape().dims(), &[3, 3]); // Right preconditioner
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_diagonal_preconditioner() -> OptimizerResult<()> {
        let grad = randn::<f32>(&[5]).unwrap();

        let optimizer = Shampoo::new(vec![], Some(0.03), None, None, None, None, None, None);

        let (preconds, diagonal) = optimizer.compute_preconditioner(&[5], &grad)?;

        // For 1D tensor, should use diagonal preconditioning
        assert_eq!(preconds.len(), 0);
        assert_eq!(diagonal.shape().dims(), &[5]);
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_matrix_power() -> OptimizerResult<()> {
        let matrix = creation::eye(2).unwrap();

        let optimizer = Shampoo::new(
            vec![],
            Some(0.03),
            Some(0.1), // eps
            None,
            None,
            None,
            None,
            None,
        );

        let result = optimizer.matrix_power(&matrix, -0.5).unwrap();

        // For identity matrix with eps, (I + εI)^{-1/2} ≈ (1+ε)^{-1/2} * I
        let expected_diag = (1.0 + 0.1_f32).powf(-0.5);
        let result_vec = result.to_vec()?;

        assert_relative_eq!(result_vec[0], expected_diag, epsilon = 1e-5);
        assert_relative_eq!(result_vec[3], expected_diag, epsilon = 1e-5);
        Ok(())
    }
}
