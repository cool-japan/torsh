//! K-FAC (Kronecker-Factored Approximate Curvature) optimizer

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_linalg::solve;
use torsh_tensor::{
    creation::{eye, zeros_like},
    Tensor,
};

/// K-FAC optimizer
///
/// K-FAC approximates the Fisher Information Matrix using Kronecker factorization,
/// which is more scalable than full natural gradient methods for neural networks.
///
/// Reference: "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
/// (Martens & Grosse, 2015)
pub struct KFAC {
    base: BaseOptimizer,
    lr: f32,
    momentum: f32,
    damping: f32,
    kfac_update_freq: usize,
    stat_decay: f32,
    #[allow(dead_code)]
    kl_clip: f32,
    step_count: usize,
    use_preconditioning: bool,
}

impl KFAC {
    /// Create a new K-FAC optimizer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        momentum: Option<f32>,
        damping: Option<f32>,
        kfac_update_freq: Option<usize>,
        stat_decay: Option<f32>,
        kl_clip: Option<f32>,
        use_preconditioning: Option<bool>,
    ) -> Self {
        let lr = lr.unwrap_or(0.001);
        let momentum = momentum.unwrap_or(0.9);
        let damping = damping.unwrap_or(0.003);
        let kfac_update_freq = kfac_update_freq.unwrap_or(10);
        let stat_decay = stat_decay.unwrap_or(0.95);
        let kl_clip = kl_clip.unwrap_or(0.001);
        let use_preconditioning = use_preconditioning.unwrap_or(true);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("damping".to_string(), damping);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "KFAC".to_string(),
            defaults,
        };

        Self {
            base,
            lr,
            momentum,
            damping,
            kfac_update_freq,
            stat_decay,
            kl_clip,
            step_count: 0,
            use_preconditioning,
        }
    }

    /// Builder pattern for K-FAC optimizer
    pub fn builder() -> KFACBuilder {
        KFACBuilder::default()
    }

    /// Compute Kronecker factorization for a layer
    /// For a linear layer with weight W (input_size x output_size), we approximate:
    /// F ≈ A ⊗ G where A is the activation covariance and G is the gradient covariance
    fn compute_kronecker_factors(
        &self,
        weight: &Tensor,
        _grad: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let shape = weight.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "K-FAC currently only supports 2D weight matrices".to_string(),
            ));
        }

        let input_size = dims[0];
        let output_size = dims[1];

        // For simplicity, we approximate the activations and gradients
        // In a real implementation, these would come from forward/backward hooks

        // A: Activation covariance matrix (input_size x input_size)
        // We approximate this with identity for now
        let activation_cov = eye(input_size);

        // G: Gradient covariance matrix (output_size x output_size)
        // We use a simplified approximation since outer product is not available
        let grad_cov = eye(output_size);

        Ok((activation_cov?, grad_cov?))
    }

    /// Apply Kronecker factorized preconditioning
    fn apply_kfac_preconditioning(
        &self,
        grad: &Tensor,
        a_inv: &Tensor,
        g_inv: &Tensor,
    ) -> Result<Tensor> {
        // Preconditioned gradient: vec(A^{-1} * G * G^{-1})
        // For matrix G with gradient dL/dG, the preconditioned gradient is:
        // A^{-1} * (dL/dG) * G^{-1}

        let preconditioned = a_inv.matmul(grad)?.matmul(g_inv)?;
        Ok(preconditioned)
    }

    /// Compute matrix inverse with damping for numerical stability
    fn damped_inverse(&self, matrix: &Tensor) -> Result<Tensor> {
        let shape = matrix.shape();
        let dims = shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Add damping to diagonal
        let damping_matrix = eye(dims[0])?.mul_scalar(self.damping)?;
        let damped_matrix = matrix.add(&damping_matrix)?;

        // For simplicity, we approximate the inverse with reciprocal for diagonal matrices
        // In practice, you'd use proper matrix inversion (Cholesky, LU, etc.)
        damped_matrix.reciprocal()
    }
}

impl Optimizer for KFAC {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let should_update_kfac = self.step_count % self.kfac_update_freq == 0;

        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());

                // Only handle 2D parameters (weight matrices) for now
                if param.shape().ndim() != 2 {
                    // Fall back to regular gradient descent for non-2D parameters
                    let update = grad.mul_scalar(self.lr)?;
                    *param = param.sub(&update)?;
                    continue;
                }

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    state.insert("momentum_buffer".to_string(), zeros_like(&param)?);
                    state.insert("activation_cov".to_string(), eye(param.shape().dims()[0])?);
                    state.insert("gradient_cov".to_string(), eye(param.shape().dims()[1])?);
                    state.insert("a_inv".to_string(), eye(param.shape().dims()[0])?);
                    state.insert("g_inv".to_string(), eye(param.shape().dims()[1])?);
                    state.insert("step".to_string(), zeros_like(&param)?);
                }

                let mut momentum_buffer = state.get("momentum_buffer").unwrap().clone();
                let mut activation_cov = state.get("activation_cov").unwrap().clone();
                let mut gradient_cov = state.get("gradient_cov").unwrap().clone();
                let mut a_inv = state.get("a_inv").unwrap().clone();
                let mut g_inv = state.get("g_inv").unwrap().clone();
                let mut step_tensor = state.get("step").unwrap().clone();

                // Increment step count
                step_tensor.add_scalar_(1.0)?;

                // Update Kronecker factors if needed
                if should_update_kfac {
                    // Extract data to avoid borrowing conflicts
                    let param_data = param.clone();
                    let _grad_data = grad.clone();
                    let stat_decay = self.stat_decay;
                    let damping = self.damping;

                    // Temporarily drop param to avoid borrowing conflicts
                    drop(param);

                    // Compute Kronecker factors inline to avoid borrowing conflicts
                    let (new_a, new_g) = {
                        let shape = param_data.shape();
                        let dims = shape.dims();
                        if dims.len() != 2 {
                            return Err(OptimizerError::TensorError(TorshError::InvalidArgument(
                                "K-FAC only supports 2D weight matrices".to_string(),
                            )));
                        }

                        // For simplicity, use identity matrices as placeholders
                        // In a full implementation, these would be computed from activations and gradients
                        let input_size = dims[0];
                        let output_size = dims[1];

                        // Create identity-like matrices (simplified implementation)
                        let a = Tensor::ones(&[input_size, input_size], param_data.device())?;
                        let g = Tensor::ones(&[output_size, output_size], param_data.device())?;
                        (a, g)
                    };

                    // Update covariance matrices with exponential moving average
                    activation_cov = activation_cov
                        .mul_scalar(stat_decay)?
                        .add(&new_a.mul_scalar(1.0 - stat_decay)?)?;
                    gradient_cov = gradient_cov
                        .mul_scalar(stat_decay)?
                        .add(&new_g.mul_scalar(1.0 - stat_decay)?)?;

                    // Compute damped inverses manually to avoid borrowing self
                    let damped_a = activation_cov.add_scalar(damping)?;
                    let damped_g = gradient_cov.add_scalar(damping)?;
                    a_inv = solve::inv(&damped_a)?;
                    g_inv = solve::inv(&damped_g)?;

                    // Re-acquire parameter lock
                    param = param_arc.write();
                }

                // Re-acquire gradient if param was dropped
                let grad = if should_update_kfac {
                    param.grad().unwrap()
                } else {
                    grad
                };

                // Apply preconditioning if enabled (inline to avoid borrowing issues)
                let preconditioned_grad = if self.use_preconditioning {
                    // Manual implementation of KFAC preconditioning
                    grad.matmul(&g_inv)?
                        .transpose(0, 1)?
                        .matmul(&a_inv)?
                        .transpose(0, 1)?
                } else {
                    grad
                };

                // Apply momentum
                if self.momentum != 0.0 {
                    momentum_buffer = momentum_buffer
                        .mul_scalar(self.momentum)?
                        .add(&preconditioned_grad)?;
                    let update = momentum_buffer.mul_scalar(self.lr)?;
                    *param = param.sub(&update)?;
                } else {
                    let update = preconditioned_grad.mul_scalar(self.lr)?;
                    *param = param.sub(&update)?;
                }

                // Update state
                state.insert("momentum_buffer".to_string(), momentum_buffer);
                state.insert("activation_cov".to_string(), activation_cov);
                state.insert("gradient_cov".to_string(), gradient_cov);
                state.insert("a_inv".to_string(), a_inv);
                state.insert("g_inv".to_string(), g_inv);
                state.insert("step".to_string(), step_tensor);
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

/// Builder for K-FAC optimizer
pub struct KFACBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    momentum: f32,
    damping: f32,
    kfac_update_freq: usize,
    stat_decay: f32,
    kl_clip: f32,
    use_preconditioning: bool,
}

impl Default for KFACBuilder {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            lr: 0.001,
            momentum: 0.9,
            damping: 0.003,
            kfac_update_freq: 10,
            stat_decay: 0.95,
            kl_clip: 0.001,
            use_preconditioning: true,
        }
    }
}

impl KFACBuilder {
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    pub fn kfac_update_freq(mut self, freq: usize) -> Self {
        self.kfac_update_freq = freq;
        self
    }

    pub fn stat_decay(mut self, decay: f32) -> Self {
        self.stat_decay = decay;
        self
    }

    pub fn kl_clip(mut self, clip: f32) -> Self {
        self.kl_clip = clip;
        self
    }

    pub fn use_preconditioning(mut self, use_preconditioning: bool) -> Self {
        self.use_preconditioning = use_preconditioning;
        self
    }

    pub fn build(self) -> KFAC {
        KFAC::new(
            self.params,
            Some(self.lr),
            Some(self.momentum),
            Some(self.damping),
            Some(self.kfac_update_freq),
            Some(self.stat_decay),
            Some(self.kl_clip),
            Some(self.use_preconditioning),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_kfac_creation() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 5]).unwrap()));
        let params = vec![param];

        let optimizer = KFAC::new(
            params,
            Some(0.001),
            Some(0.9),
            Some(0.003),
            Some(10),
            Some(0.95),
            Some(0.001),
            Some(true),
        );

        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.damping, 0.003);
        assert_eq!(optimizer.kfac_update_freq, 10);
    }

    #[test]
    fn test_kfac_builder() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[8, 4]).unwrap()));
        let params = vec![param];

        let optimizer = KFAC::builder()
            .params(vec![Arc::new(RwLock::new(randn::<f32>(&[10, 5]).unwrap()))])
            .lr(0.01)
            .momentum(0.8)
            .damping(0.01)
            .kfac_update_freq(5)
            .stat_decay(0.9)
            .kl_clip(0.01)
            .use_preconditioning(false)
            .build();

        assert_eq!(optimizer.lr, 0.01);
        assert_eq!(optimizer.momentum, 0.8);
        assert_eq!(optimizer.damping, 0.01);
        assert_eq!(optimizer.kfac_update_freq, 5);
        assert!(!optimizer.use_preconditioning);
    }

    #[test]
    fn test_kfac_step() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[4, 3]).unwrap()));
        let initial_param = param.read().clone();

        // Set up gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[4, 3]).unwrap();
            p.set_grad(Some(grad));
        }

        let params = vec![param.clone()];
        let mut optimizer = KFAC::new(params, Some(0.01), None, None, None, None, None, None);

        // Perform optimization step
        optimizer.step().unwrap();

        // Check that parameter changed
        let final_param = param.read().clone();
        let diff = initial_param.sub(&final_param).unwrap();
        let norm = diff.norm().unwrap().to_vec().unwrap()[0];
        assert!(
            norm > 0.0,
            "Parameter should change after optimization step"
        );
    }

    #[test]
    #[allow(dead_code)]
    fn test_kronecker_factors() {
        let weight = randn::<f32>(&[3, 2]).unwrap();
        let grad = randn::<f32>(&[3, 2]).unwrap();

        let optimizer = KFAC::new(vec![], Some(0.001), None, None, None, None, None, None);

        let (a, g) = optimizer.compute_kronecker_factors(&weight, &grad).unwrap();

        // Check dimensions
        assert_eq!(a.shape().dims(), &[3, 3]);
        assert_eq!(g.shape().dims(), &[2, 2]);
    }

    #[test]
    #[allow(dead_code)]
    fn test_damped_inverse() -> OptimizerResult<()> {
        let matrix = eye(3).unwrap();

        let optimizer = KFAC::new(vec![], Some(0.001), None, Some(0.1), None, None, None, None);

        let inv = optimizer.damped_inverse(&matrix).unwrap();

        // For identity matrix with damping, inverse should approximately be 1/(1+damping) * I
        let expected_diag = 1.0 / (1.0 + 0.1);
        let inv_vec = inv.to_vec()?;

        // Check diagonal elements (positions 0, 4, 8 for 3x3 matrix)
        assert_relative_eq!(inv_vec[0], expected_diag, epsilon = 1e-5);
        assert_relative_eq!(inv_vec[4], expected_diag, epsilon = 1e-5);
        assert_relative_eq!(inv_vec[8], expected_diag, epsilon = 1e-5);

        Ok(())
    }
}
