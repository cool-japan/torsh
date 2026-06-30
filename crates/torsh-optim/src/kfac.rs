//! K-FAC (Kronecker-Factored Approximate Curvature) optimizer

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
use parking_lot::RwLock;
use std::collections::HashMap;
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

    /// Compute Kronecker factorization for a layer.
    ///
    /// For a linear layer with weight `W` stored as `[input_size, output_size]`,
    /// K-FAC approximates the Fisher block as `F ≈ A ⊗ G`, where `A` is the input
    /// (activation) covariance and `G` is the pre-activation gradient covariance.
    ///
    /// Without forward/backward hooks the true activation statistics are
    /// unavailable, so we use the gradient-only approximation that estimates both
    /// factors from the empirical second moment of the weight gradient `Gw`:
    ///     A ≈ (1 / output_size) · Gw · Gwᵀ   -> [input_size, input_size]
    ///     G ≈ (1 / input_size)  · Gwᵀ · Gw   -> [output_size, output_size]
    ///
    /// Both factors are symmetric positive semi-definite, matching the structure
    /// of the true Kronecker factors and genuinely carrying curvature information.
    ///
    /// This is an associated function (no `&self`) so it can be called from inside
    /// `step()`'s `&mut self.base.param_groups` loop without a borrow conflict.
    fn compute_kronecker_factors(weight: &Tensor, grad: &Tensor) -> Result<(Tensor, Tensor)> {
        let shape = weight.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "K-FAC currently only supports 2D weight matrices".to_string(),
            ));
        }

        let grad_shape = grad.shape();
        if grad_shape.dims() != dims {
            return Err(TorshError::Other(format!(
                "K-FAC gradient shape {:?} must match weight shape {:?}",
                grad_shape.dims(),
                dims
            )));
        }

        let input_size = dims[0];
        let output_size = dims[1];

        if input_size == 0 || output_size == 0 {
            return Err(TorshError::Other(
                "K-FAC weight matrix dimensions must be non-zero".to_string(),
            ));
        }

        let grad_t = grad.t()?; // [output_size, input_size]

        // A: input covariance estimate (input_size x input_size)
        let activation_cov = grad.matmul(&grad_t)?.mul_scalar(1.0 / output_size as f32)?;

        // G: output (gradient) covariance estimate (output_size x output_size)
        let grad_cov = grad_t.matmul(grad)?.mul_scalar(1.0 / input_size as f32)?;

        Ok((activation_cov, grad_cov))
    }

    /// Apply Kronecker factorized preconditioning: `A^{-1} · Gw · G^{-1}`.
    ///
    /// Associated function (no `&self`) so it can be invoked from `step()`'s
    /// mutable-borrow loop without conflicting borrows.
    fn apply_kfac_preconditioning(grad: &Tensor, a_inv: &Tensor, g_inv: &Tensor) -> Result<Tensor> {
        // Preconditioned gradient: A^{-1} * (dL/dW) * G^{-1}.
        let preconditioned = a_inv.matmul(grad)?.matmul(g_inv)?;
        Ok(preconditioned)
    }

    /// Compute matrix inverse with Tikhonov damping for numerical stability.
    ///
    /// Returns `(matrix + damping · I)^{-1}` using a true dense matrix inverse
    /// (the same `solve::inv` routine the optimizer's `step()` uses), not an
    /// element-wise reciprocal. The damping keeps the system well-conditioned even
    /// when `matrix` is singular or near-singular, exactly as required for the
    /// K-FAC preconditioner.
    ///
    /// Associated function taking `damping` explicitly (rather than reading
    /// `self.damping`) so it can be called from `step()` without borrowing `self`.
    fn damped_inverse(matrix: &Tensor, damping: f32) -> Result<Tensor> {
        let shape = matrix.shape();
        let dims = shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Add damping to the diagonal: matrix + damping · I.
        let damping_matrix = eye(dims[0])?.mul_scalar(damping)?;
        let damped_matrix = matrix.add(&damping_matrix)?;

        // Proper dense matrix inversion (LU-based) rather than a diagonal-only
        // reciprocal approximation.
        Ok(solve::inv(&damped_matrix)?)
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

                let grad = param
                    .grad()
                    .expect("gradient should exist after has_grad check");
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

                let mut momentum_buffer = state
                    .get("momentum_buffer")
                    .expect("momentum_buffer state should exist")
                    .clone();
                let mut activation_cov = state
                    .get("activation_cov")
                    .expect("activation_cov state should exist")
                    .clone();
                let mut gradient_cov = state
                    .get("gradient_cov")
                    .expect("gradient_cov state should exist")
                    .clone();
                let mut a_inv = state
                    .get("a_inv")
                    .expect("a_inv state should exist")
                    .clone();
                let mut g_inv = state
                    .get("g_inv")
                    .expect("g_inv state should exist")
                    .clone();
                let mut step_tensor = state.get("step").expect("step state should exist").clone();

                // Increment step count
                step_tensor.add_scalar_(1.0)?;

                // Update Kronecker factors if needed
                if should_update_kfac {
                    // Snapshot the data we need and copy the scalar hyper-parameters
                    // so the factor/inverse helpers (associated functions) never need
                    // to borrow `self` while `self.base.param_groups` is borrowed by
                    // the surrounding loop.
                    let param_data = param.clone();
                    let grad_data = grad.clone();
                    let stat_decay = self.stat_decay;
                    let damping = self.damping;

                    // Temporarily drop param to avoid borrowing conflicts
                    drop(param);

                    // Estimate the Kronecker factors A and G from the current weight
                    // gradient (see `compute_kronecker_factors` for the math). This
                    // genuinely injects curvature information into the preconditioner,
                    // unlike the previous `Tensor::ones` placeholder that made K-FAC
                    // behave like plain SGD.
                    let (new_a, new_g) = Self::compute_kronecker_factors(&param_data, &grad_data)?;

                    // Update covariance matrices with an exponential moving average so
                    // the factors track curvature statistics across steps.
                    activation_cov = activation_cov
                        .mul_scalar(stat_decay)?
                        .add(&new_a.mul_scalar(1.0 - stat_decay)?)?;
                    gradient_cov = gradient_cov
                        .mul_scalar(stat_decay)?
                        .add(&new_g.mul_scalar(1.0 - stat_decay)?)?;

                    // Damped (Tikhonov) inverses of the running factors.
                    a_inv = Self::damped_inverse(&activation_cov, damping)?;
                    g_inv = Self::damped_inverse(&gradient_cov, damping)?;

                    // Re-acquire parameter lock
                    param = param_arc.write();
                }

                // Re-acquire gradient if param was dropped
                let grad = if should_update_kfac {
                    param.grad().expect("gradient should exist after update")
                } else {
                    grad
                };

                // Apply the Kronecker-factored preconditioner A^{-1} · Gw · G^{-1}.
                let preconditioned_grad = if self.use_preconditioning {
                    Self::apply_kfac_preconditioning(&grad, &a_inv, &g_inv)?
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

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base.parameters()
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
    use torsh_core::device::DeviceType;
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
        let norm = diff
            .norm()
            .unwrap()
            .to_vec()
            .expect("tensor to vec conversion should succeed")[0];
        assert!(
            norm > 0.0,
            "Parameter should change after optimization step"
        );
    }

    #[test]
    fn test_kronecker_factors() {
        let weight = randn::<f32>(&[3, 2]).unwrap();
        let grad = randn::<f32>(&[3, 2]).unwrap();

        let (a, g) = KFAC::compute_kronecker_factors(&weight, &grad).unwrap();

        // Check dimensions
        assert_eq!(a.shape().dims(), &[3, 3]);
        assert_eq!(g.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_kronecker_factors_carry_curvature_not_placeholder() {
        // The factors must be the gradient-based curvature estimate, NOT the old
        // `Tensor::ones` / identity placeholder. Use a deterministic gradient so we
        // can compute the expected factors exactly.
        //
        // weight/grad shape: [input_size=2, output_size=2]
        //   Gw = [[1, 2],
        //         [3, 4]]
        //   A = (1/output) Gw Gwᵀ = 0.5 * [[5, 11], [11, 25]]  = [[2.5, 5.5], [5.5, 12.5]]
        //   G = (1/input)  Gwᵀ Gw = 0.5 * [[10, 14], [14, 20]] = [[5.0, 7.0], [7.0, 10.0]]
        let weight = Tensor::from_data(vec![0.0; 4], vec![2, 2], DeviceType::Cpu).unwrap();
        let grad =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let (a, g) = KFAC::compute_kronecker_factors(&weight, &grad).unwrap();

        let a_vals = a.to_vec().unwrap();
        let g_vals = g.to_vec().unwrap();

        let expected_a = [2.5f32, 5.5, 5.5, 12.5];
        let expected_g = [5.0f32, 7.0, 7.0, 10.0];

        for (got, want) in a_vals.iter().zip(expected_a.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-5);
        }
        for (got, want) in g_vals.iter().zip(expected_g.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-5);
        }

        // Sanity: factors are symmetric and not the all-ones placeholder.
        assert_relative_eq!(a_vals[1], a_vals[2], epsilon = 1e-6);
        assert_relative_eq!(g_vals[1], g_vals[2], epsilon = 1e-6);
        assert!(a_vals.iter().any(|v| (v - 1.0).abs() > 1e-3));
    }

    #[test]
    fn test_damped_inverse() -> OptimizerResult<()> {
        let matrix = eye(3).unwrap();

        let damping = 0.1f32;
        let inv = KFAC::damped_inverse(&matrix, damping).unwrap();

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
