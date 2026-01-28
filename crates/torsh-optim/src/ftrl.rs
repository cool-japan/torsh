//! FTRL (Follow-The-Regularized-Leader) optimizer

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::{
    creation::{self, zeros_like},
    Tensor,
};

/// FTRL optimizer
///
/// Follow-The-Regularized-Leader (FTRL) is an online learning algorithm
/// particularly effective for sparse feature spaces and large-scale problems.
/// It combines the benefits of FOBOS and RDA algorithms.
///
/// Reference: "Ad Click Prediction: a View from the Trenches" (McMahan et al., 2013)
pub struct FTRL {
    base: BaseOptimizer,
    alpha: f32,
    beta: f32,
    l1: f32,
    l2: f32,
}

impl FTRL {
    /// Create a new FTRL optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        alpha: Option<f32>,
        beta: Option<f32>,
        l1: Option<f32>,
        l2: Option<f32>,
    ) -> Self {
        let alpha = alpha.unwrap_or(1.0);
        let beta = beta.unwrap_or(1.0);
        let l1 = l1.unwrap_or(0.0);
        let l2 = l2.unwrap_or(0.0);

        let mut defaults = HashMap::new();
        defaults.insert("alpha".to_string(), alpha);
        defaults.insert("beta".to_string(), beta);
        defaults.insert("l1".to_string(), l1);
        defaults.insert("l2".to_string(), l2);

        let param_group = ParamGroup::new(params, alpha); // Use alpha as lr for compatibility

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "FTRL".to_string(),
            defaults,
        };

        Self {
            base,
            alpha,
            beta,
            l1,
            l2,
        }
    }

    /// Builder pattern for FTRL optimizer
    pub fn builder() -> FTRLBuilder {
        FTRLBuilder::default()
    }

    /// Apply L1 regularization (soft thresholding)
    fn apply_l1_regularization(&self, z: &Tensor, sigma: &Tensor) -> Result<Tensor> {
        if self.l1 == 0.0 {
            return Ok(z.clone());
        }

        // Soft thresholding: sign(z) * max(0, |z| - l1 * sigma)
        let abs_z = z.abs()?;
        let threshold = sigma.mul_scalar(self.l1)?;
        let sign_z = z.sign()?;

        // Element-wise max with 0
        let diff = abs_z.sub(&threshold)?;
        let zeros = Tensor::zeros(z.shape().dims(), z.device())?;
        let regularized = diff.maximum(&zeros)?;

        sign_z.mul_op(&regularized)
    }

    /// Compute learning rate for FTRL
    fn compute_learning_rate(&self, n: &Tensor) -> Result<Tensor> {
        // η_t = α / (β + √(Σ g_i²))
        let sqrt_n = n.sqrt()?;
        let denom = sqrt_n.add_scalar(self.beta)?;
        let lr = denom.reciprocal()?.mul_scalar(self.alpha)?;
        Ok(lr)
    }
}

impl Optimizer for FTRL {
    fn step(&mut self) -> OptimizerResult<()> {
        // Extract values to avoid borrowing conflicts
        let alpha = self.alpha;
        let beta = self.beta;
        let l1 = self.l1;
        let l2 = self.l2;

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

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    let z_init = zeros_like(&param).map_err(OptimizerError::TensorError)?;
                    let n_init = zeros_like(&param).map_err(OptimizerError::TensorError)?;
                    state.insert("z".to_string(), z_init); // Accumulated z values
                    state.insert("n".to_string(), n_init); // Accumulated squared gradients
                }

                let mut z = state.get("z").expect("z state should exist").clone();
                let mut n = state.get("n").expect("n state should exist").clone();

                // Current parameter value
                let w = param.clone();

                // Compute learning rate: η_t = α / (β + √(Σ g_i²))
                let n_sqrt = n.sqrt()?;
                let denom = n_sqrt.add_scalar(beta)?;
                let eta = denom.reciprocal()?.mul_scalar(alpha)?;

                // Update z: z_t = z_{t-1} + g_t - (η_t - η_{t-1}) * w_t
                // For first step, η_{t-1} = 0, so this simplifies to z_t = z_{t-1} + g_t - η_t * w_t
                let eta_w = eta.mul_op(&w)?;
                z = z.add(&grad)?.sub(&eta_w)?;

                // Update n: n_t = n_{t-1} + g_t²
                let grad_sq = grad.pow(2.0)?;
                n = n.add(&grad_sq)?;

                // Compute new learning rate with updated n
                let n_sqrt = n.sqrt()?;
                let denom = n_sqrt.add_scalar(beta)?;
                let new_eta = denom.reciprocal()?.mul_scalar(alpha)?;

                // Apply L1 regularization to z
                let sigma = new_eta.add_scalar(l2)?; // σ = η + λ₂

                // Apply L1 regularization: max(0, |z| - λ₁/σ) * sign(z)
                let threshold = sigma.reciprocal()?.mul_scalar(l1)?;
                let abs_z = z.abs()?;
                let sign_z = z.sign()?;
                let diff = abs_z.sub(&threshold)?;
                let zeros = Tensor::zeros(z.shape().dims(), z.device())?;
                let regularized = diff.maximum(&zeros)?;
                let z_reg = sign_z.mul_op(&regularized)?;

                // Update parameter: w_{t+1} = -z_reg / σ
                let new_param = z_reg.neg()?.div(&sigma)?;
                *param = new_param;

                // Update state
                state.insert("z".to_string(), z);
                state.insert("n".to_string(), n);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        // FTRL doesn't have a fixed learning rate, return alpha as approximation
        vec![self.alpha; self.base.param_groups.len()]
    }

    fn set_lr(&mut self, lr: f32) {
        // Set alpha parameter
        self.alpha = lr;
        for group in &mut self.base.param_groups {
            group.lr = lr;
        }
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

/// Builder for FTRL optimizer
pub struct FTRLBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    alpha: f32,
    beta: f32,
    l1: f32,
    l2: f32,
}

impl Default for FTRLBuilder {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            alpha: 1.0,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
        }
    }
}

impl FTRLBuilder {
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn l1(mut self, l1: f32) -> Self {
        self.l1 = l1;
        self
    }

    pub fn l2(mut self, l2: f32) -> Self {
        self.l2 = l2;
        self
    }

    /// Set both L1 and L2 regularization
    pub fn regularization(mut self, l1: f32, l2: f32) -> Self {
        self.l1 = l1;
        self.l2 = l2;
        self
    }

    /// Preset for sparse features (high L1, low L2)
    pub fn sparse_preset(mut self) -> Self {
        self.l1 = 0.1;
        self.l2 = 0.01;
        self.alpha = 0.1;
        self.beta = 1.0;
        self
    }

    /// Preset for dense features (low L1, moderate L2)
    pub fn dense_preset(mut self) -> Self {
        self.l1 = 0.01;
        self.l2 = 0.1;
        self.alpha = 1.0;
        self.beta = 1.0;
        self
    }

    pub fn build(self) -> FTRL {
        FTRL::new(
            self.params,
            Some(self.alpha),
            Some(self.beta),
            Some(self.l1),
            Some(self.l2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::{randn, tensor_1d};

    #[test]
    fn test_ftrl_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let params = vec![param];

        let optimizer = FTRL::new(params, Some(1.0), Some(1.0), Some(0.1), Some(0.01));

        assert_eq!(optimizer.alpha, 1.0);
        assert_eq!(optimizer.beta, 1.0);
        assert_eq!(optimizer.l1, 0.1);
        assert_eq!(optimizer.l2, 0.01);
        Ok(())
    }

    #[test]
    fn test_ftrl_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let params = vec![param];

        let optimizer = FTRL::builder()
            .params(params)
            .alpha(0.5)
            .beta(2.0)
            .l1(0.05)
            .l2(0.1)
            .build();

        assert_eq!(optimizer.alpha, 0.5);
        assert_eq!(optimizer.beta, 2.0);
        assert_eq!(optimizer.l1, 0.05);
        assert_eq!(optimizer.l2, 0.1);
        Ok(())
    }

    #[test]
    fn test_ftrl_presets() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3])?));
        let params = vec![param];

        let sparse_optimizer = FTRL::builder()
            .params(params.clone())
            .sparse_preset()
            .build();

        assert_eq!(sparse_optimizer.l1, 0.1);
        assert_eq!(sparse_optimizer.l2, 0.01);
        assert_eq!(sparse_optimizer.alpha, 0.1);

        let dense_optimizer = FTRL::builder().params(params).dense_preset().build();

        assert_eq!(dense_optimizer.l1, 0.01);
        assert_eq!(dense_optimizer.l2, 0.1);
        assert_eq!(dense_optimizer.alpha, 1.0);
        Ok(())
    }

    #[test]
    fn test_ftrl_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3])?));
        let initial_param = param.read().clone();

        // Set up gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[3, 3])?;
            p.set_grad(Some(grad));
        }

        let params = vec![param.clone()];
        let mut optimizer = FTRL::new(params, Some(0.1), Some(1.0), Some(0.01), Some(0.01));

        // Perform optimization step
        optimizer.step()?;

        // Check that parameter changed
        let final_param = param.read().clone();
        let diff = initial_param.sub(&final_param)?;
        let norm = diff.norm()?.to_vec()?[0];
        assert!(
            norm > 0.0,
            "Parameter should change after optimization step"
        );
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_learning_rate_computation() -> OptimizerResult<()> {
        let n = creation::ones(&[2, 2])?;

        let optimizer = FTRL::new(
            vec![],
            Some(1.0), // alpha
            Some(1.0), // beta
            None,
            None,
        );

        let lr = optimizer.compute_learning_rate(&n)?;

        // η = α / (β + √n) = 1.0 / (1.0 + √1.0) = 1.0 / 2.0 = 0.5
        let expected = 0.5;
        let lr_vec = lr.to_vec()?;
        for &val in &lr_vec {
            assert_relative_eq!(val, expected, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_l1_regularization() -> OptimizerResult<()> {
        let z = tensor_1d(&[1.5, -2.0, 0.5, -0.3])?;
        let sigma = creation::ones(&[4])?;

        let optimizer = FTRL::new(
            vec![],
            None,
            None,
            Some(1.0), // l1 = 1.0
            None,
        );

        let result = optimizer.apply_l1_regularization(&z, &sigma)?;
        let result_vec = result.to_vec()?;

        // Expected: sign(z) * max(0, |z| - l1 * sigma)
        // For l1 = 1.0, sigma = 1.0:
        // z[0] = 1.5 -> sign(1.5) * max(0, 1.5 - 1.0) = 1 * 0.5 = 0.5
        // z[1] = -2.0 -> sign(-2.0) * max(0, 2.0 - 1.0) = -1 * 1.0 = -1.0
        // z[2] = 0.5 -> sign(0.5) * max(0, 0.5 - 1.0) = 1 * 0.0 = 0.0
        // z[3] = -0.3 -> sign(-0.3) * max(0, 0.3 - 1.0) = -1 * 0.0 = 0.0

        assert_relative_eq!(result_vec[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(result_vec[1], -1.0, epsilon = 1e-6);
        assert_relative_eq!(result_vec[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result_vec[3], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_ftrl_sparsity() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(creation::ones(&[5])?));

        // Set up small gradient
        {
            let mut p = param.write();
            let grad = tensor_1d(&[0.1, 0.2, 0.05, 0.3, 0.01])?;
            p.set_grad(Some(grad));
        }

        let params = vec![param.clone()];
        let mut optimizer = FTRL::builder()
            .params(params)
            .alpha(0.1)
            .beta(1.0)
            .l1(0.5) // High L1 to encourage sparsity
            .l2(0.01)
            .build();

        // Perform multiple optimization steps
        for _ in 0..10 {
            optimizer.step()?;
        }

        // Check that some parameters became sparse (close to zero)
        let final_param = param.read().clone();
        let param_vec = final_param.to_vec()?;

        // With high L1 regularization, some small parameters should become zero
        let zero_count = param_vec.iter().filter(|&&x| x.abs() < 1e-3).count();
        assert!(
            zero_count > 0,
            "FTRL should induce sparsity with L1 regularization"
        );
        Ok(())
    }
}
