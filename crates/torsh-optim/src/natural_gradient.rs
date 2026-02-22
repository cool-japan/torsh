//! Natural Gradient optimizer

use crate::{optimizer::BaseOptimizer, Optimizer, OptimizerResult, OptimizerState, ParamGroup};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::{creation::zeros_like, Tensor};

/// Natural Gradient optimizer
///
/// Natural gradients use the Fisher Information Matrix to precondition gradients,
/// providing better-conditioned updates especially for neural networks and probabilistic models.
///
/// Reference: "Natural Gradient Works Efficiently in Learning" (Amari, 1998)
pub struct NaturalGradient {
    base: BaseOptimizer,
    lr: f32,
    momentum: f32,
    damping: f32,
    fisher_update_freq: usize,
    use_empirical_fisher: bool,
    fisher_ema_decay: f32,
    step_count: usize,
}

impl NaturalGradient {
    /// Create a new Natural Gradient optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        momentum: Option<f32>,
        damping: Option<f32>,
        fisher_update_freq: Option<usize>,
        use_empirical_fisher: Option<bool>,
        fisher_ema_decay: Option<f32>,
    ) -> Self {
        let lr = lr.unwrap_or(0.03);
        let momentum = momentum.unwrap_or(0.9);
        let damping = damping.unwrap_or(0.001);
        let fisher_update_freq = fisher_update_freq.unwrap_or(1);
        let use_empirical_fisher = use_empirical_fisher.unwrap_or(true);
        let fisher_ema_decay = fisher_ema_decay.unwrap_or(0.95);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("damping".to_string(), damping);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "NaturalGradient".to_string(),
            defaults,
        };

        Self {
            base,
            lr,
            momentum,
            damping,
            fisher_update_freq,
            use_empirical_fisher,
            fisher_ema_decay,
            step_count: 0,
        }
    }

    /// Builder pattern for Natural Gradient optimizer
    pub fn builder() -> NaturalGradientBuilder {
        NaturalGradientBuilder::default()
    }

    /// Compute Fisher Information Matrix (simplified approximation)
    fn compute_fisher_matrix(&self, param: &Tensor, grad: &Tensor) -> Result<Tensor> {
        // For neural networks, we use a diagonal approximation of the Fisher matrix
        // F_ii ≈ E[g_i^2] where g_i is the gradient of log-likelihood w.r.t. parameter i

        if self.use_empirical_fisher {
            // Empirical Fisher: use squared gradients
            Ok(grad.pow(2.0)?)
        } else {
            // True Fisher would require second-order information
            // For now, we approximate with squared gradients plus some regularization
            let reg = param.mul_scalar(1e-6)?;
            let fisher_approx = grad.pow(2.0)?;
            Ok(fisher_approx.add(&reg)?)
        }
    }

    /// Apply natural gradient update
    fn apply_natural_gradient_update(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        fisher: &Tensor,
    ) -> Result<()> {
        // Natural gradient update: θ_{t+1} = θ_t - η * F^{-1} * ∇L
        // We approximate F^{-1} with element-wise division for diagonal Fisher matrix

        let damped_fisher = fisher.add_scalar(self.damping)?;
        let natural_grad = grad.div(&damped_fisher)?;
        let update = natural_grad.mul_scalar(self.lr)?;

        *param = param.sub(&update)?;
        Ok(())
    }
}

impl Optimizer for NaturalGradient {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let should_update_fisher = self.step_count % self.fisher_update_freq == 0;

        // Extract values to avoid borrowing conflicts
        let lr = self.lr;
        let momentum = self.momentum;
        let damping = self.damping;
        let fisher_ema_decay = self.fisher_ema_decay;

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

                // Extract data early to avoid borrow conflicts
                let _param_data = param.clone();
                let grad_data = grad.clone();

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    state.insert("momentum_buffer".to_string(), zeros_like(&param)?);
                    state.insert("fisher_matrix".to_string(), zeros_like(&param)?);
                    state.insert("step".to_string(), zeros_like(&param)?);
                }

                let mut momentum_buffer = state
                    .get("momentum_buffer")
                    .expect("momentum_buffer state should exist")
                    .clone();
                let mut fisher_matrix = state
                    .get("fisher_matrix")
                    .expect("fisher_matrix state should exist")
                    .clone();
                let mut step_tensor = state.get("step").expect("step state should exist").clone();

                // Increment step count
                step_tensor.add_scalar_(1.0)?;

                // Update Fisher matrix if needed
                let mut new_fisher_opt = None;
                if should_update_fisher {
                    // Temporarily drop the mutable borrow
                    drop(param);

                    // Compute Fisher matrix inline: F = g ⊗ g (outer product approximation)
                    let new_fisher = grad_data.mul_op(&grad_data)?;
                    new_fisher_opt = Some(new_fisher);
                }

                let fisher_was_updated = new_fisher_opt.is_some();
                if let Some(new_fisher) = new_fisher_opt.clone() {
                    if step_tensor.to_vec()?[0] == 1.0 {
                        // First step: initialize Fisher matrix
                        fisher_matrix = new_fisher;
                    } else {
                        // Update Fisher matrix with exponential moving average
                        fisher_matrix = fisher_matrix
                            .mul_scalar(fisher_ema_decay)?
                            .add(&new_fisher.mul_scalar(1.0 - fisher_ema_decay)?)?;
                    }
                }

                // Compute natural gradient
                let damped_fisher = fisher_matrix.add_scalar(damping)?;
                let natural_grad = grad_data.div(&damped_fisher)?;

                // Re-acquire parameter lock if it was dropped
                let mut param = if fisher_was_updated {
                    param_arc.write()
                } else {
                    // Re-acquire the lock since we dropped it earlier
                    param_arc.write()
                };

                // Apply momentum
                if momentum != 0.0 {
                    momentum_buffer = momentum_buffer.mul_scalar(momentum)?.add(&natural_grad)?;
                    let update = momentum_buffer.mul_scalar(lr)?;
                    *param = param.sub(&update)?;
                } else {
                    let update = natural_grad.mul_scalar(lr)?;
                    *param = param.sub(&update)?;
                }

                // Update state
                state.insert("momentum_buffer".to_string(), momentum_buffer);
                state.insert("fisher_matrix".to_string(), fisher_matrix);
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

/// Builder for Natural Gradient optimizer
pub struct NaturalGradientBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    momentum: f32,
    damping: f32,
    fisher_update_freq: usize,
    use_empirical_fisher: bool,
    fisher_ema_decay: f32,
}

impl Default for NaturalGradientBuilder {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            lr: 0.03,
            momentum: 0.9,
            damping: 0.001,
            fisher_update_freq: 1,
            use_empirical_fisher: true,
            fisher_ema_decay: 0.95,
        }
    }
}

impl NaturalGradientBuilder {
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

    pub fn fisher_update_freq(mut self, freq: usize) -> Self {
        self.fisher_update_freq = freq;
        self
    }

    pub fn use_empirical_fisher(mut self, use_empirical: bool) -> Self {
        self.use_empirical_fisher = use_empirical;
        self
    }

    pub fn fisher_ema_decay(mut self, decay: f32) -> Self {
        self.fisher_ema_decay = decay;
        self
    }

    pub fn build(self) -> NaturalGradient {
        NaturalGradient::new(
            self.params,
            Some(self.lr),
            Some(self.momentum),
            Some(self.damping),
            Some(self.fisher_update_freq),
            Some(self.use_empirical_fisher),
            Some(self.fisher_ema_decay),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_natural_gradient_creation() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()));
        let params = vec![param];

        let optimizer = NaturalGradient::new(
            params,
            Some(0.01),
            Some(0.9),
            Some(0.001),
            Some(1),
            Some(true),
            Some(0.95),
        );

        assert_eq!(optimizer.lr, 0.01);
        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.damping, 0.001);
    }

    #[test]
    fn test_natural_gradient_builder() {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5]).unwrap()));
        let params = vec![param];

        let optimizer = NaturalGradient::builder()
            .params(params)
            .lr(0.02)
            .momentum(0.95)
            .damping(0.01)
            .fisher_update_freq(5)
            .use_empirical_fisher(false)
            .fisher_ema_decay(0.9)
            .build();

        assert_eq!(optimizer.lr, 0.02);
        assert_eq!(optimizer.momentum, 0.95);
        assert_eq!(optimizer.damping, 0.01);
        assert_eq!(optimizer.fisher_update_freq, 5);
        assert!(!optimizer.use_empirical_fisher);
        assert_eq!(optimizer.fisher_ema_decay, 0.9);
    }

    #[test]
    fn test_natural_gradient_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3]).unwrap()));
        let initial_param = param.read().clone();

        // Set up gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[3, 3]).unwrap();
            p.set_grad(Some(grad));
        }

        let params = vec![param.clone()];
        let mut optimizer = NaturalGradient::new(params, Some(0.01), None, None, None, None, None);

        // Perform optimization step
        optimizer.step().unwrap();

        // Check that parameter changed
        let final_param = param.read().clone();
        let diff = initial_param.sub(&final_param).unwrap();
        let norm = diff.norm()?.to_vec()?[0];
        assert!(
            norm > 0.0,
            "Parameter should change after optimization step"
        );
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_fisher_matrix_computation() -> OptimizerResult<()> {
        let param = randn::<f32>(&[2, 2]).unwrap();
        let grad = randn::<f32>(&[2, 2]).unwrap();

        let optimizer =
            NaturalGradient::new(vec![], Some(0.01), None, None, None, Some(true), None);

        let fisher = optimizer.compute_fisher_matrix(&param, &grad).unwrap();

        // Fisher matrix should be positive (squared gradients)
        let fisher_vec = fisher.to_vec()?;
        for &val in &fisher_vec {
            assert!(val >= 0.0, "Fisher matrix elements should be non-negative");
        }

        Ok(())
    }
}
