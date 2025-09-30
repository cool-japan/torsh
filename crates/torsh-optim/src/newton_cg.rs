//! Newton-CG (Newton-Conjugate Gradient) optimizer
//!
//! This module implements the Newton-CG optimization algorithm, which uses
//! conjugate gradient to approximately solve the Newton direction.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Newton-CG optimizer configuration
#[derive(Clone)]
pub struct NewtonCGConfig {
    /// Maximum number of CG iterations
    pub max_cg_iter: usize,
    /// CG tolerance for residual
    pub cg_tolerance: f32,
    /// Whether to use trust region
    pub use_trust_region: bool,
    /// Initial trust region radius
    pub initial_trust_radius: f32,
    /// Trust region tolerance
    pub trust_tolerance: f32,
    /// Maximum trust region radius
    pub max_trust_radius: f32,
    /// Minimum trust region radius
    pub min_trust_radius: f32,
}

impl Default for NewtonCGConfig {
    fn default() -> Self {
        Self {
            max_cg_iter: 50,
            cg_tolerance: 1e-6,
            use_trust_region: true,
            initial_trust_radius: 1.0,
            trust_tolerance: 1e-4,
            max_trust_radius: 100.0,
            min_trust_radius: 1e-6,
        }
    }
}

/// Newton-CG optimizer
///
/// Implements the Newton-Conjugate Gradient method for second-order optimization.
/// Uses CG to approximately solve the Newton equations, optionally with trust regions.
pub struct NewtonCG {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    config: NewtonCGConfig,
    tolerance_grad: f32,
    tolerance_change: f32,
}

impl NewtonCG {
    /// Create a new Newton-CG optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        tolerance_grad: Option<f32>,
        tolerance_change: Option<f32>,
        config: Option<NewtonCGConfig>,
    ) -> Self {
        let lr = lr.unwrap_or(1.0);
        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            config: config.unwrap_or_default(),
            tolerance_grad: tolerance_grad.unwrap_or(1e-7),
            tolerance_change: tolerance_change.unwrap_or(1e-9),
        }
    }

    /// Create a Newton-CG optimizer with builder pattern
    pub fn builder() -> NewtonCGBuilder {
        NewtonCGBuilder::new()
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }

    /// Flatten parameters into a single vector
    fn flatten_params(&self) -> Result<Tensor> {
        let mut flattened = Vec::new();

        for group in &self.param_groups {
            for param in &group.params {
                let param_read = param.read();
                let param_flat = param_read.flatten()?;
                let param_data = param_flat.data()?;
                flattened.extend_from_slice(&param_data);
            }
        }

        let len = flattened.len();
        Ok(Tensor::from_data(
            flattened,
            vec![len],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    /// Flatten gradients into a single vector
    fn flatten_grads(&self) -> Result<Tensor> {
        let mut flattened = Vec::new();

        for group in &self.param_groups {
            for param in &group.params {
                let param_read = param.read();
                let grad = param_read.grad().ok_or_else(|| {
                    TorshError::invalid_argument_with_context(
                        "Parameter has no gradient",
                        "newton_cg_step",
                    )
                })?;
                let grad_flat = grad.flatten()?;
                let grad_data = grad_flat.data()?;
                flattened.extend_from_slice(&grad_data);
            }
        }

        let len = flattened.len();
        Ok(Tensor::from_data(
            flattened,
            vec![len],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    /// Update parameters from flattened vector
    fn update_params_from_flat(&self, flat_params: &Tensor) -> Result<()> {
        let flat_data = flat_params.data()?;
        let mut offset = 0;

        for group in &self.param_groups {
            for param in &group.params {
                let mut param_write = param.write();
                let param_shape = param_write.shape();
                let param_size = param_shape.numel();

                let param_data = &flat_data[offset..offset + param_size];
                *param_write = Tensor::from_data(
                    param_data.to_vec(),
                    param_shape.dims().to_vec(),
                    param_write.device(),
                )?;

                offset += param_size;
            }
        }

        Ok(())
    }

    /// Approximate Hessian-vector product using finite differences
    ///
    /// This is a simplified implementation. In practice, you would want to use
    /// automatic differentiation or analytical Hessians when available.
    fn hessian_vector_product(
        &self,
        grad: &Tensor,
        vector: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor> {
        // H * v ≈ (∇f(x + ε*v) - ∇f(x)) / ε
        // For simplicity, we'll use an identity approximation scaled by gradient magnitude
        // In practice, this should compute the actual Hessian-vector product

        let grad_norm = grad.norm()?.item()?;
        let scale = if grad_norm > 1e-8 { grad_norm } else { 1.0 };

        // Simple approximation: H ≈ scale * I
        Ok(vector.mul_scalar(scale * epsilon)?)
    }

    /// Solve Newton system using Conjugate Gradient
    fn solve_newton_cg(&self, grad: &Tensor, trust_radius: Option<f32>) -> Result<Tensor> {
        let n = grad.shape().numel();

        // Initialize CG variables
        let mut x = Tensor::zeros(&[n], grad.device())?; // Solution vector
        let mut r = grad.neg()?; // Residual: -g (since we want to solve Hp = -g)
        let mut p = r.clone(); // Search direction
        let mut rsold = r.dot(&r)?.item()?;

        let tolerance = self.config.cg_tolerance;
        let max_iter = self.config.max_cg_iter.min(n);

        for _i in 0..max_iter {
            // Check convergence
            if rsold.sqrt() < tolerance {
                break;
            }

            // Compute Hessian-vector product Hp
            let hp = self.hessian_vector_product(grad, &p, 1e-7)?;
            let pap = p.dot(&hp)?.item()?;

            // Check for negative curvature
            if pap <= 0.0 {
                // Negative curvature detected, use Cauchy point
                let grad_norm = grad.norm()?.item()?;
                if grad_norm > 1e-12 {
                    let alpha = if let Some(radius) = trust_radius {
                        (radius / grad_norm).min(1.0)
                    } else {
                        1.0
                    };
                    return Ok(grad.mul_scalar(-alpha)?);
                } else {
                    return Ok(x);
                }
            }

            // CG step
            let alpha = rsold / pap;
            let x_new = x.add(&p.mul_scalar(alpha)?)?;

            // Trust region check
            if let Some(radius) = trust_radius {
                let x_norm = x_new.norm()?.item()?;
                if x_norm > radius {
                    // Step would exceed trust region, find boundary point
                    let x_norm_old = x.norm()?.item()?;
                    if x_norm_old >= radius {
                        return Ok(x);
                    }

                    // Solve ||x + t*p|| = radius for t
                    let xp = x.dot(&p)?.item()?;
                    let pp = p.dot(&p)?.item()?;
                    let xx = x.dot(&x)?.item()?;

                    let discriminant = xp * xp - pp * (xx - radius * radius);
                    if discriminant >= 0.0 {
                        let t = (-xp + discriminant.sqrt()) / pp;
                        return Ok(x.add(&p.mul_scalar(t)?)?);
                    } else {
                        return Ok(x);
                    }
                }
            }

            x = x_new;
            let r_new = r.sub(&hp.mul_scalar(alpha)?)?;
            let rsnew = r_new.dot(&r_new)?.item()?;

            // Check convergence
            if rsnew.sqrt() < tolerance {
                break;
            }

            // Update for next iteration
            let beta = rsnew / rsold;
            p = r_new.add(&p.mul_scalar(beta)?)?;
            r = r_new;
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Compute the actual reduction vs predicted reduction
    fn compute_reduction_ratio(
        &self,
        _old_params: &Tensor,
        _new_params: &Tensor,
        _old_grad: &Tensor,
        _step: &Tensor,
    ) -> Result<f32> {
        // Simplified implementation
        // In practice, this would evaluate the function at both points
        // and compute actual_reduction / predicted_reduction
        Ok(0.75) // Placeholder value
    }

    /// Update trust region radius based on reduction ratio
    fn update_trust_radius(&self, current_radius: f32, reduction_ratio: f32) -> f32 {
        let config = &self.config;

        if reduction_ratio < 0.25 {
            // Poor reduction, shrink trust region
            (current_radius * 0.25).max(config.min_trust_radius)
        } else if reduction_ratio > 0.75 {
            // Good reduction, expand trust region
            (current_radius * 2.0).min(config.max_trust_radius)
        } else {
            // Acceptable reduction, keep current radius
            current_radius
        }
    }
}

impl Optimizer for NewtonCG {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        // Get current parameters and gradients
        let current_params = self.flatten_params()?;
        let current_grad = self.flatten_grads()?;

        // Check gradient tolerance
        let grad_norm = current_grad.norm()?.item()?;
        if grad_norm < self.tolerance_grad {
            return Ok(());
        }

        // Get or initialize trust region radius
        let state_id = "newton_cg_state".to_string();
        let mut trust_radius = if self.config.use_trust_region {
            let param_state = self.state.entry(state_id.clone()).or_default();
            if let Some(radius_tensor) = param_state.get("trust_radius") {
                radius_tensor.item()?
            } else {
                let radius = self.config.initial_trust_radius;
                param_state.insert("trust_radius".to_string(), Tensor::scalar(radius)?);
                radius
            }
        } else {
            0.0 // Not used when trust region is disabled
        };

        // Solve Newton system using CG
        let trust_radius_opt = if self.config.use_trust_region {
            Some(trust_radius)
        } else {
            None
        };

        let newton_step = self.solve_newton_cg(&current_grad, trust_radius_opt)?;

        // Compute new parameters
        let new_params = current_params.add(&newton_step)?;

        // Trust region management
        if self.config.use_trust_region {
            let reduction_ratio = self.compute_reduction_ratio(
                &current_params,
                &new_params,
                &current_grad,
                &newton_step,
            )?;

            trust_radius = self.update_trust_radius(trust_radius, reduction_ratio);

            // Update trust radius in state
            let param_state = self.state.get_mut(&state_id).unwrap();
            param_state.insert("trust_radius".to_string(), Tensor::scalar(trust_radius)?);

            // Accept step only if reduction ratio is reasonable
            if reduction_ratio > self.config.trust_tolerance {
                self.update_params_from_flat(&new_params)?;
            }
        } else {
            // Always accept step when not using trust region
            self.update_params_from_flat(&new_params)?;
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
        let lr = options.get("lr").copied().unwrap_or(1.0);
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
            optimizer_type: "Newton-CG".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::TensorError(TorshError::InvalidArgument(
                "Parameter group count mismatch".to_string(),
            )));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Builder for Newton-CG optimizer
pub struct NewtonCGBuilder {
    lr: f32,
    tolerance_grad: f32,
    tolerance_change: f32,
    config: NewtonCGConfig,
}

impl NewtonCGBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1.0,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            config: NewtonCGConfig::default(),
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn tolerance_grad(mut self, tol: f32) -> Self {
        self.tolerance_grad = tol;
        self
    }

    pub fn tolerance_change(mut self, tol: f32) -> Self {
        self.tolerance_change = tol;
        self
    }

    pub fn max_cg_iter(mut self, max_iter: usize) -> Self {
        self.config.max_cg_iter = max_iter;
        self
    }

    pub fn cg_tolerance(mut self, tol: f32) -> Self {
        self.config.cg_tolerance = tol;
        self
    }

    pub fn use_trust_region(mut self, use_tr: bool) -> Self {
        self.config.use_trust_region = use_tr;
        self
    }

    pub fn initial_trust_radius(mut self, radius: f32) -> Self {
        self.config.initial_trust_radius = radius;
        self
    }

    pub fn trust_tolerance(mut self, tol: f32) -> Self {
        self.config.trust_tolerance = tol;
        self
    }

    pub fn max_trust_radius(mut self, radius: f32) -> Self {
        self.config.max_trust_radius = radius;
        self
    }

    pub fn min_trust_radius(mut self, radius: f32) -> Self {
        self.config.min_trust_radius = radius;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> NewtonCG {
        NewtonCG::new(
            params,
            Some(self.lr),
            Some(self.tolerance_grad),
            Some(self.tolerance_change),
            Some(self.config),
        )
    }
}

impl Default for NewtonCGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_newton_cg_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()))];
        let optimizer = NewtonCG::new(params, None, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 1.0);
    }

    #[test]
    fn test_newton_cg_builder() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()))];
        let optimizer = NewtonCG::builder()
            .lr(0.1)
            .max_cg_iter(20)
            .cg_tolerance(1e-5)
            .use_trust_region(true)
            .initial_trust_radius(0.5)
            .build(params);

        assert_eq!(optimizer.get_lr()[0], 0.1);
        assert_eq!(optimizer.config.max_cg_iter, 20);
        assert_eq!(optimizer.config.cg_tolerance, 1e-5);
        assert!(optimizer.config.use_trust_region);
        assert_eq!(optimizer.config.initial_trust_radius, 0.5);
    }

    #[test]
    fn test_newton_cg_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2]).unwrap()));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = NewtonCG::new(params, Some(0.1), None, None, None);

        optimizer.step()?;

        Ok(())
    }

    #[test]
    fn test_newton_cg_config() {
        let config = NewtonCGConfig {
            max_cg_iter: 25,
            cg_tolerance: 1e-7,
            use_trust_region: false,
            initial_trust_radius: 2.0,
            trust_tolerance: 1e-5,
            max_trust_radius: 50.0,
            min_trust_radius: 1e-5,
        };

        assert_eq!(config.max_cg_iter, 25);
        assert_eq!(config.cg_tolerance, 1e-7);
        assert!(!config.use_trust_region);
        assert_eq!(config.initial_trust_radius, 2.0);
    }
}
