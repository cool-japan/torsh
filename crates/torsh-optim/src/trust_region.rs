//! Trust Region optimization methods
//!
//! This module provides trust region optimization algorithms including
//! Trust Region Newton, Trust Region with CG, and general trust region frameworks.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Trust region update strategy
#[derive(Clone, Copy, Debug)]
pub enum TrustRegionStrategy {
    /// Standard trust region with reduction ratio
    Standard,
    /// Adaptive trust region based on gradient norms
    Adaptive,
    /// Conservative trust region with smaller expansion
    Conservative,
    /// Aggressive trust region with larger expansion
    Aggressive,
}

/// Trust region configuration
#[derive(Clone)]
pub struct TrustRegionConfig {
    /// Initial trust region radius
    pub initial_radius: f32,
    /// Maximum trust region radius
    pub max_radius: f32,
    /// Minimum trust region radius
    pub min_radius: f32,
    /// Acceptance threshold for reduction ratio
    pub eta1: f32,
    /// Expansion threshold for reduction ratio
    pub eta2: f32,
    /// Radius reduction factor
    pub gamma1: f32,
    /// Radius expansion factor
    pub gamma2: f32,
    /// Trust region strategy
    pub strategy: TrustRegionStrategy,
    /// Maximum number of iterations per step
    pub max_iter: usize,
    /// Tolerance for gradient norm
    pub tolerance_grad: f32,
    /// Tolerance for step size
    pub tolerance_step: f32,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            min_radius: 1e-6,
            eta1: 0.25,
            eta2: 0.75,
            gamma1: 0.25,
            gamma2: 2.0,
            strategy: TrustRegionStrategy::Standard,
            max_iter: 100,
            tolerance_grad: 1e-6,
            tolerance_step: 1e-8,
        }
    }
}

/// Trust region subproblem solver
#[derive(Clone, Copy, Debug)]
pub enum SubproblemSolver {
    /// Cauchy point (steepest descent)
    CauchyPoint,
    /// Dogleg method
    Dogleg,
    /// Conjugate Gradient (Steihaug-Toint)
    ConjugateGradient,
    /// Two-dimensional subspace minimization
    TwoDSubspace,
}

/// Trust region method implementation
pub struct TrustRegionMethod {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    config: TrustRegionConfig,
    solver: SubproblemSolver,
}

impl TrustRegionMethod {
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        config: Option<TrustRegionConfig>,
        solver: Option<SubproblemSolver>,
    ) -> Self {
        let lr = lr.unwrap_or(1.0);
        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            config: config.unwrap_or_default(),
            solver: solver.unwrap_or(SubproblemSolver::Dogleg),
        }
    }

    pub fn builder() -> TrustRegionBuilder {
        TrustRegionBuilder::new()
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
                        "trust_region_step",
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

    /// Compute Cauchy point (steepest descent direction projected onto trust region)
    fn cauchy_point(&self, grad: &Tensor, radius: f32) -> Result<Tensor> {
        let grad_norm = grad.norm()?.item()?;

        if grad_norm < self.config.tolerance_grad {
            return Ok(Tensor::zeros(grad.shape().dims(), grad.device())?);
        }

        // Cauchy point: -τ * Δ / ||g|| * g, where τ is chosen to satisfy ||s|| ≤ Δ
        let tau = (radius / grad_norm).min(1.0);
        Ok(grad.mul_scalar(-tau * radius / grad_norm)?)
    }

    /// Approximate Hessian using gradient differences (simplified)
    fn approximate_hessian(&self, grad: &Tensor) -> Result<Tensor> {
        // Simplified Hessian approximation: H ≈ ||g|| * I
        // In practice, you would use L-BFGS, finite differences, or exact Hessian
        let grad_norm = grad.norm()?.item()?;
        let scale = if grad_norm > 1e-8 { grad_norm } else { 1.0 };

        // Return diagonal approximation as a vector (diagonal elements)
        let n = grad.shape().numel();
        Ok(Tensor::from_data(vec![scale; n], vec![n], grad.device())?)
    }

    /// Solve trust region subproblem using dogleg method
    fn dogleg(&self, grad: &Tensor, hessian_diag: &Tensor, radius: f32) -> Result<Tensor> {
        let grad_norm = grad.norm()?.item()?;

        if grad_norm < self.config.tolerance_grad {
            return Ok(Tensor::zeros(grad.shape().dims(), grad.device())?);
        }

        // Cauchy point
        let cauchy_step = self.cauchy_point(grad, radius)?;
        let cauchy_norm = cauchy_step.norm()?.item()?;

        // If Cauchy point is on the boundary, return it
        if cauchy_norm >= radius - self.config.tolerance_step {
            return Ok(cauchy_step);
        }

        // Newton step (simplified): -H^(-1) * g
        // For diagonal Hessian: -g[i] / H[i][i]
        let hessian_data = hessian_diag.data()?;
        let grad_data = grad.data()?;
        let newton_data: Vec<f32> = grad_data
            .iter()
            .zip(hessian_data.iter())
            .map(|(g, h)| if h.abs() > 1e-12 { -g / h } else { -g })
            .collect();

        let newton_step =
            Tensor::from_data(newton_data, grad.shape().dims().to_vec(), grad.device())?;
        let newton_norm = newton_step.norm()?.item()?;

        // If Newton step is within trust region, return it
        if newton_norm <= radius {
            return Ok(newton_step);
        }

        // Dogleg path: find intersection with trust region boundary
        // s(τ) = τ * cauchy_step + (1-τ) * newton_step for τ ∈ [0, 1]
        // Solve ||s(τ)|| = radius

        let diff = newton_step.sub(&cauchy_step)?;
        let a = diff.dot(&diff)?.item()?;
        let b = 2.0 * cauchy_step.dot(&diff)?.item()?;
        let c = cauchy_norm * cauchy_norm - radius * radius;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Ok(cauchy_step);
        }

        let tau = (-b + discriminant.sqrt()) / (2.0 * a);
        let tau = tau.clamp(0.0, 1.0);

        // s = τ * cauchy + (1-τ) * newton
        let result = cauchy_step
            .mul_scalar(tau)?
            .add(&newton_step.mul_scalar(1.0 - tau)?)?;
        Ok(result)
    }

    /// Solve trust region subproblem using conjugate gradient (Steihaug-Toint)
    fn conjugate_gradient_tr(
        &self,
        grad: &Tensor,
        hessian_diag: &Tensor,
        radius: f32,
    ) -> Result<Tensor> {
        let n = grad.shape().numel();
        let tolerance = self.config.tolerance_grad;
        let max_iter = n.min(50); // Limit CG iterations

        // Initialize
        let mut x = Tensor::zeros(&[n], grad.device())?;
        let mut r = grad.neg()?; // -g
        let mut p = r.clone();
        let mut rsold = r.dot(&r)?.item()?;

        for _i in 0..max_iter {
            if rsold.sqrt() < tolerance {
                break;
            }

            // Approximate Hessian-vector product using diagonal approximation
            let hp_data: Vec<f32> = {
                let hessian_data = hessian_diag.data()?;
                let p_data = p.data()?;
                p_data
                    .iter()
                    .zip(hessian_data.iter())
                    .map(|(p_val, h_val)| p_val * h_val)
                    .collect()
            };
            let hp = Tensor::from_data(hp_data, grad.shape().dims().to_vec(), grad.device())?;

            let pap = p.dot(&hp)?.item()?;

            // Check for negative curvature
            if pap <= 0.0 {
                // Find boundary of trust region along direction p
                let x_norm_sq = x.dot(&x)?.item()?;
                let xp = x.dot(&p)?.item()?;
                let p_norm_sq = p.dot(&p)?.item()?;

                let discriminant = xp * xp + p_norm_sq * (radius * radius - x_norm_sq);
                if discriminant >= 0.0 {
                    let alpha = (-xp + discriminant.sqrt()) / p_norm_sq;
                    return Ok(x.add(&p.mul_scalar(alpha)?)?);
                } else {
                    return Ok(x);
                }
            }

            let alpha = rsold / pap;
            let x_new = x.add(&p.mul_scalar(alpha)?)?;

            // Check trust region constraint
            let x_norm = x_new.norm()?.item()?;
            if x_norm >= radius {
                // Find intersection with trust region boundary
                let x_norm_old_sq = x.dot(&x)?.item()?;
                let xp = x.dot(&p)?.item()?;
                let p_norm_sq = p.dot(&p)?.item()?;

                let discriminant = xp * xp + p_norm_sq * (radius * radius - x_norm_old_sq);
                if discriminant >= 0.0 {
                    let tau = (-xp + discriminant.sqrt()) / p_norm_sq;
                    return Ok(x.add(&p.mul_scalar(tau)?)?);
                } else {
                    return Ok(x);
                }
            }

            x = x_new;
            let r_new = r.sub(&hp.mul_scalar(alpha)?)?;
            let rsnew = r_new.dot(&r_new)?.item()?;

            if rsnew.sqrt() < tolerance {
                break;
            }

            let beta = rsnew / rsold;
            p = r_new.add(&p.mul_scalar(beta)?)?;
            r = r_new;
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Solve trust region subproblem
    fn solve_subproblem(&self, grad: &Tensor, radius: f32) -> Result<Tensor> {
        match self.solver {
            SubproblemSolver::CauchyPoint => self.cauchy_point(grad, radius),
            SubproblemSolver::Dogleg => {
                let hessian_diag = self.approximate_hessian(grad)?;
                self.dogleg(grad, &hessian_diag, radius)
            }
            SubproblemSolver::ConjugateGradient => {
                let hessian_diag = self.approximate_hessian(grad)?;
                self.conjugate_gradient_tr(grad, &hessian_diag, radius)
            }
            SubproblemSolver::TwoDSubspace => {
                // Simplified: fall back to dogleg
                let hessian_diag = self.approximate_hessian(grad)?;
                self.dogleg(grad, &hessian_diag, radius)
            }
        }
    }

    /// Compute model decrease (simplified)
    fn model_decrease(&self, grad: &Tensor, step: &Tensor) -> Result<f32> {
        // m(0) - m(s) ≈ -g^T s - 0.5 s^T H s
        // Simplified: just use -g^T s
        let decrease = -grad.dot(step)?.item()?;
        Ok(decrease.max(0.0))
    }

    /// Compute actual function decrease (placeholder)
    fn actual_decrease(&self, _old_params: &Tensor, _new_params: &Tensor) -> Result<f32> {
        // This would evaluate the actual function at both points
        // For now, return a placeholder value
        Ok(0.5) // Placeholder
    }

    /// Update trust region radius
    fn update_radius(&self, current_radius: f32, reduction_ratio: f32) -> f32 {
        let config = &self.config;

        match config.strategy {
            TrustRegionStrategy::Standard => {
                if reduction_ratio < config.eta1 {
                    (current_radius * config.gamma1).max(config.min_radius)
                } else if reduction_ratio > config.eta2 {
                    (current_radius * config.gamma2).min(config.max_radius)
                } else {
                    current_radius
                }
            }
            TrustRegionStrategy::Adaptive => {
                // More aggressive adaptation based on reduction ratio
                if reduction_ratio < 0.1 {
                    (current_radius * 0.1).max(config.min_radius)
                } else if reduction_ratio > 0.9 {
                    (current_radius * 3.0).min(config.max_radius)
                } else {
                    current_radius * (0.5 + reduction_ratio)
                }
            }
            TrustRegionStrategy::Conservative => {
                // Smaller changes to radius
                if reduction_ratio < config.eta1 {
                    (current_radius * 0.5).max(config.min_radius)
                } else if reduction_ratio > config.eta2 {
                    (current_radius * 1.5).min(config.max_radius)
                } else {
                    current_radius
                }
            }
            TrustRegionStrategy::Aggressive => {
                // Larger changes to radius
                if reduction_ratio < config.eta1 {
                    (current_radius * 0.1).max(config.min_radius)
                } else if reduction_ratio > 0.5 {
                    (current_radius * 4.0).min(config.max_radius)
                } else {
                    current_radius
                }
            }
        }
    }
}

impl Optimizer for TrustRegionMethod {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        // Get current parameters and gradients
        let current_params = self.flatten_params()?;
        let current_grad = self.flatten_grads()?;

        // Check convergence
        let grad_norm = current_grad.norm()?.item()?;
        if grad_norm < self.config.tolerance_grad {
            return Ok(());
        }

        // Get or initialize trust region radius
        let state_id = "trust_region_state".to_string();
        let mut radius = {
            let param_state = self.state.entry(state_id.clone()).or_default();
            if let Some(radius_tensor) = param_state.get("radius") {
                radius_tensor.item()?
            } else {
                let initial_radius = self.config.initial_radius;
                param_state.insert("radius".to_string(), Tensor::scalar(initial_radius)?);
                initial_radius
            }
        };

        // Trust region iteration
        for _iter in 0..self.config.max_iter {
            // Solve trust region subproblem
            let step = self.solve_subproblem(&current_grad, radius)?;
            let step_norm = step.norm()?.item()?;

            // Check if step is too small
            if step_norm < self.config.tolerance_step {
                break;
            }

            // Compute new parameters
            let new_params = current_params.add(&step)?;

            // Compute reduction ratio
            let model_dec = self.model_decrease(&current_grad, &step)?;
            let actual_dec = self.actual_decrease(&current_params, &new_params)?;

            let reduction_ratio = if model_dec > 1e-12 {
                actual_dec / model_dec
            } else {
                0.0
            };

            // Accept or reject step
            if reduction_ratio > self.config.eta1 {
                // Accept step
                self.update_params_from_flat(&new_params)?;
            }

            // Update trust region radius
            radius = self.update_radius(radius, reduction_ratio);

            // Store updated radius
            let param_state = self.state.get_mut(&state_id).unwrap();
            param_state.insert("radius".to_string(), Tensor::scalar(radius)?);

            // Break if step was accepted and radius is reasonable
            if reduction_ratio > self.config.eta1 {
                break;
            }

            // Break if radius becomes too small
            if radius < self.config.min_radius {
                break;
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
            optimizer_type: "TrustRegion".to_string(),
            version: "0.1.0".to_string(),
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

/// Builder for trust region optimizers
pub struct TrustRegionBuilder {
    lr: f32,
    config: TrustRegionConfig,
    solver: SubproblemSolver,
}

impl TrustRegionBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1.0,
            config: TrustRegionConfig::default(),
            solver: SubproblemSolver::Dogleg,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn initial_radius(mut self, radius: f32) -> Self {
        self.config.initial_radius = radius;
        self
    }

    pub fn max_radius(mut self, radius: f32) -> Self {
        self.config.max_radius = radius;
        self
    }

    pub fn min_radius(mut self, radius: f32) -> Self {
        self.config.min_radius = radius;
        self
    }

    pub fn strategy(mut self, strategy: TrustRegionStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn solver(mut self, solver: SubproblemSolver) -> Self {
        self.solver = solver;
        self
    }

    pub fn tolerance_grad(mut self, tol: f32) -> Self {
        self.config.tolerance_grad = tol;
        self
    }

    pub fn tolerance_step(mut self, tol: f32) -> Self {
        self.config.tolerance_step = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> TrustRegionMethod {
        TrustRegionMethod::new(params, Some(self.lr), Some(self.config), Some(self.solver))
    }
}

impl Default for TrustRegionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_trust_region_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = TrustRegionMethod::new(params, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 1.0);
        Ok(())
    }

    #[test]
    fn test_trust_region_builder() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = TrustRegionMethod::builder()
            .lr(0.1)
            .initial_radius(0.5)
            .strategy(TrustRegionStrategy::Adaptive)
            .solver(SubproblemSolver::ConjugateGradient)
            .build(params);

        assert_eq!(optimizer.get_lr()[0], 0.1);
        assert_eq!(optimizer.config.initial_radius, 0.5);
        assert!(matches!(
            optimizer.config.strategy,
            TrustRegionStrategy::Adaptive
        ));
        assert!(matches!(
            optimizer.solver,
            SubproblemSolver::ConjugateGradient
        ));
        Ok(())
    }

    #[test]
    fn test_cauchy_point() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = TrustRegionMethod::new(params, None, None, None);

        let grad = Tensor::from_data(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![4],
            torsh_core::device::DeviceType::Cpu,
        )?;
        let radius = 1.0;

        let cauchy = optimizer.cauchy_point(&grad, radius)?;
        let cauchy_norm = cauchy.norm()?.item()?;

        // Cauchy point should be within trust region
        assert!(cauchy_norm <= radius + 1e-6);
        Ok(())
    }

    #[test]
    fn test_trust_region_config() {
        let config = TrustRegionConfig {
            initial_radius: 2.0,
            max_radius: 50.0,
            min_radius: 1e-5,
            eta1: 0.1,
            eta2: 0.9,
            gamma1: 0.1,
            gamma2: 3.0,
            strategy: TrustRegionStrategy::Aggressive,
            max_iter: 50,
            tolerance_grad: 1e-8,
            tolerance_step: 1e-10,
        };

        assert_eq!(config.initial_radius, 2.0);
        assert_eq!(config.eta1, 0.1);
        assert_eq!(config.eta2, 0.9);
        assert!(matches!(config.strategy, TrustRegionStrategy::Aggressive));
    }
}
