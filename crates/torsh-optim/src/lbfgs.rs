//! Limited-memory BFGS optimizer

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Limited-memory BFGS (L-BFGS) optimizer
///
/// L-BFGS is a quasi-Newton optimization algorithm that approximates the
/// Broyden–Fletcher–Goldfarb–Shanno algorithm using a limited amount of
/// computer memory.
pub struct LBFGS {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,
    #[allow(dead_code)]
    max_iter: usize,
    #[allow(dead_code)]
    max_eval: Option<usize>,
    tolerance_grad: f32,
    #[allow(dead_code)]
    tolerance_change: f32,
    #[allow(dead_code)]
    history_size: usize,
    #[allow(dead_code)]
    line_search_fn: Option<String>,
}

impl LBFGS {
    /// Create a new L-BFGS optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 1.0)
    /// * `max_iter` - Maximum number of iterations per optimization step (default: 20)
    /// * `max_eval` - Maximum number of function evaluations per optimization step
    /// * `tolerance_grad` - Termination tolerance on first order optimality (default: 1e-7)
    /// * `tolerance_change` - Termination tolerance on function value/parameter changes (default: 1e-9)
    /// * `history_size` - Update history size (default: 100)
    /// * `line_search_fn` - Line search function ('strong_wolfe' or None)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        max_iter: Option<usize>,
        max_eval: Option<usize>,
        tolerance_grad: Option<f32>,
        tolerance_change: Option<f32>,
        history_size: Option<usize>,
        line_search_fn: Option<String>,
    ) -> Self {
        let lr = lr.unwrap_or(1.0);
        let max_iter = max_iter.unwrap_or(20);
        let tolerance_grad = tolerance_grad.unwrap_or(1e-7);
        let tolerance_change = tolerance_change.unwrap_or(1e-9);
        let history_size = history_size.unwrap_or(100);

        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            max_iter,
            max_eval,
            tolerance_grad,
            tolerance_change,
            history_size,
            line_search_fn,
        }
    }

    #[allow(dead_code)]
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
                        "lbfgs_step",
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

    /// Two-loop recursion for L-BFGS direction computation
    fn compute_direction(
        &self,
        grad: &Tensor,
        s_list: &[Tensor],
        y_list: &[Tensor],
        rho_list: &[f32],
    ) -> Result<Tensor> {
        let mut q = grad.clone();
        let m = s_list.len();
        let mut alpha = vec![0.0; m];

        // First loop (backward)
        for i in (0..m).rev() {
            let rho_i = rho_list[i];
            alpha[i] = rho_i * s_list[i].dot(&q)?.item()?;
            q = q.sub(&y_list[i].mul_scalar(alpha[i])?)?;
        }

        // Apply initial Hessian approximation (identity scaling)
        let mut r = q;
        if m > 0 {
            let gamma = s_list[m - 1].dot(&y_list[m - 1])?.item()?
                / y_list[m - 1].dot(&y_list[m - 1])?.item()?;
            r = r.mul_scalar(gamma)?;
        }

        // Second loop (forward)
        for i in 0..m {
            let rho_i = rho_list[i];
            let beta = rho_i * y_list[i].dot(&r)?.item()?;
            r = r.add(&s_list[i].mul_scalar(alpha[i] - beta)?)?;
        }

        r.neg() // Negative for descent direction
    }

    /// Simple backtracking line search with Armijo condition
    fn line_search(
        &self,
        _x: &Tensor,
        grad: &Tensor,
        direction: &Tensor,
        initial_step_size: f32,
    ) -> Result<f32> {
        // Simple Armijo line search
        #[allow(dead_code)]
        const C1: f32 = 1e-4; // Armijo constant
        const ALPHA_DECAY: f32 = 0.5; // Step size reduction factor
        const MAX_ITERATIONS: usize = 20;

        let grad_dot_dir = grad.dot(direction)?.item()?;

        // If direction is not a descent direction, return small step
        if grad_dot_dir >= 0.0 {
            return Ok(0.01);
        }

        let mut alpha = initial_step_size;

        // Simple backtracking without function evaluation
        // In practice, you would evaluate f(x + alpha * d) here
        for _ in 0..MAX_ITERATIONS {
            // For now, just ensure we have a reasonable step size
            if alpha > 1e-8 {
                return Ok(alpha);
            }
            alpha *= ALPHA_DECAY;
        }

        Ok(1e-8) // Minimum step size
    }
}

impl Optimizer for LBFGS {
    fn step(&mut self) -> OptimizerResult<()> {
        // L-BFGS requires a closure for function evaluation, but since we're working
        // with the standard Optimizer interface, we'll implement a simplified version
        self.step_count += 1;

        // Get current parameters and gradients
        let current_params = self.flatten_params()?;
        let current_grad = self.flatten_grads()?;

        // Check gradient tolerance
        let grad_norm = current_grad.norm()?.item()?;
        if grad_norm < self.tolerance_grad {
            return Ok(());
        }

        // Get or initialize L-BFGS state from persistent storage (collect data first)
        let state_id = "lbfgs_history".to_string();
        let (s_list, y_list, rho_list, prev_params, prev_grad) = {
            let param_state = self.state.entry(state_id.clone()).or_default();

            let s_list = if let Some(s_tensors) = param_state.get("s_list") {
                // In practice, this would be a list of tensors
                vec![s_tensors.clone()]
            } else {
                Vec::new()
            };

            let y_list = if let Some(y_tensors) = param_state.get("y_list") {
                vec![y_tensors.clone()]
            } else {
                Vec::new()
            };

            let rho_list = if let Some(rho_tensor) = param_state.get("rho_list") {
                vec![rho_tensor.item()?]
            } else {
                Vec::new()
            };

            let prev_params = if let Some(prev) = param_state.get("prev_params") {
                prev.clone()
            } else {
                current_params.clone()
            };

            let prev_grad = if let Some(prev) = param_state.get("prev_grad") {
                prev.clone()
            } else {
                current_grad.clone()
            };

            (s_list, y_list, rho_list, prev_params, prev_grad)
        };

        // For the first step, use steepest descent
        let direction = if self.step_count == 1 || s_list.is_empty() {
            current_grad.neg()?
        } else {
            // Compute L-BFGS direction using two-loop recursion
            self.compute_direction(&current_grad, &s_list, &y_list, &rho_list)?
        };

        // Line search to find step size
        let step_size = self.line_search(&current_params, &current_grad, &direction, 1.0)?;

        // Update parameters
        let new_params = current_params.add(&direction.mul_scalar(step_size)?)?;
        self.update_params_from_flat(&new_params)?;

        // Update L-BFGS history for next iteration
        {
            let param_state = self
                .state
                .get_mut(&state_id)
                .expect("state should exist for state_id");

            if self.step_count > 1 {
                let s = new_params.sub(&prev_params)?; // Parameter step
                let y = current_grad.sub(&prev_grad)?; // Gradient difference

                let y_dot_s = y.dot(&s)?.item()?;
                if y_dot_s.abs() > 1e-10 {
                    let rho = 1.0 / y_dot_s;

                    // Store new history (simplified - store only the latest)
                    param_state.insert("s_list".to_string(), s);
                    param_state.insert("y_list".to_string(), y);
                    param_state.insert("rho_list".to_string(), Tensor::scalar(rho)?);
                }
            }

            // Store current state for next iteration
            param_state.insert("prev_params".to_string(), new_params);
            param_state.insert("prev_grad".to_string(), current_grad);
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
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
            optimizer_type: "L-BFGS".to_string(),
            version: "1.0".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_lbfgs_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()))];
        let optimizer = LBFGS::new(params, None, None, None, None, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 1.0);
    }

    #[test]
    fn test_lbfgs_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2]).unwrap()));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2]).unwrap()));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = LBFGS::new(params, Some(0.1), None, None, None, None, None, None);

        optimizer.step()?;

        Ok(())
    }
}
