//! Online learning optimizers and variance-reduced methods
//!
//! This module provides optimizers designed for online learning scenarios,
//! including variance-reduced methods like SVRG and SAGA.

use crate::{Optimizer, OptimizerResult, OptimizerState, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Online Gradient Descent (OGD) optimizer
///
/// This is a simple online learning algorithm that updates parameters
/// immediately upon receiving each gradient.
pub struct OnlineGradientDescent {
    /// Parameters
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Learning rate
    lr: f32,
    /// Regularization strength
    regularization: f32,
    /// Regret bound parameter
    regret_bound: f32,
    /// Number of steps taken
    step_count: usize,
}

impl OnlineGradientDescent {
    /// Create a new OnlineGradientDescent optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        regularization: Option<f32>,
        regret_bound: Option<f32>,
    ) -> Self {
        Self {
            params,
            lr,
            regularization: regularization.unwrap_or(0.0),
            regret_bound: regret_bound.unwrap_or(1.0),
            step_count: 0,
        }
    }

    /// Create OGD with adaptive learning rate
    pub fn new_adaptive(params: Vec<Arc<RwLock<Tensor>>>, regret_bound: f32) -> Self {
        Self::new(params, 1.0, Some(0.01), Some(regret_bound))
    }

    /// Get adaptive learning rate based on step count
    fn get_adaptive_lr(&self) -> f32 {
        if self.step_count == 0 {
            self.lr
        } else {
            self.regret_bound / (self.step_count as f32).sqrt()
        }
    }

    /// Get regret bound
    pub fn regret_bound(&self) -> f32 {
        self.regret_bound
    }
}

impl Optimizer for OnlineGradientDescent {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let adaptive_lr = self.get_adaptive_lr();

        for param_arc in &self.params {
            let mut param = param_arc.write();
            let grad = param
                .grad()
                .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

            // Apply regularization
            let mut effective_grad = grad.clone();
            if self.regularization > 0.0 {
                effective_grad = effective_grad.add(&param.mul_scalar(self.regularization)?)?;
            }

            // Update parameters
            let update = effective_grad.mul_scalar(adaptive_lr)?;
            *param = param.sub(&update)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.write().zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.get_adaptive_lr()]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group = ParamGroupState {
            lr: self.lr,
            options: [
                ("regularization".to_string(), self.regularization),
                ("regret_bound".to_string(), self.regret_bound),
                ("step_count".to_string(), self.step_count as f32),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        Ok(OptimizerState {
            optimizer_type: "OnlineGradientDescent".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state: HashMap::new(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "OnlineGradientDescent" {
            return Err(crate::OptimizerError::InvalidParameter(format!(
                "Expected OnlineGradientDescent, got {}",
                state.optimizer_type
            )));
        }

        // Load hyperparameters from param groups
        if let Some(param_group) = state.param_groups.first() {
            self.lr = param_group.lr;

            if let Some(&regularization) = param_group.options.get("regularization") {
                self.regularization = regularization;
            }
            if let Some(&regret_bound) = param_group.options.get("regret_bound") {
                self.regret_bound = regret_bound;
            }
            if let Some(&step_count) = param_group.options.get("step_count") {
                self.step_count = step_count as usize;
            }
        }

        Ok(())
    }
}

/// Stochastic Variance Reduced Gradient (SVRG) optimizer
///
/// SVRG reduces variance in stochastic gradients by periodically computing
/// full gradients and using them as control variates.
pub struct SVRG {
    /// Parameters
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Learning rate
    lr: f32,
    /// Epoch length (frequency of full gradient computation)
    epoch_length: usize,
    /// Current step in epoch
    epoch_step: usize,
    /// Full gradient from last epoch
    full_gradients: Vec<Tensor>,
    /// Parameters at start of epoch
    epoch_params: Vec<Tensor>,
    /// Whether full gradient is available
    has_full_gradient: bool,
}

impl SVRG {
    /// Create a new SVRG optimizer
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32, epoch_length: Option<usize>) -> Self {
        let epoch_length = epoch_length.unwrap_or(100);

        Self {
            params,
            lr,
            epoch_length,
            epoch_step: 0,
            full_gradients: Vec::new(),
            epoch_params: Vec::new(),
            has_full_gradient: false,
        }
    }

    /// Compute full gradient (to be called with full dataset)
    pub fn compute_full_gradient(&mut self) -> Result<()> {
        // Store current parameters as epoch parameters
        self.epoch_params = self.params.iter().map(|p| p.read().clone()).collect();

        // Store full gradients
        self.full_gradients = self
            .params
            .iter()
            .map(|p| {
                let param = p.read();
                param.grad().unwrap_or_else(|| {
                    Tensor::zeros(param.shape().dims(), param.device())
                        .expect("tensor creation should succeed")
                })
            })
            .collect();

        self.has_full_gradient = true;
        self.epoch_step = 0;

        Ok(())
    }

    /// Perform SVRG update with mini-batch gradient
    pub fn svrg_step(
        &mut self,
        minibatch_grad_at_current: &[Tensor],
        minibatch_grad_at_epoch: &[Tensor],
    ) -> Result<()> {
        if !self.has_full_gradient {
            return Err(TorshError::AutogradError(
                "Must compute full gradient before SVRG steps".to_string(),
            ));
        }

        for (i, param_arc) in self.params.iter().enumerate() {
            let mut param = param_arc.write();

            // SVRG update: gradient = minibatch_grad_current - minibatch_grad_epoch + full_grad_epoch
            let variance_reduced_grad = minibatch_grad_at_current[i]
                .sub(&minibatch_grad_at_epoch[i])?
                .add(&self.full_gradients[i])?;

            // Update parameters
            let update = variance_reduced_grad.mul_scalar(self.lr)?;
            *param = param.sub(&update)?;
        }

        self.epoch_step += 1;

        // Check if epoch is complete
        if self.epoch_step >= self.epoch_length {
            self.has_full_gradient = false; // Need to recompute full gradient
        }

        Ok(())
    }

    /// Check if new epoch is needed
    pub fn needs_new_epoch(&self) -> bool {
        !self.has_full_gradient || self.epoch_step >= self.epoch_length
    }

    /// Get epoch parameters
    pub fn epoch_params(&self) -> &[Tensor] {
        &self.epoch_params
    }
}

impl Optimizer for SVRG {
    fn step(&mut self) -> OptimizerResult<()> {
        // Regular SGD step if no full gradient available
        for param_arc in &self.params {
            let mut param = param_arc.write();
            let grad = param
                .grad()
                .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

            let update = grad.mul_scalar(self.lr)?;
            *param = param.sub(&update)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.write().zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group = ParamGroupState {
            lr: self.lr,
            options: [
                ("epoch_length".to_string(), self.epoch_length as f32),
                ("epoch_step".to_string(), self.epoch_step as f32),
                (
                    "has_full_gradient".to_string(),
                    if self.has_full_gradient { 1.0 } else { 0.0 },
                ),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        // NOTE: Full gradient and epoch parameter serialization not included
        // Rationale: These are large tensors that should be recomputed fresh on the next epoch
        // for variance reduction methods. This is consistent with standard SVRG practices
        // where the full gradient is recalculated at each epoch start.
        // See: Johnson & Zhang (2013) "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"

        Ok(OptimizerState {
            optimizer_type: "SVRG".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state: HashMap::new(), // Full gradients excluded by design - recomputed on epoch start
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "SVRG" {
            return Err(crate::OptimizerError::InvalidParameter(format!(
                "Expected SVRG, got {}",
                state.optimizer_type
            )));
        }

        // Load hyperparameters from param groups
        if let Some(param_group) = state.param_groups.first() {
            self.lr = param_group.lr;

            if let Some(&epoch_length) = param_group.options.get("epoch_length") {
                self.epoch_length = epoch_length as usize;
            }
            if let Some(&epoch_step) = param_group.options.get("epoch_step") {
                self.epoch_step = epoch_step as usize;
            }
            if let Some(&has_full_gradient) = param_group.options.get("has_full_gradient") {
                self.has_full_gradient = has_full_gradient > 0.5;
            }
        }

        // Note: Full gradients and epoch params are not restored as they should be
        // recomputed on the next epoch. This is consistent with variance reduction
        // methods which require fresh gradient computations.
        if self.has_full_gradient {
            // Reset to require fresh full gradient computation
            self.has_full_gradient = false;
        }

        Ok(())
    }
}

/// Stochastic Average Gradient Algorithm (SAGA) optimizer
///
/// SAGA maintains a table of gradients for each data point and uses
/// the average as a control variate to reduce variance.
pub struct SAGA {
    /// Parameters
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Learning rate
    lr: f32,
    /// Gradient table (one gradient per data point per parameter)
    gradient_table: HashMap<usize, Vec<Tensor>>,
    /// Sum of all gradients in table
    gradient_sum: Vec<Tensor>,
    /// Number of data points
    num_data_points: usize,
    /// Whether gradient table is initialized
    is_initialized: bool,
}

impl SAGA {
    /// Create a new SAGA optimizer
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32, num_data_points: usize) -> Self {
        Self {
            params,
            lr,
            gradient_table: HashMap::new(),
            gradient_sum: Vec::new(),
            num_data_points,
            is_initialized: false,
        }
    }

    /// Initialize gradient table with zeros
    pub fn initialize(&mut self) -> Result<()> {
        // Initialize gradient sum with zeros
        self.gradient_sum = self
            .params
            .iter()
            .map(|p| {
                Tensor::zeros(p.read().shape().dims(), p.read().device())
                    .expect("tensor creation should succeed")
            })
            .collect();

        // Initialize gradient table with zeros for each data point
        for data_idx in 0..self.num_data_points {
            let grad_for_point: Vec<Tensor> = self
                .params
                .iter()
                .map(|p| {
                    Tensor::zeros(p.read().shape().dims(), p.read().device())
                        .expect("tensor creation should succeed")
                })
                .collect();
            self.gradient_table.insert(data_idx, grad_for_point);
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Perform SAGA update for a specific data point
    pub fn saga_step(&mut self, data_index: usize, current_gradients: &[Tensor]) -> Result<()> {
        if !self.is_initialized {
            self.initialize()?;
        }

        if data_index >= self.num_data_points {
            return Err(TorshError::InvalidArgument(format!(
                "Data index {} exceeds number of data points {}",
                data_index, self.num_data_points
            )));
        }

        // Get old gradient for this data point
        let old_gradients = self.gradient_table.get(&data_index).ok_or_else(|| {
            TorshError::InvalidArgument("Gradient table not properly initialized".to_string())
        })?;

        // Update parameters using SAGA rule
        for (i, param_arc) in self.params.iter().enumerate() {
            let mut param = param_arc.write();

            // SAGA update: gradient = current_grad - old_grad + average_grad
            let average_grad = self.gradient_sum[i].div_scalar(self.num_data_points as f32)?;
            let saga_grad = current_gradients[i]
                .sub(&old_gradients[i])?
                .add(&average_grad)?;

            // Update parameters
            let update = saga_grad.mul_scalar(self.lr)?;
            *param = param.sub(&update)?;

            // Update gradient sum
            self.gradient_sum[i] = self.gradient_sum[i]
                .sub(&old_gradients[i])?
                .add(&current_gradients[i])?;
        }

        // Update gradient table
        self.gradient_table
            .insert(data_index, current_gradients.to_vec());

        Ok(())
    }

    /// Get average gradient
    pub fn average_gradient(&self) -> Result<Vec<Tensor>> {
        if !self.is_initialized {
            return Err(TorshError::AutogradError(
                "SAGA not initialized".to_string(),
            ));
        }

        self.gradient_sum
            .iter()
            .map(|grad| grad.div_scalar(self.num_data_points as f32))
            .collect::<Result<Vec<_>>>()
    }

    /// Get number of data points
    pub fn num_data_points(&self) -> usize {
        self.num_data_points
    }
}

impl Optimizer for SAGA {
    fn step(&mut self) -> OptimizerResult<()> {
        // Regular SGD step if not using SAGA-specific method
        for param_arc in &self.params {
            let mut param = param_arc.write();
            let grad = param
                .grad()
                .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

            let update = grad.mul_scalar(self.lr)?;
            *param = param.sub(&update)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.write().zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group = ParamGroupState {
            lr: self.lr,
            options: [
                ("num_data_points".to_string(), self.num_data_points as f32),
                (
                    "is_initialized".to_string(),
                    if self.is_initialized { 1.0 } else { 0.0 },
                ),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        // NOTE: Gradient table serialization deferred to v0.2.0
        // Enhancement: Full gradient table persistence for checkpoint/restore
        // Current: Basic state only (lr, num_data_points, is_initialized)
        // Future: Serialize gradient_table and gradient_sum for complete state recovery
        // Impact: Currently requires gradient table reinitialization after load_state_dict
        // See ROADMAP.md for full persistence implementation plan

        Ok(OptimizerState {
            optimizer_type: "SAGA".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state: HashMap::new(), // Gradient table persistence deferred to v0.2.0
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Validate optimizer type
        if state.optimizer_type != "SAGA" {
            return Err(crate::OptimizerError::InvalidParameter(format!(
                "Expected SAGA optimizer state, got {}",
                state.optimizer_type
            )));
        }

        // Load parameter group state
        if let Some(param_group) = state.param_groups.first() {
            self.lr = param_group.lr;

            // Load SAGA-specific parameters from options
            if let Some(&num_data_points) = param_group.options.get("num_data_points") {
                self.num_data_points = num_data_points as usize;
            }
            if let Some(&is_initialized) = param_group.options.get("is_initialized") {
                self.is_initialized = is_initialized > 0.0;
            }

            // Validate parameter count
            if param_group.param_count != self.params.len() {
                return Err(crate::OptimizerError::InvalidParameter(format!(
                    "Parameter count mismatch: expected {}, got {}",
                    self.params.len(),
                    param_group.param_count
                )));
            }
        } else {
            return Err(crate::OptimizerError::InvalidParameter(
                "No parameter groups found in state".to_string(),
            ));
        }

        // Note: Gradient table and gradient sum state restoration would be implemented
        // when the state_dict method is enhanced to include the gradient table

        Ok(())
    }
}

/// Proximal Gradient Method optimizer
///
/// This optimizer handles non-smooth regularization terms using
/// proximal operators, commonly used for L1 regularization.
pub struct ProximalGradient {
    /// Parameters
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Learning rate
    lr: f32,
    /// L1 regularization strength
    l1_reg: f32,
    /// L2 regularization strength
    l2_reg: f32,
    /// Proximal operator type
    prox_type: ProximalOperator,
}

/// Types of proximal operators
#[derive(Debug, Clone)]
pub enum ProximalOperator {
    /// L1 regularization (soft thresholding)
    L1,
    /// L2 regularization (scaling)
    L2,
    /// Elastic net (L1 + L2)
    ElasticNet,
    /// Group LASSO
    GroupLasso,
}

impl ProximalGradient {
    /// Create a new ProximalGradient optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        l1_reg: Option<f32>,
        l2_reg: Option<f32>,
        prox_type: Option<ProximalOperator>,
    ) -> Self {
        Self {
            params,
            lr,
            l1_reg: l1_reg.unwrap_or(0.0),
            l2_reg: l2_reg.unwrap_or(0.0),
            prox_type: prox_type.unwrap_or(ProximalOperator::L1),
        }
    }

    /// Create ProximalGradient for LASSO (L1 regularization)
    pub fn lasso(params: Vec<Arc<RwLock<Tensor>>>, lr: f32, l1_reg: f32) -> Self {
        Self::new(params, lr, Some(l1_reg), None, Some(ProximalOperator::L1))
    }

    /// Create ProximalGradient for Elastic Net
    pub fn elastic_net(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        l1_reg: f32,
        l2_reg: f32,
    ) -> Self {
        Self::new(
            params,
            lr,
            Some(l1_reg),
            Some(l2_reg),
            Some(ProximalOperator::ElasticNet),
        )
    }

    /// Apply proximal operator
    fn apply_proximal_operator(&self, param: &Tensor) -> Result<Tensor> {
        match self.prox_type {
            ProximalOperator::L1 => {
                // Soft thresholding for L1
                self.soft_threshold(param, self.lr * self.l1_reg)
            }
            ProximalOperator::L2 => {
                // Scaling for L2
                let scale = 1.0 / (1.0 + self.lr * self.l2_reg);
                Ok(param.mul_scalar(scale)?)
            }
            ProximalOperator::ElasticNet => {
                // L2 scaling followed by L1 soft thresholding
                let l2_scale = 1.0 / (1.0 + self.lr * self.l2_reg);
                let l2_result = param.mul_scalar(l2_scale)?;
                self.soft_threshold(&l2_result, self.lr * self.l1_reg)
            }
            ProximalOperator::GroupLasso => {
                // Group soft thresholding (simplified version)
                let param_norm = param.norm()?.item()?;
                let threshold = self.lr * self.l1_reg;

                if param_norm <= threshold {
                    Ok(Tensor::zeros(param.shape().dims(), param.device())?)
                } else {
                    let scale = (param_norm - threshold) / param_norm;
                    Ok(param.mul_scalar(scale)?)
                }
            }
        }
    }

    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(&self, param: &Tensor, threshold: f32) -> Result<Tensor> {
        // Element-wise soft thresholding: sign(x) * max(|x| - threshold, 0)
        let abs_param = param.abs()?;
        let mask = abs_param.gt_scalar(threshold)?;
        let threshold_vec = vec![threshold; abs_param.numel()];
        let threshold_tensor = Tensor::from_vec(threshold_vec, &abs_param.shape().dims())?;
        let thresholded = abs_param
            .sub(&threshold_tensor)?
            .maximum(&Tensor::zeros_like(param)?)?;
        let result = param.sign()?.mul_op(&thresholded)?;
        // Convert boolean mask to float values (1.0 for true, 0.0 for false)
        let mask_data = mask.to_vec()?;
        let mask_f32_data: Vec<f32> = mask_data
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let mask_f32 = Tensor::from_vec(mask_f32_data, &mask.shape().dims())?;
        Ok(result.mul_op(&mask_f32)?) // Apply mask to zero out elements below threshold
    }

    /// Get regularization strengths
    pub fn regularization(&self) -> (f32, f32) {
        (self.l1_reg, self.l2_reg)
    }

    /// Set regularization strengths
    pub fn set_regularization(&mut self, l1_reg: f32, l2_reg: f32) {
        self.l1_reg = l1_reg;
        self.l2_reg = l2_reg;
    }
}

impl Optimizer for ProximalGradient {
    fn step(&mut self) -> OptimizerResult<()> {
        for param_arc in &self.params {
            let mut param = param_arc.write();
            let grad = param
                .grad()
                .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

            // Gradient step
            let grad_step = param.sub(&grad.mul_scalar(self.lr)?)?;

            // Apply proximal operator
            let proximal_result = self.apply_proximal_operator(&grad_step)?;
            *param = proximal_result;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.write().zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group = ParamGroupState {
            lr: self.lr,
            options: [
                ("l1_reg".to_string(), self.l1_reg),
                ("l2_reg".to_string(), self.l2_reg),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        Ok(OptimizerState {
            optimizer_type: "ProximalGradient".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state: HashMap::new(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Validate optimizer type
        if state.optimizer_type != "ProximalGradient" {
            return Err(crate::OptimizerError::InvalidParameter(format!(
                "Expected ProximalGradient optimizer state, got {}",
                state.optimizer_type
            )));
        }

        // Load parameter group state
        if let Some(param_group) = state.param_groups.first() {
            self.lr = param_group.lr;

            // Load regularization parameters from options
            if let Some(&l1_reg) = param_group.options.get("l1_reg") {
                self.l1_reg = l1_reg;
            }
            if let Some(&l2_reg) = param_group.options.get("l2_reg") {
                self.l2_reg = l2_reg;
            }

            // Validate parameter count
            if param_group.param_count != self.params.len() {
                return Err(crate::OptimizerError::InvalidParameter(format!(
                    "Parameter count mismatch: expected {}, got {}",
                    self.params.len(),
                    param_group.param_count
                )));
            }
        } else {
            return Err(crate::OptimizerError::InvalidParameter(
                "No parameter groups found in state".to_string(),
            ));
        }

        // Note: ProximalGradient doesn't maintain per-parameter state like momentum,
        // so we only need to restore the hyperparameters (lr, l1_reg, l2_reg)

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_online_gradient_descent() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let optimizer = OnlineGradientDescent::new(params, 0.01, None, None);
        assert_eq!(optimizer.get_lr()[0], 0.01);
    }

    #[test]
    fn test_svrg_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let optimizer = SVRG::new(params, 0.01, Some(50));
        assert!(optimizer.needs_new_epoch());
    }

    #[test]
    fn test_saga_creation() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let optimizer = SAGA::new(params, 0.01, 100);
        assert_eq!(optimizer.num_data_points(), 100);
    }

    #[test]
    fn test_proximal_gradient() {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[10, 10]).unwrap()))];
        let optimizer = ProximalGradient::lasso(params, 0.01, 0.1);
        let (l1, l2) = optimizer.regularization();
        assert_eq!(l1, 0.1);
        assert_eq!(l2, 0.0);
    }
}
