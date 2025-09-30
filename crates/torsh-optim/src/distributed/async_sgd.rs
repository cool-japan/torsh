//! Asynchronous SGD optimizer for distributed training
//!
//! This module provides an asynchronous SGD optimizer that allows workers to update
//! parameters independently without waiting for synchronization, which can lead to
//! faster convergence in scenarios with high communication latency.

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Asynchronous SGD optimizer for distributed training
///
/// This optimizer implements asynchronous SGD for distributed training where
/// workers can update parameters independently without waiting for synchronization.
/// This can lead to faster convergence in scenarios with high communication latency.
pub struct AsyncSGD {
    /// Learning rate
    lr: f32,
    /// Momentum factor (0 for vanilla SGD)
    momentum: Option<f32>,
    /// Weight decay (L2 penalty)
    weight_decay: Option<f32>,
    /// Dampening for momentum
    dampening: Option<f32>,
    /// Whether to enable Nesterov momentum
    nesterov: bool,
    /// Parameter tensors
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Momentum buffers for each parameter
    momentum_buffers: HashMap<String, Tensor>,
    /// Asynchronous update configuration
    async_config: AsyncConfig,
    /// Parameter staleness tracking
    staleness_tracker: StalenessTracker,
}

/// Configuration for asynchronous SGD
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Maximum allowed staleness (number of updates a parameter can lag behind)
    pub max_staleness: usize,
    /// Staleness penalty factor (reduces effective learning rate for stale parameters)
    pub staleness_penalty: f32,
    /// Whether to use bounded staleness
    pub bounded_staleness: bool,
    /// Asynchronous update frequency (updates per synchronization)
    pub async_frequency: usize,
    /// Whether to enable parameter mixing
    pub parameter_mixing: bool,
    /// Mixing ratio for parameter averaging
    pub mixing_ratio: f32,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            max_staleness: 10,
            staleness_penalty: 0.9,
            bounded_staleness: true,
            async_frequency: 1,
            parameter_mixing: false,
            mixing_ratio: 0.1,
        }
    }
}

/// Tracks parameter staleness for asynchronous updates
#[derive(Debug)]
struct StalenessTracker {
    /// Global update counter
    global_updates: u64,
    /// Per-parameter update counters
    parameter_updates: HashMap<String, u64>,
    /// Staleness history for adaptive learning rates
    staleness_history: HashMap<String, Vec<u64>>,
}

impl StalenessTracker {
    fn new() -> Self {
        Self {
            global_updates: 0,
            parameter_updates: HashMap::new(),
            staleness_history: HashMap::new(),
        }
    }

    /// Record a parameter update
    fn record_update(&mut self, param_id: &str) {
        self.global_updates += 1;
        self.parameter_updates
            .insert(param_id.to_string(), self.global_updates);

        // Update staleness history
        let history = self
            .staleness_history
            .entry(param_id.to_string())
            .or_insert_with(Vec::new);
        history.push(self.global_updates);

        // Keep only recent history (last 100 updates)
        if history.len() > 100 {
            history.drain(0..history.len() - 100);
        }
    }

    /// Get staleness for a parameter
    fn get_staleness(&self, param_id: &str) -> u64 {
        if let Some(&last_update) = self.parameter_updates.get(param_id) {
            self.global_updates.saturating_sub(last_update)
        } else {
            self.global_updates
        }
    }

    /// Get adaptive learning rate based on staleness
    fn get_adaptive_lr(&self, param_id: &str, base_lr: f32, config: &AsyncConfig) -> f32 {
        let staleness = self.get_staleness(param_id);

        if staleness == 0 {
            return base_lr;
        }

        // Apply staleness penalty
        let penalty = config.staleness_penalty.powf(staleness as f32);
        base_lr * penalty
    }
}

impl AsyncSGD {
    /// Create a new AsyncSGD optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        dampening: Option<f32>,
        nesterov: bool,
        async_config: Option<AsyncConfig>,
    ) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            params,
            momentum_buffers: HashMap::new(),
            async_config: async_config.unwrap_or_default(),
            staleness_tracker: StalenessTracker::new(),
        }
    }

    /// Create AsyncSGD with default asynchronous configuration
    pub fn new_async(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, None, None, None, false, None)
    }

    /// Perform asynchronous parameter update
    pub fn async_step(&mut self, param_id: &str) -> Result<()> {
        // Find the parameter by ID
        let param_arc = self
            .params
            .iter()
            .find(|p| {
                // Use memory address as ID for now - could be improved with proper naming
                format!("{:p}", p.as_ref()) == param_id
            })
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!("Parameter with ID {} not found", param_id))
            })?;

        let mut param = param_arc.write();
        let grad = param
            .grad()
            .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

        // Get adaptive learning rate based on staleness
        let adaptive_lr =
            self.staleness_tracker
                .get_adaptive_lr(param_id, self.lr, &self.async_config);

        // Check staleness bounds
        if self.async_config.bounded_staleness {
            let staleness = self.staleness_tracker.get_staleness(param_id);
            if staleness > self.async_config.max_staleness as u64 {
                // Skip update if too stale
                return Ok(());
            }
        }

        // Apply weight decay if specified
        let mut effective_grad = grad.clone();
        if let Some(decay) = self.weight_decay {
            effective_grad = effective_grad.add(&param.mul_scalar(decay)?)?;
        }

        // Apply momentum if specified
        if let Some(momentum) = self.momentum {
            let param_key = format!("{:p}", param_arc.as_ref());

            if let Some(buf) = self.momentum_buffers.get(&param_key) {
                let dampening = self.dampening.unwrap_or(0.0);
                let new_buf = buf
                    .mul_scalar(momentum)?
                    .add(&effective_grad.mul_scalar(1.0 - dampening)?)?;

                effective_grad = if self.nesterov {
                    effective_grad.add(&new_buf.mul_scalar(momentum)?)?
                } else {
                    new_buf.clone()
                };

                self.momentum_buffers.insert(param_key, new_buf);
            } else {
                self.momentum_buffers
                    .insert(param_key, effective_grad.clone());
            }
        }

        // Update parameters
        let update = effective_grad.mul_scalar(adaptive_lr)?;
        *param = param.sub(&update)?;

        // Record the update
        self.staleness_tracker.record_update(param_id);

        Ok(())
    }

    /// Get staleness information for all parameters
    pub fn staleness_info(&self) -> HashMap<String, u64> {
        self.params
            .iter()
            .map(|p| {
                let param_id = format!("{:p}", p.as_ref());
                let staleness = self.staleness_tracker.get_staleness(&param_id);
                (param_id, staleness)
            })
            .collect()
    }

    /// Get asynchronous configuration
    pub fn async_config(&self) -> &AsyncConfig {
        &self.async_config
    }

    /// Update asynchronous configuration
    pub fn set_async_config(&mut self, config: AsyncConfig) {
        self.async_config = config;
    }

    /// Perform parameter mixing (average with other workers)
    pub fn mix_parameters(&mut self, other_params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        if !self.async_config.parameter_mixing {
            return Ok(());
        }

        let mixing_ratio = self.async_config.mixing_ratio;

        for (param_arc, other_param_arc) in self.params.iter().zip(other_params.iter()) {
            let mut param = param_arc.write();
            let other_param = other_param_arc.read();

            // Mix parameters: param = (1 - ratio) * param + ratio * other_param
            let mixed = param
                .mul_scalar(1.0 - mixing_ratio)?
                .add(&other_param.mul_scalar(mixing_ratio)?)?;

            *param = mixed;
        }

        Ok(())
    }
}

impl Optimizer for AsyncSGD {
    fn step(&mut self) -> OptimizerResult<()> {
        // For synchronous step, update all parameters
        let param_ids: Vec<String> = self
            .params
            .iter()
            .map(|param_arc| format!("{:p}", param_arc.as_ref()))
            .collect();
        for param_id in param_ids {
            self.async_step(&param_id)?;
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
                ("momentum".to_string(), self.momentum.unwrap_or(0.0)),
                ("weight_decay".to_string(), self.weight_decay.unwrap_or(0.0)),
                ("dampening".to_string(), self.dampening.unwrap_or(0.0)),
                (
                    "nesterov".to_string(),
                    if self.nesterov { 1.0 } else { 0.0 },
                ),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        // Include momentum buffers in state
        let mut state = HashMap::new();
        for (param_id, momentum_buffer) in &self.momentum_buffers {
            let mut param_state = HashMap::new();
            param_state.insert("momentum_buffer".to_string(), momentum_buffer.clone());
            state.insert(param_id.clone(), param_state);
        }

        // Include async configuration and staleness tracking in global state
        let mut global_state = HashMap::new();

        // AsyncConfig fields
        global_state.insert(
            "max_staleness".to_string(),
            self.async_config.max_staleness as f32,
        );
        global_state.insert(
            "staleness_penalty".to_string(),
            self.async_config.staleness_penalty,
        );
        global_state.insert(
            "bounded_staleness".to_string(),
            if self.async_config.bounded_staleness {
                1.0
            } else {
                0.0
            },
        );
        global_state.insert(
            "async_frequency".to_string(),
            self.async_config.async_frequency as f32,
        );
        global_state.insert(
            "parameter_mixing".to_string(),
            if self.async_config.parameter_mixing {
                1.0
            } else {
                0.0
            },
        );
        global_state.insert("mixing_ratio".to_string(), self.async_config.mixing_ratio);

        // Staleness tracking state
        global_state.insert(
            "global_updates".to_string(),
            self.staleness_tracker.global_updates as f32,
        );

        Ok(OptimizerState {
            optimizer_type: "AsyncSGD".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Validate optimizer type
        if state.optimizer_type != "AsyncSGD" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected AsyncSGD optimizer state, got {}",
                state.optimizer_type
            )));
        }

        // Load parameter group state
        if let Some(param_group) = state.param_groups.first() {
            self.lr = param_group.lr;

            // Load hyperparameters from options
            if let Some(&momentum) = param_group.options.get("momentum") {
                self.momentum = if momentum > 0.0 { Some(momentum) } else { None };
            }
            if let Some(&weight_decay) = param_group.options.get("weight_decay") {
                self.weight_decay = if weight_decay > 0.0 {
                    Some(weight_decay)
                } else {
                    None
                };
            }
            if let Some(&dampening) = param_group.options.get("dampening") {
                self.dampening = if dampening > 0.0 {
                    Some(dampening)
                } else {
                    None
                };
            }
            if let Some(&nesterov) = param_group.options.get("nesterov") {
                self.nesterov = nesterov > 0.0;
            }

            // Validate parameter count
            if param_group.param_count != self.params.len() {
                return Err(OptimizerError::InvalidParameter(format!(
                    "Parameter count mismatch: expected {}, got {}",
                    self.params.len(),
                    param_group.param_count
                )));
            }
        } else {
            return Err(OptimizerError::InvalidParameter(
                "No parameter groups found in state".to_string(),
            ));
        }

        // Load momentum buffers from state
        self.momentum_buffers.clear();
        for (param_id, param_state) in state.state {
            if let Some(momentum_buffer) = param_state.get("momentum_buffer") {
                self.momentum_buffers
                    .insert(param_id, momentum_buffer.clone());
            }
        }

        // Load async configuration from global state
        if let Some(&max_staleness) = state.global_state.get("max_staleness") {
            self.async_config.max_staleness = max_staleness as usize;
        }
        if let Some(&staleness_penalty) = state.global_state.get("staleness_penalty") {
            self.async_config.staleness_penalty = staleness_penalty;
        }
        if let Some(&bounded_staleness) = state.global_state.get("bounded_staleness") {
            self.async_config.bounded_staleness = bounded_staleness > 0.0;
        }
        if let Some(&async_frequency) = state.global_state.get("async_frequency") {
            self.async_config.async_frequency = async_frequency as usize;
        }
        if let Some(&parameter_mixing) = state.global_state.get("parameter_mixing") {
            self.async_config.parameter_mixing = parameter_mixing > 0.0;
        }
        if let Some(&mixing_ratio) = state.global_state.get("mixing_ratio") {
            self.async_config.mixing_ratio = mixing_ratio;
        }

        // Load staleness tracking state
        if let Some(&global_updates) = state.global_state.get("global_updates") {
            self.staleness_tracker.global_updates = global_updates as u64;
        }

        // Note: parameter_updates and staleness_history are reset since they're
        // runtime tracking state that will be rebuilt during training
        self.staleness_tracker.parameter_updates.clear();
        self.staleness_tracker.staleness_history.clear();

        Ok(())
    }
}

/// Utility functions for asynchronous distributed training
pub mod utils {
    use super::*;

    /// Create AsyncSGD with commonly used settings for distributed training
    pub fn create_async_sgd_distributed(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        world_size: usize,
    ) -> AsyncSGD {
        let async_config = AsyncConfig {
            max_staleness: world_size * 2, // Allow up to 2x world size staleness
            staleness_penalty: 0.9,
            bounded_staleness: true,
            async_frequency: 1,
            parameter_mixing: world_size > 4, // Enable mixing for large clusters
            mixing_ratio: 0.1 / (world_size as f32).sqrt(), // Adaptive mixing ratio
        };

        AsyncSGD::new(
            params,
            lr,
            Some(0.9),
            Some(1e-4),
            None,
            false,
            Some(async_config),
        )
    }

    /// Synchronize AsyncSGD optimizers across workers (simplified simulation)
    pub fn synchronize_async_sgd_workers(workers: &mut [AsyncSGD]) -> Result<()> {
        if workers.is_empty() {
            return Ok(());
        }

        // Simple parameter averaging across workers
        let num_workers = workers.len();

        // Get parameter references from first worker to determine structure
        let param_count = workers[0].params.len();

        // For each parameter position
        for param_idx in 0..param_count {
            // Collect all parameter values from all workers
            let mut param_sum: Option<Tensor> = None;

            for worker in workers.iter() {
                if param_idx < worker.params.len() {
                    let param = worker.params[param_idx].read();
                    if let Some(ref mut sum) = param_sum {
                        *sum = sum.add(&*param)?;
                    } else {
                        param_sum = Some(param.clone());
                    }
                }
            }

            // Average and distribute back to all workers
            if let Some(sum) = param_sum {
                let average = sum.div_scalar(num_workers as f32)?;

                for worker in workers.iter_mut() {
                    if param_idx < worker.params.len() {
                        let mut param = worker.params[param_idx].write();
                        *param = average.clone();
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate global staleness statistics across workers
    pub fn global_staleness_stats(workers: &[AsyncSGD]) -> HashMap<String, f32> {
        let mut stats = HashMap::new();

        if workers.is_empty() {
            return stats;
        }

        let mut total_staleness = 0.0;
        let mut max_staleness: f32 = 0.0;
        let mut param_count = 0;

        for worker in workers {
            let staleness_info = worker.staleness_info();
            for staleness in staleness_info.values() {
                let staleness_f32 = *staleness as f32;
                total_staleness += staleness_f32;
                max_staleness = max_staleness.max(staleness_f32);
                param_count += 1;
            }
        }

        if param_count > 0 {
            stats.insert(
                "average_staleness".to_string(),
                total_staleness / param_count as f32,
            );
            stats.insert("max_staleness".to_string(), max_staleness);
            stats.insert("total_parameters".to_string(), param_count as f32);
        }

        stats
    }
}
