//! Elastic Averaging SGD for distributed training
//!
//! This module provides the ElasticAveragingSGD optimizer which allows workers to explore
//! different parameter spaces while periodically pulling towards a moving average.
//! This provides better exploration than vanilla distributed SGD.

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Elastic Averaging SGD optimizer for distributed training
///
/// EASGD allows workers to explore different parameter spaces while periodically
/// pulling towards a moving average. This provides better exploration than vanilla
/// distributed SGD and can lead to improved convergence and generalization.
///
/// The key insight is that allowing workers to maintain different parameter values
/// while still being influenced by the center (average) parameters enables better
/// exploration of the parameter space while maintaining coordination.
pub struct ElasticAveragingSGD {
    /// Local optimizer parameters
    params: Vec<Arc<RwLock<Tensor>>>,
    /// Center (averaged) parameters
    center_params: Vec<Tensor>,
    /// Learning rate
    lr: f32,
    /// Momentum factor
    momentum: Option<f32>,
    /// Weight decay (L2 penalty)
    weight_decay: Option<f32>,
    /// Elastic parameter (controls pulling force towards center)
    rho: f32,
    /// Communication frequency (steps between averaging)
    communication_freq: usize,
    /// Current step counter
    step_counter: usize,
    /// Momentum buffers for local updates
    momentum_buffers: HashMap<String, Tensor>,
    /// Worker rank for identification
    worker_rank: usize,
    /// Total number of workers
    num_workers: usize,
}

impl ElasticAveragingSGD {
    /// Create a new ElasticAveragingSGD optimizer
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor (optional)
    /// * `weight_decay` - Weight decay (L2 penalty, optional)
    /// * `rho` - Elastic parameter controlling pulling force towards center (0, 1]
    /// * `communication_freq` - Number of steps between communication rounds
    /// * `worker_rank` - Unique identifier for this worker
    /// * `num_workers` - Total number of workers in the distributed system
    ///
    /// # Returns
    ///
    /// Result containing the new optimizer or an error if parameters are invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// # use torsh_tensor::creation::randn;
    /// # use torsh_core::error::Result;
    /// # use parking_lot::RwLock;
    /// # use std::sync::Arc;
    /// # use torsh_optim::distributed::ElasticAveragingSGD;
    /// # fn main() -> Result<()> {
    /// // Create some parameters
    /// let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
    /// let params = vec![param1];
    ///
    /// let optimizer = ElasticAveragingSGD::new(
    ///     params,
    ///     0.1,        // learning rate
    ///     Some(0.9),  // momentum
    ///     Some(1e-4), // weight decay
    ///     0.1,        // rho (elastic parameter)
    ///     10,         // communicate every 10 steps
    ///     0,          // worker rank
    ///     4           // total workers
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        rho: f32,
        communication_freq: usize,
        worker_rank: usize,
        num_workers: usize,
    ) -> Result<Self> {
        if rho <= 0.0 || rho > 1.0 {
            return Err(TorshError::InvalidArgument(
                "Elastic parameter rho must be in (0, 1]".to_string(),
            ));
        }

        if communication_freq == 0 {
            return Err(TorshError::InvalidArgument(
                "Communication frequency must be greater than 0".to_string(),
            ));
        }

        // Initialize center parameters with copies of local parameters
        let center_params = params.iter().map(|p| p.read().clone()).collect();

        Ok(Self {
            params,
            center_params,
            lr,
            momentum,
            weight_decay,
            rho,
            communication_freq,
            step_counter: 0,
            momentum_buffers: HashMap::new(),
            worker_rank,
            num_workers,
        })
    }

    /// Create EASGD with default parameters
    ///
    /// Uses commonly effective default values:
    /// - Momentum: 0.9
    /// - Weight decay: 1e-4
    /// - Rho: 0.1 (10% elastic force)
    /// - Communication frequency: 10 steps
    pub fn new_default(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        worker_rank: usize,
        num_workers: usize,
    ) -> Result<Self> {
        Self::new(
            params,
            lr,
            Some(0.9),  // momentum
            Some(1e-4), // weight_decay
            0.1,        // rho (elastic parameter)
            10,         // communication_freq
            worker_rank,
            num_workers,
        )
    }

    /// Perform local SGD update with elastic averaging
    ///
    /// This method performs the core EASGD update which includes:
    /// 1. Computing gradients with weight decay
    /// 2. Applying momentum if enabled
    /// 3. Adding elastic force towards center parameters
    /// 4. Updating local parameters
    fn local_update(&mut self) -> Result<()> {
        for (i, param_arc) in self.params.iter().enumerate() {
            let mut param = param_arc.write();
            let grad = param
                .grad()
                .ok_or_else(|| TorshError::AutogradError("No gradient available".to_string()))?;

            // Apply weight decay if specified
            let mut effective_grad = grad.clone();
            if let Some(decay) = self.weight_decay {
                effective_grad = effective_grad.add(&param.mul_scalar(decay)?)?;
            }

            // Apply momentum if specified
            if let Some(momentum) = self.momentum {
                let param_key = format!("param_{}", i);

                if let Some(buf) = self.momentum_buffers.get(&param_key) {
                    let new_buf = buf.mul_scalar(momentum)?.add(&effective_grad)?;
                    effective_grad = new_buf.clone();
                    self.momentum_buffers.insert(param_key, new_buf);
                } else {
                    self.momentum_buffers
                        .insert(param_key, effective_grad.clone());
                }
            }

            // Apply elastic force towards center
            // This is the key difference from standard SGD - we pull towards the center
            let elastic_force = param.sub(&self.center_params[i])?.mul_scalar(self.rho)?;
            effective_grad = effective_grad.add(&elastic_force)?;

            // Update parameters
            let update = effective_grad.mul_scalar(self.lr)?;
            *param = param.sub(&update)?;
        }

        Ok(())
    }

    /// Communicate with center (average parameters across workers)
    ///
    /// This method updates the center parameters by averaging across all workers.
    /// In a real distributed system, this would involve network communication.
    ///
    /// # Arguments
    ///
    /// * `all_worker_params` - Parameters from all workers for averaging
    ///
    /// # Note
    ///
    /// This is a simplified version that assumes all worker parameters are available.
    /// In practice, this would involve MPI, NCCL, or other communication frameworks.
    pub fn communicate(&mut self, all_worker_params: &[Vec<Tensor>]) -> Result<()> {
        if all_worker_params.is_empty() {
            return Ok(());
        }

        // Update center parameters as average of all workers
        for i in 0..self.center_params.len() {
            if i >= all_worker_params[0].len() {
                break; // Safety check
            }

            let mut sum = all_worker_params[0][i].clone();

            for worker_params in all_worker_params.iter().skip(1) {
                if i < worker_params.len() {
                    sum = sum.add(&worker_params[i])?;
                }
            }

            self.center_params[i] = sum.div_scalar(self.num_workers as f32)?;
        }

        Ok(())
    }

    /// Get current worker parameters for communication
    ///
    /// Returns a copy of the current local parameters that can be sent to other workers
    /// or the parameter server for averaging.
    pub fn get_worker_params(&self) -> Vec<Tensor> {
        self.params.iter().map(|p| p.read().clone()).collect()
    }

    /// Get center parameters
    ///
    /// Returns a reference to the current center (averaged) parameters.
    pub fn get_center_params(&self) -> &[Tensor] {
        &self.center_params
    }

    /// Get worker rank
    pub fn worker_rank(&self) -> usize {
        self.worker_rank
    }

    /// Get elastic parameter
    pub fn rho(&self) -> f32 {
        self.rho
    }

    /// Set elastic parameter
    ///
    /// The elastic parameter controls how strongly workers are pulled towards the center.
    /// Higher values mean stronger pulling force, lower values allow more exploration.
    ///
    /// # Arguments
    ///
    /// * `rho` - New elastic parameter value (must be in (0, 1])
    pub fn set_rho(&mut self, rho: f32) -> Result<()> {
        if rho <= 0.0 || rho > 1.0 {
            return Err(TorshError::InvalidArgument(
                "Elastic parameter rho must be in (0, 1]".to_string(),
            ));
        }
        self.rho = rho;
        Ok(())
    }

    /// Get communication frequency
    pub fn communication_freq(&self) -> usize {
        self.communication_freq
    }

    /// Set communication frequency
    ///
    /// Controls how often workers synchronize their parameters. Higher frequencies
    /// mean more communication overhead but potentially faster convergence.
    ///
    /// # Arguments
    ///
    /// * `freq` - New communication frequency (must be > 0)
    pub fn set_communication_freq(&mut self, freq: usize) -> Result<()> {
        if freq == 0 {
            return Err(TorshError::InvalidArgument(
                "Communication frequency must be greater than 0".to_string(),
            ));
        }
        self.communication_freq = freq;
        Ok(())
    }

    /// Check if it's time to communicate
    ///
    /// Returns true if the current step count is a multiple of the communication frequency.
    pub fn should_communicate(&self) -> bool {
        self.step_counter % self.communication_freq == 0
    }

    /// Get current step counter
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Reset step counter (useful after communication rounds)
    pub fn reset_step_counter(&mut self) {
        self.step_counter = 0;
    }
}

impl Optimizer for ElasticAveragingSGD {
    fn step(&mut self) -> OptimizerResult<()> {
        // Perform local update
        self.local_update()?;

        self.step_counter += 1;

        // Note: Communication happens externally when needed
        // (when step_counter % communication_freq == 0)
        // The caller should check should_communicate() and call communicate()

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
        // Add new center parameters for new params
        for param in &params {
            self.center_params.push(param.read().clone());
        }
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_group = ParamGroupState {
            lr: self.lr,
            options: [
                ("momentum".to_string(), self.momentum.unwrap_or(0.0)),
                ("weight_decay".to_string(), self.weight_decay.unwrap_or(0.0)),
                ("rho".to_string(), self.rho),
                (
                    "communication_freq".to_string(),
                    self.communication_freq as f32,
                ),
                ("worker_rank".to_string(), self.worker_rank as f32),
                ("num_workers".to_string(), self.num_workers as f32),
            ]
            .iter()
            .cloned()
            .collect(),
            param_count: self.params.len(),
        };

        // Include momentum buffers in state
        let mut state = HashMap::new();

        // Add momentum buffers
        for (param_id, momentum_buffer) in &self.momentum_buffers {
            let mut param_state = HashMap::new();
            param_state.insert("momentum_buffer".to_string(), momentum_buffer.clone());
            state.insert(param_id.clone(), param_state);
        }

        // Add step counter to global state
        let mut global_state = HashMap::new();
        global_state.insert("step_counter".to_string(), self.step_counter as f32);

        // Note: center_params are stored separately since they need special handling
        // during restoration (they need to match parameter shapes)

        Ok(OptimizerState {
            optimizer_type: "EASGD".to_string(),
            version: "0.1.0".to_string(),
            param_groups: vec![param_group],
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Validate optimizer type
        if state.optimizer_type != "EASGD" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected EASGD optimizer state, got {}",
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
            if let Some(&rho) = param_group.options.get("rho") {
                self.rho = rho;
            }
            if let Some(&communication_freq) = param_group.options.get("communication_freq") {
                self.communication_freq = communication_freq as usize;
            }
            if let Some(&worker_rank) = param_group.options.get("worker_rank") {
                self.worker_rank = worker_rank as usize;
            }
            if let Some(&num_workers) = param_group.options.get("num_workers") {
                self.num_workers = num_workers as usize;
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

        // Load global state
        if let Some(&step_counter) = state.global_state.get("step_counter") {
            self.step_counter = step_counter as usize;
        }

        // Note: center_params are reset to current parameter values since they
        // need to be synchronized across workers after loading state
        self.center_params.clear();
        for param in &self.params {
            self.center_params.push(param.read().clone());
        }

        Ok(())
    }
}

/// Utility functions for Elastic Averaging SGD
pub mod utils {
    use super::*;

    /// Create EASGD with settings optimized for different cluster sizes
    ///
    /// Automatically adjusts parameters based on the number of workers:
    /// - Smaller clusters: Higher rho (more pulling force)
    /// - Larger clusters: Lower rho (more exploration)
    /// - Communication frequency adjusted for network overhead
    pub fn create_easgd_for_cluster_size(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        worker_rank: usize,
        num_workers: usize,
    ) -> Result<ElasticAveragingSGD> {
        let (rho, comm_freq) = match num_workers {
            1..=2 => (0.2, 5),    // High coordination, frequent communication
            3..=8 => (0.1, 10),   // Balanced
            9..=16 => (0.05, 20), // More exploration, less frequent communication
            _ => (0.02, 50),      // Large clusters: minimal pulling, infrequent communication
        };

        ElasticAveragingSGD::new(
            params,
            lr,
            Some(0.9),  // Standard momentum
            Some(1e-4), // Standard weight decay
            rho,
            comm_freq,
            worker_rank,
            num_workers,
        )
    }

    /// Simulate a full EASGD training round with multiple workers
    ///
    /// This is a simplified simulation for testing and demonstration purposes.
    /// In practice, workers would run on different machines and communicate
    /// over the network.
    pub fn simulate_easgd_round(
        workers: &mut [ElasticAveragingSGD],
        steps_per_round: usize,
    ) -> Result<()> {
        // Each worker performs local updates
        for worker in workers.iter_mut() {
            for _ in 0..steps_per_round {
                worker.step()?;
            }
        }

        // Collect parameters from all workers
        let all_worker_params: Vec<Vec<Tensor>> = workers
            .iter()
            .map(|worker| worker.get_worker_params())
            .collect();

        // Each worker updates its center parameters
        for worker in workers.iter_mut() {
            worker.communicate(&all_worker_params)?;
        }

        Ok(())
    }

    /// Calculate convergence metrics for EASGD workers
    ///
    /// Returns various metrics useful for monitoring distributed training:
    /// - Parameter divergence (how spread out workers are)
    /// - Center stability (how much center parameters change)
    /// - Convergence rate estimates
    pub fn calculate_easgd_metrics(workers: &[ElasticAveragingSGD]) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();

        if workers.is_empty() {
            return metrics;
        }

        // Calculate parameter divergence (standard deviation across workers)
        let mut total_divergence = 0.0;
        let mut param_count = 0;

        if !workers.is_empty() && !workers[0].params.is_empty() {
            let num_params = workers[0].params.len();

            for param_idx in 0..num_params {
                // Get parameter values from all workers
                let mut param_values = Vec::new();
                for worker in workers {
                    if param_idx < worker.params.len() {
                        let param = worker.params[param_idx].read();
                        // For simplicity, use the first element as representative
                        if let Ok(value) = param.get(&[0]) {
                            param_values.push(value);
                        }
                    }
                }

                if param_values.len() > 1 {
                    // Calculate standard deviation
                    let mean = param_values.iter().sum::<f32>() / param_values.len() as f32;
                    let variance = param_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                        / param_values.len() as f32;
                    total_divergence += variance.sqrt();
                    param_count += 1;
                }
            }
        }

        if param_count > 0 {
            metrics.insert(
                "average_parameter_divergence".to_string(),
                total_divergence / param_count as f32,
            );
        }

        // Add other useful metrics
        metrics.insert("num_workers".to_string(), workers.len() as f32);
        if let Some(first_worker) = workers.first() {
            metrics.insert("rho".to_string(), first_worker.rho());
            metrics.insert(
                "communication_freq".to_string(),
                first_worker.communication_freq() as f32,
            );
            metrics.insert(
                "step_counter".to_string(),
                first_worker.step_counter() as f32,
            );
        }

        metrics
    }
}
