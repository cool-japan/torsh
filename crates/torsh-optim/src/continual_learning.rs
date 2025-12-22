//! Continual Learning Optimizers
//!
//! This module implements optimization algorithms for lifelong learning scenarios,
//! preventing catastrophic forgetting when learning sequential tasks.
//!
//! # Key Concepts
//!
//! - **Catastrophic Forgetting**: The tendency of neural networks to forget previously
//!   learned tasks when learning new ones.
//! - **Parameter Importance**: Identifying which parameters are critical for past tasks
//!   and should be protected from large updates.
//! - **Task-Specific Learning**: Adapting learning strategies based on task sequence.
//!
//! # Algorithms
//!
//! ## EWC (Elastic Weight Consolidation)
//!
//! Protects important parameters by adding a quadratic penalty based on the Fisher
//! Information Matrix. Parameters critical for previous tasks receive higher penalties.
//!
//! Loss: L_new = L_task + (λ/2) Σ F_i (θ_i - θ*_i)²
//!
//! ## SI (Synaptic Intelligence)
//!
//! Online continual learning that accumulates parameter importance during training.
//! Updates importance based on path integral of parameter changes.
//!
//! ## MAS (Memory Aware Synapses)
//!
//! Uses gradient magnitude at optimal parameters to estimate importance,
//! avoiding the need for data from previous tasks.
//!
//! ## PackNet
//!
//! Packs multiple tasks into a single network by dynamically allocating
//! and protecting subnetworks for each task.
//!
//! # References
//!
//! - Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks"
//! - Zenke et al. (2017). "Continual Learning Through Synaptic Intelligence"
//! - Aljundi et al. (2018). "Memory Aware Synapses"
//! - Mallya & Lazebnik (2018). "PackNet: Adding Multiple Tasks to a Single Network"

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::Tensor;

// ============================================================================
// EWC (Elastic Weight Consolidation)
// ============================================================================

/// EWC configuration
#[derive(Debug, Clone)]
pub struct EWCConfig {
    /// Importance weight (λ) for Fisher penalty
    pub importance: f32,
    /// Sample size for Fisher matrix estimation
    pub fisher_sample_size: usize,
    /// Use diagonal Fisher approximation
    pub diagonal_fisher: bool,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            importance: 1000.0,
            fisher_sample_size: 200,
            diagonal_fisher: true,
        }
    }
}

/// Elastic Weight Consolidation optimizer
///
/// Prevents catastrophic forgetting by adding regularization based on
/// parameter importance computed from the Fisher Information Matrix.
pub struct EWCOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: EWCConfig,
    /// Fisher information per parameter
    fisher_information: HashMap<String, Tensor>,
    /// Optimal parameters from previous task
    optimal_params: HashMap<String, Tensor>,
    /// Current task ID
    current_task: usize,
    /// Parameter groups reference
    param_groups: Vec<Arc<RwLock<Tensor>>>,
}

impl<O: Optimizer> EWCOptimizer<O> {
    /// Create a new EWC optimizer
    pub fn new(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
        config: EWCConfig,
    ) -> OptimizerResult<Self> {
        Ok(Self {
            base_optimizer,
            config,
            fisher_information: HashMap::new(),
            optimal_params: HashMap::new(),
            current_task: 0,
            param_groups: params,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
    ) -> OptimizerResult<Self> {
        Self::new(base_optimizer, params, EWCConfig::default())
    }

    /// Consolidate current task (compute Fisher and save optimal parameters)
    pub fn consolidate_task(&mut self) -> OptimizerResult<()> {
        // Save current parameters as optimal
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let param_read = param.read();
            self.optimal_params
                .insert(param_key.clone(), param_read.clone());
        }

        // Compute Fisher information (simplified diagonal approximation)
        self.compute_fisher_diagonal()?;

        self.current_task += 1;
        Ok(())
    }

    /// Compute diagonal Fisher information matrix
    fn compute_fisher_diagonal(&mut self) -> OptimizerResult<()> {
        // For each parameter, compute F_ii = E[∂L/∂θ_i]²
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            // Get gradient (squared for Fisher diagonal)
            let param_read = param.read();
            if let Some(grad) = param_read.grad() {
                let fisher = grad.mul(&grad)?; // Element-wise square

                // If Fisher already exists, accumulate
                if let Some(existing_fisher) = self.fisher_information.get(&param_key) {
                    let accumulated = existing_fisher.add(&fisher)?;
                    self.fisher_information.insert(param_key, accumulated);
                } else {
                    self.fisher_information.insert(param_key, fisher);
                }
            }
        }

        Ok(())
    }

    /// Apply EWC penalty to gradients
    fn apply_ewc_penalty(&mut self) -> OptimizerResult<()> {
        if self.optimal_params.is_empty() {
            // No previous task, no penalty
            return Ok(());
        }

        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let (Some(fisher), Some(optimal)) = (
                self.fisher_information.get(&param_key),
                self.optimal_params.get(&param_key),
            ) {
                let mut param_write = param.write();

                // Compute EWC penalty: λ * F * (θ - θ*)
                let diff = param_write.sub(optimal)?;
                let penalty = fisher.mul(&diff)?;
                let scaled_penalty = penalty.mul_scalar(self.config.importance)?;

                // Add penalty to gradient
                if let Some(grad) = param_write.grad() {
                    let new_grad = grad.add(&scaled_penalty)?;
                    param_write.set_grad(Some(new_grad));
                }
            }
        }

        Ok(())
    }
}

impl<O: Optimizer> Optimizer for EWCOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Apply EWC penalty to gradients
        self.apply_ewc_penalty()?;

        // Call base optimizer
        self.base_optimizer.step()
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("EWC({})", state.optimizer_type);
        state
            .global_state
            .insert("current_task".to_string(), self.current_task as f32);
        state
            .global_state
            .insert("importance".to_string(), self.config.importance);
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// SI (Synaptic Intelligence)
// ============================================================================

/// Synaptic Intelligence configuration
#[derive(Debug, Clone)]
pub struct SIConfig {
    /// Damping parameter (ξ)
    pub damping: f32,
    /// Importance regularization strength
    pub importance: f32,
}

impl Default for SIConfig {
    fn default() -> Self {
        Self {
            damping: 0.1,
            importance: 1.0,
        }
    }
}

/// Synaptic Intelligence optimizer
///
/// Online continual learning that tracks parameter importance during training
/// using path integral of gradient times parameter change.
pub struct SIOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: SIConfig,
    /// Path integral accumulator (ω)
    path_integral: HashMap<String, Tensor>,
    /// Previous parameters
    prev_params: HashMap<String, Tensor>,
    /// Consolidated importance per task
    importance: HashMap<String, Tensor>,
    /// Current task ID
    current_task: usize,
    /// Parameter groups
    param_groups: Vec<Arc<RwLock<Tensor>>>,
}

impl<O: Optimizer> SIOptimizer<O> {
    /// Create a new SI optimizer
    pub fn new(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
        config: SIConfig,
    ) -> OptimizerResult<Self> {
        let mut prev_params = HashMap::new();
        let mut path_integral = HashMap::new();

        // Initialize previous parameters and path integral
        for (i, param) in params.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let param_read = param.read();
            prev_params.insert(param_key.clone(), param_read.clone());

            let shape_owned = param_read.shape().dims().to_vec();
            drop(param_read);
            let zeros = torsh_tensor::creation::zeros(&shape_owned)?;
            path_integral.insert(param_key, zeros);
        }

        Ok(Self {
            base_optimizer,
            config,
            path_integral,
            prev_params,
            importance: HashMap::new(),
            current_task: 0,
            param_groups: params,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
    ) -> OptimizerResult<Self> {
        Self::new(base_optimizer, params, SIConfig::default())
    }

    /// Update path integral
    fn update_path_integral(&mut self) -> OptimizerResult<()> {
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            let param_read = param.read();
            if let Some(grad) = param_read.grad() {
                if let Some(prev_param) = self.prev_params.get(&param_key) {
                    // Δθ = θ_new - θ_old
                    let delta = param_read.sub(prev_param)?;

                    // ω += -∂L/∂θ * Δθ
                    let contribution = grad.mul(&delta)?;
                    let neg_contribution = contribution.mul_scalar(-1.0)?;

                    if let Some(omega) = self.path_integral.get_mut(&param_key) {
                        *omega = omega.add(&neg_contribution)?;
                    }

                    // Update previous parameters
                    self.prev_params.insert(param_key, param_read.clone());
                }
            }
        }

        Ok(())
    }

    /// Consolidate task importance
    pub fn consolidate_task(&mut self) -> OptimizerResult<()> {
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let Some(omega) = self.path_integral.get(&param_key) {
                // Compute importance: Ω = ω / (Δθ² + ξ)
                if let Some(prev_param) = self.prev_params.get(&param_key) {
                    let param_read = param.read();
                    let delta = param_read.sub(prev_param)?;
                    let delta_sq = delta.mul(&delta)?;
                    let denom = delta_sq.add_scalar(self.config.damping)?;

                    let task_importance = omega.div(&denom)?;

                    // Accumulate importance
                    if let Some(existing) = self.importance.get(&param_key) {
                        let accumulated = existing.add(&task_importance)?;
                        self.importance.insert(param_key.clone(), accumulated);
                    } else {
                        self.importance.insert(param_key.clone(), task_importance);
                    }

                    // Reset path integral
                    let shape_owned = param_read.shape().dims().to_vec();
                    let zeros = torsh_tensor::creation::zeros(&shape_owned)?;
                    self.path_integral.insert(param_key, zeros);
                }
            }
        }

        self.current_task += 1;
        Ok(())
    }

    /// Apply SI penalty
    fn apply_si_penalty(&mut self) -> OptimizerResult<()> {
        if self.importance.is_empty() {
            return Ok(());
        }

        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let (Some(importance), Some(prev_param)) = (
                self.importance.get(&param_key),
                self.prev_params.get(&param_key),
            ) {
                let mut param_write = param.write();

                // Penalty: Ω * (θ - θ_prev)
                let diff = param_write.sub(prev_param)?;
                let penalty = importance.mul(&diff)?;
                let scaled_penalty = penalty.mul_scalar(self.config.importance)?;

                if let Some(grad) = param_write.grad() {
                    let new_grad = grad.add(&scaled_penalty)?;
                    param_write.set_grad(Some(new_grad));
                }
            }
        }

        Ok(())
    }
}

impl<O: Optimizer> Optimizer for SIOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Update path integral before step
        self.update_path_integral()?;

        // Apply SI penalty
        self.apply_si_penalty()?;

        // Call base optimizer
        self.base_optimizer.step()
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("SI({})", state.optimizer_type);
        state
            .global_state
            .insert("current_task".to_string(), self.current_task as f32);
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// MAS (Memory Aware Synapses)
// ============================================================================

/// MAS configuration
#[derive(Debug, Clone)]
pub struct MASConfig {
    /// Importance regularization strength
    pub importance: f32,
    /// Number of samples for importance estimation
    pub n_samples: usize,
}

impl Default for MASConfig {
    fn default() -> Self {
        Self {
            importance: 1.0,
            n_samples: 100,
        }
    }
}

/// Memory Aware Synapses optimizer
///
/// Estimates parameter importance using gradient magnitude at optimal parameters,
/// avoiding the need for data from previous tasks.
pub struct MASOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: MASConfig,
    /// Parameter importance
    importance: HashMap<String, Tensor>,
    /// Optimal parameters from previous tasks
    optimal_params: HashMap<String, Tensor>,
    /// Current task ID
    current_task: usize,
    /// Parameter groups
    param_groups: Vec<Arc<RwLock<Tensor>>>,
}

impl<O: Optimizer> MASOptimizer<O> {
    /// Create a new MAS optimizer
    pub fn new(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
        config: MASConfig,
    ) -> OptimizerResult<Self> {
        Ok(Self {
            base_optimizer,
            config,
            importance: HashMap::new(),
            optimal_params: HashMap::new(),
            current_task: 0,
            param_groups: params,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(
        base_optimizer: O,
        params: Vec<Arc<RwLock<Tensor>>>,
    ) -> OptimizerResult<Self> {
        Self::new(base_optimizer, params, MASConfig::default())
    }

    /// Compute importance using output gradient magnitude
    pub fn compute_importance(&mut self) -> OptimizerResult<()> {
        // Accumulate gradient magnitudes
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            let param_read = param.read();
            if let Some(grad) = param_read.grad() {
                // Importance = |∂L/∂θ|
                let grad_abs = grad.abs()?;

                if let Some(existing) = self.importance.get(&param_key) {
                    let accumulated = existing.add(&grad_abs)?;
                    self.importance.insert(param_key, accumulated);
                } else {
                    self.importance.insert(param_key, grad_abs);
                }
            }
        }

        Ok(())
    }

    /// Consolidate task (save optimal parameters)
    pub fn consolidate_task(&mut self) -> OptimizerResult<()> {
        // Save current parameters
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let param_read = param.read();
            self.optimal_params.insert(param_key, param_read.clone());
        }

        self.current_task += 1;
        Ok(())
    }

    /// Apply MAS penalty
    fn apply_mas_penalty(&mut self) -> OptimizerResult<()> {
        if self.importance.is_empty() {
            return Ok(());
        }

        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let (Some(importance), Some(optimal)) = (
                self.importance.get(&param_key),
                self.optimal_params.get(&param_key),
            ) {
                let mut param_write = param.write();

                // Penalty: Ω * (θ - θ*)
                let diff = param_write.sub(optimal)?;
                let penalty = importance.mul(&diff)?;
                let scaled_penalty = penalty.mul_scalar(self.config.importance)?;

                if let Some(grad) = param_write.grad() {
                    let new_grad = grad.add(&scaled_penalty)?;
                    param_write.set_grad(Some(new_grad));
                }
            }
        }

        Ok(())
    }
}

impl<O: Optimizer> Optimizer for MASOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Apply MAS penalty
        self.apply_mas_penalty()?;

        // Call base optimizer
        self.base_optimizer.step()
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("MAS({})", state.optimizer_type);
        state
            .global_state
            .insert("current_task".to_string(), self.current_task as f32);
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_ewc_config_default() {
        let config = EWCConfig::default();
        assert_eq!(config.importance, 1000.0);
        assert_eq!(config.fisher_sample_size, 200);
        assert!(config.diagonal_fisher);
    }

    #[test]
    fn test_ewc_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);

        let optimizer = EWCOptimizer::with_defaults(base, vec![param])?;
        assert_eq!(optimizer.current_task, 0);
        Ok(())
    }

    #[test]
    fn test_ewc_consolidate_task() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));

        // Set gradient for Fisher computation
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[5, 5])?;
            p.set_grad(Some(grad));
        }

        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut optimizer = EWCOptimizer::with_defaults(base, vec![param])?;

        optimizer.consolidate_task()?;
        assert_eq!(optimizer.current_task, 1);
        assert!(!optimizer.optimal_params.is_empty());
        Ok(())
    }

    #[test]
    fn test_si_config_default() {
        let config = SIConfig::default();
        assert_eq!(config.damping, 0.1);
        assert_eq!(config.importance, 1.0);
    }

    #[test]
    fn test_si_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);

        let optimizer = SIOptimizer::with_defaults(base, vec![param])?;
        assert_eq!(optimizer.current_task, 0);
        Ok(())
    }

    #[test]
    fn test_si_consolidate_task() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[3, 3])?;
            p.set_grad(Some(grad));
        }

        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut optimizer = SIOptimizer::with_defaults(base, vec![param])?;

        optimizer.consolidate_task()?;
        assert_eq!(optimizer.current_task, 1);
        Ok(())
    }

    #[test]
    fn test_mas_config_default() {
        let config = MASConfig::default();
        assert_eq!(config.importance, 1.0);
        assert_eq!(config.n_samples, 100);
    }

    #[test]
    fn test_mas_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);

        let optimizer = MASOptimizer::with_defaults(base, vec![param])?;
        assert_eq!(optimizer.current_task, 0);
        Ok(())
    }

    #[test]
    fn test_mas_compute_importance() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[5, 5])?;
            p.set_grad(Some(grad));
        }

        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut optimizer = MASOptimizer::with_defaults(base, vec![param])?;

        optimizer.compute_importance()?;
        assert!(!optimizer.importance.is_empty());
        Ok(())
    }

    #[test]
    fn test_ewc_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[2, 2])?;
            p.set_grad(Some(grad));
        }

        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut optimizer = EWCOptimizer::with_defaults(base, vec![param])?;

        // Step should succeed
        optimizer.step()?;
        Ok(())
    }
}
