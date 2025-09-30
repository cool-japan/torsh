//! Optimization algorithms for ToRSh
//!
//! This crate provides PyTorch-compatible optimizers built on top of scirs2-optim.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod adabelief;
pub mod adabound;
pub mod adadelta;
pub mod adagrad;
pub mod adahessian;
pub mod adam;
pub mod adamax;
pub mod advanced;
pub mod asgd;
pub mod bayesian_optimization;
pub mod benchmarks;
pub mod checkpointing;
pub mod composition;
pub mod cross_framework_validation;
pub mod debugging;
pub mod differential_privacy;
pub mod distributed;
pub mod evolutionary_strategies;
pub mod ftrl;
pub mod fused_kernels;
pub mod grad_accumulation;
pub mod gradient_free;
pub mod hyperparameter_tuning;
pub mod kfac;
pub mod lamb;
pub mod lazy_updates;
pub mod lbfgs;
pub mod lookahead;
pub mod low_precision;
pub mod lr_scheduler;
pub mod lr_scheduler_additional;
pub mod lr_scheduler_enhanced;
pub mod memory_efficient;
pub mod memory_mapped;
pub mod mixed_precision;
pub mod nadam;
pub mod natural_gradient;
pub mod neural_optimizer;
pub mod newton_cg;
pub mod numerical_stability_tests;
pub mod online_learning;
pub mod optimizer;
pub mod radam;
pub mod ranger;
pub mod rmsprop;
pub mod robustness;
pub mod rprop;
pub mod sgd;
pub mod shampoo;
pub mod sparse_adam;
pub mod sparse_updates;
pub mod state_dict_ops;
pub mod stress_tests;
pub mod trust_region;
pub mod yellowfin;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Optimizer-specific error type
#[derive(Debug, thiserror::Error)]
pub enum OptimizerError {
    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] torsh_core::error::TorshError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("State error: {0}")]
    StateError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Memory map error: {0}")]
    MemoryMapError(String),
}

impl From<OptimizerError> for torsh_core::error::TorshError {
    fn from(err: OptimizerError) -> Self {
        match err {
            OptimizerError::TensorError(e) => e,
            OptimizerError::InvalidParameter(msg) => {
                torsh_core::error::TorshError::InvalidArgument(msg)
            }
            OptimizerError::SerializationError(msg) => {
                torsh_core::error::TorshError::SerializationError(msg)
            }
            OptimizerError::IoError(e) => torsh_core::error::TorshError::IoError(e.to_string()),
            OptimizerError::CheckpointError(msg) => {
                torsh_core::error::TorshError::RuntimeError(msg)
            }
            OptimizerError::ConfigError(msg) => torsh_core::error::TorshError::ConfigError(msg),
            OptimizerError::StateError(msg) => torsh_core::error::TorshError::RuntimeError(msg),
            OptimizerError::InvalidInput(msg) => {
                torsh_core::error::TorshError::InvalidArgument(msg)
            }
            OptimizerError::NumericalError(msg) => torsh_core::error::TorshError::RuntimeError(msg),
            OptimizerError::MemoryMapError(msg) => torsh_core::error::TorshError::RuntimeError(msg),
        }
    }
}

/// Result type for optimizer operations
pub type OptimizerResult<T> = std::result::Result<T, OptimizerError>;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

// Re-export scirs2 optimizer functionality
// use scirs2::optim as sci_optim;

/// Base optimizer trait
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self) -> OptimizerResult<()>;

    /// Zero all gradients
    fn zero_grad(&mut self);

    /// Get the current learning rate
    fn get_lr(&self) -> Vec<f32>;

    /// Set the learning rate
    fn set_lr(&mut self, lr: f32);

    /// Add a parameter group
    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>);

    /// Get state dict for serialization
    fn state_dict(&self) -> OptimizerResult<OptimizerState>;

    /// Load state dict
    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()>;
}

/// Optimizer state for serialization
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Optimizer type identifier
    pub optimizer_type: String,
    /// Version of the state format
    pub version: String,
    /// Parameter group states
    pub param_groups: Vec<ParamGroupState>,
    /// Per-parameter optimizer state (keyed by parameter ID)
    pub state: HashMap<String, HashMap<String, Tensor>>,
    /// Global optimizer state
    pub global_state: HashMap<String, f32>,
}

/// Parameter group state
#[derive(Debug, Clone)]
pub struct ParamGroupState {
    /// Learning rate for this group
    pub lr: f32,
    /// Additional options for this group
    pub options: HashMap<String, f32>,
    /// Number of parameters in this group (for validation)
    pub param_count: usize,
}

impl OptimizerState {
    /// Create a new empty optimizer state
    pub fn new(optimizer_type: String) -> Self {
        Self {
            optimizer_type,
            version: VERSION.to_string(),
            param_groups: Vec::new(),
            state: HashMap::new(),
            global_state: HashMap::new(),
        }
    }

    /// Validate the state structure
    pub fn validate(&self) -> Result<()> {
        if self.optimizer_type.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Optimizer type cannot be empty".to_string(),
            ));
        }

        // Check that all parameter groups are valid
        for (i, group) in self.param_groups.iter().enumerate() {
            if !group.lr.is_finite() || group.lr <= 0.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "Invalid learning rate in group {}",
                    i
                )));
            }
        }

        // Check that all state values are finite
        for (param_id, param_state) in &self.state {
            for (state_name, tensor) in param_state {
                // For now, just check that the keys are valid
                if param_id.is_empty() || state_name.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "State keys cannot be empty".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get the total number of parameters across all groups
    pub fn total_param_count(&self) -> usize {
        self.param_groups.iter().map(|g| g.param_count).sum()
    }

    /// Check if state is compatible with another state (same structure)
    pub fn is_compatible_with(&self, other: &OptimizerState) -> bool {
        self.optimizer_type == other.optimizer_type
            && self.param_groups.len() == other.param_groups.len()
            && self
                .param_groups
                .iter()
                .zip(other.param_groups.iter())
                .all(|(a, b)| a.param_count == b.param_count)
    }
}

impl ParamGroupState {
    /// Create a new parameter group state
    pub fn new(lr: f32, param_count: usize) -> Self {
        Self {
            lr,
            options: HashMap::new(),
            param_count,
        }
    }

    /// Create from a ParamGroup
    pub fn from_param_group(group: &ParamGroup) -> Self {
        Self {
            lr: group.lr,
            options: group.options.clone(),
            param_count: group.params.len(),
        }
    }

    /// Get an option value with a default
    pub fn get_option(&self, key: &str, default: f32) -> f32 {
        self.options.get(key).copied().unwrap_or(default)
    }

    /// Set an option value
    pub fn set_option(&mut self, key: String, value: f32) {
        self.options.insert(key, value);
    }
}

/// Parameter group
#[derive(Debug, Clone)]
pub struct ParamGroup {
    pub params: Vec<Arc<RwLock<Tensor>>>,
    pub lr: f32,
    pub options: HashMap<String, f32>,
}

/// Builder for creating parameter groups with various options
#[derive(Debug)]
pub struct ParamGroupBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    options: HashMap<String, f32>,
}

impl ParamGroupBuilder {
    /// Create a new parameter group builder
    pub fn new(lr: f32) -> Self {
        Self {
            params: Vec::new(),
            lr,
            options: HashMap::new(),
        }
    }

    /// Add parameters to the group
    pub fn params(mut self, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        self.params = params;
        self
    }

    /// Add a single parameter to the group
    pub fn add_param(mut self, param: Arc<RwLock<Tensor>>) -> Self {
        self.params.push(param);
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.options
            .insert("weight_decay".to_string(), weight_decay);
        self
    }

    /// Set epsilon
    pub fn eps(mut self, eps: f32) -> Self {
        self.options.insert("eps".to_string(), eps);
        self
    }

    /// Set a custom option
    pub fn option(mut self, key: String, value: f32) -> Self {
        self.options.insert(key, value);
        self
    }

    /// Set options from OptimizerOptions
    pub fn from_options(mut self, options: &OptimizerOptions) -> Self {
        self.lr = options.lr;
        self.options = options.to_hashmap();
        self.options.remove("lr"); // lr is stored separately
        self
    }

    /// Build the parameter group
    pub fn build(self) -> ParamGroup {
        ParamGroup {
            params: self.params,
            lr: self.lr,
            options: self.options,
        }
    }
}

impl ParamGroup {
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self {
            params,
            lr,
            options: HashMap::new(),
        }
    }

    pub fn with_options(mut self, options: HashMap<String, f32>) -> Self {
        self.options = options;
        self
    }

    /// Add a single parameter to the group
    pub fn add_param(&mut self, param: Arc<RwLock<Tensor>>) {
        self.params.push(param);
    }

    /// Get a specific option value, falling back to a default
    pub fn get_option(&self, key: &str, default: f32) -> f32 {
        self.options.get(key).copied().unwrap_or(default)
    }

    /// Set a specific option value
    pub fn set_option(&mut self, key: String, value: f32) {
        self.options.insert(key, value);
    }

    /// Get the number of parameters in this group
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Check if this group has any parameters
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get all parameters that have gradients
    pub fn params_with_grads(&self) -> Vec<&Arc<RwLock<Tensor>>> {
        self.params
            .iter()
            .filter(|param| param.read().has_grad())
            .collect()
    }

    /// Validate that all parameters in the group are valid
    pub fn validate(&self) -> bool {
        !self.params.is_empty() && self.lr.is_finite() && self.lr > 0.0
    }

    /// Get parameter count for each unique shape in the group
    pub fn get_shape_counts(&self) -> HashMap<Vec<usize>, usize> {
        let mut shape_counts = HashMap::new();
        for param in &self.params {
            let shape = param.read().shape().dims().to_vec();
            *shape_counts.entry(shape).or_insert(0) += 1;
        }
        shape_counts
    }

    /// Get total number of parameters (not tensors, but individual parameters)
    pub fn total_param_count(&self) -> usize {
        self.params.iter().map(|param| param.read().numel()).sum()
    }

    /// Clear gradients for all parameters in this group
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.write().zero_grad();
        }
    }

    /// Check if any parameter in the group has gradients
    pub fn has_any_grads(&self) -> bool {
        self.params.iter().any(|param| param.read().has_grad())
    }

    /// Get gradient norm for all parameters in the group
    pub fn grad_norm(&self) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for param in &self.params {
            let param_guard = param.read();
            if let Some(grad) = param_guard.grad() {
                let grad_norm = grad.norm().map_err(|e| {
                    TorshError::Other(format!("Failed to compute gradient norm: {}", e))
                })?;
                let norm_value = grad_norm.to_vec().map_err(|e| {
                    TorshError::Other(format!("Failed to extract norm value: {}", e))
                })?[0];
                total_norm_sq += norm_value * norm_value;
            }
        }

        Ok(total_norm_sq.sqrt())
    }

    /// Apply gradient clipping to all parameters in the group
    pub fn clip_grads(&self, max_norm: f32) -> Result<f32> {
        let total_norm = self.grad_norm()?;

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in &self.params {
                let mut param_guard = param.write();
                if let Some(grad) = param_guard.grad() {
                    let clipped_grad = grad.mul_scalar(scale).map_err(|e| {
                        TorshError::Other(format!("Failed to clip gradient: {}", e))
                    })?;
                    param_guard.set_grad(Some(clipped_grad));
                }
            }
        }

        Ok(total_norm)
    }
}

/// Common optimizer options
#[derive(Debug, Clone)]
pub struct OptimizerOptions {
    pub lr: f32,
    pub weight_decay: f32,
    pub eps: f32,
    pub maximize: bool,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 0.0,
            eps: 1e-8,
            maximize: false,
        }
    }
}

impl OptimizerOptions {
    /// Create new optimizer options with specified learning rate
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon value for numerical stability
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set maximize flag (for maximization problems)
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Convert to HashMap for compatibility with parameter groups
    pub fn to_hashmap(&self) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        map.insert("lr".to_string(), self.lr);
        map.insert("weight_decay".to_string(), self.weight_decay);
        map.insert("eps".to_string(), self.eps);
        map.insert(
            "maximize".to_string(),
            if self.maximize { 1.0 } else { 0.0 },
        );
        map
    }

    /// Create from HashMap
    pub fn from_hashmap(map: &HashMap<String, f32>) -> Self {
        Self {
            lr: map.get("lr").copied().unwrap_or(1e-3),
            weight_decay: map.get("weight_decay").copied().unwrap_or(0.0),
            eps: map.get("eps").copied().unwrap_or(1e-8),
            maximize: map.get("maximize").copied().unwrap_or(0.0) > 0.0,
        }
    }

    /// Validate the options are reasonable
    pub fn validate(&self) -> Result<()> {
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Learning rate must be positive and finite".to_string(),
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Weight decay must be non-negative and finite".to_string(),
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Epsilon must be positive and finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Create standardized state dict for any optimizer
    pub fn create_standard_state_dict(
        optimizer_type: &str,
        version: Option<&str>,
        param_groups: &[ParamGroup],
        state: &HashMap<String, HashMap<String, Tensor>>,
        global_state: Option<HashMap<String, f32>>,
    ) -> OptimizerState {
        let param_group_states = param_groups
            .iter()
            .map(|g| ParamGroupState::from_param_group(g))
            .collect();

        let mut optimizer_state = OptimizerState {
            optimizer_type: optimizer_type.to_string(),
            version: version.unwrap_or("1.0").to_string(),
            param_groups: param_group_states,
            state: state.clone(),
            global_state: global_state.unwrap_or_default(),
        };

        optimizer_state
    }

    /// Validate state dict compatibility between optimizers
    pub fn validate_state_compatibility(
        current_groups: &[ParamGroup],
        state_groups: &[ParamGroupState],
    ) -> Result<()> {
        if current_groups.len() != state_groups.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Parameter group count mismatch: expected {}, got {}",
                current_groups.len(),
                state_groups.len()
            )));
        }

        for (i, (current_group, state_group)) in
            current_groups.iter().zip(state_groups.iter()).enumerate()
        {
            if current_group.params.len() != state_group.param_count {
                return Err(TorshError::InvalidArgument(format!(
                    "Parameter count mismatch in group {}: expected {}, got {}",
                    i,
                    current_group.params.len(),
                    state_group.param_count
                )));
            }
        }

        Ok(())
    }
}

/// Prelude module for convenient imports
/// Convergence testing utilities
#[cfg(test)]
pub mod convergence_tests {
    use super::*;
    use parking_lot::RwLock;
    use std::ops::Add;
    use std::sync::Arc;
    use torsh_tensor::{
        creation::{randn, zeros},
        Tensor,
    };

    /// Test that an optimizer can minimize a simple quadratic function
    pub fn test_quadratic_convergence<O: Optimizer>(
        create_optimizer: impl Fn(Vec<Arc<RwLock<Tensor>>>) -> O,
        tolerance: f32,
        max_iterations: usize,
    ) -> Result<()> {
        // Create a simple quadratic function: f(x) = x^2 + y^2
        let x = Arc::new(RwLock::new(Tensor::scalar(2.0)?));
        let y = Arc::new(RwLock::new(Tensor::scalar(2.0)?));
        let params = vec![x.clone(), y.clone()];

        let mut optimizer = create_optimizer(params);

        for i in 0..max_iterations {
            // Compute gradients: df/dx = 2x, df/dy = 2y
            {
                let x_val = x.read().clone();
                let y_val = y.read().clone();

                let x_grad = x_val.mul_scalar(2.0)?;
                let y_grad = y_val.mul_scalar(2.0)?;

                x.write().set_grad(Some(x_grad));
                y.write().set_grad(Some(y_grad));
            }

            // Optimizer step
            optimizer
                .step()
                .map_err(|e| TorshError::Other(format!("Optimizer step failed: {}", e)))?;

            // Check convergence
            let x_val = x.read().to_vec()?[0];
            let y_val = y.read().to_vec()?[0];
            let loss = x_val * x_val + y_val * y_val;

            if loss < tolerance {
                return Ok(());
            }

            // Clear gradients for next iteration
            optimizer.zero_grad();
        }

        Err(TorshError::Other(format!(
            "Failed to converge within {} iterations",
            max_iterations
        )))
    }

    /// Test that an optimizer can minimize a linear regression problem
    pub fn test_linear_regression_convergence<O: Optimizer>(
        create_optimizer: impl Fn(Vec<Arc<RwLock<Tensor>>>) -> O,
        tolerance: f32,
        max_iterations: usize,
    ) -> Result<()> {
        // Create a simple linear regression problem: y = 2x + 1 + noise
        let true_weight = 2.0;
        let true_bias = 1.0;

        // Generate synthetic data
        let n_samples = 100;
        let x_data = randn::<f32>(&[n_samples, 1])?;
        let noise = randn::<f32>(&[n_samples, 1])?.mul_scalar(0.1)?;
        let y_data = x_data
            .mul_scalar(true_weight)?
            .add_scalar(true_bias)?
            .add(&noise)?;

        // Initialize parameters
        let weight = Arc::new(RwLock::new(zeros(&[1, 1])?));
        let bias = Arc::new(RwLock::new(zeros(&[1])?));
        let params = vec![weight.clone(), bias.clone()];

        let mut optimizer = create_optimizer(params);

        for i in 0..max_iterations {
            // Forward pass: y_pred = x * weight + bias
            let w_val = weight.read().clone();
            let b_val = bias.read().clone();

            let y_pred = x_data.matmul(&w_val)?.add(&b_val)?;

            // Compute loss: MSE = mean((y_pred - y_true)^2)
            let diff = y_pred.sub(&y_data)?;
            let loss_tensor = diff.pow(2.0)?.mean(Some(&[0]), false)?;
            let loss = loss_tensor.to_vec()?[0];

            // Compute gradients
            let grad_scale = 2.0 / n_samples as f32;
            let weight_grad = x_data
                .transpose(0, 1)?
                .matmul(&diff)?
                .mul_scalar(grad_scale)?;
            let bias_grad = diff.sum()?.mul_scalar(grad_scale)?;

            weight.write().set_grad(Some(weight_grad));
            bias.write().set_grad(Some(bias_grad));

            // Optimizer step
            optimizer
                .step()
                .map_err(|e| TorshError::Other(format!("Optimizer step failed: {}", e)))?;

            // Check convergence
            if loss < tolerance {
                // Verify the learned parameters are close to true values
                let learned_weight = weight.read().to_vec()?[0];
                let learned_bias = bias.read().to_vec()?[0];

                if (learned_weight - true_weight).abs() < 0.1
                    && (learned_bias - true_bias).abs() < 0.1
                {
                    return Ok(());
                }
            }

            // Clear gradients for next iteration
            optimizer.zero_grad();
        }

        Err(TorshError::Other(format!(
            "Failed to converge within {} iterations",
            max_iterations
        )))
    }

    /// Test that an optimizer maintains consistent behavior across multiple runs
    pub fn test_optimizer_consistency<O: Optimizer>(
        create_optimizer: impl Fn(Vec<Arc<RwLock<Tensor>>>) -> O,
        n_runs: usize,
        tolerance: f32,
    ) -> Result<()> {
        let mut final_values = Vec::new();

        for run in 0..n_runs {
            let param = Arc::new(RwLock::new(Tensor::scalar(1.0)?));
            let params = vec![param.clone()];
            let mut optimizer = create_optimizer(params);

            // Run for a fixed number of steps
            for _ in 0..10 {
                {
                    let param_val = param.read().clone();
                    let grad = param_val.mul_scalar(2.0)?; // Simple gradient
                    param.write().set_grad(Some(grad));
                }

                optimizer
                    .step()
                    .map_err(|e| TorshError::Other(format!("Optimizer step failed: {}", e)))?;
                optimizer.zero_grad();
            }

            final_values.push(param.read().to_vec()?[0]);
        }

        // Check that all runs produce similar results
        let mean_value = final_values.iter().sum::<f32>() / final_values.len() as f32;
        for &value in &final_values {
            if (value - mean_value).abs() > tolerance {
                return Err(TorshError::Other(format!(
                    "Inconsistent optimizer behavior: values vary by more than {}",
                    tolerance
                )));
            }
        }

        Ok(())
    }
}

pub mod prelude {
    pub use crate::adabelief::AdaBelief;
    pub use crate::adabound::AdaBound;
    pub use crate::adadelta::AdaDelta;
    pub use crate::adagrad::AdaGrad;
    pub use crate::adahessian::{AdaHessian, AdaHessianBuilder};
    pub use crate::adam::{Adam, AdamW};
    pub use crate::adamax::AdaMax;
    pub use crate::asgd::ASGD;
    pub use crate::checkpointing::{
        Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata, CheckpointStatistics,
        CheckpointSupport, CheckpointingOptimizer,
    };
    pub use crate::composition::{
        CombinationMethod, ComposedOptimizer, CompositionBuilder, CompositionStrategy,
        OptimizerMetrics, SwitchCriterion, VotingMethod,
    };
    pub use crate::debugging::{
        AnalysisReport, AnalyzerConfig, ConvergenceTracker, GradientFlowPoint, GradientStatistics,
        HyperparameterSensitivity, OptimizationRecommendation, OptimizationStep, OptimizerAnalyzer,
        ParameterStatistics, RecommendationCategory, SensitivityReport, SensitivityResult,
        Severity,
    };
    pub use crate::distributed::{
        utils as distributed_utils, AsyncConfig, AsyncSGD, CommunicationStats, DistributedBackend,
        DistributedConfig, DistributedOptimizer, ElasticAveragingSGD, SyncStrategy,
    };
    pub use crate::ftrl::{FTRLBuilder, FTRL};
    pub use crate::fused_kernels::{
        fused_adadelta_step, fused_adagrad_step, fused_adam_step, fused_rmsprop_step,
        fused_sgd_step, FusedKernelSupport, FusedStats,
    };
    pub use crate::grad_accumulation::{
        with_gradient_accumulation, AccumulatingOptimizer, GradientAccumulationSupport,
        GradientAccumulator,
    };
    pub use crate::kfac::{KFACBuilder, KFAC};
    pub use crate::lamb::LAMB;
    pub use crate::lazy_updates::{
        LazyUpdateConfig, LazyUpdateDecision, LazyUpdateManager, LazyUpdateOptimizer,
        LazyUpdateStatistics, LazyUpdateSupport, ParameterImportance, PendingUpdate,
        UpdatePriority,
    };
    pub use crate::lbfgs::LBFGS;
    pub use crate::lookahead::{lookahead_adam, lookahead_radam, lookahead_sgd, Lookahead};
    pub use crate::low_precision::{
        LowPrecisionConvertible, LowPrecisionOptimizer, LowPrecisionState, PrecisionType,
        StateStatistics,
    };
    pub use crate::lr_scheduler::{
        CosineAnnealingLR, ExponentialLR, LRScheduler, OneCycleLR, ReduceLROnPlateau, StepLR,
    };
    pub use crate::lr_scheduler_additional::{
        ConstantLR, CosineAnnealingWarmRestarts, CyclicLR, LinearLR, MultiStepLR, PolynomialLR,
    };
    pub use crate::lr_scheduler_enhanced::{
        utils as lr_enhanced_utils, AdaptiveLRScheduler, AdaptiveSchedulerStats, AdaptiveStrategy,
        CosineAnnealingWarmRestartsWithWarmup, PolynomialDecayWithWarmup, WarmupStrategy,
    };
    pub use crate::memory_efficient::{
        CircularBuffer, MemoryConfig, MemoryEfficientAdam, MemoryEfficientLBFGS,
        MemoryEfficientOptimizerBuilder, MemoryPool,
    };
    pub use crate::memory_mapped::{
        MemoryMappedConfig, MemoryMappedFile, MemoryMappedOptimizer, MemoryMappedStateStorage,
        MemoryMappedSupport, StorageStatistics,
    };
    pub use crate::mixed_precision::{
        with_mixed_precision, MixedPrecisionConfig, MixedPrecisionOptimizer,
    };
    pub use crate::nadam::NAdam;
    pub use crate::natural_gradient::{NaturalGradient, NaturalGradientBuilder};
    pub use crate::newton_cg::{NewtonCG, NewtonCGBuilder, NewtonCGConfig};
    pub use crate::online_learning::{
        OnlineGradientDescent, ProximalGradient, ProximalOperator, SAGA, SVRG,
    };
    pub use crate::radam::RAdam;
    pub use crate::ranger::{Ranger, RangerBuilder};
    pub use crate::rmsprop::RMSprop;
    pub use crate::rprop::Rprop;
    pub use crate::sgd::SGD;
    pub use crate::shampoo::{Shampoo, ShampooBuilder};
    pub use crate::sparse_adam::SparseAdam;
    pub use crate::state_dict_ops::{
        CompressionMethod, CompressionStats, MemoryEstimate, SerializationFormat, StateDictConfig,
        StateDictManager,
    };
    pub use crate::trust_region::{
        SubproblemSolver, TrustRegionBuilder, TrustRegionConfig, TrustRegionMethod,
        TrustRegionStrategy,
    };
    pub use crate::yellowfin::{YellowFin, YellowFinBuilder, YellowFinConfig};
    pub use crate::{Optimizer, OptimizerOptions, OptimizerState, ParamGroup, ParamGroupBuilder};
    pub use crate::{OptimizerError, OptimizerResult};
}

// Re-export commonly used types
pub use adam::{Adam, AdamW};
pub use distributed::{DistributedBackend, DistributedConfig, DistributedOptimizer, SyncStrategy};
pub use sgd::SGD;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_group() {
        let params = vec![];
        let group = ParamGroup::new(params, 0.01);
        assert_eq!(group.lr, 0.01);
    }
}
