//! Core infrastructure for functional neural network operations
//!
//! This module provides the foundational components for the functional API,
//! including configuration, validation, numerical stability, and performance utilities.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// FUNCTIONAL API CONFIGURATION AND STANDARDS
// =============================================================================

/// Standard configuration for functional operations
#[derive(Debug, Clone)]
pub struct FunctionalConfig {
    /// Enable input validation (slower but safer)
    pub validate_inputs: bool,
    /// Numerical stability epsilon
    pub eps: f32,
    /// Whether to use in-place operations when possible
    pub inplace: bool,
    /// Memory optimization level
    pub memory_opt: MemoryOptLevel,
}

/// Memory optimization levels for functional operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOptLevel {
    /// No memory optimization (fastest but uses more memory)
    None,
    /// Balanced memory usage and performance
    Balanced,
    /// Maximum memory efficiency (slower but minimal memory usage)
    Maximum,
}

impl Default for FunctionalConfig {
    fn default() -> Self {
        Self {
            validate_inputs: true,
            eps: 1e-8,
            inplace: false,
            memory_opt: MemoryOptLevel::Balanced,
        }
    }
}

/// Functional operation result with enhanced error context
pub type FuncResult<T> = Result<T>;

/// Builder pattern for functional operations with configuration
#[derive(Debug, Clone)]
pub struct FunctionalBuilder {
    config: FunctionalConfig,
}

impl FunctionalBuilder {
    /// Create a new functional builder with default configuration
    pub fn new() -> Self {
        Self {
            config: FunctionalConfig::default(),
        }
    }

    /// Enable or disable input validation
    pub fn validate(mut self, validate: bool) -> Self {
        self.config.validate_inputs = validate;
        self
    }

    /// Set numerical stability epsilon
    pub fn eps(mut self, eps: f32) -> Self {
        self.config.eps = eps;
        self
    }

    /// Enable in-place operations
    pub fn inplace(mut self, inplace: bool) -> Self {
        self.config.inplace = inplace;
        self
    }

    /// Set memory optimization level
    pub fn memory_opt(mut self, level: MemoryOptLevel) -> Self {
        self.config.memory_opt = level;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> FunctionalConfig {
        self.config
    }
}

impl Default for FunctionalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Global default configuration for functional operations
static DEFAULT_CONFIG: FunctionalConfig = FunctionalConfig {
    validate_inputs: true,
    eps: 1e-8,
    inplace: false,
    memory_opt: MemoryOptLevel::Balanced,
};

/// Get the global default configuration
pub fn default_config() -> &'static FunctionalConfig {
    &DEFAULT_CONFIG
}

/// Create a functional builder with optimized defaults
pub fn optimized() -> FunctionalBuilder {
    FunctionalBuilder::new()
        .memory_opt(MemoryOptLevel::Maximum)
        .inplace(true)
}

/// Create a functional builder with safe defaults
pub fn safe() -> FunctionalBuilder {
    FunctionalBuilder::new()
        .validate(true)
        .memory_opt(MemoryOptLevel::None)
}

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/// Standardized input validation utilities
pub mod validation {
    use super::*;
    use torsh_core::TensorElement;

    /// Validate tensor is not empty
    pub fn validate_not_empty<T: TensorElement>(tensor: &Tensor<T>, name: &str) -> FuncResult<()> {
        if tensor.shape().numel() == 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Tensor '{}' cannot be empty",
                name
            )));
        }
        Ok(())
    }

    /// Validate tensor has expected number of dimensions
    pub fn validate_ndim(tensor: &Tensor, expected: usize, name: &str) -> FuncResult<()> {
        let actual = tensor.shape().dims().len();
        if actual != expected {
            return Err(TorshError::InvalidArgument(format!(
                "Tensor '{}' expected {}D, got {}D",
                name, expected, actual
            )));
        }
        Ok(())
    }

    /// Validate tensor has minimum number of dimensions
    pub fn validate_min_ndim(tensor: &Tensor, min_dims: usize, name: &str) -> FuncResult<()> {
        let actual = tensor.shape().dims().len();
        if actual < min_dims {
            return Err(TorshError::InvalidArgument(format!(
                "Tensor '{}' requires at least {}D, got {}D",
                name, min_dims, actual
            )));
        }
        Ok(())
    }

    /// Validate dimensions are compatible
    pub fn validate_compatible_shapes(a: &Tensor, b: &Tensor, op_name: &str) -> FuncResult<()> {
        let a_shape_binding = a.shape();
        let b_shape_binding = b.shape();
        let a_shape = a_shape_binding.dims();
        let b_shape = b_shape_binding.dims();

        // Simple compatibility check - more sophisticated broadcasting logic can be added
        if a_shape != b_shape && a.shape().numel() != 1 && b.shape().numel() != 1 {
            // Allow different shapes only if they are broadcastable
            if !are_broadcastable(a_shape, b_shape) {
                return Err(TorshError::InvalidArgument(format!(
                    "Incompatible shapes for {}: {:?} vs {:?}",
                    op_name, a_shape, b_shape
                )));
            }
        }
        Ok(())
    }

    /// Check if two shapes are broadcastable
    fn are_broadcastable(a: &[usize], b: &[usize]) -> bool {
        let max_len = a.len().max(b.len());
        let a_padded: Vec<usize> = (0..max_len)
            .map(|i| {
                if i < max_len - a.len() {
                    1
                } else {
                    a[i - (max_len - a.len())]
                }
            })
            .collect();
        let b_padded: Vec<usize> = (0..max_len)
            .map(|i| {
                if i < max_len - b.len() {
                    1
                } else {
                    b[i - (max_len - b.len())]
                }
            })
            .collect();

        a_padded
            .iter()
            .zip(b_padded.iter())
            .all(|(a_dim, b_dim)| *a_dim == *b_dim || *a_dim == 1 || *b_dim == 1)
    }

    /// Validate parameter is in valid range
    pub fn validate_range<T: PartialOrd + core::fmt::Display>(
        value: T,
        min: T,
        max: T,
        name: &str,
    ) -> FuncResult<()> {
        if value < min || value > max {
            return Err(TorshError::InvalidArgument(format!(
                "Parameter '{}' value {} not in range [{}, {}]",
                name, value, min, max
            )));
        }
        Ok(())
    }

    /// Validate parameter is positive
    pub fn validate_positive<T: PartialOrd + Default + core::fmt::Display>(
        value: T,
        name: &str,
    ) -> FuncResult<()> {
        if value <= T::default() {
            return Err(TorshError::InvalidArgument(format!(
                "Parameter '{}' must be positive, got {}",
                name, value
            )));
        }
        Ok(())
    }
}

// =============================================================================
// NUMERICAL STABILITY UTILITIES
// =============================================================================

/// Standardized numerical utilities
pub mod numerics {
    use super::*;

    /// Create epsilon tensor for numerical stability
    pub fn epsilon_tensor(like: &Tensor, eps: f32) -> FuncResult<Tensor> {
        torsh_tensor::creation::full_like(like, eps)
    }

    /// Clamp tensor values to prevent numerical issues
    pub fn safe_clamp(tensor: &Tensor, min_val: f32, max_val: f32) -> FuncResult<Tensor> {
        // Implement clamp using max and min operations: clamp(x, min, max) = max(min, min(x, max))
        let max_tensor = torsh_tensor::creation::full_like(tensor, max_val)?;
        let min_tensor = torsh_tensor::creation::full_like(tensor, min_val)?;
        let clamped_max = tensor.minimum(&max_tensor)?;
        clamped_max.maximum(&min_tensor)
    }

    /// Safe division with numerical stability
    pub fn safe_div(numerator: &Tensor, denominator: &Tensor, eps: f32) -> FuncResult<Tensor> {
        let eps_tensor = epsilon_tensor(denominator, eps)?;
        let safe_denom = denominator.add(&eps_tensor)?;
        numerator.div(&safe_denom)
    }

    /// Safe square root with numerical stability
    pub fn safe_sqrt(tensor: &Tensor, eps: f32) -> FuncResult<Tensor> {
        let eps_tensor = epsilon_tensor(tensor, eps)?;
        let safe_tensor = tensor.add(&eps_tensor)?;
        safe_tensor.sqrt()
    }

    /// Safe reciprocal square root
    pub fn safe_rsqrt(tensor: &Tensor, eps: f32) -> FuncResult<Tensor> {
        let eps_tensor = epsilon_tensor(tensor, eps)?;
        let safe_tensor = tensor.add(&eps_tensor)?;
        safe_tensor.rsqrt()
    }
}

// =============================================================================
// PERFORMANCE UTILITIES
// =============================================================================

/// Performance utilities for functional operations
pub mod performance {
    use super::*;

    /// Execute operation with memory optimization
    pub fn with_memory_opt<F, T>(config: &FunctionalConfig, op: F) -> FuncResult<T>
    where
        F: FnOnce() -> FuncResult<T>,
    {
        // Future: implement memory pool management, garbage collection hints, etc.
        match config.memory_opt {
            MemoryOptLevel::None => op(),
            MemoryOptLevel::Balanced => {
                // Future: balanced optimization
                op()
            }
            MemoryOptLevel::Maximum => {
                // Future: maximum memory efficiency
                op()
            }
        }
    }

    /// Check if in-place operation is beneficial
    pub fn should_use_inplace(config: &FunctionalConfig, tensor_size: usize) -> bool {
        config.inplace && tensor_size > 1000 // Simple heuristic
    }
}

// =============================================================================
// MACROS FOR CONSISTENT PATTERNS
// =============================================================================

/// Macro for consistent function validation patterns
#[macro_export]
macro_rules! validate_inputs {
    ($config:expr, $($validation:expr),*) => {
        if $config.validate_inputs {
            $( $validation?; )*
        }
    };
}

/// Macro for consistent error wrapping
#[macro_export]
macro_rules! func_error {
    ($op:expr, $context:expr) => {
        $op.map_err(|e| torsh_core::error::TorshError::RuntimeError(format!("{}: {}", $context, e)))
    };
}

// =============================================================================
// ACTIVATION FUNCTION CONFIGURATION
// =============================================================================

/// Activation function configuration
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Base functional configuration
    pub base: FunctionalConfig,
    /// Whether to clamp outputs to prevent numerical issues
    pub clamp_output: bool,
    /// Output clamping range
    pub clamp_range: (f32, f32),
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            base: FunctionalConfig::default(),
            clamp_output: false,
            clamp_range: (-10.0, 10.0),
        }
    }
}

/// Standardized activation function trait
pub trait Activation {
    /// Apply activation function
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor>;

    /// Apply activation with configuration
    fn apply_with_config(&self, input: &Tensor, config: &ActivationConfig) -> FuncResult<Tensor> {
        let result = self.apply(input)?;
        if config.clamp_output {
            let (min_val, max_val) = config.clamp_range;
            numerics::safe_clamp(&result, min_val, max_val)
        } else {
            Ok(result)
        }
    }
}
