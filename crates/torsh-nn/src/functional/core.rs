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

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // =========================================================================
    // CONFIGURATION TESTS
    // =========================================================================

    #[test]
    fn test_functional_config_default() {
        let config = FunctionalConfig::default();
        assert!(config.validate_inputs);
        assert_relative_eq!(config.eps, 1e-8);
        assert!(!config.inplace);
        assert_eq!(config.memory_opt, MemoryOptLevel::Balanced);
    }

    #[test]
    fn test_functional_builder_basic() {
        let config = FunctionalBuilder::new()
            .validate(false)
            .eps(1e-6)
            .inplace(true)
            .memory_opt(MemoryOptLevel::Maximum)
            .build();

        assert!(!config.validate_inputs);
        assert_relative_eq!(config.eps, 1e-6);
        assert!(config.inplace);
        assert_eq!(config.memory_opt, MemoryOptLevel::Maximum);
    }

    #[test]
    fn test_functional_builder_optimized() {
        let config = optimized().build();
        assert!(config.inplace);
        assert_eq!(config.memory_opt, MemoryOptLevel::Maximum);
    }

    #[test]
    fn test_functional_builder_safe() {
        let config = safe().build();
        assert!(config.validate_inputs);
        assert_eq!(config.memory_opt, MemoryOptLevel::None);
    }

    #[test]
    fn test_activation_config_default() {
        let config = ActivationConfig::default();
        assert!(!config.clamp_output);
        assert_eq!(config.clamp_range, (-10.0, 10.0));
    }

    // =========================================================================
    // VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_validate_not_empty_valid() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        validation::validate_not_empty(&tensor, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_not_empty_invalid() {
        let tensor: Tensor = Tensor::from_vec(vec![], &[0]).unwrap();
        let result = validation::validate_not_empty(&tensor, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ndim_valid() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0; 12], &[3, 4])?;
        validation::validate_ndim(&tensor, 2, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_ndim_invalid() {
        let tensor = Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let result = validation::validate_ndim(&tensor, 3, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_min_ndim_valid() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0; 24], &[2, 3, 4])?;
        validation::validate_min_ndim(&tensor, 2, "test")?;
        validation::validate_min_ndim(&tensor, 3, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_min_ndim_invalid() {
        let tensor = Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap();
        let result = validation::validate_min_ndim(&tensor, 3, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_range_valid() -> Result<()> {
        validation::validate_range(5.0, 0.0, 10.0, "test")?;
        validation::validate_range(0.0, 0.0, 10.0, "test")?;
        validation::validate_range(10.0, 0.0, 10.0, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_range_invalid() {
        assert!(validation::validate_range(-1.0, 0.0, 10.0, "test").is_err());
        assert!(validation::validate_range(11.0, 0.0, 10.0, "test").is_err());
    }

    #[test]
    fn test_validate_positive_valid() -> Result<()> {
        validation::validate_positive(1.0, "test")?;
        validation::validate_positive(0.001, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_positive_invalid() {
        assert!(validation::validate_positive(0.0, "test").is_err());
        assert!(validation::validate_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_compatible_shapes_same() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0; 12], &[3, 4])?;
        let b = Tensor::from_vec(vec![2.0; 12], &[3, 4])?;
        validation::validate_compatible_shapes(&a, &b, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_compatible_shapes_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0; 12], &[3, 4])?;
        let scalar = Tensor::from_vec(vec![2.0], &[1])?;
        validation::validate_compatible_shapes(&a, &scalar, "test")?;
        Ok(())
    }

    #[test]
    fn test_validate_compatible_shapes_broadcastable() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0; 12], &[3, 4])?;
        let b = Tensor::from_vec(vec![2.0; 4], &[1, 4])?;
        validation::validate_compatible_shapes(&a, &b, "test")?;
        Ok(())
    }

    // =========================================================================
    // NUMERICAL UTILITIES TESTS
    // =========================================================================

    #[test]
    fn test_epsilon_tensor() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let eps = numerics::epsilon_tensor(&tensor, 1e-5)?;

        let eps_data = eps.to_vec()?;
        assert_eq!(eps_data.len(), 3);
        for &val in eps_data.iter() {
            assert_relative_eq!(val, 1e-5, epsilon = 1e-10);
        }
        Ok(())
    }

    #[test]
    fn test_safe_clamp_basic() -> Result<()> {
        let tensor = Tensor::from_vec(vec![-5.0, 0.0, 5.0, 10.0, 15.0], &[5])?;
        let clamped = numerics::safe_clamp(&tensor, 0.0, 10.0)?;

        let clamped_data = clamped.to_vec()?;
        assert_relative_eq!(clamped_data[0], 0.0, epsilon = 1e-6); // -5 clamped to 0
        assert_relative_eq!(clamped_data[1], 0.0, epsilon = 1e-6); // 0 stays 0
        assert_relative_eq!(clamped_data[2], 5.0, epsilon = 1e-6); // 5 stays 5
        assert_relative_eq!(clamped_data[3], 10.0, epsilon = 1e-6); // 10 stays 10
        assert_relative_eq!(clamped_data[4], 10.0, epsilon = 1e-6); // 15 clamped to 10
        Ok(())
    }

    #[test]
    fn test_safe_clamp_negative_range() -> Result<()> {
        let tensor = Tensor::from_vec(vec![-10.0, -5.0, 0.0, 5.0], &[4])?;
        let clamped = numerics::safe_clamp(&tensor, -6.0, -2.0)?;

        let clamped_data = clamped.to_vec()?;
        assert_relative_eq!(clamped_data[0], -6.0, epsilon = 1e-6); // -10 clamped to -6
        assert_relative_eq!(clamped_data[1], -5.0, epsilon = 1e-6); // -5 stays -5
        assert_relative_eq!(clamped_data[2], -2.0, epsilon = 1e-6); // 0 clamped to -2
        assert_relative_eq!(clamped_data[3], -2.0, epsilon = 1e-6); // 5 clamped to -2
        Ok(())
    }

    #[test]
    fn test_safe_div_basic() -> Result<()> {
        let numerator = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
        let denominator = Tensor::from_vec(vec![2.0, 4.0, 5.0], &[3])?;
        let result = numerics::safe_div(&numerator, &denominator, 1e-8)?;

        let result_data = result.to_vec()?;
        assert_relative_eq!(result_data[0], 5.0, epsilon = 1e-5);
        assert_relative_eq!(result_data[1], 5.0, epsilon = 1e-5);
        assert_relative_eq!(result_data[2], 6.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_safe_div_near_zero() -> Result<()> {
        let numerator = Tensor::from_vec(vec![1.0], &[1])?;
        let denominator = Tensor::from_vec(vec![0.0], &[1])?;
        let result = numerics::safe_div(&numerator, &denominator, 1e-8)?;

        // Should not crash, should give very large but finite result
        let result_data = result.to_vec()?;
        assert!(result_data[0].is_finite());
        Ok(())
    }

    #[test]
    fn test_safe_sqrt_basic() -> Result<()> {
        let tensor = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3])?;
        let result = numerics::safe_sqrt(&tensor, 1e-8)?;

        let result_data = result.to_vec()?;
        assert_relative_eq!(result_data[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result_data[1], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result_data[2], 4.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_safe_sqrt_near_zero() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0], &[1])?;
        let result = numerics::safe_sqrt(&tensor, 1e-8)?;

        // Should not crash, epsilon prevents negative sqrt
        let result_data = result.to_vec()?;
        assert!(result_data[0] > 0.0);
        Ok(())
    }

    #[test]
    fn test_safe_rsqrt_basic() -> Result<()> {
        let tensor = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3])?;
        let result = numerics::safe_rsqrt(&tensor, 1e-8)?;

        let result_data = result.to_vec()?;
        assert_relative_eq!(result_data[0], 0.5, epsilon = 1e-5); // 1/sqrt(4) = 0.5
        assert_relative_eq!(result_data[1], 1.0 / 3.0, epsilon = 1e-5); // 1/sqrt(9) = 1/3
        assert_relative_eq!(result_data[2], 0.25, epsilon = 1e-5); // 1/sqrt(16) = 0.25
        Ok(())
    }

    #[test]
    fn test_safe_rsqrt_near_zero() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0], &[1])?;
        let result = numerics::safe_rsqrt(&tensor, 1e-8)?;

        // Should not crash, epsilon prevents division by zero
        let result_data = result.to_vec()?;
        assert!(result_data[0].is_finite());
        Ok(())
    }

    // =========================================================================
    // PERFORMANCE UTILITIES TESTS
    // =========================================================================

    #[test]
    fn test_with_memory_opt_none() -> Result<()> {
        let config = FunctionalBuilder::new()
            .memory_opt(MemoryOptLevel::None)
            .build();

        let result = performance::with_memory_opt(&config, || Ok(42))?;
        assert_eq!(result, 42);
        Ok(())
    }

    #[test]
    fn test_with_memory_opt_balanced() -> Result<()> {
        let config = FunctionalBuilder::new()
            .memory_opt(MemoryOptLevel::Balanced)
            .build();

        let result = performance::with_memory_opt(&config, || Ok(42))?;
        assert_eq!(result, 42);
        Ok(())
    }

    #[test]
    fn test_with_memory_opt_maximum() -> Result<()> {
        let config = FunctionalBuilder::new()
            .memory_opt(MemoryOptLevel::Maximum)
            .build();

        let result = performance::with_memory_opt(&config, || Ok(42))?;
        assert_eq!(result, 42);
        Ok(())
    }

    #[test]
    fn test_should_use_inplace_small() {
        let config = FunctionalBuilder::new().inplace(true).build();
        assert!(!performance::should_use_inplace(&config, 100));
        assert!(!performance::should_use_inplace(&config, 1000));
    }

    #[test]
    fn test_should_use_inplace_large() {
        let config = FunctionalBuilder::new().inplace(true).build();
        assert!(performance::should_use_inplace(&config, 1001));
        assert!(performance::should_use_inplace(&config, 10000));
    }

    #[test]
    fn test_should_use_inplace_disabled() {
        let config = FunctionalBuilder::new().inplace(false).build();
        assert!(!performance::should_use_inplace(&config, 10000));
    }
}
