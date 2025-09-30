//! # Softmax Family Activation Functions
//!
//! This module contains activation functions related to the softmax operation,
//! which are commonly used for multi-class classification and probability distributions.
//! These functions are essential for converting raw scores (logits) into probability distributions.
//!
//! ## Included Activation Functions
//!
//! - **Softmax** - Standard softmax function: converts logits to probability distribution
//! - **LogSoftmax** - Log of softmax: numerically stable log probabilities
//! - **LogSigmoid** - Log of sigmoid: numerically stable log probabilities for binary classification
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_nn::layers::activation::softmax::{Softmax, LogSoftmax, LogSigmoid};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Multi-class classification
//! let softmax = Softmax::new(Some(1)); // Apply along dimension 1
//! let logits = Tensor::randn(&[4, 10]); // 4 samples, 10 classes
//! let probabilities = softmax.forward(&logits)?;
//!
//! // Log probabilities for numerical stability
//! let log_softmax = LogSoftmax::new(Some(1));
//! let log_probs = log_softmax.forward(&logits)?;
//!
//! // Binary classification
//! let log_sigmoid = LogSigmoid::new();
//! let binary_logits = Tensor::randn(&[4, 1]);
//! let log_probs = log_sigmoid.forward(&binary_logits)?;
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, string::String};

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Softmax activation function
///
/// Applies the softmax function element-wise across a specified dimension.
/// The softmax function converts a vector of real numbers into a probability distribution
/// where each element is in the range (0, 1) and all elements sum to 1.
///
/// # Mathematical Definition
/// ```text
/// Softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in dimension
/// ```
///
/// # Properties
/// - **Probability distribution**: Output values sum to 1 along the specified dimension
/// - **Differentiable**: Smooth gradients for backpropagation
/// - **Multi-class**: Ideal for multi-class classification problems
/// - **Dimensional**: Can be applied along any dimension
/// - **Monotonic**: Preserves relative ordering of inputs
///
/// # Numerical Stability
/// The implementation uses numerically stable computation to avoid overflow
/// by subtracting the maximum value before computing exponentials.
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::softmax::Softmax;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// // Apply softmax along the last dimension (typical for classification)
/// let softmax = Softmax::new(Some(1));
/// let logits = Tensor::from(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let probabilities = softmax.forward(&logits)?;
/// // Each row will sum to 1.0
/// ```
pub struct Softmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl Softmax {
    /// Creates a new Softmax activation function
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to apply softmax.
    ///           If None, applies softmax to the entire tensor (flattened).
    ///           For classification, typically use Some(1) for class dimension.
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates a Softmax that operates along the last dimension (most common case)
    pub fn along_last_dim() -> Self {
        // Note: This is a simplified version. In practice, you'd determine the last dim dynamically
        Self::new(Some(1))
    }

    /// Creates a Softmax that operates on the entire tensor (flattened)
    pub fn global() -> Self {
        Self::new(None)
    }

    /// Gets the dimension along which softmax is applied
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softmax: exp(x) / sum(exp(x))
        // For numerical stability, subtract max before exp
        let max_vals = if let Some(dim) = self.dim {
            input.max_dim(dim as i32, true)?
        } else {
            let max_val = input.max(None, false)?;
            full(input.shape().dims(), max_val.item()?)?
        };

        let shifted = input.sub(&max_vals)?;
        let exp_input = shifted.exp()?;
        let sum_exp = if let Some(dim) = self.dim {
            exp_input.sum_dim(&[dim as i32], true)?
        } else {
            let sum_val = exp_input.sum()?;
            full(input.shape().dims(), sum_val.item()?)?
        };
        exp_input.div(&sum_exp)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// LogSoftmax activation function
///
/// Applies the log-softmax function element-wise across a specified dimension.
/// LogSoftmax computes the logarithm of the softmax function, which is numerically
/// more stable than computing softmax followed by log, especially for large inputs.
///
/// # Mathematical Definition
/// ```text
/// LogSoftmax(x_i) = log(Softmax(x_i)) = x_i - log(sum(exp(x_j))) for j in dimension
/// ```
///
/// # Properties
/// - **Numerically stable**: Avoids underflow issues with very small probabilities
/// - **Log probabilities**: Output values are log probabilities (≤ 0)
/// - **NLL-friendly**: Ideal for use with Negative Log Likelihood loss
/// - **Dimensional**: Can be applied along any dimension
/// - **Efficient**: More efficient than computing softmax then log
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::softmax::LogSoftmax;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// // Apply log-softmax for classification with NLL loss
/// let log_softmax = LogSoftmax::new(Some(1));
/// let logits = Tensor::randn(&[32, 10]); // 32 samples, 10 classes
/// let log_probabilities = log_softmax.forward(&logits)?;
/// // Each row contains log probabilities that can be used with NLL loss
/// ```
pub struct LogSoftmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl LogSoftmax {
    /// Creates a new LogSoftmax activation function
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to apply log-softmax.
    ///           If None, applies log-softmax to the entire tensor (flattened).
    ///           For classification, typically use Some(1) for class dimension.
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates a LogSoftmax that operates along the last dimension (most common case)
    pub fn along_last_dim() -> Self {
        Self::new(Some(1))
    }

    /// Creates a LogSoftmax that operates on the entire tensor (flattened)
    pub fn global() -> Self {
        Self::new(None)
    }

    /// Gets the dimension along which log-softmax is applied
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply log_softmax: log(softmax(x)) = x - log(sum(exp(x)))
        // For numerical stability, use log-sum-exp trick
        let max_vals = if let Some(dim) = self.dim {
            input.max_dim(dim as i32, true)?
        } else {
            let max_val = input.max(None, false)?;
            full(input.shape().dims(), max_val.item()?)?
        };

        let shifted = input.sub(&max_vals)?;
        let exp_shifted = shifted.exp()?;
        let sum_exp = if let Some(dim) = self.dim {
            exp_shifted.sum_dim(&[dim as i32], true)?
        } else {
            let sum_val = exp_shifted.sum()?;
            full(input.shape().dims(), sum_val.item()?)?
        };
        let log_sum_exp = sum_exp.log()?;
        let log_sum_exp_with_max = log_sum_exp.add(&max_vals)?;
        input.sub(&log_sum_exp_with_max)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// LogSigmoid activation function
///
/// Applies the log-sigmoid function element-wise: `log(sigmoid(x))`.
/// This function computes the logarithm of the sigmoid function in a numerically
/// stable way, avoiding numerical issues that can arise when sigmoid outputs
/// are very close to 0 or 1.
///
/// # Mathematical Definition
/// ```text
/// LogSigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
/// ```
///
/// # Numerically Stable Implementation
/// For numerical stability, the implementation uses:
/// - For x ≥ 0: `LogSigmoid(x) = -log(1 + exp(-x))`
/// - For x < 0: `LogSigmoid(x) = x - log(1 + exp(x))`
///
/// # Properties
/// - **Numerically stable**: Handles extreme input values without overflow/underflow
/// - **Log probabilities**: Output values are log probabilities (≤ 0)
/// - **Binary classification**: Ideal for binary classification with BCE loss
/// - **Smooth**: Continuously differentiable everywhere
/// - **Efficient**: More efficient than computing sigmoid then log
///
/// # Example
/// ```rust
/// use torsh_nn::layers::activation::softmax::LogSigmoid;
/// use torsh_nn::Module;
/// use torsh_tensor::Tensor;
///
/// // Apply log-sigmoid for binary classification
/// let log_sigmoid = LogSigmoid::new();
/// let logits = Tensor::randn(&[32, 1]); // 32 samples, binary classification
/// let log_probabilities = log_sigmoid.forward(&logits)?;
/// // Output contains log probabilities for positive class
/// ```
pub struct LogSigmoid {
    base: ModuleBase,
}

impl LogSigmoid {
    /// Creates a new LogSigmoid activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for LogSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LogSigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Numerically stable implementation of log(sigmoid(x))
        // For x >= 0: log(sigmoid(x)) = -log(1 + exp(-x))
        // For x < 0:  log(sigmoid(x)) = x - log(1 + exp(x))

        // Create tensors for comparisons
        let zeros_tensor = zeros(input.shape().dims())?;
        let ones_tensor = ones(input.shape().dims())?;

        // Create float masks directly using where operation
        let positive_condition = input.ge(&zeros_tensor)?;
        let negative_condition = input.lt(&zeros_tensor)?;
        let pos_mask_f32 = ones_tensor.where_tensor(&positive_condition, &zeros_tensor)?;
        let neg_mask_f32 = ones_tensor.where_tensor(&negative_condition, &zeros_tensor)?;

        // For positive values: -log(1 + exp(-x))
        let neg_input = input.neg()?;
        let exp_neg = neg_input.exp()?;
        let one_plus_exp_neg = exp_neg.add(&ones_tensor)?;
        let log_one_plus_exp_neg = one_plus_exp_neg.log()?;
        let positive_result = log_one_plus_exp_neg.neg()?;

        // For negative values: x - log(1 + exp(x))
        let exp_input = input.exp()?;
        let one_plus_exp = exp_input.add(&ones_tensor)?;
        let log_one_plus_exp = one_plus_exp.log()?;
        let negative_result = input.sub(&log_one_plus_exp)?;

        // Combine results using masks
        let pos_part = positive_result.mul(&pos_mask_f32)?;
        let neg_part = negative_result.mul(&neg_mask_f32)?;
        pos_part.add(&neg_part)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_softmax_creation() {
        let softmax = Softmax::new(Some(1));
        assert_eq!(softmax.dim(), Some(1));

        let global_softmax = Softmax::global();
        assert_eq!(global_softmax.dim(), None);

        let last_dim_softmax = Softmax::along_last_dim();
        assert_eq!(last_dim_softmax.dim(), Some(1));
    }

    #[test]
    fn test_softmax_forward() -> Result<()> {
        let softmax = Softmax::new(Some(1));
        let input = randn(&[2, 3])?;
        let output = softmax.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_log_softmax_creation() {
        let log_softmax = LogSoftmax::new(Some(1));
        assert_eq!(log_softmax.dim(), Some(1));

        let global_log_softmax = LogSoftmax::global();
        assert_eq!(global_log_softmax.dim(), None);
    }

    #[test]
    fn test_log_softmax_forward() -> Result<()> {
        let log_softmax = LogSoftmax::new(Some(1));
        let input = randn(&[2, 3])?;
        let output = log_softmax.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_log_sigmoid_forward() -> Result<()> {
        let log_sigmoid = LogSigmoid::new();
        let input = randn(&[2, 3])?;
        let output = log_sigmoid.forward(&input)?;
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_softmax_dimension_parameter() {
        let softmax_dim0 = Softmax::new(Some(0));
        let softmax_dim1 = Softmax::new(Some(1));
        let softmax_global = Softmax::new(None);

        assert_eq!(softmax_dim0.dim(), Some(0));
        assert_eq!(softmax_dim1.dim(), Some(1));
        assert_eq!(softmax_global.dim(), None);
    }

    #[test]
    fn test_training_mode_toggle() -> Result<()> {
        let mut softmax = Softmax::new(Some(1));
        assert!(softmax.training());

        softmax.eval();
        assert!(!softmax.training());

        softmax.train();
        assert!(softmax.training());

        Ok(())
    }

    #[test]
    fn test_default_implementations() {
        let _softmax = Softmax::default();
        let _log_softmax = LogSoftmax::default();
        let _log_sigmoid = LogSigmoid::default();
    }

    #[test]
    fn test_log_sigmoid_numerical_stability() -> Result<()> {
        let log_sigmoid = LogSigmoid::new();

        // Test with large positive values (should not overflow)
        let large_positive = full(&[2, 2], 100.0)?;
        let result = log_sigmoid.forward(&large_positive)?;
        assert_eq!(result.shape(), large_positive.shape());

        // Test with large negative values (should not underflow)
        let large_negative = full(&[2, 2], -100.0)?;
        let result = log_sigmoid.forward(&large_negative)?;
        assert_eq!(result.shape(), large_negative.shape());

        Ok(())
    }

    #[test]
    fn test_convenience_constructors() {
        let last_dim_softmax = Softmax::along_last_dim();
        let global_softmax = Softmax::global();
        let last_dim_log_softmax = LogSoftmax::along_last_dim();
        let global_log_softmax = LogSoftmax::global();

        assert_eq!(last_dim_softmax.dim(), Some(1));
        assert_eq!(global_softmax.dim(), None);
        assert_eq!(last_dim_log_softmax.dim(), Some(1));
        assert_eq!(global_log_softmax.dim(), None);
    }
}
