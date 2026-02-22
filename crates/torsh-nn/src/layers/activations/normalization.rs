//! Normalization and Smooth Activation Functions
//!
//! This module contains activation functions that are commonly used for
//! normalization and probability distributions, as well as smooth alternatives
//! to traditional activation functions:
//! - Probability distributions (Softmax, LogSoftmax)
//! - Smooth activation functions (Softplus, Softsign)

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

// =============================================================================
// PROBABILITY DISTRIBUTION ACTIVATIONS
// =============================================================================

/// Softmax activation function
///
/// Applies the Softmax function to an n-dimensional input Tensor:
/// Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
///
/// The Softmax function is often used as the last activation function of
/// a neural network to normalize the output of a network to a probability
/// distribution over predicted output classes.
///
/// # Parameters
/// - `dim`: The dimension along which Softmax will be computed (None for all dimensions)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::normalization::Softmax;
/// use torsh_nn::Module;
///
/// // Apply softmax over the last dimension
/// let softmax = Softmax::new(Some(1));
/// let output = softmax.forward(&input_tensor)?;
///
/// // Apply softmax over all dimensions
/// let softmax_all = Softmax::new(None);
/// let output_all = softmax_all.forward(&input_tensor)?;
/// ```
pub struct Softmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl Softmax {
    /// Creates a new Softmax activation function
    ///
    /// # Arguments
    /// * `dim` - The dimension along which Softmax will be computed.
    ///          If None, softmax is applied over all dimensions.
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates a new Softmax that operates on the last dimension
    pub fn last_dim() -> Self {
        // Will be resolved dynamically based on input tensor
        Self::new(Some(1))
    }

    /// Creates a new Softmax that operates on all dimensions
    pub fn all_dims() -> Self {
        Self::new(None)
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
        // For numerical stability, subtract the maximum value first
        let max_vals = if let Some(dim) = self.dim {
            input.max_dim(&[dim as i32], true)?
        } else {
            let max_val = input.max()?;
            full(input.shape().dims(), max_val)?
        };

        let shifted_input = input.sub(&max_vals)?;
        let exp_input = shifted_input.exp()?;

        let sum_exp = if let Some(dim) = self.dim {
            exp_input.sum_dim(&[dim as i32], true)?
        } else {
            let sum_val = exp_input.sum()?;
            full(input.shape().dims(), sum_val)?
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

impl std::fmt::Debug for Softmax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Softmax").field("dim", &self.dim).finish()
    }
}

/// Log Softmax activation function
///
/// Applies the Log Softmax function to an n-dimensional input Tensor:
/// LogSoftmax(x_i) = log(exp(x_i) / sum_j(exp(x_j))) = x_i - log(sum_j(exp(x_j)))
///
/// The LogSoftmax function is a more numerically stable version of the
/// composition of log and softmax. It's commonly used in classification
/// problems with negative log-likelihood loss.
///
/// # Parameters
/// - `dim`: The dimension along which LogSoftmax will be computed (None for all dimensions)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::normalization::LogSoftmax;
/// use torsh_nn::Module;
///
/// let log_softmax = LogSoftmax::new(Some(1));
/// let output = log_softmax.forward(&input_tensor)?;
/// ```
pub struct LogSoftmax {
    base: ModuleBase,
    dim: Option<usize>,
}

impl LogSoftmax {
    /// Creates a new LogSoftmax activation function
    ///
    /// # Arguments
    /// * `dim` - The dimension along which LogSoftmax will be computed.
    ///          If None, log_softmax is applied over all dimensions.
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }

    /// Creates a new LogSoftmax that operates on the last dimension
    pub fn last_dim() -> Self {
        Self::new(Some(1))
    }

    /// Creates a new LogSoftmax that operates on all dimensions
    pub fn all_dims() -> Self {
        Self::new(None)
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
            input.max_dim(&[dim as i32], true)?
        } else {
            let max_val = input.max()?;
            full(input.shape().dims(), max_val)?
        };

        let shifted_input = input.sub(&max_vals)?;
        let exp_shifted = shifted_input.exp()?;

        let sum_exp = if let Some(dim) = self.dim {
            exp_shifted.sum_dim(&[dim as i32], true)?
        } else {
            let sum_val = exp_shifted.sum()?;
            full(input.shape().dims(), sum_val)?
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

impl std::fmt::Debug for LogSoftmax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogSoftmax")
            .field("dim", &self.dim)
            .finish()
    }
}

// =============================================================================
// SMOOTH ACTIVATION FUNCTIONS
// =============================================================================

/// Softplus activation function
///
/// Applies the element-wise function: Softplus(x) = (1/β) * log(1 + exp(β * x))
///
/// Softplus is a smooth approximation to the ReLU function. It can be used
/// to constrain the output of a machine to always be positive.
///
/// # Parameters
/// - `beta`: The β value for the Softplus formulation (default: 1.0)
/// - `threshold`: Values above this threshold will be approximated linearly (default: 20.0)
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::normalization::Softplus;
/// use torsh_nn::Module;
///
/// let softplus = Softplus::new(1.0, 20.0);
/// let output = softplus.forward(&input_tensor)?;
/// ```
pub struct Softplus {
    base: ModuleBase,
    beta: f32,
    threshold: f32,
}

impl Softplus {
    /// Creates a new Softplus activation function
    ///
    /// # Arguments
    /// * `beta` - The β value for the Softplus formulation
    /// * `threshold` - Values above this threshold will be approximated linearly
    pub fn new(beta: f32, threshold: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            beta,
            threshold,
        }
    }

    /// Creates a new Softplus with default parameters (beta=1.0, threshold=20.0)
    pub fn default_params() -> Self {
        Self::new(1.0, 20.0)
    }

    /// Creates a new Softplus with custom beta and default threshold
    pub fn with_beta(beta: f32) -> Self {
        Self::new(beta, 20.0)
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::default_params()
    }
}

impl Module for Softplus {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softplus: (1/β) * log(1 + exp(β * x))
        // Use threshold for numerical stability
        let beta_x = input.scalar_mul(self.beta)?;
        let threshold_tensor = full(input.shape().dims(), self.threshold)?;

        // For values above threshold, use linear approximation to avoid overflow
        let above_threshold = beta_x.gt(&threshold_tensor)?;

        // Linear part: x (for large values)
        let linear_part = input.clone();

        // Softplus part: (1/β) * log(1 + exp(β * x))
        let exp_beta_x = beta_x.exp()?;
        let ones_tensor = ones(input.shape().dims())?;
        let one_plus_exp = ones_tensor.add(&exp_beta_x)?;
        let log_part = one_plus_exp.log()?;
        let softplus_part = log_part.scalar_mul(1.0 / self.beta)?;

        // Combine using mask
        let mask_data: Vec<bool> = above_threshold.to_vec()?;
        let selected_values: Vec<f32> = mask_data
            .iter()
            .zip(
                linear_part
                    .to_vec()?
                    .iter()
                    .zip(softplus_part.to_vec()?.iter()),
            )
            .map(
                |(&use_linear, (&linear_val, &softplus_val))| {
                    if use_linear {
                        linear_val
                    } else {
                        softplus_val
                    }
                },
            )
            .collect();

        Tensor::from_data(
            selected_values,
            input.shape().dims().to_vec(),
            input.device(),
        )
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

impl std::fmt::Debug for Softplus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Softplus")
            .field("beta", &self.beta)
            .field("threshold", &self.threshold)
            .finish()
    }
}

/// Softsign activation function
///
/// Applies the element-wise function: Softsign(x) = x / (1 + |x|)
///
/// Softsign is similar to Tanh but has a different asymptotic behavior.
/// The output range is (-1, 1), and the function approaches the asymptotes
/// more slowly than Tanh.
///
/// # Examples
/// ```rust
/// use torsh_nn::layers::activations::normalization::Softsign;
/// use torsh_nn::Module;
///
/// let softsign = Softsign::new();
/// let output = softsign.forward(&input_tensor)?;
/// ```
pub struct Softsign {
    base: ModuleBase,
}

impl Softsign {
    /// Creates a new Softsign activation function
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Softsign {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply softsign: x / (1 + |x|)
        let abs_input = input.abs()?;
        let ones_tensor = ones(input.shape().dims())?;
        let one_plus_abs = ones_tensor.add(&abs_input)?;
        input.div(&one_plus_abs)
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

impl std::fmt::Debug for Softsign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Softsign").finish()
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Numerically stable log-sum-exp operation
///
/// Computes log(sum(exp(x))) in a numerically stable way by factoring out
/// the maximum value: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
pub fn log_sum_exp(input: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    let max_vals = if let Some(d) = dim {
        input.max_dim(&[d as i32], keepdim)?
    } else {
        let max_val = input.max()?;
        if keepdim {
            full(input.shape().dims(), max_val)?
        } else {
            Tensor::from_data(vec![max_val], vec![1], input.device())?
        }
    };

    let shifted = input.sub(&max_vals)?;
    let exp_shifted = shifted.exp()?;

    let sum_exp = if let Some(d) = dim {
        exp_shifted.sum_dim(&[d as i32], keepdim)?
    } else {
        let sum_val = exp_shifted.sum()?;
        if keepdim {
            full(input.shape().dims(), sum_val)?
        } else {
            Tensor::from_data(vec![sum_val], vec![1], input.device())?
        }
    };

    let log_sum = sum_exp.log()?;
    max_vals.add(&log_sum)
}

/// Numerically stable softmax operation
///
/// Computes softmax(x) = exp(x) / sum(exp(x)) using the log-sum-exp trick
/// for numerical stability.
pub fn stable_softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let log_sum_exp_val = log_sum_exp(input, dim, true)?;
    let log_softmax_val = input.sub(&log_sum_exp_val)?;
    log_softmax_val.exp()
}

/// Numerically stable log softmax operation
///
/// Computes log(softmax(x)) = x - log(sum(exp(x))) using the log-sum-exp trick
/// for numerical stability.
pub fn stable_log_softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let log_sum_exp_val = log_sum_exp(input, dim, true)?;
    input.sub(&log_sum_exp_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::*;

    #[test]
    fn test_softmax_forward() {
        let softmax = Softmax::new(None);
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let output = softmax.forward(&input).unwrap();
        let output_vec = output.to_vec().expect("tensor to vec conversion should succeed");

        // Check that outputs sum to 1
        let sum: f32 = output_vec.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // Check that all outputs are positive
        assert!(output_vec.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_log_softmax_forward() {
        let log_softmax = LogSoftmax::new(None);
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let output = log_softmax.forward(&input).unwrap();
        let output_vec = output.to_vec().expect("tensor to vec conversion should succeed");

        // Check that all outputs are negative (since they're log probabilities)
        assert!(output_vec.iter().all(|&x| x <= 0.0));

        // Check consistency with softmax
        let softmax = Softmax::new(None);
        let softmax_output = softmax.forward(&input).unwrap();
        let log_softmax_exp: Vec<f32> = output_vec.iter().map(|&x| x.exp()).collect();
        let softmax_vec = softmax_output.to_vec().expect("tensor to vec conversion should succeed");

        for (actual, expected) in log_softmax_exp.iter().zip(softmax_vec.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_softplus_forward() {
        let softplus = Softplus::new(1.0, 20.0);
        let input = Tensor::from_data(vec![-2.0, 0.0, 2.0], vec![3], DeviceType::Cpu).unwrap();
        let output = softplus.forward(&input).unwrap();
        let output_vec = output.to_vec().expect("tensor to vec conversion should succeed");

        // Check that all outputs are positive
        assert!(output_vec.iter().all(|&x| x > 0.0));

        // Check that softplus(0) ≈ ln(2)
        assert_relative_eq!(output_vec[1], 2.0_f32.ln(), epsilon = 1e-3);

        // Check that for large positive inputs, softplus(x) ≈ x
        let large_input = Tensor::from_data(vec![10.0], vec![1], DeviceType::Cpu).unwrap();
        let large_output = softplus.forward(&large_input).unwrap();
        assert_relative_eq!(large_output.to_vec().expect("tensor to vec conversion should succeed")[0], 10.0, epsilon = 1e-3);
    }

    #[test]
    fn test_softsign_forward() {
        let softsign = Softsign::new();
        let input = Tensor::from_data(vec![-2.0, 0.0, 2.0], vec![3], DeviceType::Cpu).unwrap();
        let output = softsign.forward(&input).unwrap();
        let output_vec = output.to_vec().expect("tensor to vec conversion should succeed");

        // Check that softsign(0) = 0
        assert_relative_eq!(output_vec[1], 0.0, epsilon = 1e-5);

        // Check that softsign(-x) = -softsign(x)
        assert_relative_eq!(output_vec[0], -output_vec[2], epsilon = 1e-5);

        // Check range (-1, 1)
        assert!(output_vec.iter().all(|&x| x > -1.0 && x < 1.0));

        // Manual calculation: softsign(2) = 2 / (1 + 2) = 2/3
        assert_relative_eq!(output_vec[2], 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_log_sum_exp_utility() {
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let result = log_sum_exp(&input, None, false).unwrap();
        let result_val = result.to_vec().expect("tensor to vec conversion should succeed")[0];

        // Manual calculation: log(e^1 + e^2 + e^3)
        let expected = (1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp()).ln();
        assert_relative_eq!(result_val, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_stable_softmax_utility() {
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let result = stable_softmax(&input, None).unwrap();
        let result_vec = result.to_vec().expect("tensor to vec conversion should succeed");

        // Check that outputs sum to 1
        let sum: f32 = result_vec.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // Test with large values (potential overflow)
        let large_input =
            Tensor::from_data(vec![100.0, 101.0, 102.0], vec![3], DeviceType::Cpu).unwrap();
        let large_result = stable_softmax(&large_input, None).unwrap();
        let large_sum: f32 = large_result.to_vec().expect("tensor to vec conversion should succeed").iter().sum();
        assert_relative_eq!(large_sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_module_interface() {
        let mut softmax = Softmax::new(Some(0));

        // Test training mode
        assert!(softmax.training()); // Should be true by default
        softmax.eval();
        assert!(!softmax.training());
        softmax.train();
        assert!(softmax.training());

        // Test parameters (should be empty for activation functions)
        assert!(softmax.parameters().is_empty());
        assert!(softmax.named_parameters().is_empty());
    }

    #[test]
    fn test_softmax_dimensions() {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let softmax_all = Softmax::new(None);
        let output_all = softmax_all.forward(&input).unwrap();
        let sum_all: f32 = output_all.to_vec().expect("tensor to vec conversion should succeed").iter().sum();
        assert_relative_eq!(sum_all, 1.0, epsilon = 1e-5);

        // Test dimension-specific softmax
        let softmax_dim1 = Softmax::new(Some(1));
        let output_dim1 = softmax_dim1.forward(&input).unwrap();

        // Each row should sum to 1
        let output_2d = output_dim1.to_vec().expect("tensor to vec conversion should succeed");
        let row1_sum: f32 = output_2d[0..3].iter().sum();
        let row2_sum: f32 = output_2d[3..6].iter().sum();

        assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-5);
        assert_relative_eq!(row2_sum, 1.0, epsilon = 1e-5);
    }
}
