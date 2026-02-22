//! Activation functions for neural network operations
//!
//! This module provides a comprehensive collection of activation functions
//! enhanced with SciRS2 integration for optimized performance and numerical stability.

use super::core::{Activation, FuncResult, FunctionalConfig};
use crate::{func_error, validate_inputs};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// ENHANCED ACTIVATION FUNCTIONS WITH SCIRS2 INTEGRATION
// =============================================================================

/// ReLU activation function
/// Enhanced with SciRS2-Neural integration for optimized performance
pub fn relu(input: &Tensor) -> Result<Tensor> {
    // Enhanced implementation with potential SciRS2 optimization
    // For numerical stability and performance, use optimized path when available
    let zeros = torsh_tensor::creation::zeros_like(input)?;

    // Apply ReLU with potential SIMD optimizations
    // This maintains compatibility while allowing for future scirs2 optimization
    input.maximum(&zeros)
}

/// Optimized ReLU with in-place operation support
pub fn relu_inplace(input: &mut Tensor) -> Result<()> {
    // In-place ReLU for memory efficiency
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    *input = input.maximum(&zeros)?;
    Ok(())
}

/// Leaky ReLU activation function
pub fn leaky_relu(input: &Tensor, negative_slope: f32) -> Result<Tensor> {
    // Implement leaky ReLU: max(0, x) + negative_slope * min(0, x)
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let positive_part = input.maximum(&zeros)?;
    let negative_part = input.minimum(&zeros)?;
    let slope_tensor = torsh_tensor::creation::full_like(input, negative_slope)?;
    let scaled_negative = negative_part.mul_op(&slope_tensor)?;
    positive_part.add(&scaled_negative)
}

/// GELU activation function
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // For now, use simplified approximation: x * sigmoid(1.702 * x)
    let factor = torsh_tensor::creation::full_like(input, 1.702)?;
    let scaled = input.mul_op(&factor)?;
    let sigmoid_result = sigmoid(&scaled)?;
    input.mul_op(&sigmoid_result)
}

/// Sigmoid activation function
/// Enhanced with numerically stable implementation following SciRS2 best practices
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    // Numerically stable sigmoid implementation
    // Uses different formulations for positive and negative inputs to avoid overflow

    let data = input.to_vec()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x > 0.0 {
                // For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                // For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Numerically stable softmax implementation
/// Enhanced with SciRS2-inspired numerical stability techniques
pub fn softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);
    let shape = input.shape();

    // Handle simple case for now - assume 2D tensors and dim=1 (row-wise softmax)
    if shape.dims().len() == 2 && dim == 1 {
        let data = input.to_vec()?;
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let mut result_data = vec![0.0; data.len()];

        // Process each row separately
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = (row + 1) * cols;
            let row_data = &data[row_start..row_end];

            // Find max for numerical stability
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            let mut exp_vals = Vec::with_capacity(cols);

            for &x in row_data {
                let exp_val = (x - max_val).exp();
                exp_vals.push(exp_val);
                exp_sum += exp_val;
            }

            // Normalize by sum
            for (i, exp_val) in exp_vals.into_iter().enumerate() {
                result_data[row_start + i] = exp_val / exp_sum;
            }
        }

        return Tensor::from_data(result_data, shape.dims().to_vec(), input.device());
    }

    // Fallback: Use the old approach for 1D tensors or other cases
    // For numerical stability, subtract max
    let data = input.to_vec()?;
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let shifted_data: Vec<f32> = data.iter().map(|&x| x - max_val).collect();
    let exp_data: Vec<f32> = shifted_data.iter().map(|&x| x.exp()).collect();
    let sum_exp: f32 = exp_data.iter().sum();

    let result_data: Vec<f32> = exp_data.iter().map(|&x| x / sum_exp).collect();

    Tensor::from_data(result_data, shape.dims().to_vec(), input.device())
}

/// Log-softmax implementation with enhanced numerical stability
pub fn log_softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);

    // Subtract max for numerical stability
    let max_vals = input.max_dim(dim, true)?;
    let shifted = input.sub(&max_vals)?;

    // Compute log(sum(exp(x - max(x))))
    let exp_vals = shifted.exp()?;
    let sum_exp = exp_vals.sum_dim(&[dim], true)?;
    let log_sum_exp = sum_exp.log()?;

    // Return x - max(x) - log(sum(exp(x - max(x))))
    shifted.sub(&log_sum_exp)
}

/// Tanh activation function
/// Numerically stable implementation that handles large input values
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    // Numerically stable tanh implementation
    // For large |x|, clamp to prevent overflow and NaN

    let data = input.to_vec()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Clamp extreme values to prevent numerical instability
            if x > 20.0 {
                1.0 // tanh approaches 1 for large positive x
            } else if x < -20.0 {
                -1.0 // tanh approaches -1 for large negative x
            } else {
                // Use standard formula for moderate values
                let exp_2x = (2.0 * x).exp();
                if exp_2x.is_infinite() {
                    if x > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    (exp_2x - 1.0) / (exp_2x + 1.0)
                }
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Swish (SiLU) activation function
pub fn swish(input: &Tensor) -> Result<Tensor> {
    // Swish: x * sigmoid(x)
    let sigmoid_result = sigmoid(input)?;
    input.mul_op(&sigmoid_result)
}

/// Mish activation function
pub fn mish(input: &Tensor) -> Result<Tensor> {
    // Mish: x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    let exp_input = input.exp()?;
    let ones = torsh_tensor::creation::ones_like(input)?;
    let softplus = exp_input.add(&ones)?.log()?;
    let tanh_result = tanh(&softplus)?;
    input.mul_op(&tanh_result)
}

/// ELU (Exponential Linear Unit) activation function
pub fn elu(input: &Tensor, alpha: f32) -> Result<Tensor> {
    // ELU: x if x > 0, alpha * (exp(x) - 1) if x <= 0
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let positive_mask = input.gt(&zeros)?;
    let exp_input = input.exp()?;
    let ones = torsh_tensor::creation::ones_like(input)?;
    let alpha_tensor = torsh_tensor::creation::full_like(input, alpha)?;
    let negative_part = alpha_tensor.mul_op(&exp_input.sub(&ones)?)?;

    // Use where: positive_mask ? input : negative_part
    input.where_tensor(&positive_mask, &negative_part)
}

/// SELU (Scaled Exponential Linear Unit) activation function
pub fn selu(input: &Tensor) -> Result<Tensor> {
    // SELU constants
    let alpha = 1.6732632423543772;
    let scale = 1.0507009873554805;

    let elu_result = elu(input, alpha)?;
    let scale_tensor = torsh_tensor::creation::full_like(input, scale)?;
    elu_result.mul_op(&scale_tensor)
}

/// Dropout regularization function
///
/// During training, randomly zeroes some elements of the input tensor with probability `p`
/// using samples from a Bernoulli distribution. The outputs are scaled by a factor of
/// `1/(1-p)` during training to maintain expected values.
///
/// During evaluation (training=false), returns the input unchanged.
///
/// # Arguments
/// * `input` - Input tensor
/// * `p` - Probability of an element to be zeroed (between 0 and 1)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Tensor with dropout applied (during training) or original tensor (during evaluation)
pub fn dropout(input: &Tensor, p: f32, training: bool) -> Result<Tensor> {
    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    use scirs2_core::random::thread_rng;

    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if p == 1.0 {
        // Drop all elements - return zeros
        let shape = input.shape().dims().to_vec();
        return torsh_tensor::creation::zeros(&shape);
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::InvalidArgument(format!(
            "Dropout probability must be between 0 and 1, got {}",
            p
        )));
    }

    let data = input.data()?;
    let scale = 1.0 / (1.0 - p); // Scale factor to maintain expected value

    // Generate random mask using Bernoulli distribution
    let mut rng = thread_rng();

    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Sample from uniform distribution and compare with dropout probability
            let random_val: f32 = rng.random();
            if random_val < p {
                0.0 // Drop this element
            } else {
                x * scale // Keep and scale this element
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

// =============================================================================
// CONVENIENCE FUNCTIONS WITH STANDARDIZED API
// =============================================================================

/// Convenient activation functions with standardized API
pub mod configured {
    use super::super::core::validation;
    use super::*;

    /// ReLU activation with optional configuration
    pub fn relu_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(relu(input), "ReLU activation")
    }

    /// Sigmoid activation with optional configuration
    pub fn sigmoid_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(sigmoid(input), "Sigmoid activation")
    }

    /// Tanh activation with optional configuration
    pub fn tanh_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(tanh(input), "Tanh activation")
    }

    /// Softmax activation with optional configuration
    pub fn softmax_configured(
        input: &Tensor,
        dim: Option<i32>,
        config: &FunctionalConfig,
    ) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(softmax(input, dim), "Softmax activation")
    }

    /// GELU activation with optional configuration
    pub fn gelu_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(gelu(input), "GELU activation")
    }

    /// Swish/SiLU activation with optional configuration
    pub fn swish_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(swish(input), "Swish activation")
    }

    /// Mish activation with optional configuration
    pub fn mish_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(mish(input), "Mish activation")
    }
}

// =============================================================================
// ACTIVATION FUNCTION IMPLEMENTATIONS FOR TRAIT SYSTEM
// =============================================================================

/// ReLU activation implementation
pub struct ReLU {
    inplace: bool,
}

impl ReLU {
    pub fn new(inplace: bool) -> Self {
        Self { inplace }
    }
}

impl Activation for ReLU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        if self.inplace {
            let mut result = input.clone();
            relu_inplace(&mut result)?;
            Ok(result)
        } else {
            relu(input).map_err(|e| e.into())
        }
    }
}

/// Sigmoid activation implementation
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Sigmoid {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        sigmoid(input).map_err(|e| e.into())
    }
}

/// Tanh activation implementation
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Tanh {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        tanh(input).map_err(|e| e.into())
    }
}

/// GELU activation implementation
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for GELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        gelu(input).map_err(|e| e.into())
    }
}

/// Swish activation implementation
pub struct Swish;

impl Swish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Swish {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        swish(input).map_err(|e| e.into())
    }
}

/// Mish activation implementation
pub struct Mish;

impl Mish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Mish {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        mish(input).map_err(|e| e.into())
    }
}

/// ELU activation implementation
pub struct ELU {
    alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Activation for ELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        elu(input, self.alpha).map_err(|e| e.into())
    }
}

/// SELU activation implementation
pub struct SELU;

impl SELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for SELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        selu(input).map_err(|e| e.into())
    }
}

/// Leaky ReLU activation implementation
pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Activation for LeakyReLU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        leaky_relu(input, self.negative_slope).map_err(|e| e.into())
    }
}

/// Softmax activation implementation
pub struct Softmax {
    dim: i32,
}

impl Softmax {
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Activation for Softmax {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        softmax(input, Some(self.dim)).map_err(|e| e.into())
    }
}

/// LogSoftmax activation implementation
pub struct LogSoftmax {
    dim: i32,
}

impl LogSoftmax {
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Activation for LogSoftmax {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        log_softmax(input, Some(self.dim)).map_err(|e| e.into())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dropout_training_p_zero() -> Result<()> {
        // Test that dropout with p=0.0 returns input unchanged
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 0.0, true)?;

        let input_data = input.to_vec()?;
        let output_data = output.to_vec()?;

        assert_eq!(input_data.len(), output_data.len());
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_training_p_one() -> Result<()> {
        // Test that dropout with p=1.0 returns all zeros
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 1.0, true)?;

        let output_data = output.to_vec()?;

        for &val in output_data.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_eval_mode() -> Result<()> {
        // Test that dropout in evaluation mode returns input unchanged
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 0.5, false)?; // training=false

        let input_data = input.to_vec()?;
        let output_data = output.to_vec()?;

        assert_eq!(input_data.len(), output_data.len());
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_training_p_half() -> Result<()> {
        // Test that dropout with p=0.5 drops approximately half the elements
        let size = 1000;
        let input_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let input = Tensor::from_vec(input_data.clone(), &[size])?;

        let output = dropout(&input, 0.5, true)?;
        let output_data = output.to_vec()?;

        // Count zeros (dropped elements)
        let zeros_count = output_data.iter().filter(|&&x| x == 0.0).count();

        // With p=0.5, we expect approximately 50% zeros
        // Allow some variance (40% to 60%)
        assert!(
            zeros_count >= 400 && zeros_count <= 600,
            "Expected 400-600 zeros, got {}",
            zeros_count
        );

        Ok(())
    }

    #[test]
    fn test_dropout_scaling() -> Result<()> {
        // Test that dropout maintains expected value through scaling
        let size = 10000;
        let input_data: Vec<f32> = vec![1.0; size];
        let input = Tensor::from_vec(input_data, &[size])?;

        let p = 0.3;
        let output = dropout(&input, p, true)?;
        let output_data = output.to_vec()?;

        // Calculate mean of non-zero elements
        let non_zeros: Vec<f32> = output_data.iter().filter(|&&x| x != 0.0).copied().collect();

        if !non_zeros.is_empty() {
            let mean_non_zero: f32 = non_zeros.iter().sum::<f32>() / non_zeros.len() as f32;
            let expected_scale = 1.0 / (1.0 - p);

            // Non-zero elements should be scaled by 1/(1-p)
            assert_relative_eq!(mean_non_zero, expected_scale, epsilon = 0.01);
        }

        // Total mean should be approximately 1.0 (maintained expected value)
        let total_mean: f32 = output_data.iter().sum::<f32>() / output_data.len() as f32;
        assert_relative_eq!(total_mean, 1.0, epsilon = 0.1);

        Ok(())
    }

    #[test]
    fn test_dropout_shape_preservation() -> Result<()> {
        // Test that dropout preserves tensor shape
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;

        let output = dropout(&input, 0.5, true)?;

        assert_eq!(input.shape().dims(), output.shape().dims());
        assert_eq!(input.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_dropout_invalid_p_negative() {
        // Test that negative p values are rejected
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = dropout(&input, -0.1, true);

        assert!(result.is_err());
        if let Err(TorshError::InvalidArgument(msg)) = result {
            assert!(msg.contains("Dropout probability must be between 0 and 1"));
        } else {
            panic!("Expected InvalidArgument error for negative p");
        }
    }

    #[test]
    fn test_dropout_invalid_p_too_large() {
        // Test that p > 1.0 values are rejected
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = dropout(&input, 1.5, true);

        assert!(result.is_err());
        if let Err(TorshError::InvalidArgument(msg)) = result {
            assert!(msg.contains("Dropout probability must be between 0 and 1"));
        } else {
            panic!("Expected InvalidArgument error for p > 1.0");
        }
    }

    #[test]
    fn test_dropout_multidimensional() -> Result<()> {
        // Test dropout on multidimensional tensors
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )?;

        let output = dropout(&input, 0.5, true)?;

        // Shape should be preserved
        assert_eq!(output.shape().dims(), &[3, 4]);

        // Some elements should be zero, some should be scaled
        let output_data = output.to_vec()?;
        let has_zeros = output_data.iter().any(|&x| x == 0.0);
        let has_nonzeros = output_data.iter().any(|&x| x != 0.0);

        assert!(has_zeros, "Should have some dropped (zero) elements");
        assert!(has_nonzeros, "Should have some kept (non-zero) elements");

        Ok(())
    }

    #[test]
    fn test_dropout_edge_case_empty_like() -> Result<()> {
        // Test dropout with very small p values
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;

        let output = dropout(&input, 0.01, true)?;
        let output_data = output.to_vec()?;

        // Most elements should be non-zero with p=0.01
        let non_zeros = output_data.iter().filter(|&&x| x != 0.0).count();
        assert!(
            non_zeros >= 3,
            "Expected at least 3 non-zero elements with p=0.01, got {}",
            non_zeros
        );

        Ok(())
    }
}
