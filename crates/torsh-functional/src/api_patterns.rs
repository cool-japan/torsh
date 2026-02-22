//! Standardized API patterns and conventions for torsh-functional
//!
//! This module documents and demonstrates the standardized API patterns
//! used throughout the torsh-functional crate to ensure consistency
//! and usability.

use crate::utils::*;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Standard parameter ordering for functional operations
///
/// All functions in torsh-functional follow this parameter ordering convention:
/// 1. Primary input tensor(s)
/// 2. Required parameters (sizes, dimensions, etc.)
/// 3. Optional parameters with defaults
/// 4. Configuration flags (inplace, etc.)
///
/// Example: `conv2d(input, weight, bias, stride, padding, dilation, groups)`

/// Standard documentation template for functions
///
/// All functions should include:
/// - Brief description of what the function does
/// - Mathematical formula (if applicable)
/// - Parameter descriptions with types and constraints
/// - Return value description
/// - Error conditions
/// - Usage example
/// - References to related functions

/// Standard error handling patterns
///
/// All functions should:
/// 1. Create a function context using `function_context()`
/// 2. Validate inputs using appropriate `validate_*` functions
/// 3. Use descriptive error messages with context
/// 4. Chain errors properly with `.map_err()`

/// Example function demonstrating all API conventions
///
/// Linear transformation with bias: y = xW^T + b
///
/// Formula: linear(x, W, b) = xW^T + b
///
/// # Parameters
/// * `input` - Input tensor of shape (N, *, in_features)
/// * `weight` - Weight matrix of shape (out_features, in_features)  
/// * `bias` - Optional bias vector of shape (out_features,)
///
/// # Returns
/// Output tensor of shape (N, *, out_features)
///
/// # Errors
/// Returns error if:
/// - Input tensor is empty
/// - Weight/bias shapes are incompatible with input
/// - Matrix multiplication fails
///
/// # Example
/// ```rust,no_run
/// use torsh_functional::api_patterns::example_linear;
/// use torsh_functional::random_ops::randn;
/// let input = randn(&[2, 3], None, None, None).unwrap();
/// let weight = randn(&[4, 3], None, None, None).unwrap();
/// let bias = Some(randn(&[4], None, None, None).unwrap());
/// let output = example_linear(&input, &weight, bias.as_ref()).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 4]);
/// ```
pub fn example_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> TorshResult<Tensor> {
    let context = function_context("example_linear");

    // 1. Input validation
    validate_non_empty(input, &context)?;
    validate_non_empty(weight, &context)?;
    validate_tensor_dims(weight, 2, &context)?;

    let input_shape_binding = input.shape();
    let input_shape = input_shape_binding.dims();
    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();
    let input_features = input_shape[input_shape.len() - 1];
    let weight_features = weight_shape[1];

    if input_features != weight_features {
        return Err(invalid_argument_error(
            &format!(
                "Input features ({}) don't match weight features ({})",
                input_features, weight_features
            ),
            &context,
        ));
    }

    if let Some(bias_tensor) = bias {
        validate_non_empty(bias_tensor, &context)?;
        validate_tensor_dims(bias_tensor, 1, &context)?;

        let bias_shape_binding = bias_tensor.shape();
        let bias_shape = bias_shape_binding.dims();
        if bias_shape[0] != weight_shape[0] {
            return Err(invalid_argument_error(
                &format!(
                    "Bias size ({}) doesn't match output features ({})",
                    bias_shape[0], weight_shape[0]
                ),
                &context,
            ));
        }
    }

    // 2. Core computation with error handling
    // Linear transformation: y = xW^T + b (weight needs to be transposed)
    let weight_t = weight.transpose(0, 1).map_err(|e| {
        TorshError::InvalidOperation(format!("Weight transpose failed: {}", e))
            .with_context(&context)
    })?;

    let output = input.matmul(&weight_t).map_err(|e| {
        TorshError::InvalidOperation(format!("Matrix multiplication failed: {}", e))
            .with_context(&context)
    })?;

    // 3. Optional bias addition
    if let Some(bias_tensor) = bias {
        output.add(bias_tensor).map_err(|e| {
            TorshError::InvalidOperation(format!("Bias addition failed: {}", e))
                .with_context(&context)
        })
    } else {
        Ok(output)
    }
}

/// Standard activation function pattern
///
/// All activation functions should follow this pattern:
/// - Generic over tensor element type
/// - Optional inplace parameter (last position)
/// - Comprehensive input validation
/// - Proper error context
pub fn example_activation<T>(input: &Tensor, alpha: f32, inplace: bool) -> TorshResult<Tensor>
where
    T: Copy + PartialOrd + From<f32>,
{
    let context = function_context("example_activation");

    // Validate inputs
    validate_non_empty(input, &context)?;
    validate_positive(alpha, "alpha", &context)?;

    // Handle inplace operation
    handle_inplace_operation(
        input,
        inplace,
        |tensor| {
            // Perform the actual activation computation
            tensor.relu() // Placeholder implementation
        },
        &context,
    )
}

/// Standard pooling function pattern
///
/// All pooling functions should follow this pattern:
/// - Kernel size as first required parameter
/// - Optional stride (defaults to kernel_size)
/// - Padding as optional parameter
/// - Dilation for advanced pooling
pub fn example_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    _dilation: (usize, usize),
) -> TorshResult<Tensor> {
    let context = function_context("example_pool2d");

    // Validate 4D input tensor (N, C, H, W)
    validate_tensor_dims(input, 4, &context)?;

    // Validate pooling parameters
    let kernel_slice = [kernel_size.0, kernel_size.1];
    let stride_slice = stride.unwrap_or(kernel_size);
    let stride_slice = [stride_slice.0, stride_slice.1];
    let padding_slice = [padding.0, padding.1];

    validate_pooling_params(
        input,
        &kernel_slice,
        &stride_slice,
        &padding_slice,
        &context,
    )?;

    // Placeholder implementation
    Ok(input.clone())
}

/// Standard loss function pattern
///
/// All loss functions should follow this pattern:
/// - Input and target as first parameters
/// - Reduction type as explicit enum
/// - Comprehensive shape validation
pub fn example_loss(
    input: &Tensor,
    target: &Tensor,
    reduction: crate::loss::ReductionType,
    weight: Option<&Tensor>,
) -> TorshResult<Tensor> {
    let context = function_context("example_loss");

    // Validate inputs
    validate_non_empty(input, &context)?;
    validate_non_empty(target, &context)?;
    validate_broadcastable_shapes(input, target, &context)?;

    if let Some(weight_tensor) = weight {
        validate_non_empty(weight_tensor, &context)?;
        // Additional weight validation...
    }

    // Compute loss (placeholder)
    let loss = input.sub(target)?;
    reduction.apply(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_example_linear() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )?;

        // Create weight tensor with correct dimensions [out_features, in_features] = [2, 3]
        let weight = Tensor::from_data(
            vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3], // [out_features, in_features] = [2, 3]
            DeviceType::Cpu,
        )?;

        let bias = Tensor::from_data(vec![0.1f32, 0.2], vec![2], DeviceType::Cpu)?;

        let output = example_linear(&input, &weight, Some(&bias))?;
        assert_eq!(output.shape().dims(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_validation_patterns() -> TorshResult<()> {
        // Test empty tensor validation
        let empty = Tensor::from_data(vec![0.0f32; 0], vec![0], DeviceType::Cpu)?;
        let valid = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu)?;

        let result = example_linear(&empty, &valid, None);
        assert!(result.is_err());

        Ok(())
    }
}
