//! Utility functions for torsh-functional
//!
//! This module contains helper functions and common patterns used
//! throughout the functional API.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Validates that input tensors have compatible shapes for element-wise operations
pub fn validate_elementwise_shapes(a: &Tensor, b: &Tensor) -> TorshResult<()> {
    let binding_a = a.shape();
    let shape_a = binding_a.dims();
    let binding_b = b.shape();
    let shape_b = binding_b.dims();

    if shape_a != shape_b {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Tensor shapes are not compatible for element-wise operation: {:?} vs {:?}",
                shape_a, shape_b
            ),
            "elementwise_operation",
        ));
    }

    Ok(())
}

/// Validates that a value is within a specified range
pub fn validate_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    param_name: &str,
    context: &str,
) -> TorshResult<()> {
    if value < min || value > max {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "{} must be in range [{}, {}], got {}",
                param_name, min, max, value
            ),
            context,
        ));
    }
    Ok(())
}

/// Validates that a tensor is not empty
pub fn validate_non_empty(tensor: &Tensor, context: &str) -> TorshResult<()> {
    if tensor.numel() == 0 {
        return Err(TorshError::invalid_argument_with_context(
            "Tensor cannot be empty",
            context,
        ));
    }
    Ok(())
}

/// Validates dimension index for a tensor
pub fn validate_dimension(tensor: &Tensor, dim: i32, context: &str) -> TorshResult<()> {
    let ndim = tensor.shape().ndim() as i32;
    let normalized_dim = if dim < 0 { dim + ndim } else { dim };

    if normalized_dim < 0 || normalized_dim >= ndim {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Dimension {} is out of range for tensor with {} dimensions",
                dim, ndim
            ),
            context,
        ));
    }
    Ok(())
}

/// Validates that a parameter is positive
pub fn validate_positive<T: PartialOrd + std::fmt::Display + Copy>(
    value: T,
    param_name: &str,
    context: &str,
) -> TorshResult<()>
where
    T: From<f32>,
{
    let zero = T::from(0.0);
    if value <= zero {
        return Err(TorshError::invalid_argument_with_context(
            &format!("{} must be positive, got {}", param_name, value),
            context,
        ));
    }
    Ok(())
}

/// Creates a standardized context string for function errors
pub fn function_context(function_name: &str) -> String {
    function_name.to_string()
}

/// Standard parameter validation for activation functions
pub fn validate_activation_params<T: PartialOrd + std::fmt::Display + Copy>(
    input: &Tensor,
    alpha: Option<T>,
    beta: Option<T>,
    context: &str,
) -> TorshResult<()>
where
    T: From<f32>,
{
    validate_non_empty(input, context)?;

    if let Some(alpha_val) = alpha {
        validate_positive(alpha_val, "alpha", context)?;
    }

    if let Some(beta_val) = beta {
        validate_positive(beta_val, "beta", context)?;
    }

    Ok(())
}

/// Standard parameter validation for pooling operations
pub fn validate_pooling_params(
    input: &Tensor,
    kernel_size: &[usize],
    stride: &[usize],
    _padding: &[usize],
    context: &str,
) -> TorshResult<()> {
    validate_non_empty(input, context)?;

    if kernel_size.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "kernel_size cannot be empty",
            context,
        ));
    }

    if kernel_size.iter().any(|&k| k == 0) {
        return Err(TorshError::invalid_argument_with_context(
            "All kernel_size values must be positive",
            context,
        ));
    }

    if stride.iter().any(|&s| s == 0) {
        return Err(TorshError::invalid_argument_with_context(
            "All stride values must be positive",
            context,
        ));
    }

    Ok(())
}

/// Standard parameter validation for loss functions
pub fn validate_loss_params(
    input: &Tensor,
    target: &Tensor,
    reduction: &str,
    context: &str,
) -> TorshResult<()> {
    validate_non_empty(input, context)?;
    validate_non_empty(target, context)?;

    match reduction {
        "none" | "mean" | "sum" => Ok(()),
        _ => Err(TorshError::invalid_argument_with_context(
            &format!(
                "Invalid reduction '{}'. Must be 'none', 'mean', or 'sum'",
                reduction
            ),
            context,
        )),
    }
}

/// Validates tensor dimensions for specific operations
pub fn validate_tensor_dims(
    tensor: &Tensor,
    expected_dims: usize,
    context: &str,
) -> TorshResult<()> {
    let actual_dims = tensor.shape().ndim();
    if actual_dims != expected_dims {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {}D tensor, got {}D tensor",
                expected_dims, actual_dims
            ),
            context,
        ));
    }
    Ok(())
}

/// Validates tensor shapes are broadcastable
pub fn validate_broadcastable_shapes(a: &Tensor, b: &Tensor, context: &str) -> TorshResult<()> {
    let binding_a = a.shape();
    let shape_a = binding_a.dims();
    let binding_b = b.shape();
    let shape_b = binding_b.dims();

    // Simple broadcastability check (can be expanded for more complex cases)
    if shape_a.len() != shape_b.len() && shape_a != shape_b {
        // Allow different lengths if one is scalar or can be broadcast
        let a_numel = a.numel();
        let b_numel = b.numel();

        if a_numel != 1 && b_numel != 1 && shape_a != shape_b {
            return Err(TorshError::invalid_argument_with_context(
                &format!(
                    "Tensor shapes {:?} and {:?} are not broadcastable",
                    shape_a, shape_b
                ),
                context,
            ));
        }
    }

    Ok(())
}

/// Helper function to create invalid argument error with function context
pub fn invalid_argument_error(message: &str, function_name: &str) -> TorshError {
    TorshError::invalid_argument_with_context(message, function_name)
}

/// Standard function documentation format helper
pub fn create_function_docs(
    name: &str,
    description: &str,
    formula: Option<&str>,
    parameters: &[(&str, &str)],
    example: Option<&str>,
) -> String {
    let mut docs = String::new();
    docs.push_str(&format!("/// {}\n", name));
    docs.push_str("///\n");
    docs.push_str(&format!("/// {}\n", description));

    if let Some(formula) = formula {
        docs.push_str("///\n");
        docs.push_str(&format!("/// Formula: {}\n", formula));
    }

    if !parameters.is_empty() {
        docs.push_str("///\n");
        docs.push_str("/// # Parameters\n");
        for (param, desc) in parameters {
            docs.push_str(&format!("/// - `{}`: {}\n", param, desc));
        }
    }

    if let Some(example) = example {
        docs.push_str("///\n");
        docs.push_str("/// # Example\n");
        docs.push_str("/// ```rust\n");
        docs.push_str(&format!("/// {}\n", example));
        docs.push_str("/// ```\n");
    }

    docs
}

/// Standardized inplace operation handling
pub fn handle_inplace_operation<F>(
    input: &Tensor,
    inplace: bool,
    operation: F,
    _context: &str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    if inplace {
        // For true in-place operations, we would modify the tensor in place
        // For now, we perform the operation and return a new tensor
        // TODO: Implement proper in-place operations when tensor mutation is available
        operation(input)
    } else {
        operation(input)
    }
}

/// Utility for element-wise operations with inplace support
pub fn apply_elementwise_operation<F>(
    input: &Tensor,
    _inplace: bool,
    operation: F,
    _context: &str,
) -> TorshResult<Tensor>
where
    F: Fn(f32) -> f32,
{
    // For now, treat all operations as out-of-place since inplace is not fully implemented
    let data = input.data()?;
    let result_data: Vec<f32> = data.iter().map(|&x| operation(x)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Utility for conditional element-wise operations (like ReLU, LeakyReLU, etc.)
pub fn apply_conditional_elementwise<F>(
    input: &Tensor,
    condition: F,
    true_op: impl Fn(f32) -> f32,
    false_op: impl Fn(f32) -> f32,
    _inplace: bool,
    _context: &str,
) -> TorshResult<Tensor>
where
    F: Fn(f32) -> bool,
{
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if condition(x) {
                true_op(x)
            } else {
                false_op(x)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Common pattern for pooling output size calculation
pub fn calculate_pooling_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1);
    (input_size + 2 * padding - effective_kernel_size) / stride + 1
}

/// Common pattern for 2D pooling output size calculation
pub fn calculate_pooling_output_size_2d(
    input_size: (usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> (usize, usize) {
    let out_h =
        calculate_pooling_output_size(input_size.0, kernel_size.0, stride.0, padding.0, dilation.0);
    let out_w =
        calculate_pooling_output_size(input_size.1, kernel_size.1, stride.1, padding.1, dilation.1);
    (out_h, out_w)
}

/// Common pattern for 3D pooling output size calculation
pub fn calculate_pooling_output_size_3d(
    input_size: (usize, usize, usize),
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
) -> (usize, usize, usize) {
    let out_d =
        calculate_pooling_output_size(input_size.0, kernel_size.0, stride.0, padding.0, dilation.0);
    let out_h =
        calculate_pooling_output_size(input_size.1, kernel_size.1, stride.1, padding.1, dilation.1);
    let out_w =
        calculate_pooling_output_size(input_size.2, kernel_size.2, stride.2, padding.2, dilation.2);
    (out_d, out_h, out_w)
}

/// Utility for creating tensors with same shape and device as input
pub fn create_tensor_like(
    reference: &Tensor,
    data: Vec<f32>,
    shape: Option<Vec<usize>>,
) -> TorshResult<Tensor> {
    let tensor_shape = match shape {
        Some(s) => s,
        None => reference.shape().dims().to_vec(),
    };

    Tensor::from_data(data, tensor_shape, reference.device())
}

/// Common pattern for element-wise tensor operations with broadcasting
pub fn apply_binary_elementwise<F>(
    a: &Tensor,
    b: &Tensor,
    operation: F,
    _context: &str,
) -> TorshResult<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    validate_elementwise_shapes(a, b)?;

    let data_a = a.data()?;
    let data_b = b.data()?;

    let result_data: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(&x, &y)| operation(x, y))
        .collect();

    create_tensor_like(a, result_data, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_validate_range() -> TorshResult<()> {
        // Valid range
        validate_range(5.0, 0.0, 10.0, "value", "test")?;

        // Invalid range - too small
        let result = validate_range(-1.0, 0.0, 10.0, "value", "test");
        assert!(result.is_err());

        // Invalid range - too large
        let result = validate_range(15.0, 0.0, 10.0, "value", "test");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_validate_non_empty() -> TorshResult<()> {
        // Non-empty tensor
        let tensor = zeros(&[2, 3])?;
        validate_non_empty(&tensor, "test")?;

        // Empty tensor
        let empty_tensor = zeros(&[0])?;
        let result = validate_non_empty(&empty_tensor, "test");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_validate_dimension() -> TorshResult<()> {
        let tensor = zeros(&[2, 3, 4])?;

        // Valid dimensions
        validate_dimension(&tensor, 0, "test")?;
        validate_dimension(&tensor, 1, "test")?;
        validate_dimension(&tensor, 2, "test")?;
        validate_dimension(&tensor, -1, "test")?; // Last dimension
        validate_dimension(&tensor, -2, "test")?; // Second to last

        // Invalid dimensions
        let result = validate_dimension(&tensor, 3, "test");
        assert!(result.is_err());

        let result = validate_dimension(&tensor, -4, "test");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_validate_positive() -> TorshResult<()> {
        // Valid positive value
        validate_positive(1.5, "value", "test")?;

        // Invalid zero value
        let result = validate_positive(0.0, "value", "test");
        assert!(result.is_err());

        // Invalid negative value
        let result = validate_positive(-1.0, "value", "test");
        assert!(result.is_err());

        Ok(())
    }
}
