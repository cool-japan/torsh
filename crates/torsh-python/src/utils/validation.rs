//! Input validation utilities for Python bindings

use crate::error::PyResult;
use pyo3::prelude::*;

/// Validate that a shape is valid (all dimensions > 0)
pub fn validate_shape(shape: &[usize]) -> PyResult<()> {
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid shape: dimension {} cannot be zero",
                i
            )));
        }
    }
    Ok(())
}

/// Validate that an index is within bounds for a given dimension
pub fn validate_index(index: i64, dim_size: usize) -> PyResult<usize> {
    let positive_index = if index < 0 {
        let abs_index = (-index) as usize;
        if abs_index > dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for dimension with size {}",
                index, dim_size
            )));
        }
        dim_size - abs_index
    } else {
        let pos_index = index as usize;
        if pos_index >= dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for dimension with size {}",
                index, dim_size
            )));
        }
        pos_index
    };
    Ok(positive_index)
}

/// Validate that dimensions are compatible for broadcasting
pub fn validate_broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> PyResult<Vec<usize>> {
    let mut result_shape = Vec::new();
    let max_dims = shape1.len().max(shape2.len());

    for i in 0..max_dims {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result_shape.push(dim1.max(dim2));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                shape1, shape2
            )));
        }
    }

    result_shape.reverse();
    Ok(result_shape)
}

/// Validate that a learning rate is positive
pub fn validate_learning_rate(lr: f32) -> PyResult<()> {
    if lr <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Learning rate must be positive",
        ));
    }
    Ok(())
}

/// Validate that momentum is in valid range [0, 1]
pub fn validate_momentum(momentum: f32) -> PyResult<()> {
    if !(0.0..=1.0).contains(&momentum) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Momentum must be in range [0, 1]",
        ));
    }
    Ok(())
}

/// Validate that weight decay is non-negative
pub fn validate_weight_decay(weight_decay: f32) -> PyResult<()> {
    if weight_decay < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Weight decay must be non-negative",
        ));
    }
    Ok(())
}

/// Validate that epsilon is positive
pub fn validate_epsilon(eps: f32) -> PyResult<()> {
    if eps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Epsilon must be positive",
        ));
    }
    Ok(())
}

/// Validate beta parameters for Adam-like optimizers
pub fn validate_betas(betas: (f32, f32)) -> PyResult<()> {
    let (beta1, beta2) = betas;
    if !(0.0..1.0).contains(&beta1) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Beta1 must be in range [0, 1)",
        ));
    }
    if !(0.0..1.0).contains(&beta2) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Beta2 must be in range [0, 1)",
        ));
    }
    Ok(())
}

/// Validate that tensor dimensions match for operations
pub fn validate_tensor_shapes_match(shape1: &[usize], shape2: &[usize]) -> PyResult<()> {
    if shape1 != shape2 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Tensor shapes do not match: {:?} vs {:?}",
            shape1, shape2
        )));
    }
    Ok(())
}

/// Validate that a dimension index is valid for a tensor
pub fn validate_dimension(dim: i32, ndim: usize) -> PyResult<usize> {
    let positive_dim = if dim < 0 {
        let abs_dim = (-dim) as usize;
        if abs_dim > ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim, ndim
            )));
        }
        ndim - abs_dim
    } else {
        let pos_dim = dim as usize;
        if pos_dim >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim, ndim
            )));
        }
        pos_dim
    };
    Ok(positive_dim)
}

/// Validate that parameters list is not empty
pub fn validate_parameters_not_empty<T>(params: &[T]) -> PyResult<()> {
    if params.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Parameters list cannot be empty",
        ));
    }
    Ok(())
}

/// Validate dropout probability is in valid range [0, 1]
pub fn validate_dropout_probability(p: f32) -> PyResult<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Dropout probability must be in range [0, 1], got {}",
            p
        )));
    }
    Ok(())
}

/// Validate kernel size is positive
pub fn validate_kernel_size(kernel_size: usize, name: &str) -> PyResult<()> {
    if kernel_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be positive, got 0",
            name
        )));
    }
    Ok(())
}

/// Validate stride is positive
pub fn validate_stride(stride: usize, name: &str) -> PyResult<()> {
    if stride == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be positive, got 0",
            name
        )));
    }
    Ok(())
}

/// Validate that input tensor has expected number of dimensions
pub fn validate_tensor_ndim(
    actual_ndim: usize,
    expected_ndim: usize,
    op_name: &str,
) -> PyResult<()> {
    if actual_ndim != expected_ndim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} expects {}D input, got {}D",
            op_name, expected_ndim, actual_ndim
        )));
    }
    Ok(())
}

/// Validate that input tensor has at least minimum number of dimensions
pub fn validate_tensor_min_ndim(
    actual_ndim: usize,
    min_ndim: usize,
    op_name: &str,
) -> PyResult<()> {
    if actual_ndim < min_ndim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} expects at least {}D input, got {}D",
            op_name, min_ndim, actual_ndim
        )));
    }
    Ok(())
}

/// Validate that number of features matches expected value
pub fn validate_num_features(
    actual_features: usize,
    expected_features: usize,
    layer_name: &str,
) -> PyResult<()> {
    if actual_features != expected_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} expected {} features, got {}",
            layer_name, expected_features, actual_features
        )));
    }
    Ok(())
}

/// Validate that a value is finite (not NaN or infinity)
pub fn validate_finite(value: f32, name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be finite, got {}",
            name, value
        )));
    }
    Ok(())
}

/// Validate that a range is valid (start < end)
pub fn validate_range(start: usize, end: usize, name: &str) -> PyResult<()> {
    if start >= end {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid range for {}: start ({}) must be less than end ({})",
            name, start, end
        )));
    }
    Ok(())
}

/// Validate pooling output size calculation
pub fn validate_pooling_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PyResult<usize> {
    if kernel_size == 0 || stride == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Kernel size and stride must be positive",
        ));
    }

    let effective_kernel = dilation * (kernel_size - 1) + 1;
    if input_size + 2 * padding < effective_kernel {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Input size {} (with padding {}) is too small for kernel size {} (with dilation {})",
            input_size, padding, kernel_size, dilation
        )));
    }

    let output_size = (input_size + 2 * padding - effective_kernel) / stride + 1;
    Ok(output_size)
}

/// Validate convolution parameters
pub fn validate_conv_params(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> PyResult<()> {
    if in_channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "in_channels must be positive",
        ));
    }
    if out_channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "out_channels must be positive",
        ));
    }
    if kernel_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "kernel_size must be positive",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // Shape Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_shape_valid() {
        assert!(validate_shape(&[1, 2, 3]).is_ok());
        assert!(validate_shape(&[10, 20, 30, 40]).is_ok());
    }

    #[test]
    fn test_validate_shape_with_zero() {
        assert!(validate_shape(&[1, 0, 3]).is_err());
        assert!(validate_shape(&[0]).is_err());
    }

    #[test]
    fn test_validate_shape_empty() {
        // Empty shape (scalar) should be valid
        assert!(validate_shape(&[]).is_ok());
    }

    // =============================================================================
    // Index Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_index_positive() {
        assert_eq!(validate_index(0, 10).unwrap(), 0);
        assert_eq!(validate_index(5, 10).unwrap(), 5);
        assert_eq!(validate_index(9, 10).unwrap(), 9);
    }

    #[test]
    fn test_validate_index_negative() {
        assert_eq!(validate_index(-1, 10).unwrap(), 9);
        assert_eq!(validate_index(-5, 10).unwrap(), 5);
        assert_eq!(validate_index(-10, 10).unwrap(), 0);
    }

    #[test]
    fn test_validate_index_out_of_bounds_positive() {
        assert!(validate_index(10, 10).is_err());
        assert!(validate_index(100, 10).is_err());
    }

    #[test]
    fn test_validate_index_out_of_bounds_negative() {
        assert!(validate_index(-11, 10).is_err());
        assert!(validate_index(-100, 10).is_err());
    }

    // =============================================================================
    // Broadcasting Tests
    // =============================================================================

    #[test]
    fn test_validate_broadcast_shapes_compatible() {
        assert_eq!(
            validate_broadcast_shapes(&[3, 4], &[3, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            validate_broadcast_shapes(&[3, 1], &[3, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            validate_broadcast_shapes(&[1, 4], &[3, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            validate_broadcast_shapes(&[3, 4], &[4]).unwrap(),
            vec![3, 4]
        );
    }

    #[test]
    fn test_validate_broadcast_shapes_incompatible() {
        assert!(validate_broadcast_shapes(&[3, 4], &[3, 5]).is_err());
        assert!(validate_broadcast_shapes(&[2, 3], &[3, 4]).is_err());
    }

    // =============================================================================
    // Learning Rate Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_learning_rate_valid() {
        assert!(validate_learning_rate(0.001).is_ok());
        assert!(validate_learning_rate(0.1).is_ok());
        assert!(validate_learning_rate(1.0).is_ok());
        assert!(validate_learning_rate(10.0).is_ok());
    }

    #[test]
    fn test_validate_learning_rate_invalid() {
        assert!(validate_learning_rate(0.0).is_err());
        assert!(validate_learning_rate(-0.1).is_err());
    }

    // =============================================================================
    // Momentum Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_momentum_valid() {
        assert!(validate_momentum(0.0).is_ok());
        assert!(validate_momentum(0.5).is_ok());
        assert!(validate_momentum(0.9).is_ok());
        assert!(validate_momentum(1.0).is_ok());
    }

    #[test]
    fn test_validate_momentum_invalid() {
        assert!(validate_momentum(-0.1).is_err());
        assert!(validate_momentum(1.1).is_err());
    }

    // =============================================================================
    // Weight Decay Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_weight_decay_valid() {
        assert!(validate_weight_decay(0.0).is_ok());
        assert!(validate_weight_decay(0.01).is_ok());
        assert!(validate_weight_decay(1.0).is_ok());
    }

    #[test]
    fn test_validate_weight_decay_invalid() {
        assert!(validate_weight_decay(-0.1).is_err());
    }

    // =============================================================================
    // Epsilon Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_epsilon_valid() {
        assert!(validate_epsilon(1e-8).is_ok());
        assert!(validate_epsilon(1e-5).is_ok());
        assert!(validate_epsilon(0.1).is_ok());
    }

    #[test]
    fn test_validate_epsilon_invalid() {
        assert!(validate_epsilon(0.0).is_err());
        assert!(validate_epsilon(-1e-8).is_err());
    }

    // =============================================================================
    // Beta Parameters Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_betas_valid() {
        assert!(validate_betas((0.0, 0.0)).is_ok());
        assert!(validate_betas((0.9, 0.999)).is_ok());
        assert!(validate_betas((0.5, 0.5)).is_ok());
    }

    #[test]
    fn test_validate_betas_invalid() {
        assert!(validate_betas((-0.1, 0.5)).is_err());
        assert!(validate_betas((0.5, 1.0)).is_err());
        assert!(validate_betas((1.0, 0.5)).is_err());
        assert!(validate_betas((1.1, 0.5)).is_err());
    }

    // =============================================================================
    // Tensor Shape Matching Tests
    // =============================================================================

    #[test]
    fn test_validate_tensor_shapes_match_valid() {
        assert!(validate_tensor_shapes_match(&[3, 4], &[3, 4]).is_ok());
        assert!(validate_tensor_shapes_match(&[], &[]).is_ok());
    }

    #[test]
    fn test_validate_tensor_shapes_match_invalid() {
        assert!(validate_tensor_shapes_match(&[3, 4], &[3, 5]).is_err());
        assert!(validate_tensor_shapes_match(&[3, 4], &[4, 3]).is_err());
    }

    // =============================================================================
    // Dimension Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_dimension_positive() {
        assert_eq!(validate_dimension(0, 4).unwrap(), 0);
        assert_eq!(validate_dimension(2, 4).unwrap(), 2);
        assert_eq!(validate_dimension(3, 4).unwrap(), 3);
    }

    #[test]
    fn test_validate_dimension_negative() {
        assert_eq!(validate_dimension(-1, 4).unwrap(), 3);
        assert_eq!(validate_dimension(-2, 4).unwrap(), 2);
        assert_eq!(validate_dimension(-4, 4).unwrap(), 0);
    }

    #[test]
    fn test_validate_dimension_out_of_bounds() {
        assert!(validate_dimension(4, 4).is_err());
        assert!(validate_dimension(-5, 4).is_err());
    }

    // =============================================================================
    // Parameters Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_parameters_not_empty_valid() {
        assert!(validate_parameters_not_empty(&[1, 2, 3]).is_ok());
    }

    #[test]
    fn test_validate_parameters_not_empty_invalid() {
        let empty: &[i32] = &[];
        assert!(validate_parameters_not_empty(empty).is_err());
    }

    // =============================================================================
    // Dropout Probability Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_dropout_probability_valid() {
        assert!(validate_dropout_probability(0.0).is_ok());
        assert!(validate_dropout_probability(0.5).is_ok());
        assert!(validate_dropout_probability(1.0).is_ok());
    }

    #[test]
    fn test_validate_dropout_probability_invalid() {
        assert!(validate_dropout_probability(-0.1).is_err());
        assert!(validate_dropout_probability(1.1).is_err());
    }

    // =============================================================================
    // Kernel Size Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_kernel_size_valid() {
        assert!(validate_kernel_size(1, "kernel").is_ok());
        assert!(validate_kernel_size(3, "kernel").is_ok());
        assert!(validate_kernel_size(5, "kernel").is_ok());
    }

    #[test]
    fn test_validate_kernel_size_invalid() {
        assert!(validate_kernel_size(0, "kernel").is_err());
    }

    // =============================================================================
    // Stride Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_stride_valid() {
        assert!(validate_stride(1, "stride").is_ok());
        assert!(validate_stride(2, "stride").is_ok());
    }

    #[test]
    fn test_validate_stride_invalid() {
        assert!(validate_stride(0, "stride").is_err());
    }

    // =============================================================================
    // Tensor NDim Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_tensor_ndim_valid() {
        assert!(validate_tensor_ndim(4, 4, "conv2d").is_ok());
        assert!(validate_tensor_ndim(2, 2, "linear").is_ok());
    }

    #[test]
    fn test_validate_tensor_ndim_invalid() {
        assert!(validate_tensor_ndim(3, 4, "conv2d").is_err());
        assert!(validate_tensor_ndim(5, 4, "conv2d").is_err());
    }

    // =============================================================================
    // Tensor Min NDim Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_tensor_min_ndim_valid() {
        assert!(validate_tensor_min_ndim(4, 2, "operation").is_ok());
        assert!(validate_tensor_min_ndim(2, 2, "operation").is_ok());
    }

    #[test]
    fn test_validate_tensor_min_ndim_invalid() {
        assert!(validate_tensor_min_ndim(1, 2, "operation").is_err());
    }

    // =============================================================================
    // Number of Features Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_num_features_valid() {
        assert!(validate_num_features(64, 64, "BatchNorm").is_ok());
    }

    #[test]
    fn test_validate_num_features_invalid() {
        assert!(validate_num_features(32, 64, "BatchNorm").is_err());
    }

    // =============================================================================
    // Finite Value Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_finite_valid() {
        assert!(validate_finite(0.0, "value").is_ok());
        assert!(validate_finite(1.0, "value").is_ok());
        assert!(validate_finite(-1.0, "value").is_ok());
    }

    #[test]
    fn test_validate_finite_invalid() {
        assert!(validate_finite(f32::NAN, "value").is_err());
        assert!(validate_finite(f32::INFINITY, "value").is_err());
        assert!(validate_finite(f32::NEG_INFINITY, "value").is_err());
    }

    // =============================================================================
    // Range Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_range_valid() {
        assert!(validate_range(0, 10, "range").is_ok());
        assert!(validate_range(5, 10, "range").is_ok());
    }

    #[test]
    fn test_validate_range_invalid() {
        assert!(validate_range(10, 10, "range").is_err());
        assert!(validate_range(10, 5, "range").is_err());
    }

    // =============================================================================
    // Pooling Output Size Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_pooling_output_size_valid() {
        // Input: 28, Kernel: 2, Stride: 2, Padding: 0, Dilation: 1
        // Output: (28 + 0 - 2) / 2 + 1 = 14
        assert_eq!(validate_pooling_output_size(28, 2, 2, 0, 1).unwrap(), 14);

        // Input: 32, Kernel: 3, Stride: 1, Padding: 1, Dilation: 1
        // Output: (32 + 2 - 3) / 1 + 1 = 32
        assert_eq!(validate_pooling_output_size(32, 3, 1, 1, 1).unwrap(), 32);
    }

    #[test]
    fn test_validate_pooling_output_size_invalid_zero_kernel() {
        assert!(validate_pooling_output_size(28, 0, 2, 0, 1).is_err());
    }

    #[test]
    fn test_validate_pooling_output_size_invalid_zero_stride() {
        assert!(validate_pooling_output_size(28, 2, 0, 0, 1).is_err());
    }

    #[test]
    fn test_validate_pooling_output_size_invalid_too_small() {
        // Input: 2, Kernel: 5, Stride: 1, Padding: 0, Dilation: 1
        // Input too small for kernel
        assert!(validate_pooling_output_size(2, 5, 1, 0, 1).is_err());
    }

    // =============================================================================
    // Convolution Parameters Validation Tests
    // =============================================================================

    #[test]
    fn test_validate_conv_params_valid() {
        assert!(validate_conv_params(3, 64, 3).is_ok());
        assert!(validate_conv_params(64, 128, 5).is_ok());
    }

    #[test]
    fn test_validate_conv_params_invalid_in_channels() {
        assert!(validate_conv_params(0, 64, 3).is_err());
    }

    #[test]
    fn test_validate_conv_params_invalid_out_channels() {
        assert!(validate_conv_params(3, 0, 3).is_err());
    }

    #[test]
    fn test_validate_conv_params_invalid_kernel_size() {
        assert!(validate_conv_params(3, 64, 0).is_err());
    }
}
