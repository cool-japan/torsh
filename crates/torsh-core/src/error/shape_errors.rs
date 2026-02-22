//! Shape-related error types for ToRSh
//!
//! This module contains all error variants related to tensor shape mismatches,
//! broadcasting incompatibilities, and shape validation failures.

use crate::error::core::{format_shape, ErrorLocation};
use thiserror::Error;

/// Shape-related error variants
#[derive(Error, Debug, Clone)]
pub enum ShapeError {
    /// Shape mismatch in tensor operations
    #[error(
        "Shape mismatch: expected {}, got {}",
        format_shape(expected),
        format_shape(got)
    )]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Broadcasting incompatible shapes
    #[error(
        "Broadcasting error: incompatible shapes {} and {}",
        format_shape(shape1),
        format_shape(shape2)
    )]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    /// Detailed broadcasting error with comprehensive error handling
    #[error("Broadcasting error: {0}")]
    DetailedBroadcastError(String),

    /// Matrix multiplication shape mismatch
    #[error(
        "Matrix multiplication shape error: left shape {} is incompatible with right shape {}",
        format_shape(left),
        format_shape(right)
    )]
    MatmulShapeError { left: Vec<usize>, right: Vec<usize> },

    /// Incompatible dimensions for concatenation
    #[error("Concatenation shape error: incompatible shapes at dimension {dim}")]
    ConcatShapeError { shapes: Vec<Vec<usize>>, dim: usize },

    /// Reshape error - incompatible element count
    #[error("Reshape error: cannot reshape tensor with {original_elements} elements to shape {} ({target_elements} elements)", format_shape(target_shape))]
    ReshapeError {
        original_elements: usize,
        target_shape: Vec<usize>,
        target_elements: usize,
    },

    /// Invalid shape specification
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Dimension count mismatch in tensor operations
    #[error(
        "Dimension mismatch: expected {expected} dimensions, got {got} in operation '{operation}'"
    )]
    DimensionMismatch {
        expected: usize,
        got: usize,
        operation: String,
    },

    /// Convolution operation shape error with detailed context
    #[error("Convolution error: input shape {} incompatible with kernel shape {} (stride: {}, padding: {}, dilation: {})",
        format_shape(input_shape), format_shape(kernel_shape), format_shape(stride), format_shape(padding), format_shape(dilation))]
    ConvolutionShapeError {
        input_shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },

    /// Linear layer operation shape error
    #[error("Linear layer error: input shape {} incompatible with weight shape {} (expected input features: {}, got: {})",
        format_shape(input_shape), format_shape(weight_shape), expected_features, actual_features)]
    LinearShapeError {
        input_shape: Vec<usize>,
        weight_shape: Vec<usize>,
        expected_features: usize,
        actual_features: usize,
    },

    /// Pooling operation shape error
    #[error("Pooling error: input shape {} incompatible with pooling parameters (kernel: {}, stride: {}, padding: {})",
        format_shape(input_shape), format_shape(kernel_size), format_shape(stride), format_shape(padding))]
    PoolingShapeError {
        input_shape: Vec<usize>,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },

    /// Indexing operation shape error
    #[error("Indexing error: tensor shape {} incompatible with indices (dimension: {}, index: {}, max_size: {})",
        format_shape(tensor_shape), dimension, index, max_size)]
    IndexingShapeError {
        tensor_shape: Vec<usize>,
        dimension: usize,
        index: usize,
        max_size: usize,
    },

    /// Batch operation shape error
    #[error("Batch operation error: tensors have incompatible shapes for batching (shapes: {})",
        shapes.iter().map(|s| format!("{}", format_shape(s))).collect::<Vec<_>>().join(", "))]
    BatchShapeError {
        shapes: Vec<Vec<usize>>,
        operation: String,
    },

    /// Element-wise operation shape error
    #[error(
        "Element-wise operation '{}' error: incompatible shapes {} and {}",
        operation,
        format_shape(left_shape),
        format_shape(right_shape)
    )]
    ElementWiseShapeError {
        operation: String,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },

    /// Reduction operation shape error
    #[error("Reduction operation '{}' error: cannot reduce tensor shape {} along dimension {} (dimension out of bounds)",
        operation, format_shape(tensor_shape), dimension)]
    ReductionShapeError {
        operation: String,
        tensor_shape: Vec<usize>,
        dimension: usize,
    },

    /// Shape mismatch with location information
    #[error(
        "Shape mismatch at {location}: expected {}, got {}",
        format_shape(expected),
        format_shape(got)
    )]
    ShapeMismatchWithLocation {
        expected: Vec<usize>,
        got: Vec<usize>,
        location: ErrorLocation,
    },
}

impl ShapeError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }

    /// Create a broadcasting error
    pub fn broadcast_error(shape1: &[usize], shape2: &[usize]) -> Self {
        Self::BroadcastError {
            shape1: shape1.to_vec(),
            shape2: shape2.to_vec(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, got: usize, operation: &str) -> Self {
        Self::DimensionMismatch {
            expected,
            got,
            operation: operation.to_string(),
        }
    }

    /// Create a matmul shape error
    pub fn matmul_shape_error(left: &[usize], right: &[usize]) -> Self {
        Self::MatmulShapeError {
            left: left.to_vec(),
            right: right.to_vec(),
        }
    }

    /// Create a reshape error
    pub fn reshape_error(
        original_elements: usize,
        target_shape: &[usize],
        target_elements: usize,
    ) -> Self {
        Self::ReshapeError {
            original_elements,
            target_shape: target_shape.to_vec(),
            target_elements,
        }
    }

    /// Create a convolution shape error
    pub fn convolution_shape_error(
        input_shape: &[usize],
        kernel_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Self {
        Self::ConvolutionShapeError {
            input_shape: input_shape.to_vec(),
            kernel_shape: kernel_shape.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
        }
    }

    /// Create a linear layer shape error
    pub fn linear_shape_error(
        input_shape: &[usize],
        weight_shape: &[usize],
        expected_features: usize,
        actual_features: usize,
    ) -> Self {
        Self::LinearShapeError {
            input_shape: input_shape.to_vec(),
            weight_shape: weight_shape.to_vec(),
            expected_features,
            actual_features,
        }
    }

    /// Create an element-wise operation shape error
    pub fn element_wise_shape_error(
        operation: &str,
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Self {
        Self::ElementWiseShapeError {
            operation: operation.to_string(),
            left_shape: left_shape.to_vec(),
            right_shape: right_shape.to_vec(),
        }
    }

    /// Get the error category
    pub fn category(&self) -> crate::error::core::ErrorCategory {
        crate::error::core::ErrorCategory::Shape
    }

    /// Get the error severity
    pub fn severity(&self) -> crate::error::core::ErrorSeverity {
        match self {
            Self::ShapeMismatch { .. }
            | Self::BroadcastError { .. }
            | Self::MatmulShapeError { .. }
            | Self::ConvolutionShapeError { .. } => crate::error::core::ErrorSeverity::High,
            Self::DimensionMismatch { .. } | Self::ReshapeError { .. } => {
                crate::error::core::ErrorSeverity::Medium
            }
            _ => crate::error::core::ErrorSeverity::Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_error() {
        let error = ShapeError::shape_mismatch(&[2, 3], &[3, 2]);
        match error {
            ShapeError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(got, vec![3, 2]);
            }
            _ => panic!("Expected ShapeMismatch variant"),
        }
    }

    #[test]
    fn test_broadcast_error() {
        let error = ShapeError::broadcast_error(&[2, 3], &[4, 5]);
        match error {
            ShapeError::BroadcastError { shape1, shape2 } => {
                assert_eq!(shape1, vec![2, 3]);
                assert_eq!(shape2, vec![4, 5]);
            }
            _ => panic!("Expected BroadcastError variant"),
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let error = ShapeError::dimension_mismatch(3, 2, "add");
        match error {
            ShapeError::DimensionMismatch {
                expected,
                got,
                operation,
            } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
                assert_eq!(operation, "add");
            }
            _ => panic!("Expected DimensionMismatch variant"),
        }
    }

    #[test]
    fn test_matmul_shape_error() {
        let error = ShapeError::matmul_shape_error(&[2, 3], &[4, 5]);
        match error {
            ShapeError::MatmulShapeError { left, right } => {
                assert_eq!(left, vec![2, 3]);
                assert_eq!(right, vec![4, 5]);
            }
            _ => panic!("Expected MatmulShapeError variant"),
        }
    }

    #[test]
    fn test_reshape_error() {
        let error = ShapeError::reshape_error(6, &[2, 4], 8);
        match error {
            ShapeError::ReshapeError {
                original_elements,
                target_shape,
                target_elements,
            } => {
                assert_eq!(original_elements, 6);
                assert_eq!(target_shape, vec![2, 4]);
                assert_eq!(target_elements, 8);
            }
            _ => panic!("Expected ReshapeError variant"),
        }
    }

    #[test]
    fn test_error_severity() {
        let shape_mismatch = ShapeError::shape_mismatch(&[2, 3], &[3, 2]);
        assert_eq!(
            shape_mismatch.severity(),
            crate::error::core::ErrorSeverity::High
        );

        let dimension_mismatch = ShapeError::dimension_mismatch(3, 2, "add");
        assert_eq!(
            dimension_mismatch.severity(),
            crate::error::core::ErrorSeverity::Medium
        );
    }

    #[test]
    fn test_error_category() {
        let error = ShapeError::shape_mismatch(&[2, 3], &[3, 2]);
        assert_eq!(error.category(), crate::error::core::ErrorCategory::Shape);
    }

    #[test]
    fn test_error_display() {
        let error = ShapeError::shape_mismatch(&[2, 3], &[3, 2]);
        let error_string = format!("{}", error);
        assert!(error_string.contains("Shape mismatch"));
        assert!(error_string.contains("[2, 3]"));
        assert!(error_string.contains("[3, 2]"));
    }

    #[test]
    fn test_convolution_shape_error() {
        let error = ShapeError::convolution_shape_error(
            &[1, 3, 32, 32],
            &[64, 3, 3, 3],
            &[1, 1],
            &[0, 0],
            &[1, 1],
        );
        let error_string = format!("{}", error);
        assert!(error_string.contains("Convolution error"));
    }

    #[test]
    fn test_linear_shape_error() {
        let error = ShapeError::linear_shape_error(&[2, 512], &[256, 512], 512, 256);
        let error_string = format!("{}", error);
        assert!(error_string.contains("Linear layer error"));
        assert!(error_string.contains("expected input features: 512"));
        assert!(error_string.contains("got: 256"));
    }
}
