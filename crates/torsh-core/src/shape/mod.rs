//! Shape and stride utilities for tensors
//!
//! This module provides comprehensive shape manipulation, validation, and optimization
//! utilities for tensor operations. The module is organized into focused components
//! for better maintainability and performance.

// Core shape types and fundamental operations
pub mod const_generic;
pub mod core;

// Shape construction and builder patterns (to be implemented)
// pub mod builder;

// Shape operations and transformations (to be implemented)
// pub mod operations;

// Validation traits and utilities (to be implemented)
// pub mod validation;

// Caching and performance optimization (to be implemented)
// pub mod caching;

// Utility functions and helpers (to be implemented)
// pub mod utils;

// Re-export core types for backward compatibility
pub use core::Shape;

// Re-export const generic types for compile-time shape checking
pub use const_generic::{
    common, utils as const_utils, BroadcastCompatible, ConstShape, ConstShapeOps, MatMulCompatible,
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, ReshapeInto, ShapeRank, SqueezeOps, TransposeOps,
    UnsqueezeOps,
};

// Type aliases for convenience and backward compatibility
pub type ShapeRef<'a> = &'a Shape;

/// Common shape patterns for validation and creation
pub mod patterns {
    use super::Shape;
    use crate::error::Result;

    /// Common neural network layer shapes
    pub struct NeuralNetShapes;

    impl NeuralNetShapes {
        /// Create a fully connected layer input shape [batch_size, features]
        pub fn fully_connected_input(batch_size: usize, features: usize) -> Result<Shape> {
            Shape::from_2d(batch_size, features)
        }

        /// Create a convolutional layer input shape [batch_size, channels, height, width]
        pub fn conv2d_input(
            batch_size: usize,
            channels: usize,
            height: usize,
            width: usize,
        ) -> Result<Shape> {
            Shape::from_4d(batch_size, channels, height, width)
        }

        /// Create a sequence shape [batch_size, sequence_length, features]
        pub fn sequence(batch_size: usize, seq_len: usize, features: usize) -> Result<Shape> {
            Shape::from_3d(batch_size, seq_len, features)
        }

        /// Create an image batch shape [batch_size, channels, height, width]
        pub fn image_batch(
            batch_size: usize,
            channels: usize,
            height: usize,
            width: usize,
        ) -> Result<Shape> {
            Shape::from_4d(batch_size, channels, height, width)
        }
    }

    /// Common computer vision shapes
    pub struct VisionShapes;

    impl VisionShapes {
        /// Create RGB image shape [3, height, width]
        pub fn rgb_image(height: usize, width: usize) -> Result<Shape> {
            Shape::from_3d(3, height, width)
        }

        /// Create grayscale image shape [1, height, width]
        pub fn grayscale_image(height: usize, width: usize) -> Result<Shape> {
            Shape::from_3d(1, height, width)
        }

        /// Create bounding box shape [num_boxes, 4] for (x, y, w, h)
        pub fn bounding_boxes(num_boxes: usize) -> Result<Shape> {
            Shape::from_2d(num_boxes, 4)
        }

        /// Create keypoints shape [num_keypoints, 2] for (x, y) coordinates
        pub fn keypoints(num_keypoints: usize) -> Result<Shape> {
            Shape::from_2d(num_keypoints, 2)
        }
    }

    /// Common NLP shapes
    pub struct NlpShapes;

    impl NlpShapes {
        /// Create token indices shape [batch_size, sequence_length]
        pub fn token_indices(batch_size: usize, seq_len: usize) -> Result<Shape> {
            Shape::from_2d(batch_size, seq_len)
        }

        /// Create attention mask shape [batch_size, sequence_length]
        pub fn attention_mask(batch_size: usize, seq_len: usize) -> Result<Shape> {
            Shape::from_2d(batch_size, seq_len)
        }

        /// Create embeddings shape [vocab_size, embedding_dim]
        pub fn embeddings(vocab_size: usize, embedding_dim: usize) -> Result<Shape> {
            Shape::from_2d(vocab_size, embedding_dim)
        }

        /// Create transformer hidden states shape [batch_size, sequence_length, hidden_size]
        pub fn transformer_hidden(
            batch_size: usize,
            seq_len: usize,
            hidden_size: usize,
        ) -> Result<Shape> {
            Shape::from_3d(batch_size, seq_len, hidden_size)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_neural_net_shapes() {
            let fc_input = NeuralNetShapes::fully_connected_input(32, 784).unwrap();
            assert_eq!(fc_input.dims(), &[32, 784]);

            let conv_input = NeuralNetShapes::conv2d_input(16, 3, 224, 224).unwrap();
            assert_eq!(conv_input.dims(), &[16, 3, 224, 224]);

            let sequence = NeuralNetShapes::sequence(8, 50, 512).unwrap();
            assert_eq!(sequence.dims(), &[8, 50, 512]);
        }

        #[test]
        fn test_vision_shapes() {
            let rgb = VisionShapes::rgb_image(224, 224).unwrap();
            assert_eq!(rgb.dims(), &[3, 224, 224]);

            let grayscale = VisionShapes::grayscale_image(128, 128).unwrap();
            assert_eq!(grayscale.dims(), &[1, 128, 128]);

            let boxes = VisionShapes::bounding_boxes(10).unwrap();
            assert_eq!(boxes.dims(), &[10, 4]);
        }

        #[test]
        fn test_nlp_shapes() {
            let tokens = NlpShapes::token_indices(4, 128).unwrap();
            assert_eq!(tokens.dims(), &[4, 128]);

            let embeddings = NlpShapes::embeddings(30000, 768).unwrap();
            assert_eq!(embeddings.dims(), &[30000, 768]);

            let hidden = NlpShapes::transformer_hidden(2, 50, 512).unwrap();
            assert_eq!(hidden.dims(), &[2, 50, 512]);
        }
    }
}

/// Shape manipulation utilities
pub mod utils {
    use super::Shape;
    use crate::error::Result;

    /// Calculate the number of elements that would result from reshaping
    pub fn calculate_reshape_numel(current_shape: &Shape, new_dims: &[i32]) -> Result<usize> {
        let current_numel = current_shape.numel();

        // Count -1 dimensions (inferred dimensions)
        let inferred_count = new_dims.iter().filter(|&&d| d == -1).count();
        if inferred_count > 1 {
            return Err(crate::error::TorshError::InvalidOperation(
                "Cannot infer more than one dimension".to_string(),
            ));
        }

        if inferred_count == 0 {
            // No inference needed, just calculate product
            let new_numel: usize = new_dims
                .iter()
                .map(|&d| d as usize)
                .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    crate::error::TorshError::InvalidOperation(
                        "Shape dimensions would overflow".to_string(),
                    )
                })?;

            if new_numel != current_numel {
                return Err(crate::error::TorshError::InvalidOperation(format!(
                    "Cannot reshape tensor with {} elements to shape with {} elements",
                    current_numel, new_numel
                )));
            }
            Ok(new_numel)
        } else {
            // Calculate product of known dimensions
            let known_product: usize = new_dims
                .iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    crate::error::TorshError::InvalidOperation(
                        "Shape dimensions would overflow".to_string(),
                    )
                })?;

            if !current_numel.is_multiple_of(known_product) {
                return Err(crate::error::TorshError::InvalidOperation(format!(
                    "Cannot infer dimension: {} elements cannot be evenly divided by {}",
                    current_numel, known_product
                )));
            }

            Ok(current_numel)
        }
    }

    /// Infer the missing dimension in a reshape operation
    pub fn infer_reshape_dimension(current_shape: &Shape, new_dims: &[i32]) -> Result<Vec<usize>> {
        let current_numel = current_shape.numel();

        let inferred_indices: Vec<usize> = new_dims
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == -1)
            .map(|(i, _)| i)
            .collect();

        if inferred_indices.len() > 1 {
            return Err(crate::error::TorshError::InvalidOperation(
                "Cannot infer more than one dimension".to_string(),
            ));
        }

        let mut result_dims = Vec::with_capacity(new_dims.len());

        if inferred_indices.is_empty() {
            // No inference needed
            for &dim in new_dims {
                result_dims.push(dim as usize);
            }
        } else {
            // Calculate the inferred dimension
            let known_product: usize = new_dims
                .iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    crate::error::TorshError::InvalidOperation(
                        "Shape dimensions would overflow".to_string(),
                    )
                })?;

            if !current_numel.is_multiple_of(known_product) {
                return Err(crate::error::TorshError::InvalidOperation(format!(
                    "Cannot infer dimension: {} elements cannot be evenly divided by {}",
                    current_numel, known_product
                )));
            }

            let inferred_dim = current_numel / known_product;

            for &dim in new_dims.iter() {
                if dim == -1 {
                    result_dims.push(inferred_dim);
                } else {
                    result_dims.push(dim as usize);
                }
            }
        }

        Ok(result_dims)
    }

    /// Check if a shape can be viewed as another shape (same number of elements)
    pub fn can_view_as(from_shape: &Shape, to_shape: &Shape) -> bool {
        from_shape.numel() == to_shape.numel()
    }

    /// Calculate strides for a contiguous tensor with given shape
    pub fn calculate_contiguous_strides(shape: &Shape) -> Vec<usize> {
        let dims = shape.dims();
        if dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// Check if strides represent a contiguous tensor
    pub fn is_contiguous_strides(shape: &Shape, strides: &[usize]) -> bool {
        if shape.dims().len() != strides.len() {
            return false;
        }

        let expected_strides = calculate_contiguous_strides(shape);
        strides == expected_strides
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_reshape_calculation() {
            let shape = Shape::new(vec![2, 3, 4]);
            let numel = calculate_reshape_numel(&shape, &[6, 4]).unwrap();
            assert_eq!(numel, 24);

            let inferred = infer_reshape_dimension(&shape, &[-1, 4]).unwrap();
            assert_eq!(inferred, vec![6, 4]);

            let inferred2 = infer_reshape_dimension(&shape, &[2, -1]).unwrap();
            assert_eq!(inferred2, vec![2, 12]);
        }

        #[test]
        fn test_view_compatibility() {
            let shape1 = Shape::new(vec![2, 3, 4]);
            let shape2 = Shape::new(vec![6, 4]);
            assert!(can_view_as(&shape1, &shape2));

            let shape3 = Shape::new(vec![2, 5]);
            assert!(!can_view_as(&shape1, &shape3));
        }

        #[test]
        fn test_strides() {
            let shape = Shape::new(vec![2, 3, 4]);
            let strides = calculate_contiguous_strides(&shape);
            assert_eq!(strides, vec![12, 4, 1]);

            assert!(is_contiguous_strides(&shape, &strides));
            assert!(!is_contiguous_strides(&shape, &[1, 2, 3]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_integration() {
        // Test core Shape functionality
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.ndim(), 3);

        // Test pattern utilities
        let conv_shape = patterns::NeuralNetShapes::conv2d_input(1, 3, 224, 224).unwrap();
        assert_eq!(conv_shape.dims(), &[1, 3, 224, 224]);

        // Test utility functions
        let strides = utils::calculate_contiguous_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }
}
