//! Shape Utility Functions for Common Patterns
//!
//! This module provides convenient utility functions for common shape operations
//! and transformations, making it easier to work with tensor shapes in common
//! machine learning scenarios.
//!
//! # Features
//!
//! - **Shape Creation**: Helpers for creating common shapes (scalar, vector, matrix, batch)
//! - **Shape Transformation**: Flatten, unflatten, squeeze, unsqueeze operations
//! - **Shape Manipulation**: Insert/remove dimensions, permute, expand
//! - **Shape Comparison**: Compatible shapes, broadcastable checks
//! - **Common Patterns**: Image shapes, sequence shapes, batch shapes
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::shape_utils::{image_shape, batch_shape, flatten_from, unsqueeze_at};
//! use torsh_core::Shape;
//!
//! // Create common shapes
//! let img = image_shape(224, 224, 3);  // [224, 224, 3]
//! let batch_img = batch_shape(32, &img);  // [32, 224, 224, 3]
//!
//! // Transform shapes
//! let flattened = flatten_from(&batch_img, 1);  // [32, 150528]
//! let expanded = unsqueeze_at(&img, 0);  // [1, 224, 224, 3]
//! ```

use crate::error::{Result, TorshError};
use crate::shape::Shape;

/// Create a scalar shape (0-dimensional tensor)
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::scalar_shape;
///
/// let shape = scalar_shape();
/// let empty: &[usize] = &[];
/// assert_eq!(shape.dims(), empty);
/// assert!(shape.is_scalar());
/// ```
pub fn scalar_shape() -> Shape {
    Shape::new(vec![])
}

/// Create a vector shape (1-dimensional tensor)
///
/// # Arguments
///
/// * `size` - Number of elements in the vector
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::vector_shape;
///
/// let shape = vector_shape(128);
/// assert_eq!(shape.dims(), &[128]);
/// ```
pub fn vector_shape(size: usize) -> Shape {
    Shape::new(vec![size])
}

/// Create a matrix shape (2-dimensional tensor)
///
/// # Arguments
///
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::matrix_shape;
///
/// let shape = matrix_shape(64, 128);
/// assert_eq!(shape.dims(), &[64, 128]);
/// ```
pub fn matrix_shape(rows: usize, cols: usize) -> Shape {
    Shape::new(vec![rows, cols])
}

/// Create an image shape (height, width, channels)
///
/// # Arguments
///
/// * `height` - Image height
/// * `width` - Image width
/// * `channels` - Number of channels (e.g., 3 for RGB, 1 for grayscale)
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::image_shape;
///
/// let rgb = image_shape(224, 224, 3);
/// assert_eq!(rgb.dims(), &[224, 224, 3]);
///
/// let grayscale = image_shape(28, 28, 1);
/// assert_eq!(grayscale.dims(), &[28, 28, 1]);
/// ```
pub fn image_shape(height: usize, width: usize, channels: usize) -> Shape {
    Shape::new(vec![height, width, channels])
}

/// Create a batch shape by prepending batch dimension
///
/// # Arguments
///
/// * `batch_size` - Size of the batch dimension
/// * `base_shape` - Base shape to add batch dimension to
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::{batch_shape, image_shape};
///
/// let img = image_shape(224, 224, 3);
/// let batch = batch_shape(32, &img);
/// assert_eq!(batch.dims(), &[32, 224, 224, 3]);
/// ```
pub fn batch_shape(batch_size: usize, base_shape: &Shape) -> Shape {
    let mut dims = vec![batch_size];
    dims.extend_from_slice(base_shape.dims());
    Shape::new(dims)
}

/// Create a sequence shape (sequence_length, features)
///
/// # Arguments
///
/// * `seq_len` - Length of the sequence
/// * `features` - Number of features per timestep
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::sequence_shape;
///
/// let seq = sequence_shape(100, 512);  // 100 timesteps, 512 features
/// assert_eq!(seq.dims(), &[100, 512]);
/// ```
pub fn sequence_shape(seq_len: usize, features: usize) -> Shape {
    Shape::new(vec![seq_len, features])
}

/// Flatten shape from a given dimension
///
/// All dimensions from `start_dim` onwards are collapsed into a single dimension.
///
/// # Arguments
///
/// * `shape` - Shape to flatten
/// * `start_dim` - Dimension to start flattening from (inclusive)
///
/// # Returns
///
/// New shape with dimensions flattened, or error if start_dim is invalid
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::flatten_from;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![32, 3, 224, 224]);
/// let flattened = flatten_from(&shape, 1).unwrap();
/// assert_eq!(flattened.dims(), &[32, 150528]);  // 3*224*224 = 150528
/// ```
pub fn flatten_from(shape: &Shape, start_dim: usize) -> Result<Shape> {
    let dims = shape.dims();

    if start_dim > dims.len() {
        return Err(TorshError::InvalidShape(format!(
            "start_dim {} is out of bounds for shape with {} dimensions",
            start_dim,
            dims.len()
        )));
    }

    if start_dim == dims.len() {
        return Ok(shape.clone());
    }

    let mut new_dims = Vec::with_capacity(start_dim + 1);

    // Keep dimensions before start_dim
    new_dims.extend_from_slice(&dims[..start_dim]);

    // Compute product of dimensions from start_dim onwards
    let flattened_size: usize = dims[start_dim..].iter().product();
    new_dims.push(flattened_size);

    Ok(Shape::new(new_dims))
}

/// Unsqueeze (add) a dimension at the specified position
///
/// # Arguments
///
/// * `shape` - Original shape
/// * `dim` - Position to insert new dimension
///
/// # Returns
///
/// New shape with dimension added, or error if dim is invalid
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::unsqueeze_at;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![3, 224, 224]);
/// let unsqueezed = unsqueeze_at(&shape, 0).unwrap();
/// assert_eq!(unsqueezed.dims(), &[1, 3, 224, 224]);
///
/// let unsqueezed_end = unsqueeze_at(&shape, 3).unwrap();
/// assert_eq!(unsqueezed_end.dims(), &[3, 224, 224, 1]);
/// ```
pub fn unsqueeze_at(shape: &Shape, dim: usize) -> Result<Shape> {
    let dims = shape.dims();

    if dim > dims.len() {
        return Err(TorshError::InvalidShape(format!(
            "dim {} is out of bounds for unsqueeze (max: {})",
            dim,
            dims.len()
        )));
    }

    let mut new_dims = Vec::with_capacity(dims.len() + 1);
    new_dims.extend_from_slice(&dims[..dim]);
    new_dims.push(1);
    new_dims.extend_from_slice(&dims[dim..]);

    Ok(Shape::new(new_dims))
}

/// Squeeze (remove) dimensions of size 1
///
/// # Arguments
///
/// * `shape` - Original shape
/// * `dim` - Optional specific dimension to squeeze (None means squeeze all size-1 dims)
///
/// # Returns
///
/// New shape with size-1 dimensions removed
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::squeeze;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![1, 3, 1, 224, 224]);
/// let squeezed = squeeze(&shape, None).unwrap();
/// assert_eq!(squeezed.dims(), &[3, 224, 224]);
///
/// let shape2 = Shape::new(vec![1, 3, 1, 224]);
/// let squeezed_dim = squeeze(&shape2, Some(2)).unwrap();
/// assert_eq!(squeezed_dim.dims(), &[1, 3, 224]);
/// ```
pub fn squeeze(shape: &Shape, dim: Option<usize>) -> Result<Shape> {
    let dims = shape.dims();

    if let Some(d) = dim {
        // Squeeze specific dimension
        if d >= dims.len() {
            return Err(TorshError::InvalidShape(format!(
                "dim {} is out of bounds for shape with {} dimensions",
                d,
                dims.len()
            )));
        }

        if dims[d] != 1 {
            return Err(TorshError::InvalidShape(format!(
                "Cannot squeeze dimension {} with size {}",
                d, dims[d]
            )));
        }

        let mut new_dims = Vec::with_capacity(dims.len() - 1);
        new_dims.extend_from_slice(&dims[..d]);
        new_dims.extend_from_slice(&dims[d + 1..]);

        Ok(Shape::new(new_dims))
    } else {
        // Squeeze all dimensions of size 1
        let new_dims: Vec<usize> = dims.iter().copied().filter(|&d| d != 1).collect();
        Ok(Shape::new(new_dims))
    }
}

/// Expand shape by adding size-1 dimensions to match target rank
///
/// # Arguments
///
/// * `shape` - Original shape
/// * `target_rank` - Desired number of dimensions
///
/// # Returns
///
/// New shape with leading size-1 dimensions added
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::expand_to_rank;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![224, 224]);
/// let expanded = expand_to_rank(&shape, 4).unwrap();
/// assert_eq!(expanded.dims(), &[1, 1, 224, 224]);
/// ```
pub fn expand_to_rank(shape: &Shape, target_rank: usize) -> Result<Shape> {
    let dims = shape.dims();

    if dims.len() > target_rank {
        return Err(TorshError::InvalidShape(format!(
            "Shape rank {} is already greater than target rank {}",
            dims.len(),
            target_rank
        )));
    }

    if dims.len() == target_rank {
        return Ok(shape.clone());
    }

    let num_prepend = target_rank - dims.len();
    let mut new_dims = vec![1; num_prepend];
    new_dims.extend_from_slice(dims);

    Ok(Shape::new(new_dims))
}

/// Permute (transpose) dimensions according to a permutation
///
/// # Arguments
///
/// * `shape` - Original shape
/// * `permutation` - New order of dimensions
///
/// # Returns
///
/// Permuted shape, or error if permutation is invalid
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::permute;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![32, 3, 224, 224]);
/// // NCHW to NHWC
/// let permuted = permute(&shape, &[0, 2, 3, 1]).unwrap();
/// assert_eq!(permuted.dims(), &[32, 224, 224, 3]);
/// ```
pub fn permute(shape: &Shape, permutation: &[usize]) -> Result<Shape> {
    let dims = shape.dims();

    if permutation.len() != dims.len() {
        return Err(TorshError::InvalidShape(format!(
            "Permutation length {} doesn't match shape rank {}",
            permutation.len(),
            dims.len()
        )));
    }

    // Validate permutation is valid (all indices present exactly once)
    let mut seen = vec![false; dims.len()];
    for &idx in permutation {
        if idx >= dims.len() {
            return Err(TorshError::InvalidShape(format!(
                "Permutation index {} is out of bounds for shape with {} dimensions",
                idx,
                dims.len()
            )));
        }
        if seen[idx] {
            return Err(TorshError::InvalidShape(format!(
                "Permutation index {} appears multiple times",
                idx
            )));
        }
        seen[idx] = true;
    }

    let new_dims: Vec<usize> = permutation.iter().map(|&i| dims[i]).collect();
    Ok(Shape::new(new_dims))
}

/// Check if two shapes are compatible for element-wise operations
///
/// Shapes are compatible if they are identical or can be broadcast together.
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::are_compatible;
/// use torsh_core::Shape;
///
/// let s1 = Shape::new(vec![32, 3, 224, 224]);
/// let s2 = Shape::new(vec![32, 3, 224, 224]);
/// assert!(are_compatible(&s1, &s2));
///
/// let s3 = Shape::new(vec![1, 3, 1, 1]);
/// assert!(are_compatible(&s1, &s3));  // Can broadcast
/// ```
pub fn are_compatible(shape1: &Shape, shape2: &Shape) -> bool {
    if shape1.dims() == shape2.dims() {
        return true;
    }

    // Check if broadcastable
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();
    let max_rank = dims1.len().max(dims2.len());

    for i in 0..max_rank {
        let dim1 = dims1
            .get(dims1.len().saturating_sub(max_rank - i))
            .copied()
            .unwrap_or(1);
        let dim2 = dims2
            .get(dims2.len().saturating_sub(max_rank - i))
            .copied()
            .unwrap_or(1);

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Calculate the number of elements in a shape
///
/// # Examples
///
/// ```rust
/// use torsh_core::shape_utils::numel;
/// use torsh_core::Shape;
///
/// let shape = Shape::new(vec![32, 3, 224, 224]);
/// assert_eq!(numel(&shape), 4816896);
/// ```
pub fn numel(shape: &Shape) -> usize {
    shape.numel()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_shape() {
        let s = scalar_shape();
        let empty: &[usize] = &[];
        assert_eq!(s.dims(), empty);
        assert!(s.is_scalar());
    }

    #[test]
    fn test_vector_shape() {
        let s = vector_shape(128);
        assert_eq!(s.dims(), &[128]);
    }

    #[test]
    fn test_matrix_shape() {
        let s = matrix_shape(64, 128);
        assert_eq!(s.dims(), &[64, 128]);
    }

    #[test]
    fn test_image_shape() {
        let rgb = image_shape(224, 224, 3);
        assert_eq!(rgb.dims(), &[224, 224, 3]);

        let grayscale = image_shape(28, 28, 1);
        assert_eq!(grayscale.dims(), &[28, 28, 1]);
    }

    #[test]
    fn test_batch_shape() {
        let img = image_shape(224, 224, 3);
        let batch = batch_shape(32, &img);
        assert_eq!(batch.dims(), &[32, 224, 224, 3]);
    }

    #[test]
    fn test_sequence_shape() {
        let seq = sequence_shape(100, 512);
        assert_eq!(seq.dims(), &[100, 512]);
    }

    #[test]
    fn test_flatten_from() {
        let shape = Shape::new(vec![32, 3, 224, 224]);
        let flattened = flatten_from(&shape, 1).unwrap();
        assert_eq!(flattened.dims(), &[32, 150528]);

        let flattened_all = flatten_from(&shape, 0).unwrap();
        assert_eq!(flattened_all.dims(), &[4816896]);
    }

    #[test]
    fn test_unsqueeze_at() {
        let shape = Shape::new(vec![3, 224, 224]);
        let unsqueezed = unsqueeze_at(&shape, 0).unwrap();
        assert_eq!(unsqueezed.dims(), &[1, 3, 224, 224]);

        let unsqueezed_end = unsqueeze_at(&shape, 3).unwrap();
        assert_eq!(unsqueezed_end.dims(), &[3, 224, 224, 1]);

        // Test error case
        assert!(unsqueeze_at(&shape, 10).is_err());
    }

    #[test]
    fn test_squeeze() {
        let shape = Shape::new(vec![1, 3, 1, 224, 224]);
        let squeezed = squeeze(&shape, None).unwrap();
        assert_eq!(squeezed.dims(), &[3, 224, 224]);

        let shape2 = Shape::new(vec![1, 3, 1, 224]);
        let squeezed_dim = squeeze(&shape2, Some(2)).unwrap();
        assert_eq!(squeezed_dim.dims(), &[1, 3, 224]);

        // Test error case - squeezing non-1 dimension
        assert!(squeeze(&shape2, Some(1)).is_err());
    }

    #[test]
    fn test_expand_to_rank() {
        let shape = Shape::new(vec![224, 224]);
        let expanded = expand_to_rank(&shape, 4).unwrap();
        assert_eq!(expanded.dims(), &[1, 1, 224, 224]);

        // Already at target rank
        let same = expand_to_rank(&shape, 2).unwrap();
        assert_eq!(same.dims(), &[224, 224]);

        // Error case - already higher rank
        assert!(expand_to_rank(&shape, 1).is_err());
    }

    #[test]
    fn test_permute() {
        let shape = Shape::new(vec![32, 3, 224, 224]);
        // NCHW to NHWC
        let permuted = permute(&shape, &[0, 2, 3, 1]).unwrap();
        assert_eq!(permuted.dims(), &[32, 224, 224, 3]);

        // Error cases
        assert!(permute(&shape, &[0, 1]).is_err()); // Wrong length
        assert!(permute(&shape, &[0, 1, 2, 10]).is_err()); // Out of bounds
        assert!(permute(&shape, &[0, 1, 1, 2]).is_err()); // Duplicate index
    }

    #[test]
    fn test_are_compatible() {
        let s1 = Shape::new(vec![32, 3, 224, 224]);
        let s2 = Shape::new(vec![32, 3, 224, 224]);
        assert!(are_compatible(&s1, &s2));

        let s3 = Shape::new(vec![1, 3, 1, 1]);
        assert!(are_compatible(&s1, &s3));

        let s4 = Shape::new(vec![32, 5, 224, 224]);
        assert!(!are_compatible(&s1, &s4));
    }

    #[test]
    fn test_numel() {
        let shape = Shape::new(vec![32, 3, 224, 224]);
        assert_eq!(numel(&shape), 4816896);

        let scalar = scalar_shape();
        assert_eq!(numel(&scalar), 1);
    }
}
