//! Tensor indexing operations
//!
//! This module provides utilities for converting between different indexing formats
//! and manipulating tensor indices. These operations are fundamental for advanced
//! indexing, coordinate transformations, and implementing custom tensor operations
//! that require explicit index manipulation.

use torsh_core::Result as TorshResult;
use torsh_tensor::{creation::zeros, Tensor};

/// Convert flat index to multi-dimensional index
///
/// ## Mathematical Background
///
/// For a tensor with shape [d₁, d₂, ..., dₙ], converts flat (linear) indices to
/// multi-dimensional coordinates. Given a flat index k, computes coordinates
/// (i₁, i₂, ..., iₙ) such that:
///
/// ```text
/// k = i₁ × (d₂ × d₃ × ... × dₙ) + i₂ × (d₃ × ... × dₙ) + ... + iₙ
/// ```text
///
/// ## Algorithm: Row-Major Index Unraveling
///
/// The conversion uses row-major (C-style) ordering with strides:
/// ```text
/// stride[i] = ∏ⱼ₌ᵢ₊₁ⁿ dⱼ  (product of dimensions after i)
/// stride[n-1] = 1           (last dimension has stride 1)
/// ```text
///
/// For each dimension i:
/// ```text
/// coordinate[i] = (k mod stride[i-1]) / stride[i]
/// ```text
///
/// ## Inverse Operation
/// The inverse operation (ravel_multi_index) would be:
/// ```text
/// flat_index = Σᵢ coordinate[i] × stride[i]
/// ```text
///
/// ## Memory Layout Considerations
///
/// This function assumes row-major (C-style) memory layout:
/// - **Row-major**: rightmost index changes fastest
/// - **Column-major** (Fortran-style): leftmost index changes fastest
///
/// The choice affects stride computation and index ordering.
///
/// ## Parameters
/// * `indices` - 1D tensor containing flat indices to convert
/// * `shape` - Target tensor shape for multi-dimensional coordinates
///
/// ## Returns
/// * Vector of tensors, each containing coordinates for one dimension
///
/// ## Applications
/// - **Sparse tensor operations**: Convert between storage formats
/// - **Image processing**: Convert pixel indices to (row, col) coordinates
/// - **Advanced indexing**: Implement fancy indexing operations
/// - **Memory access patterns**: Analyze and optimize data access
/// - **Coordinate transformations**: Convert between index systems
/// - **Debugging**: Visualize tensor access patterns
///
/// ## Computational Complexity
/// - Time: O(m × n) where m = number of indices, n = number of dimensions
/// - Space: O(m × n) for output coordinates
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::unravel_index;
/// # use torsh_tensor::{Tensor, creation::tensor};
/// # use torsh_core::DeviceType;
/// // Convert flat indices to 2D coordinates
/// let indices = Tensor::from_data(vec![0.0, 1.0, 2.0, 3.0], vec![4], DeviceType::Cpu)?;
/// let shape = vec![2, 2];
/// let coords = unravel_index(&indices, &shape)?;
///
/// // coords[0] = [0, 0, 1, 1] (row coordinates)
/// // coords[1] = [0, 1, 0, 1] (column coordinates)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
///
/// ## Advanced Example: 3D Volume Indexing
/// ```rust
/// # use torsh_functional::manipulation::unravel_index;
/// # use torsh_tensor::{Tensor, creation::tensor};
/// # use torsh_core::DeviceType;
/// // Convert linear indices to 3D volume coordinates
/// let flat_indices = Tensor::from_data(vec![0.0, 7.0, 15.0, 23.0], vec![4], DeviceType::Cpu)?;
/// let volume_shape = vec![3, 4, 2]; // depth × height × width
/// let coords = unravel_index(&flat_indices, &volume_shape)?;
///
/// // For index 7 in shape [3,4,2]:
/// // 7 = 0×(4×2) + 3×2 + 1 → coordinates (0, 3, 1)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
pub fn unravel_index(indices: &Tensor, shape: &[usize]) -> TorshResult<Vec<Tensor>> {
    // Validate inputs
    let indices_shape = indices.shape();
    if indices_shape.ndim() != 1 {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Indices must be 1-dimensional",
            "unravel_index",
        ));
    }

    if shape.is_empty() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Shape cannot be empty",
            "unravel_index",
        ));
    }

    // Validate that indices are within bounds
    let total_size: usize = shape.iter().product();
    let indices_data = indices.to_vec()?;
    for &idx in &indices_data {
        if idx < 0.0 || idx as usize >= total_size {
            return Err(torsh_core::TorshError::invalid_argument_with_context(
                &format!("Index {} out of bounds for total size {}", idx, total_size),
                "unravel_index",
            ));
        }
    }

    // Calculate strides for row-major ordering
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Unravel each index
    let num_indices = indices_shape.dims()[0];
    let mut results = Vec::with_capacity(shape.len());

    // Create result tensors for each dimension
    for _ in 0..shape.len() {
        results.push(zeros(&[num_indices])?);
    }

    // Process each flat index
    for i in 0..num_indices {
        let flat_idx = indices.get(&[i])? as usize;
        let mut remaining = flat_idx;

        // Convert to multi-dimensional coordinates
        for (j, &stride) in strides.iter().enumerate() {
            let coord = remaining / stride;
            results[j].set(&[i], coord as f32)?;
            remaining %= stride;
        }
    }

    Ok(results)
}

/// Compute multi-dimensional strides for a given shape
///
/// ## Mathematical Background
///
/// Computes the strides (step sizes) for each dimension in row-major order.
/// For shape [d₁, d₂, ..., dₙ], the stride for dimension i is:
///
/// ```text
/// stride[i] = ∏ⱼ₌ᵢ₊₁ⁿ dⱼ
/// ```text
///
/// This represents how many elements to skip to move by one unit in dimension i.
///
/// ## Parameters
/// * `shape` - Tensor shape dimensions
///
/// ## Returns
/// * Vector of stride values for each dimension
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::compute_strides;
/// let shape = vec![3, 4, 2];
/// let strides = compute_strides(&shape);
/// // Returns [8, 2, 1] because:
/// // - Moving in dim 0: skip 4×2 = 8 elements
/// // - Moving in dim 1: skip 2 elements
/// // - Moving in dim 2: skip 1 element
/// ```text
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert multi-dimensional coordinates to flat indices
///
/// ## Mathematical Background
///
/// The inverse operation of `unravel_index`. Converts multi-dimensional
/// coordinates to flat (linear) indices using row-major ordering:
///
/// ```text
/// flat_index = Σᵢ coordinate[i] × stride[i]
/// ```text
///
/// Where stride[i] = ∏ⱼ₌ᵢ₊₁ⁿ shape[j]
///
/// ## Parameters
/// * `coords` - Vector of tensors containing coordinates for each dimension
/// * `shape` - Tensor shape for bounds checking and stride computation
///
/// ## Returns
/// * 1D tensor containing flat indices
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::ravel_multi_index;
/// # use torsh_tensor::{Tensor, creation::tensor};
/// # use torsh_core::DeviceType;
/// let row_coords = Tensor::from_data(vec![0.0, 0.0, 1.0, 1.0], vec![4], DeviceType::Cpu)?;
/// let col_coords = Tensor::from_data(vec![0.0, 1.0, 0.0, 1.0], vec![4], DeviceType::Cpu)?;
/// let coords = vec![row_coords, col_coords];
/// let shape = vec![2, 2];
///
/// let flat_indices = ravel_multi_index(&coords, &shape)?;
/// // Returns [0, 1, 2, 3]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
pub fn ravel_multi_index(coords: &[Tensor], shape: &[usize]) -> TorshResult<Tensor> {
    if coords.len() != shape.len() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Number of coordinate tensors must match number of dimensions",
            "ravel_multi_index",
        ));
    }

    if coords.is_empty() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Coordinate tensors cannot be empty",
            "ravel_multi_index",
        ));
    }

    // Validate all coordinate tensors have the same shape
    let coord_shape = coords[0].shape();
    for (i, coord) in coords.iter().enumerate() {
        if coord.shape().dims() != coord_shape.dims() {
            return Err(torsh_core::TorshError::invalid_argument_with_context(
                &format!(
                    "All coordinate tensors must have the same shape, but coordinate {} differs",
                    i
                ),
                "ravel_multi_index",
            ));
        }
        if coord.shape().ndim() != 1 {
            return Err(torsh_core::TorshError::invalid_argument_with_context(
                "All coordinate tensors must be 1-dimensional",
                "ravel_multi_index",
            ));
        }
    }

    let num_indices = coord_shape.dims()[0];
    let strides = compute_strides(shape);

    // Create result tensor
    let result = zeros(&[num_indices])?;

    // Convert each set of coordinates to flat index
    for i in 0..num_indices {
        let mut flat_idx = 0usize;

        for (dim, coord_tensor) in coords.iter().enumerate() {
            let coord = coord_tensor.get(&[i])? as usize;

            // Bounds checking
            if coord >= shape[dim] {
                return Err(torsh_core::TorshError::invalid_argument_with_context(
                    &format!(
                        "Coordinate {} out of bounds for dimension {} with size {}",
                        coord, dim, shape[dim]
                    ),
                    "ravel_multi_index",
                ));
            }

            flat_idx += coord * strides[dim];
        }

        result.set(&[i], flat_idx as f32)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;

    #[test]
    fn test_unravel_index_2d() -> TorshResult<()> {
        // Test unraveling indices for a 2D shape
        let indices = Tensor::from_data(vec![0.0f32, 1.0, 2.0, 3.0], vec![4], DeviceType::Cpu)?;
        let shape = vec![2, 2];
        let result = unravel_index(&indices, &shape)?;

        assert_eq!(result.len(), 2); // Should return coordinates for each dimension
        assert_eq!(result[0].shape().dims(), &[4]); // Row indices
        assert_eq!(result[1].shape().dims(), &[4]); // Column indices

        // Check specific values
        // Index 0 -> (0, 0)
        assert_eq!(result[0].get(&[0])?, 0.0);
        assert_eq!(result[1].get(&[0])?, 0.0);

        // Index 1 -> (0, 1)
        assert_eq!(result[0].get(&[1])?, 0.0);
        assert_eq!(result[1].get(&[1])?, 1.0);

        // Index 2 -> (1, 0)
        assert_eq!(result[0].get(&[2])?, 1.0);
        assert_eq!(result[1].get(&[2])?, 0.0);

        // Index 3 -> (1, 1)
        assert_eq!(result[0].get(&[3])?, 1.0);
        assert_eq!(result[1].get(&[3])?, 1.0);

        Ok(())
    }

    #[test]
    fn test_unravel_index_3d() -> TorshResult<()> {
        // Test with 3D shape
        let indices = Tensor::from_data(vec![0.0f32, 7.0, 15.0, 23.0], vec![4], DeviceType::Cpu)?;
        let shape = vec![3, 4, 2]; // Total size = 24
        let result = unravel_index(&indices, &shape)?;

        assert_eq!(result.len(), 3);

        // Index 7 -> (0, 3, 1) because 7 = 0×8 + 3×2 + 1×1
        assert_eq!(result[0].get(&[1])?, 0.0); // dim 0
        assert_eq!(result[1].get(&[1])?, 3.0); // dim 1
        assert_eq!(result[2].get(&[1])?, 1.0); // dim 2

        Ok(())
    }

    #[test]
    fn test_compute_strides() {
        let shape = vec![3, 4, 2];
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![8, 2, 1]);

        let shape = vec![5];
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![1]);

        let empty_shape: Vec<usize> = vec![];
        let strides = compute_strides(&empty_shape);
        assert_eq!(strides, Vec::<usize>::new());
    }

    #[test]
    fn test_ravel_multi_index() -> TorshResult<()> {
        // Test the inverse operation
        let row_coords = Tensor::from_data(vec![0.0, 0.0, 1.0, 1.0], vec![4], DeviceType::Cpu)?;
        let col_coords = Tensor::from_data(vec![0.0, 1.0, 0.0, 1.0], vec![4], DeviceType::Cpu)?;
        let coords = vec![row_coords, col_coords];
        let shape = vec![2, 2];

        let flat_indices = ravel_multi_index(&coords, &shape)?;

        // Should get back [0, 1, 2, 3]
        assert_eq!(flat_indices.get(&[0])?, 0.0);
        assert_eq!(flat_indices.get(&[1])?, 1.0);
        assert_eq!(flat_indices.get(&[2])?, 2.0);
        assert_eq!(flat_indices.get(&[3])?, 3.0);

        Ok(())
    }

    #[test]
    fn test_unravel_ravel_roundtrip() -> TorshResult<()> {
        // Test that unravel_index and ravel_multi_index are inverses
        let original_indices =
            Tensor::from_data(vec![0.0, 5.0, 10.0, 15.0], vec![4], DeviceType::Cpu)?;
        let shape = vec![4, 4];

        // Unravel then ravel
        let coords = unravel_index(&original_indices, &shape)?;
        let reconstructed = ravel_multi_index(&coords, &shape)?;

        // Should get back the original indices
        for i in 0..4 {
            assert_eq!(original_indices.get(&[i])?, reconstructed.get(&[i])?);
        }

        Ok(())
    }

    #[test]
    fn test_unravel_index_error_cases() {
        // Test non-1D indices
        let indices_2d =
            Tensor::from_data(vec![0.0f32, 1.0, 2.0, 3.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let shape = vec![2, 2];
        assert!(unravel_index(&indices_2d, &shape).is_err());

        // Test out of bounds index
        let indices = Tensor::from_data(vec![4.0f32], vec![1], DeviceType::Cpu).unwrap();
        let shape = vec![2, 2]; // Max valid index is 3
        assert!(unravel_index(&indices, &shape).is_err());

        // Test empty shape
        let indices = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu).unwrap();
        let empty_shape: Vec<usize> = vec![];
        assert!(unravel_index(&indices, &empty_shape).is_err());
    }

    #[test]
    fn test_ravel_multi_index_error_cases() {
        // Test mismatched number of coordinates and dimensions
        let coord = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu).unwrap();
        let coords = vec![coord];
        let shape = vec![2, 2]; // 2 dimensions but only 1 coordinate tensor
        assert!(ravel_multi_index(&coords, &shape).is_err());

        // Test out of bounds coordinate
        let coord = Tensor::from_data(vec![2.0f32], vec![1], DeviceType::Cpu).unwrap();
        let coords = vec![coord];
        let shape = vec![2]; // Max valid coordinate is 1
        assert!(ravel_multi_index(&coords, &shape).is_err());
    }

    #[test]
    fn test_edge_case_1d_tensor() -> TorshResult<()> {
        // Test with 1D tensor (should work trivially)
        let indices = Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3], DeviceType::Cpu)?;
        let shape = vec![5];
        let result = unravel_index(&indices, &shape)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get(&[0])?, 0.0);
        assert_eq!(result[0].get(&[1])?, 1.0);
        assert_eq!(result[0].get(&[2])?, 2.0);

        Ok(())
    }

    #[test]
    fn test_large_tensor_indexing() -> TorshResult<()> {
        // Test with larger tensor to verify stride calculations
        let indices = Tensor::from_data(vec![0.0, 59.0, 35.0], vec![3], DeviceType::Cpu)?;
        let shape = vec![5, 4, 3]; // Total size = 60, valid indices: 0-59
        let result = unravel_index(&indices, &shape)?;

        // Index 59 = 4×12 + 11, and 11 = 3×3 + 2, so (4, 3, 2)
        // 59 / 12 = 4 remainder 11, 11 / 3 = 3 remainder 2, so (4, 3, 2)
        assert_eq!(result[0].get(&[1])?, 4.0); // dim 0
        assert_eq!(result[1].get(&[1])?, 3.0); // dim 1
        assert_eq!(result[2].get(&[1])?, 2.0); // dim 2

        // Index 35 = 2×12 + 11, and 11 = 3×3 + 2, so (2, 3, 2)
        assert_eq!(result[0].get(&[2])?, 2.0); // dim 0
        assert_eq!(result[1].get(&[2])?, 3.0); // dim 1
        assert_eq!(result[2].get(&[2])?, 2.0); // dim 2

        Ok(())
    }
}
