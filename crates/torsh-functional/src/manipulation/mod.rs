//! Tensor manipulation operations module
//!
//! This module provides comprehensive tensor manipulation functionality organized into
//! logical categories for better maintainability and discoverability.
//!
//! # Module Organization
//!
//! - [`shape`]: Shape manipulation functions (atleast_1d, atleast_2d, atleast_3d)
//! - [`construction`]: Tensor construction operations (block_diag, cartesian_prod, meshgrid)
//! - [`splitting`]: Tensor splitting operations (split, chunk, tensor_split, hsplit, vsplit, dsplit)
//! - [`contraction`]: Tensor contraction operations (tensordot)
//! - [`indexing`]: Index manipulation utilities (unravel_index, ravel_multi_index)
//!
//! # Mathematical Foundation
//!
//! Tensor manipulation operations form the backbone of scientific computing and machine learning.
//! These operations enable:
//!
//! ## Shape Transformations
//! Shape manipulation ensures tensors have compatible dimensions for operations:
//! - **Broadcasting compatibility**: Prepare tensors for element-wise operations
//! - **Dimension alignment**: Ensure consistent tensor ranks for algorithms
//! - **API standardization**: Convert between different tensor conventions
//!
//! ## Tensor Construction
//! Advanced construction operations for creating structured tensors:
//! - **Block diagonal matrices**: Essential for multi-task learning and ensemble methods
//! - **Cartesian products**: Fundamental for grid generation and combinatorial operations
//! - **Coordinate grids**: Critical for numerical methods and interpolation
//!
//! ## Tensor Decomposition
//! Splitting operations partition tensors for distributed computing and analysis:
//! - **Data parallelism**: Distribute tensors across multiple workers
//! - **Memory management**: Process large tensors in manageable chunks
//! - **Model parallelism**: Split neural network layers across devices
//!
//! ## Multilinear Algebra
//! Contraction operations generalize matrix multiplication to higher dimensions:
//! - **Einstein notation**: Flexible specification of tensor contractions
//! - **Dimensionality reduction**: Contract specific modes of multi-way arrays
//! - **Feature interactions**: Compute tensor products for representation learning
//!
//! ## Index Transformations
//! Convert between different indexing schemes for advanced operations:
//! - **Sparse operations**: Convert between coordinate and linear indexing
//! - **Memory layouts**: Transform between row-major and column-major orderings
//! - **Advanced indexing**: Implement fancy indexing and selection operations
//!
//! # Performance Considerations
//!
//! - **Memory locality**: Operations preserve cache-friendly access patterns
//! - **Vectorization**: Leverage SIMD instructions where applicable
//! - **Zero-copy views**: Use tensor views to avoid unnecessary data copying
//! - **Batch operations**: Process multiple tensors efficiently
//!
//! # Examples
//!
//! ```rust
//! use torsh_functional::manipulation::{
//!     atleast_2d, block_diag, split, tensordot, unravel_index,
//!     SplitArg, TensorDotAxes
//! };
//! use torsh_tensor::creation::{ones, zeros};
//!
//! // Shape manipulation
//! let vector = ones(&[5])?;
//! let matrix = atleast_2d(&vector)?; // Convert to [5, 1]
//!
//! // Block diagonal construction
//! let a = ones(&[2, 2])?;
//! let b = ones(&[3, 3])?;
//! let block_matrix = block_diag(&[a, b])?; // [5, 5] block diagonal
//!
//! // Tensor splitting
//! let tensor = ones(&[12, 4])?;
//! let chunks = split(&tensor, SplitArg::Size(3), 0)?; // Split into chunks of size 3
//!
//! // Tensor contraction
//! let a = ones(&[3, 4])?;
//! let b = ones(&[4, 5])?;
//! let result = tensordot(&a, &b, TensorDotAxes::Int(1))?; // Matrix multiplication
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod construction;
pub mod contraction;
pub mod indexing;
pub mod shape;
pub mod splitting;

// Re-export shape manipulation functions
pub use shape::{atleast_1d, atleast_2d, atleast_3d};

// Re-export construction functions
pub use construction::{block_diag, cartesian_prod, meshgrid};

// Re-export splitting functions and types
pub use splitting::{chunk, dsplit, hsplit, split, tensor_split, vsplit, SplitArg, TensorSplitArg};

// Re-export contraction functions and types
pub use contraction::{tensordot, TensorDotAxes};

// Re-export indexing functions
pub use indexing::{compute_strides, ravel_multi_index, unravel_index};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;
    use torsh_core::DeviceType;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_module_integration_shape_construction() -> torsh_core::Result<()> {
        // Test integration between shape and construction operations
        let vector = ones(&[3])?;
        let matrix = atleast_2d(&vector)?;

        // Use in block diagonal construction
        let block = block_diag(&[matrix.clone(), matrix])?;
        assert_eq!(block.shape().dims(), &[6, 2]); // [3,1] + [3,1] -> [6,2]

        Ok(())
    }

    #[test]
    fn test_module_integration_split_contraction() -> torsh_core::Result<()> {
        // Test integration between splitting and contraction
        let large_matrix = randn(&[8, 6], None, None, None)?;

        // Split into smaller matrices
        let splits = split(&large_matrix, SplitArg::Sections(2), 0)?;
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape().dims(), &[4, 6]);

        // Use contraction on split results
        let other_matrix = randn(&[6, 4], None, None, None)?;
        let result = tensordot(&splits[0], &other_matrix, TensorDotAxes::Int(1))?;
        assert_eq!(result.shape().dims(), &[4, 4]);

        Ok(())
    }

    #[test]
    fn test_module_integration_indexing_construction() -> torsh_core::Result<()> {
        // Test integration between indexing and construction
        let indices =
            torsh_tensor::Tensor::from_data(vec![0.0f32, 1.0, 2.0, 3.0], vec![4], DeviceType::Cpu)?;
        let shape = vec![2, 2];

        // Unravel indices to coordinates
        let coords = unravel_index(&indices, &shape)?;
        assert_eq!(coords.len(), 2);

        // Use coordinates in mesh grid construction
        let grid = meshgrid(&coords, "ij")?;
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0].shape().dims(), &[4, 4]);

        Ok(())
    }

    #[test]
    fn test_module_integration_comprehensive_workflow() -> torsh_core::Result<()> {
        // Comprehensive test combining multiple modules

        // 1. Create base tensors with shape manipulation
        let vector1 = ones(&[4])?;
        let vector2 = ones(&[4])?;
        let matrix1 = atleast_2d(&vector1)?;
        let matrix2 = atleast_2d(&vector2)?;

        // 2. Construct block diagonal matrix
        let block_matrix = block_diag(&[matrix1, matrix2])?;
        assert_eq!(block_matrix.shape().dims(), &[8, 2]);

        // 3. Split the block matrix
        let splits = hsplit(&block_matrix, TensorSplitArg::Sections(2))?;
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].shape().dims(), &[8, 1]);

        // 4. Create coordinate grid for indexing
        let x = torsh_tensor::Tensor::from_data(vec![0.0, 1.0], vec![2], DeviceType::Cpu)?;
        let y = torsh_tensor::Tensor::from_data(vec![0.0, 1.0], vec![2], DeviceType::Cpu)?;
        let grids = meshgrid(&[x, y], "ij")?;
        assert_eq!(grids[0].shape().dims(), &[2, 2]);

        // 5. Use tensor contraction for final computation
        let weight_matrix = randn(&[1, 4], None, None, None)?;
        let result = tensordot(
            &splits[0],
            &weight_matrix,
            TensorDotAxes::Explicit(vec![1], vec![0]),
        )?;
        assert_eq!(result.shape().dims(), &[8, 4]);

        Ok(())
    }

    #[test]
    fn test_backward_compatibility() -> torsh_core::Result<()> {
        // Test that all original functions are still accessible through re-exports

        // Shape functions
        let tensor = ones(&[5])?;
        let _result1 = atleast_1d(&tensor)?;
        let _result2 = atleast_2d(&tensor)?;
        let _result3 = atleast_3d(&tensor)?;

        // Construction functions
        let matrices = vec![ones(&[2, 2])?, ones(&[3, 3])?];
        let _block = block_diag(&matrices)?;

        let tensors = vec![
            torsh_tensor::Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?,
            torsh_tensor::Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu)?,
        ];
        let _cart = cartesian_prod(&tensors)?;
        let _mesh = meshgrid(&tensors, "xy")?;

        // Splitting functions
        let tensor = ones(&[6, 4])?;
        let _splits1 = split(&tensor, SplitArg::Sections(2), 0)?;
        let _splits2 = chunk(&tensor, 3, 0)?;
        let _splits3 = tensor_split(&tensor, TensorSplitArg::Sections(2), 0)?;
        let _splits4 = hsplit(&tensor, TensorSplitArg::Sections(2))?;
        let _splits5 = vsplit(&tensor, TensorSplitArg::Sections(2))?;

        let tensor_3d = ones(&[2, 3, 6])?;
        let _splits6 = dsplit(&tensor_3d, TensorSplitArg::Sections(2))?;

        // Contraction functions
        let a = ones(&[3, 4])?;
        let b = ones(&[4, 5])?;
        let _result = tensordot(&a, &b, TensorDotAxes::Int(1))?;

        // Indexing functions
        let indices =
            torsh_tensor::Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3], DeviceType::Cpu)?;
        let shape = vec![2, 2];
        let _coords = unravel_index(&indices, &shape)?;
        let _strides = compute_strides(&shape);

        Ok(())
    }

    #[test]
    fn test_error_propagation() -> torsh_core::Result<()> {
        // Test that errors are properly propagated through the module structure

        // Invalid dimension for hsplit
        let tensor_1d = ones(&[5])?;
        assert!(hsplit(&tensor_1d, TensorSplitArg::Sections(2)).is_err());

        // Invalid axes for tensordot
        let a = ones(&[3, 4])?;
        let b = ones(&[5, 6])?;
        assert!(tensordot(&a, &b, TensorDotAxes::Int(1)).is_err());

        // Invalid indices for unravel_index
        let indices = torsh_tensor::Tensor::from_data(vec![10.0], vec![1], DeviceType::Cpu)?;
        let shape = vec![2, 2]; // Max valid index is 3
        assert!(unravel_index(&indices, &shape).is_err());

        Ok(())
    }

    #[test]
    fn test_performance_patterns() -> torsh_core::Result<()> {
        // Test patterns that should be efficient

        // Large tensor operations
        let large_tensor = randn(&[1000, 500], None, None, None)?;

        // Chunking should be efficient
        let chunks = chunk(&large_tensor, 10, 0)?;
        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].shape().dims(), &[100, 500]);

        // Block diagonal with multiple matrices
        let matrices: Vec<_> = (0..5)
            .map(|_| ones(&[100, 100]))
            .collect::<Result<Vec<_>, _>>()?;
        let block = block_diag(&matrices)?;
        assert_eq!(block.shape().dims(), &[500, 500]);

        // Tensor contraction on medium-sized tensors
        let a = randn(&[50, 100], None, None, None)?;
        let b = randn(&[100, 75], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Int(1))?;
        assert_eq!(result.shape().dims(), &[50, 75]);

        Ok(())
    }
}
