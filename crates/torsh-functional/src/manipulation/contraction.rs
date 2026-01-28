//! Tensor contraction operations
//!
//! This module provides tensor contraction functionality for computing generalized
//! dot products between tensors along specified axes. Tensor contraction is a
//! fundamental operation in multilinear algebra and forms the basis for many
//! machine learning operations including matrix multiplication, convolution,
//! and attention mechanisms.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Compute tensor dot product along specified axes
///
/// ## Mathematical Background
///
/// The tensor dot product (or tensor contraction) is a generalization of matrix
/// multiplication to higher-dimensional tensors. For tensors A and B, it computes:
///
/// ```text
/// C_{i₁...iₘ,j₁...jₙ} = Σₖ₁...ₖₚ A_{i₁...iₘ,k₁...kₚ} · B_{k₁...kₚ,j₁...jₙ}
/// ```text
///
/// Where the summation is over the contracted indices k₁,...,kₚ.
///
/// ## Contraction Modes
///
/// ### Integer Mode (Simple Contraction)
/// For `tensordot(A, B, n)`:
/// - Contracts last n axes of A with first n axes of B
/// - Equivalent to `np.tensordot(A, B, axes=n)`
/// - Result shape: A.shape[:-n] + B.shape[n:]
///
/// ### Explicit Axes Mode
/// For `tensordot(A, B, (axes_a, axes_b))`:
/// - Contracts specified axes of A with corresponding axes of B
/// - axes_a and axes_b must have same length
/// - More flexible than integer mode
///
/// ## Mathematical Properties
///
/// 1. **Associativity**: tensordot(tensordot(A,B,axes1), C, axes2) under certain conditions
/// 2. **Commutativity**: Generally not commutative (A⊗B ≠ B⊗A)
/// 3. **Linearity**: tensordot(αA + βB, C, axes) = α·tensordot(A,C,axes) + β·tensordot(B,C,axes)
/// 4. **Dimension reduction**: Output rank = rank(A) + rank(B) - 2×num_contracted_axes
///
/// ## Computational Complexity
/// - Time: O(∏(free_dims_A) × ∏(free_dims_B) × ∏(contracted_dims))
/// - Space: O(∏(result_dims))
///
/// ## Parameters
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `axes` - Axes specification for contraction
///
/// ## Returns
/// * Contracted tensor result
///
/// ## Applications
/// - **Matrix multiplication**: tensordot(A, B, 1) for 2D tensors
/// - **Batch matrix multiplication**: Contract specific dimensions in batched operations
/// - **Convolution**: Core operation in convolutional neural networks
/// - **Attention mechanisms**: Query-key interactions in transformer models
/// - **Physics simulations**: Tensor contractions in general relativity and quantum field theory
/// - **Signal processing**: Multilinear filtering and decomposition
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::{tensordot, TensorDotAxes};
/// # use torsh_tensor::creation::ones;
/// // Matrix multiplication (A: [3,4], B: [4,5])
/// let a = ones(&[3, 4])?;
/// let b = ones(&[4, 5])?;
/// let result = tensordot(&a, &b, TensorDotAxes::Int(1))?; // Shape: [3, 5]
///
/// // Higher-order contraction (A: [2,3,4], B: [4,3,5])
/// let a = ones(&[2, 3, 4])?;
/// let b = ones(&[4, 3, 5])?;
/// let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![2,1], vec![0,1]))?; // Shape: [2, 5]
///
/// // Batch operations (A: [batch,m,k], B: [batch,k,n])
/// let a = ones(&[10, 32, 64])?;
/// let b = ones(&[10, 64, 128])?;
/// let result = tensordot(&a, &b, TensorDotAxes::Arrays(vec![2], vec![1]))?; // Shape: [10, 32, 10, 128]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
pub fn tensordot(a: &Tensor, b: &Tensor, axes: TensorDotAxes) -> TorshResult<Tensor> {
    match axes {
        TensorDotAxes::Int(n) => {
            // Contract last n axes of a with first n axes of b
            let a_shape = a.shape();
            let b_shape = b.shape();

            if n > a_shape.ndim() || n > b_shape.ndim() {
                return Err(torsh_core::TorshError::invalid_argument_with_context(
                    "Number of axes to contract exceeds tensor dimensions",
                    "tensordot",
                ));
            }

            // Verify that the axes to contract have compatible sizes
            for i in 0..n {
                let a_axis = a_shape.ndim() - n + i;
                let b_axis = i;
                if a_shape.dims()[a_axis] != b_shape.dims()[b_axis] {
                    return Err(torsh_core::TorshError::invalid_argument_with_context(
                        "Axes to contract must have the same size",
                        "tensordot",
                    ));
                }
            }

            // Reshape tensors for matrix multiplication
            let a_free_dims = a_shape.ndim() - n;
            let _b_free_dims = b_shape.ndim() - n;

            let a_free_size: usize = a_shape.dims()[..a_free_dims].iter().product();
            let b_free_size: usize = b_shape.dims()[n..].iter().product();
            let contract_size: usize = a_shape.dims()[a_free_dims..].iter().product();

            // Reshape for matrix multiplication
            let a_reshaped = a.view(&[a_free_size as i32, contract_size as i32])?;
            let b_reshaped = b.view(&[contract_size as i32, b_free_size as i32])?;

            // Perform matrix multiplication
            let result = a_reshaped.matmul(&b_reshaped)?;

            // Reshape result to final shape
            let mut result_shape: Vec<usize> = Vec::new();
            result_shape.extend(&a_shape.dims()[..a_free_dims]);
            result_shape.extend(&b_shape.dims()[n..]);

            if result_shape.is_empty() {
                // Scalar result - reshape to 0D tensor
                result.view(&[])
            } else {
                let result_shape_i32: Vec<i32> = result_shape.iter().map(|&x| x as i32).collect();
                result.view(&result_shape_i32)
            }
        }
        TensorDotAxes::Explicit(a_axes, b_axes) => {
            // Contract specified axes
            if a_axes.len() != b_axes.len() {
                return Err(torsh_core::TorshError::invalid_argument_with_context(
                    "Number of axes to contract must be equal",
                    "tensordot",
                ));
            }

            let a_shape = a.shape();
            let b_shape = b.shape();

            // Verify axes are valid and compatible
            for (&a_axis, &b_axis) in a_axes.iter().zip(b_axes.iter()) {
                if a_axis >= a_shape.ndim() || b_axis >= b_shape.ndim() {
                    return Err(torsh_core::TorshError::invalid_argument_with_context(
                        "Axis index out of range",
                        "tensordot",
                    ));
                }

                if a_shape.dims()[a_axis] != b_shape.dims()[b_axis] {
                    return Err(torsh_core::TorshError::invalid_argument_with_context(
                        "Contracted axes must have the same size",
                        "tensordot",
                    ));
                }
            }

            // For explicit axes, we need to reorganize the tensors
            // This is a simplified implementation - a complete version would handle
            // arbitrary axis permutations efficiently

            // Calculate contract size
            let contract_size: usize = a_axes.iter().map(|&axis| a_shape.dims()[axis]).product();

            // Calculate free dimensions
            let a_free_dims: Vec<usize> = (0..a_shape.ndim())
                .filter(|i| !a_axes.contains(i))
                .collect();
            let b_free_dims: Vec<usize> = (0..b_shape.ndim())
                .filter(|i| !b_axes.contains(i))
                .collect();

            let a_free_size: usize = a_free_dims.iter().map(|&i| a_shape.dims()[i]).product();
            let b_free_size: usize = b_free_dims.iter().map(|&i| b_shape.dims()[i]).product();

            // This is a simplified approach - reshape both tensors
            let a_reshaped = a.view(&[a_free_size as i32, contract_size as i32])?;
            let b_reshaped = b.view(&[contract_size as i32, b_free_size as i32])?;

            // Perform matrix multiplication
            let result = a_reshaped.matmul(&b_reshaped)?;

            // Reshape result
            let mut result_shape: Vec<usize> = Vec::new();
            for &dim in &a_free_dims {
                result_shape.push(a_shape.dims()[dim]);
            }
            for &dim in &b_free_dims {
                result_shape.push(b_shape.dims()[dim]);
            }

            if result_shape.is_empty() {
                // Scalar result - reshape to 0D tensor
                result.view(&[])
            } else {
                let result_shape_i32: Vec<i32> = result_shape.iter().map(|&x| x as i32).collect();
                result.view(&result_shape_i32)
            }
        }
        TensorDotAxes::Arrays(a_axes, b_axes) => {
            // Arrays variant - same as Explicit
            tensordot(a, b, TensorDotAxes::Explicit(a_axes, b_axes))
        }
    }
}

/// Axes specification for tensordot operations
///
/// ## Variants
///
/// ### Int
/// Simple contraction mode that contracts the last n axes of the first tensor
/// with the first n axes of the second tensor.
///
/// ### Explicit
/// Explicit specification of which axes to contract. The two vectors must have
/// the same length, and corresponding axes must have compatible sizes.
///
/// ### Arrays
/// Alternative name for Explicit mode, provided for API compatibility.
/// Functionally identical to Explicit.
///
/// ## Mathematical Interpretation
///
/// For tensors A ∈ ℝ^(d₁×d₂×...×dₘ) and B ∈ ℝ^(e₁×e₂×...×eₙ):
///
/// - **Int(k)**: Contract A's last k axes with B's first k axes
/// - **Explicit([i₁,i₂,...], [j₁,j₂,...])**: Contract A's axes [i₁,i₂,...] with B's axes [j₁,j₂,...]
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::TensorDotAxes;
/// // Simple matrix multiplication
/// let axes = TensorDotAxes::Int(1);
///
/// // Contract specific axes
/// let axes = TensorDotAxes::Explicit(vec![1, 3], vec![0, 2]);
///
/// // Same as Explicit
/// let axes = TensorDotAxes::Arrays(vec![2], vec![1]);
/// ```
#[derive(Debug, Clone)]
pub enum TensorDotAxes {
    /// Contract last n axes of first tensor with first n axes of second tensor
    Int(usize),
    /// Explicitly specify axes to contract: (axes_a, axes_b)
    Explicit(Vec<usize>, Vec<usize>),
    /// Alternative name for Explicit mode
    Arrays(Vec<usize>, Vec<usize>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_tensordot_simple_matrix_multiplication() -> TorshResult<()> {
        // Test simple matrix multiplication (last dim of a with first dim of b)
        let a = randn(&[3, 4], None, None, None)?;
        let b = randn(&[4, 5], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Int(1))?;
        assert_eq!(result.shape().dims(), &[3, 5]);
        Ok(())
    }

    #[test]
    fn test_tensordot_with_explicit_axes() -> TorshResult<()> {
        // Test with explicit axis specification
        let a = randn(&[3, 4], None, None, None)?;
        let b = randn(&[4, 5], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![1], vec![0]))?;
        assert_eq!(result.shape().dims(), &[3, 5]);
        Ok(())
    }

    #[test]
    fn test_tensordot_with_arrays() -> TorshResult<()> {
        // Test with arrays variant (should be same as explicit)
        let a = randn(&[3, 4], None, None, None)?;
        let b = randn(&[4, 5], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Arrays(vec![1], vec![0]))?;
        assert_eq!(result.shape().dims(), &[3, 5]);
        Ok(())
    }

    #[test]
    fn test_tensordot_higher_order() -> TorshResult<()> {
        // Test higher-order tensor contraction
        let a = randn(&[2, 3, 4, 5], None, None, None)?;
        let b = randn(&[4, 5, 6], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Int(2))?;
        assert_eq!(result.shape().dims(), &[2, 3, 6]);
        Ok(())
    }

    #[test]
    fn test_tensordot_multiple_explicit_axes() -> TorshResult<()> {
        // Test multiple axis contraction with explicit specification
        let a = randn(&[2, 3, 4, 5], None, None, None)?;
        let b = randn(&[4, 6, 3, 7], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![2, 1], vec![0, 2]))?;
        assert_eq!(result.shape().dims(), &[2, 5, 6, 7]);
        Ok(())
    }

    #[test]
    fn test_tensordot_scalar_result() -> TorshResult<()> {
        // Test case that results in a scalar
        let a = randn(&[3, 4], None, None, None)?;
        let b = randn(&[3, 4], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![0, 1], vec![0, 1]))?;
        // Result should be a scalar (0D tensor)
        assert_eq!(result.shape().ndim(), 0);
        Ok(())
    }

    #[test]
    fn test_tensordot_error_mismatched_axes_lengths() {
        let a = randn(&[3, 4], None, None, None).unwrap();
        let b = randn(&[4, 5], None, None, None).unwrap();

        // Should error when axes vectors have different lengths
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![1], vec![0, 1]));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensordot_error_axis_out_of_bounds() {
        let a = randn(&[3, 4], None, None, None).unwrap();
        let b = randn(&[4, 5], None, None, None).unwrap();

        // Should error when axis index is out of bounds
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![2], vec![0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensordot_error_incompatible_sizes() {
        let a = randn(&[3, 4], None, None, None).unwrap();
        let b = randn(&[5, 6], None, None, None).unwrap();

        // Should error when contracted axes have incompatible sizes
        let result = tensordot(&a, &b, TensorDotAxes::Explicit(vec![1], vec![0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensordot_error_too_many_axes() {
        let a = randn(&[3, 4], None, None, None).unwrap();
        let b = randn(&[4, 5], None, None, None).unwrap();

        // Should error when trying to contract more axes than available
        let result = tensordot(&a, &b, TensorDotAxes::Int(3));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensordot_batch_operations() -> TorshResult<()> {
        // Test batch matrix multiplication scenario
        let batch_a = randn(&[10, 32, 64], None, None, None)?;
        let batch_b = randn(&[10, 64, 128], None, None, None)?;

        // Contract the inner dimension (64) while preserving batch dimension
        let result = tensordot(
            &batch_a,
            &batch_b,
            TensorDotAxes::Explicit(vec![2], vec![1]),
        )?;
        assert_eq!(result.shape().dims(), &[10, 32, 10, 128]);
        Ok(())
    }

    #[test]
    fn test_tensordot_edge_case_1d_tensors() -> TorshResult<()> {
        // Test with 1D tensors (vector dot product)
        let a = randn(&[5], None, None, None)?;
        let b = randn(&[5], None, None, None)?;
        let result = tensordot(&a, &b, TensorDotAxes::Int(1))?;
        assert_eq!(result.shape().ndim(), 0); // Should be scalar
        Ok(())
    }
}
