//! Basic linear algebra operations
//!
//! This module provides fundamental linear algebra operations including matrix
//! multiplication chains, matrix norms, and batch operations.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use super::core::NormOrd;

/// Chain matrix multiplication for multiple matrices
///
/// Efficiently computes the product of multiple matrices: A₁ × A₂ × ... × Aₙ
///
/// ## Mathematical Background
///
/// Matrix multiplication is associative but not commutative:
/// - (AB)C = A(BC) ✓
/// - AB ≠ BA in general ✗
///
/// The optimal order of multiplication can significantly reduce computational cost.
/// For matrices of sizes p×q, q×r, r×s, the cost is:
/// - (A×B)×C: pqr + prs operations
/// - A×(B×C): qrs + pqs operations
///
/// ## Parameters
/// * `matrices` - Slice of 2D tensors to multiply in sequence
///
/// ## Returns
/// * The product tensor A₁ × A₂ × ... × Aₙ
///
/// ## Errors
/// * Returns error if any matrix is not 2D
/// * Returns error if adjacent matrices have incompatible dimensions
/// * Returns error if input slice is empty
///
/// ## Example
/// ```rust
/// # use torsh_functional::linalg::chain_matmul;
/// # use torsh_tensor::creation::randn;
/// let a = randn(&[3, 4])?;
/// let b = randn(&[4, 5])?;
/// let c = randn(&[5, 2])?;
/// let result = chain_matmul(&[a, b, c])?; // Shape: [3, 2]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn chain_matmul(matrices: &[Tensor]) -> TorshResult<Tensor> {
    if matrices.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "chain_matmul requires at least one matrix",
            "chain_matmul",
        ));
    }

    if matrices.len() == 1 {
        return Ok(matrices[0].clone());
    }

    // Validate all inputs are 2D
    for (i, mat) in matrices.iter().enumerate() {
        if mat.shape().ndim() != 2 {
            return Err(TorshError::invalid_argument_with_context(
                &format!("Matrix {} is not 2D", i),
                "chain_matmul",
            ));
        }
    }

    // Validate dimensions match for multiplication
    for i in 0..matrices.len() - 1 {
        let m1_cols = matrices[i].shape().dims()[1];
        let m2_rows = matrices[i + 1].shape().dims()[0];
        if m1_cols != m2_rows {
            return Err(TorshError::invalid_argument_with_context(
                &format!(
                    "Matrix dimensions incompatible: [{} x {}] @ [{} x {}]",
                    matrices[i].shape().dims()[0],
                    m1_cols,
                    m2_rows,
                    matrices[i + 1].shape().dims()[1]
                ),
                "chain_matmul",
            ));
        }
    }

    // Perform chain multiplication
    // TODO: Use dynamic programming for optimal parenthesization
    let mut result = matrices[0].clone();
    for i in 1..matrices.len() {
        result = result.matmul(&matrices[i])?;
    }

    Ok(result)
}

/// Compute matrix norm with various norm types
///
/// ## Mathematical Background
///
/// Matrix norms provide a measure of matrix "size" and are essential for:
/// - **Condition numbers**: κ(A) = ||A|| ||A⁻¹||
/// - **Error analysis**: ||Ax - b|| ≤ ||A|| ||x|| + ||b||
/// - **Convergence analysis**: Spectral radius ρ(A) ≤ ||A|| for any norm
///
/// ### Common Matrix Norms
///
/// 1. **Frobenius norm** (default): ||A||_F = √(∑ᵢⱼ |aᵢⱼ|²)
///    - Generalizes vector Euclidean norm to matrices
///    - Invariant under orthogonal transformations
///
/// 2. **Nuclear norm**: ||A||_* = ∑ᵢ σᵢ (sum of singular values)
///    - Convex relaxation of matrix rank
///    - Used in low-rank matrix recovery
///
/// 3. **Infinity norm**: ||A||_∞ = maxᵢ ∑ⱼ |aᵢⱼ| (maximum row sum)
///    - Induced by vector infinity norm
///    - Easy to compute and interpret
///
/// 4. **p-norm**: ||A||_p = (∑ᵢⱼ |aᵢⱼ|^p)^(1/p)
///    - Generalizes other norms (p=2 gives Frobenius)
///
/// ## Parameters
/// * `tensor` - Input matrix/tensor
/// * `ord` - Norm type (defaults to Frobenius)
/// * `dim` - Dimensions to compute norm over (None for all dimensions)
/// * `keepdim` - Whether to keep reduced dimensions
///
/// ## Returns
/// * Tensor containing the computed norm(s)
pub fn norm(
    tensor: &Tensor,
    ord: Option<NormOrd>,
    dim: Option<Vec<isize>>,
    keepdim: bool,
) -> TorshResult<Tensor> {
    let ord = ord.unwrap_or(NormOrd::Fro);

    match ord {
        NormOrd::Fro => {
            // Frobenius norm: ||A||_F = √(∑ᵢⱼ |aᵢⱼ|²)
            let squared = tensor.pow(2.0)?;
            let sum = if let Some(dims) = dim {
                let mut result = squared;
                for &d in dims.iter() {
                    result = result.sum_dim(&[d as i32], keepdim)?;
                }
                result
            } else {
                squared.sum()?
            };
            sum.sqrt()
        }
        NormOrd::Nuclear => {
            // Nuclear norm (sum of singular values)
            // Delegate to the nuclear norm implementation in reduction module
            crate::reduction::norm_nuclear(tensor)
        }
        NormOrd::Inf => {
            // Infinity norm: ||A||_∞ = maxᵢ ∑ⱼ |aᵢⱼ|
            if let Some(dims) = dim {
                let abs_tensor = tensor.abs()?;
                // Sum along the specified dimensions
                let mut result = abs_tensor;
                for &d in dims.iter() {
                    result = result.sum_dim(&[d as i32], keepdim)?;
                }
                // Then take the maximum of the result
                result.max(None, false)
            } else {
                tensor.abs()?.max(None, false)
            }
        }
        NormOrd::NegInf => {
            // Negative infinity norm: ||A||_{-∞} = minᵢ ∑ⱼ |aᵢⱼ|
            if let Some(dims) = dim {
                let abs_tensor = tensor.abs()?;
                // Sum along the specified dimensions
                let mut result = abs_tensor;
                for &d in dims.iter() {
                    result = result.sum_dim(&[d as i32], keepdim)?;
                }
                // Then take the minimum of the result
                result.min()
            } else {
                tensor.abs()?.min()
            }
        }
        NormOrd::P(p) => {
            // p-norm: ||A||_p = (∑ᵢⱼ |aᵢⱼ|^p)^(1/p)
            let abs_p = tensor.abs()?.pow(p)?;
            let sum = if let Some(dims) = dim {
                let mut result = abs_p;
                for &d in dims.iter() {
                    result = result.sum_dim(&[d as i32], keepdim)?;
                }
                result
            } else {
                abs_p.sum()?
            };
            sum.pow(1.0 / p)
        }
    }
}

/// Batch matrix multiplication (BMM)
///
/// Performs matrix multiplication on batches of matrices:
/// ```
/// output[i] = input[i] @ mat2[i]
/// ```
///
/// ## Mathematical Background
///
/// Batch operations enable efficient processing of multiple independent matrix
/// operations in parallel. Each batch element is processed independently:
///
/// For tensors of shape (B, M, K) and (B, K, N):
/// - Batch size B must match
/// - Inner dimensions K must match for matrix multiplication
/// - Output shape will be (B, M, N)
///
/// ## Computational Complexity
/// - Time: O(B × M × N × K)
/// - Space: O(B × M × N) for output
///
/// ## Parameters
/// * `input` - First batch of matrices, shape (B, M, K)
/// * `mat2` - Second batch of matrices, shape (B, K, N)
///
/// ## Returns
/// * Batch of result matrices, shape (B, M, N)
///
/// ## Example
/// ```rust
/// # use torsh_functional::linalg::bmm;
/// # use torsh_tensor::creation::randn;
/// let batch1 = randn(&[10, 3, 4])?; // 10 matrices of 3×4
/// let batch2 = randn(&[10, 4, 5])?; // 10 matrices of 4×5
/// let result = bmm(&batch1, &batch2)?; // 10 matrices of 3×5
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn bmm(input: &Tensor, mat2: &Tensor) -> TorshResult<Tensor> {
    // Validate inputs are 3D tensors
    if input.shape().ndim() != 3 || mat2.shape().ndim() != 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Batch matrix multiplication requires 3D tensors (batch, rows, cols)",
            "bmm",
        ));
    }

    let input_binding = input.shape();
    let input_dims = input_binding.dims();
    let mat2_binding = mat2.shape();
    let mat2_dims = mat2_binding.dims();

    // Check batch sizes match
    if input_dims[0] != mat2_dims[0] {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Batch sizes don't match: {} vs {}",
                input_dims[0], mat2_dims[0]
            ),
            "bmm",
        ));
    }

    // Check matrix dimensions are compatible for multiplication
    if input_dims[2] != mat2_dims[1] {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Matrix dimensions incompatible: [{} x {}] @ [{} x {}]",
                input_dims[1], input_dims[2], mat2_dims[1], mat2_dims[2]
            ),
            "bmm",
        ));
    }

    // For now, use basic implementation via loops
    let batch_size = input_dims[0];
    let out_rows = input_dims[1];
    let out_cols = mat2_dims[2];

    let mut result_data = vec![0.0f32; batch_size * out_rows * out_cols];
    let input_data = input.to_vec()?;
    let mat2_data = mat2.to_vec()?;

    for b in 0..batch_size {
        for i in 0..out_rows {
            for j in 0..out_cols {
                let mut sum = 0.0f32;
                for k in 0..input_dims[2] {
                    let input_idx = b * input_dims[1] * input_dims[2] + i * input_dims[2] + k;
                    let mat2_idx = b * mat2_dims[1] * mat2_dims[2] + k * mat2_dims[2] + j;
                    sum += input_data[input_idx] * mat2_data[mat2_idx];
                }
                let result_idx = b * out_rows * out_cols + i * out_cols + j;
                result_data[result_idx] = sum;
            }
        }
    }

    Tensor::from_data(
        result_data,
        vec![batch_size, out_rows, out_cols],
        input.device(),
    )
}

/// Batch matrix-matrix addition with scaling
///
/// Computes: `beta * input + alpha * (batch1 @ batch2)`
///
/// ## Mathematical Background
///
/// This operation combines scaled matrix addition with batch matrix multiplication:
/// ```
/// output = β × input + α × (batch1 @ batch2)
/// ```
///
/// Common use cases:
/// - **Neural network layers**: Computing W₁x + b where W₁x involves batched operations
/// - **Optimization algorithms**: Momentum updates with α and β scaling factors
/// - **Numerical methods**: Iterative algorithms requiring scaled matrix operations
///
/// ## Parameters
/// * `input` - Input tensor to be scaled by beta
/// * `batch1` - First batch of matrices for multiplication
/// * `batch2` - Second batch of matrices for multiplication
/// * `beta` - Scaling factor for input tensor
/// * `alpha` - Scaling factor for matrix multiplication result
///
/// ## Returns
/// * Tensor containing β × input + α × (batch1 @ batch2)
///
/// ## Example
/// ```rust
/// # use torsh_functional::linalg::baddbmm;
/// # use torsh_tensor::creation::randn;
/// let input = randn(&[10, 3, 5])?;
/// let batch1 = randn(&[10, 3, 4])?;
/// let batch2 = randn(&[10, 4, 5])?;
/// let result = baddbmm(&input, &batch1, &batch2, 1.0, 2.0)?; // input + 2*(batch1@batch2)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn baddbmm(
    input: &Tensor,
    batch1: &Tensor,
    batch2: &Tensor,
    beta: f32,
    alpha: f32,
) -> TorshResult<Tensor> {
    // Compute batch matrix multiplication: batch1 @ batch2
    let mm_result = bmm(batch1, batch2)?;

    // Apply scaling and addition: beta * input + alpha * (batch1 @ batch2)
    let scaled_input = input.mul_scalar(beta)?;
    let scaled_mm = mm_result.mul_scalar(alpha)?;

    scaled_input.add_op(&scaled_mm)
}
