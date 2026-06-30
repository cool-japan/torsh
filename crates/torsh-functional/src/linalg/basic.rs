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
/// This function picks the cost-minimal parenthesization automatically via the
/// O(n³) matrix-chain-order dynamic program, then performs the products in that
/// order. The numeric result is identical for every valid order; only the number
/// of scalar multiplications changes. See [`matrix_chain_optimal_cost`] to query
/// the minimal cost for a dimension sequence.
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

    // Determine the cost-minimal multiplication order via the matrix-chain-order
    // dynamic program, then perform the products in that order.
    //
    // The dimension sequence has length `matrices.len() + 1`: matrix `i` has
    // shape `dim_seq[i] x dim_seq[i + 1]`. Adjacent dimensions are guaranteed
    // consistent by the validation above.
    let mut dim_seq = Vec::with_capacity(matrices.len() + 1);
    dim_seq.push(matrices[0].shape().dims()[0]);
    for mat in matrices {
        dim_seq.push(mat.shape().dims()[1]);
    }

    let (_cost, split) = matrix_chain_order(&dim_seq);
    multiply_chain(matrices, &split, 0, matrices.len() - 1)
}

/// Compute the optimal matrix-chain multiplication order via dynamic programming.
///
/// `dim_seq` is the dimension sequence of length `n + 1` describing a chain of
/// `n` matrices, where the `i`-th matrix has shape `dim_seq[i] × dim_seq[i + 1]`.
///
/// This is the classic O(n³)-time / O(n²)-space "Matrix-Chain-Order" dynamic
/// program (Cormen, Leiserson, Rivest, Stein — *Introduction to Algorithms*).
///
/// # Returns
/// A tuple `(cost, split)` where:
/// * `cost` is the minimal number of scalar multiplications required.
/// * `split[i][j]` is the index `k` for which the optimal parenthesization of
///   `Aᵢ … Aⱼ` is `(Aᵢ … Aₖ)(Aₖ₊₁ … Aⱼ)`.
///
/// Requires `dim_seq.len() >= 2` (i.e. at least one matrix); callers guarantee this.
fn matrix_chain_order(dim_seq: &[usize]) -> (u128, Vec<Vec<usize>>) {
    let n = dim_seq.len() - 1; // number of matrices in the chain

    // m[i][j] = minimal cost (scalar multiplications) to compute Aᵢ … Aⱼ.
    // A single matrix costs nothing, so the diagonal stays zero.
    let mut m = vec![vec![0u128; n]; n];
    // split[i][j] = optimal split point k for the sub-chain Aᵢ … Aⱼ.
    let mut split = vec![vec![0usize; n]; n];

    // Consider sub-chains of increasing length (2 .. n matrices).
    for chain_len in 2..=n {
        for i in 0..=(n - chain_len) {
            let j = i + chain_len - 1;
            m[i][j] = u128::MAX;
            // Try every split point k: (Aᵢ … Aₖ)(Aₖ₊₁ … Aⱼ).
            for k in i..j {
                // Multiplying the (dim_seq[i] × dim_seq[k+1]) left product by the
                // (dim_seq[k+1] × dim_seq[j+1]) right product costs
                // dim_seq[i] * dim_seq[k+1] * dim_seq[j+1] scalar multiplications.
                let cost = m[i][k]
                    + m[k + 1][j]
                    + (dim_seq[i] as u128) * (dim_seq[k + 1] as u128) * (dim_seq[j + 1] as u128);
                if cost < m[i][j] {
                    m[i][j] = cost;
                    split[i][j] = k;
                }
            }
        }
    }

    (m[0][n - 1], split)
}

/// Multiply the sub-chain `Aᵢ … Aⱼ` (inclusive) following the optimal split table.
///
/// The split table is produced by [`matrix_chain_order`]; this performs the actual
/// tensor products in the cost-minimal parenthesization. The numeric result is
/// identical to any other valid multiplication order.
fn multiply_chain(
    matrices: &[Tensor],
    split: &[Vec<usize>],
    i: usize,
    j: usize,
) -> TorshResult<Tensor> {
    if i == j {
        return Ok(matrices[i].clone());
    }
    let k = split[i][j];
    let left = multiply_chain(matrices, split, i, k)?;
    let right = multiply_chain(matrices, split, k + 1, j)?;
    left.matmul(&right)
}

/// Minimal number of scalar multiplications to evaluate a matrix chain.
///
/// Given the dimension sequence `dim_seq` (length `n + 1` for a chain of `n`
/// matrices, where matrix `i` has shape `dim_seq[i] × dim_seq[i + 1]`), this
/// returns the optimal cost computed by the matrix-chain-order dynamic program
/// used internally by [`chain_matmul`].
///
/// Returns `0` for an empty chain or a single matrix (no multiplication needed).
///
/// ## Example
/// ```rust
/// # use torsh_functional::linalg::matrix_chain_optimal_cost;
/// // A₁: 10×100, A₂: 100×5, A₃: 5×50  →  optimal 7500 (vs 75000 for the
/// // opposite parenthesization).
/// assert_eq!(matrix_chain_optimal_cost(&[10, 100, 5, 50]), 7500);
/// ```
pub fn matrix_chain_optimal_cost(dim_seq: &[usize]) -> u128 {
    if dim_seq.len() < 2 {
        return 0;
    }
    matrix_chain_order(dim_seq).0
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
/// ```text
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
/// ```text
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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    /// Build a deterministic matrix with small integer entries in `{0, 1, 2}`.
    ///
    /// Small bounded entries keep every partial sum well within f32's exact
    /// integer range (< 2²⁴), so products are bit-exact regardless of the
    /// multiplication order — which lets us compare orderings directly.
    fn pattern_matrix(rows: usize, cols: usize, seed: usize) -> TorshResult<Tensor> {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(((i + j + seed) % 3) as f32);
            }
        }
        Tensor::from_data(data, vec![rows, cols], DeviceType::Cpu)
    }

    #[test]
    fn test_matrix_chain_optimal_cost_known_values() {
        // Classic example: A1=10×100, A2=100×5, A3=5×50.
        //   (A1·A2)·A3 = 10·100·5 + 10·5·50  = 5000 + 2500  =  7500  (optimal)
        //   A1·(A2·A3) = 100·5·50 + 10·100·50 = 25000 + 50000 = 75000
        assert_eq!(matrix_chain_optimal_cost(&[10, 100, 5, 50]), 7500);

        // Reversed dimensions: here LEFT-to-right is the *expensive* order.
        // A1=50×5, A2=5×100, A3=100×10.
        //   (A1·A2)·A3 = 50·5·100 + 50·100·10 = 25000 + 50000 = 75000  (naive)
        //   A1·(A2·A3) = 5·100·10 + 50·5·10   =  5000 +  2500 =  7500  (optimal)
        // A correct DP must return 7500, NOT the naive left-to-right 75000.
        assert_eq!(matrix_chain_optimal_cost(&[50, 5, 100, 10]), 7500);

        // CLRS textbook six-matrix chain p=<30,35,15,5,10,20,25> → optimal 15125.
        assert_eq!(
            matrix_chain_optimal_cost(&[30, 35, 15, 5, 10, 20, 25]),
            15125
        );

        // Degenerate chains require no multiplication.
        assert_eq!(matrix_chain_optimal_cost(&[7, 3]), 0); // single matrix
        assert_eq!(matrix_chain_optimal_cost(&[42]), 0); // no matrix
        assert_eq!(matrix_chain_optimal_cost(&[]), 0); // empty
    }

    #[test]
    fn test_chain_matmul_uses_optimal_order_and_matches_reference() -> TorshResult<()> {
        // Reversed classic example: the optimal order A1·(A2·A3) differs from the
        // naive left-to-right (A1·A2)·A3. chain_matmul must evaluate the optimal
        // order yet still produce the same numeric product.
        let a1 = pattern_matrix(50, 5, 0)?;
        let a2 = pattern_matrix(5, 100, 1)?;
        let a3 = pattern_matrix(100, 10, 2)?;

        // Optimal cost for these dimensions is 7500 (not the naive 75000).
        assert_eq!(matrix_chain_optimal_cost(&[50, 5, 100, 10]), 7500);

        let result = chain_matmul(&[a1.clone(), a2.clone(), a3.clone()])?;
        assert_eq!(result.shape().dims(), &[50, 10]);

        // Reference computed independently via explicit left-to-right products.
        let reference = a1.matmul(&a2)?.matmul(&a3)?;
        assert_eq!(reference.shape().dims(), &[50, 10]);

        let result_data = result.to_vec()?;
        let reference_data = reference.to_vec()?;
        assert_eq!(result_data.len(), reference_data.len());
        // Sanity: the product is non-trivial (not all zeros).
        assert!(result_data.iter().any(|&v| v > 0.0));
        for (idx, (&got, &want)) in result_data.iter().zip(reference_data.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {idx}: chain_matmul={got}, reference={want}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_chain_matmul_four_matrices_matches_reference() -> TorshResult<()> {
        // Four-matrix chain to exercise deeper recursion through the split table.
        // Binary {0,1} entries keep all partial sums exact in f32.
        let dims = [4usize, 5, 3, 6, 2];
        let mut mats = Vec::new();
        for w in dims.windows(2).enumerate() {
            let (seed, pair) = w;
            let rows = pair[0];
            let cols = pair[1];
            let mut data = Vec::with_capacity(rows * cols);
            for idx in 0..rows * cols {
                data.push(((idx + seed) % 2) as f32);
            }
            mats.push(Tensor::from_data(data, vec![rows, cols], DeviceType::Cpu)?);
        }

        let result = chain_matmul(&mats)?;
        assert_eq!(result.shape().dims(), &[4, 2]);

        let reference = mats[0]
            .matmul(&mats[1])?
            .matmul(&mats[2])?
            .matmul(&mats[3])?;
        let result_data = result.to_vec()?;
        let reference_data = reference.to_vec()?;
        for (&got, &want) in result_data.iter().zip(reference_data.iter()) {
            assert!((got - want).abs() < 1e-3, "chain={got}, reference={want}");
        }
        Ok(())
    }

    #[test]
    fn test_chain_matmul_single_matrix_is_identity() -> TorshResult<()> {
        let a = pattern_matrix(3, 4, 0)?;
        let result = chain_matmul(std::slice::from_ref(&a))?;
        assert_eq!(result.to_vec()?, a.to_vec()?);
        Ok(())
    }
}
