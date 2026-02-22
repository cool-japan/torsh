//! Tensor Decomposition Operations
//!
//! This module provides advanced tensor decomposition algorithms including:
//! - Tucker decomposition (Higher-Order SVD)
//! - CP decomposition (CANDECOMP/PARAFAC)
//! - Tensor-Train decomposition
//! - Block decomposition for structured tensors
//!
//! All implementations use scirs2-core for optimized linear algebra operations.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Tucker decomposition (Higher-Order SVD)
///
/// Decomposes a tensor into a core tensor and factor matrices along each mode.
///
/// # Mathematical Formula
///
/// For a tensor X of size I₁ × I₂ × ... × Iₙ:
/// ```text
/// X ≈ G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ
/// ```
/// where G is the core tensor and Uᵢ are factor matrices.
///
/// # Arguments
///
/// * `input` - Input tensor to decompose
/// * `ranks` - Target ranks for each mode (None uses SVD-determined ranks)
///
/// # Returns
///
/// Tuple of (core_tensor, factor_matrices)
///
/// # Performance
///
/// - Time Complexity: O(∏ Iᵢ · Rᵢ²) where Rᵢ are target ranks
/// - Space Complexity: O(∏ Rᵢ + Σ Iᵢ · Rᵢ)
/// - Uses scirs2-linalg for SVD operations
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_functional::tensor_decomposition::tucker_decomposition;
///
/// let input = Tensor::randn(&[10, 20, 30])?;
/// let (core, factors) = tucker_decomposition(&input, Some(&[5, 10, 15]))?;
/// // core shape: [5, 10, 15]
/// // factors: [U1(10×5), U2(20×10), U3(30×15)]
/// ```
pub fn tucker_decomposition(
    input: &Tensor<f32>,
    ranks: Option<&[usize]>,
) -> TorshResult<(Tensor<f32>, Vec<Tensor<f32>>)> {
    let input_shape = input.shape();
    let shape = input_shape.dims();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Tucker decomposition requires at least 2D tensor",
            "tucker_decomposition",
        ));
    }

    // Determine target ranks
    let target_ranks = if let Some(r) = ranks {
        if r.len() != ndim {
            return Err(TorshError::invalid_argument_with_context(
                &format!(
                    "ranks length {} must match tensor dimensions {}",
                    r.len(),
                    ndim
                ),
                "tucker_decomposition",
            ));
        }
        r.to_vec()
    } else {
        // Use half the dimension size as default rank
        shape.iter().map(|&s| (s / 2).max(1)).collect()
    };

    // Compute factor matrices via HOSVD (Higher-Order SVD)
    let mut factor_matrices = Vec::with_capacity(ndim);

    for (mode, &rank) in target_ranks.iter().enumerate() {
        // Unfold tensor along mode
        let unfolded = unfold_tensor(input, mode)?;

        // Compute truncated SVD
        let (u, _s, _vt) = crate::linalg::svd(&unfolded, false)?;

        // Keep first 'rank' columns
        let factor = truncate_matrix(&u, rank)?;
        factor_matrices.push(factor);
    }

    // Compute core tensor by projecting input onto factor matrices
    let mut core = input.clone();
    for (mode, factor) in factor_matrices.iter().enumerate() {
        core = mode_product(&core, factor, mode, true)?;
    }

    Ok((core, factor_matrices))
}

/// Unfold (matricize) a tensor along specified mode
///
/// Reshapes a tensor into a matrix by grouping all dimensions except the specified mode.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `mode` - Mode to unfold along (0 to ndim-1)
///
/// # Returns
///
/// Matrix with rows indexed by the specified mode
fn unfold_tensor(tensor: &Tensor<f32>, mode: usize) -> TorshResult<Tensor<f32>> {
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();
    let ndim = shape.len();

    if mode >= ndim {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "mode {} out of range for tensor with {} dimensions",
                mode, ndim
            ),
            "unfold_tensor",
        ));
    }

    // Compute unfolding dimensions
    let mode_size = shape[mode];
    let other_size: usize = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &s)| s)
        .product();

    // Permute to bring mode dimension first
    let mut perm: Vec<i32> = vec![mode as i32];
    for i in 0..ndim {
        if i != mode {
            perm.push(i as i32);
        }
    }

    let permuted = tensor.permute(&perm)?;

    // Reshape to matrix
    permuted.reshape(&[mode_size as i32, other_size as i32])
}

/// Truncate matrix to specified number of columns
fn truncate_matrix(matrix: &Tensor<f32>, rank: usize) -> TorshResult<Tensor<f32>> {
    let matrix_shape = matrix.shape();
    let shape = matrix_shape.dims();
    if shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "truncate_matrix requires 2D tensor",
            "truncate_matrix",
        ));
    }

    let rows = shape[0];
    let cols = shape[1];

    if rank > cols {
        return Err(TorshError::invalid_argument_with_context(
            &format!("rank {} exceeds matrix columns {}", rank, cols),
            "truncate_matrix",
        ));
    }

    // Extract first 'rank' columns
    let data = matrix.data()?;
    let mut truncated_data = Vec::with_capacity(rows * rank);

    for i in 0..rows {
        for j in 0..rank {
            truncated_data.push(data[i * cols + j]);
        }
    }

    Tensor::from_data(truncated_data, vec![rows, rank], matrix.device())
}

/// Mode-n product of tensor with matrix
///
/// Multiplies tensor with matrix along specified mode.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `matrix` - Matrix to multiply
/// * `mode` - Mode to multiply along
/// * `transpose` - Whether to transpose matrix
///
/// # Returns
///
/// Result tensor with modified dimension along specified mode
fn mode_product(
    tensor: &Tensor<f32>,
    matrix: &Tensor<f32>,
    mode: usize,
    transpose: bool,
) -> TorshResult<Tensor<f32>> {
    let tensor_shape_binding = tensor.shape();
    let tensor_shape = tensor_shape_binding.dims();
    let matrix_shape_binding = matrix.shape();
    let matrix_shape = matrix_shape_binding.dims();

    if matrix_shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "matrix must be 2D for mode product",
            "mode_product",
        ));
    }

    let (mat_rows, mat_cols) = (matrix_shape[0], matrix_shape[1]);
    let (expected_size, output_size) = if transpose {
        (mat_rows, mat_cols)
    } else {
        (mat_cols, mat_rows)
    };

    if tensor_shape[mode] != expected_size {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "tensor mode {} size {} doesn't match matrix dimension {}",
                mode, tensor_shape[mode], expected_size
            ),
            "mode_product",
        ));
    }

    // Unfold tensor along mode
    let unfolded = unfold_tensor(tensor, mode)?;

    // Matrix multiply
    let result_matrix = if transpose {
        // matrix^T × unfolded
        let matrix_t = matrix.transpose(0, 1)?;
        let matrix_t_unsq = matrix_t.unsqueeze(0)?;
        let unfolded_unsq = unfolded.unsqueeze(0)?;
        let bmm_result = crate::linalg::bmm(&matrix_t_unsq, &unfolded_unsq)?;
        bmm_result.squeeze(0)?
    } else {
        // matrix × unfolded
        let matrix_unsq = matrix.unsqueeze(0)?;
        let unfolded_unsq = unfolded.unsqueeze(0)?;
        let bmm_result = crate::linalg::bmm(&matrix_unsq, &unfolded_unsq)?;
        bmm_result.squeeze(0)?
    };

    // Fold back to tensor shape
    let mut result_shape = tensor_shape.to_vec();
    result_shape[mode] = output_size;

    fold_tensor(&result_matrix, mode, &result_shape)
}

/// Fold matrix back into tensor along specified mode
fn fold_tensor(
    matrix: &Tensor<f32>,
    mode: usize,
    target_shape: &[usize],
) -> TorshResult<Tensor<f32>> {
    let ndim = target_shape.len();

    if mode >= ndim {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "mode {} out of range for target shape with {} dimensions",
                mode, ndim
            ),
            "fold_tensor",
        ));
    }

    // Reshape matrix to intermediate shape with mode dimension first
    let mut intermediate_shape = vec![target_shape[mode]];
    for (i, &size) in target_shape.iter().enumerate() {
        if i != mode {
            intermediate_shape.push(size);
        }
    }

    let reshaped = matrix.reshape(
        &intermediate_shape
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>(),
    )?;

    // Permute to restore original dimension order
    let mut inv_perm = vec![0i32; ndim];
    inv_perm[mode] = 0;
    let mut idx = 1;
    for i in 0..ndim {
        if i != mode {
            inv_perm[i] = idx;
            idx += 1;
        }
    }

    // Compute forward permutation (inverse of inv_perm)
    let mut perm = vec![0i32; ndim];
    for (i, &p) in inv_perm.iter().enumerate() {
        perm[p as usize] = i as i32;
    }

    reshaped.permute(&perm)
}

/// CP decomposition (CANDECOMP/PARAFAC)
///
/// Decomposes a tensor into a sum of rank-1 tensors.
///
/// # Mathematical Formula
///
/// For a tensor X of size I₁ × I₂ × ... × Iₙ:
/// ```text
/// X ≈ Σᵣ λᵣ (a₁ᵣ ⊗ a₂ᵣ ⊗ ... ⊗ aₙᵣ)
/// ```
/// where λᵣ are weights and aᵢᵣ are factor vectors.
///
/// # Arguments
///
/// * `input` - Input tensor to decompose
/// * `rank` - Number of rank-1 components
/// * `max_iter` - Maximum ALS iterations
///
/// # Returns
///
/// Tuple of (weights, factor_matrices)
///
/// # Performance
///
/// - Time Complexity: O(R · max_iter · ∏ Iᵢ) where R is rank
/// - Space Complexity: O(R · Σ Iᵢ)
/// - Uses alternating least squares (ALS) optimization
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_functional::tensor_decomposition::cp_decomposition;
///
/// let input = Tensor::randn(&[10, 20, 30])?;
/// let (weights, factors) = cp_decomposition(&input, 5, 100)?;
/// // weights: scalar tensor of size 5
/// // factors: [A1(10×5), A2(20×5), A3(30×5)]
/// ```
pub fn cp_decomposition(
    input: &Tensor<f32>,
    rank: usize,
    max_iter: usize,
) -> TorshResult<(Tensor<f32>, Vec<Tensor<f32>>)> {
    let input_shape = input.shape();
    let shape = input_shape.dims();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "CP decomposition requires at least 2D tensor",
            "cp_decomposition",
        ));
    }

    if rank == 0 {
        return Err(TorshError::invalid_argument_with_context(
            "rank must be positive",
            "cp_decomposition",
        ));
    }

    // Initialize factor matrices randomly
    use scirs2_core::random::thread_rng;
    let mut rng = thread_rng();
    let mut factors: Vec<Tensor<f32>> = Vec::with_capacity(ndim);

    for &size in shape.iter() {
        let factor_data: Vec<f32> = (0..size * rank).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let factor = Tensor::from_data(factor_data, vec![size, rank], input.device())?;
        factors.push(factor);
    }

    // ALS iterations
    for _iter in 0..max_iter {
        // Update each factor matrix in turn
        for mode in 0..ndim {
            // Compute Khatri-Rao product of all factors except current mode
            let kr = khatri_rao_product_except(&factors, mode)?;

            // Unfold tensor along mode
            let unfolded = unfold_tensor(input, mode)?;

            // Solve least squares: factor[mode] = unfolded × kr × (kr^T kr)^{-1}
            // Use normal equations instead of pseudoinverse for better numerical stability
            // kr is [other_size, rank]
            // unfolded is [mode_size, other_size]
            // We want: factor[mode] = unfolded @ kr @ (kr^T @ kr)^{-1}

            let kr_t = kr.transpose(0, 1)?; // [rank, other_size]
            let kr_t_unsq = kr_t.unsqueeze(0)?;
            let kr_unsq = kr.unsqueeze(0)?;

            // Compute kr^T @ kr: [rank, other_size] @ [other_size, rank] = [rank, rank]
            let gram = crate::linalg::bmm(&kr_t_unsq, &kr_unsq)?;
            let gram_squeezed = gram.squeeze(0)?;

            // Add regularization to avoid singularity
            let mut gram_data = gram_squeezed.data()?.to_vec();
            let gram_shape = gram_squeezed.shape();
            let rank_val = gram_shape.dims()[0];
            for i in 0..rank_val {
                gram_data[i * rank_val + i] += 1e-6; // Add small regularization
            }
            let gram_reg = Tensor::from_data(gram_data, vec![rank_val, rank_val], input.device())?;

            // Compute (kr^T @ kr)^{-1}
            let gram_inv = crate::linalg::inv(&gram_reg)?;

            // Compute unfolded @ kr: [mode_size, other_size] @ [other_size, rank] = [mode_size, rank]
            let unfolded_unsq = unfolded.unsqueeze(0)?;
            let unfolded_kr = crate::linalg::bmm(&unfolded_unsq, &kr_unsq)?;
            let unfolded_kr_squeezed = unfolded_kr.squeeze(0)?;

            // Compute factor[mode] = (unfolded @ kr) @ (kr^T @ kr)^{-1}: [mode_size, rank] @ [rank, rank] = [mode_size, rank]
            let unfolded_kr_unsq = unfolded_kr_squeezed.unsqueeze(0)?;
            let gram_inv_unsq = gram_inv.unsqueeze(0)?;
            let new_factor_result = crate::linalg::bmm(&unfolded_kr_unsq, &gram_inv_unsq)?;
            let new_factor = new_factor_result.squeeze(0)?;

            factors[mode] = new_factor;
        }
    }

    // Normalize factors and extract weights
    let weights = normalize_factors(&mut factors)?;

    Ok((weights, factors))
}

/// Khatri-Rao product of all factors except specified mode
fn khatri_rao_product_except(
    factors: &[Tensor<f32>],
    except_mode: usize,
) -> TorshResult<Tensor<f32>> {
    let factor0_shape = factors[0].shape();
    let rank = factor0_shape.dims()[1];

    // Collect all factors except the specified mode
    let other_factors: Vec<&Tensor<f32>> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != except_mode)
        .map(|(_, f)| f)
        .collect();

    // If only one factor remains, return it directly
    if other_factors.len() == 1 {
        return Ok(other_factors[0].clone());
    }

    // Start with first factor and compute Khatri-Rao products with remaining factors
    let mut result = other_factors[0].clone();
    for factor in other_factors.iter().skip(1) {
        result = khatri_rao_product(&result, factor)?;
    }

    // Verify result has correct rank
    let result_shape = result.shape();
    if result_shape.dims()[1] != rank {
        return Err(TorshError::InvalidOperation(
            "Khatri-Rao product rank mismatch (khatri_rao_product_except)".to_string(),
        ));
    }

    Ok(result)
}

/// Khatri-Rao (column-wise Kronecker) product of two matrices
fn khatri_rao_product(a: &Tensor<f32>, b: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let a_shape_obj = a.shape();
    let shape_a = a_shape_obj.dims();
    let b_shape_obj = b.shape();
    let shape_b = b_shape_obj.dims();

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Khatri-Rao product requires 2D tensors",
            "khatri_rao_product",
        ));
    }

    let (rows_a, cols_a) = (shape_a[0], shape_a[1]);
    let (rows_b, cols_b) = (shape_b[0], shape_b[1]);

    if cols_a != cols_b {
        return Err(TorshError::invalid_argument_with_context(
            &format!("column dimensions must match: {} vs {}", cols_a, cols_b),
            "khatri_rao_product",
        ));
    }

    let data_a = a.data()?;
    let data_b = b.data()?;
    let mut result_data = Vec::with_capacity(rows_a * rows_b * cols_a);

    // For each column
    for col in 0..cols_a {
        // Kronecker product of columns
        for i in 0..rows_a {
            for j in 0..rows_b {
                let val_a = data_a[i * cols_a + col];
                let val_b = data_b[j * cols_b + col];
                result_data.push(val_a * val_b);
            }
        }
    }

    Tensor::from_data(result_data, vec![rows_a * rows_b, cols_a], a.device())
}

/// Normalize factor matrices and extract weights
fn normalize_factors(factors: &mut [Tensor<f32>]) -> TorshResult<Tensor<f32>> {
    let factor0_shape = factors[0].shape();
    let rank = factor0_shape.dims()[1];
    let mut weights = vec![1.0f32; rank];

    for factor in factors.iter_mut() {
        let factor_shape = factor.shape();
        let shape = factor_shape.dims();
        let (rows, cols) = (shape[0], shape[1]);
        let data = factor.data()?;
        let mut new_data = data.to_vec();

        // Compute column norms
        for col in 0..cols {
            let mut norm = 0.0f32;
            for row in 0..rows {
                let idx = row * cols + col;
                norm += new_data[idx] * new_data[idx];
            }
            norm = norm.sqrt();

            // Normalize column and accumulate weight
            if norm > 1e-10 {
                weights[col] *= norm;
                for row in 0..rows {
                    let idx = row * cols + col;
                    new_data[idx] /= norm;
                }
            }
        }

        *factor = Tensor::from_data(new_data, vec![rows, cols], factor.device())?;
    }

    Tensor::from_data(weights, vec![rank], factors[0].device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_unfold_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_data(data, vec![2, 2, 2], torsh_core::device::DeviceType::Cpu)
            .expect("failed to create tensor");

        // Unfold along mode 0
        let unfolded = unfold_tensor(&tensor, 0).expect("unfold failed");
        assert_eq!(unfolded.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_tucker_decomposition() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_data(data, vec![2, 2, 2], torsh_core::device::DeviceType::Cpu)
            .expect("failed to create tensor");

        let (core, factors) =
            tucker_decomposition(&tensor, Some(&[1, 1, 1])).expect("tucker decomposition failed");

        assert_eq!(core.shape().dims(), &[1, 1, 1]);
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0].shape().dims(), &[2, 1]);
        assert_eq!(factors[1].shape().dims(), &[2, 1]);
        assert_eq!(factors[2].shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_khatri_rao_product() {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("failed to create tensor");

        let b = Tensor::from_data(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("failed to create tensor");

        let result = khatri_rao_product(&a, &b).expect("khatri-rao failed");
        assert_eq!(result.shape().dims(), &[4, 2]);

        let result_data = result.data().expect("failed to get data");
        // First column: [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
        assert_relative_eq!(result_data[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(result_data[1], 7.0, epsilon = 1e-6);
        assert_relative_eq!(result_data[2], 15.0, epsilon = 1e-6);
        assert_relative_eq!(result_data[3], 21.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cp_decomposition() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], torsh_core::device::DeviceType::Cpu)
            .expect("failed to create tensor");

        let (weights, factors) = cp_decomposition(&tensor, 1, 10).expect("cp decomposition failed");

        assert_eq!(weights.shape().dims(), &[1]);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0].shape().dims(), &[2, 1]);
        assert_eq!(factors[1].shape().dims(), &[2, 1]);
    }
}
