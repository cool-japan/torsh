//! Utility functions for linear algebra operations
//!
//! This module provides helper functions for common patterns in linear algebra,
//! including validation, conversion, and convenience operations.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Check if a matrix is approximately diagonal within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for off-diagonal elements (default: 1e-10)
///
/// # Returns
///
/// True if all off-diagonal elements are below tolerance
pub fn is_diagonal(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Ok(false);
    }

    let tolerance = tol.unwrap_or(1e-10);

    for i in 0..n {
        for j in 0..n {
            if i != j && tensor.get(&[i, j])?.abs() > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Check if a matrix is approximately identity within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for elements (default: 1e-8)
///
/// # Returns
///
/// True if matrix is approximately identity
pub fn is_identity(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Ok(false);
    }

    let tolerance = tol.unwrap_or(1e-8);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (tensor.get(&[i, j])? - expected).abs() > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Check if a matrix is upper triangular within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for elements below diagonal (default: 1e-10)
///
/// # Returns
///
/// True if all elements below diagonal are below tolerance
pub fn is_upper_triangular(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let tolerance = tol.unwrap_or(1e-10);

    for i in 0..m {
        for j in 0..n.min(i) {
            if tensor.get(&[i, j])?.abs() > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Check if a matrix is lower triangular within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for elements above diagonal (default: 1e-10)
///
/// # Returns
///
/// True if all elements above diagonal are below tolerance
pub fn is_lower_triangular(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let tolerance = tol.unwrap_or(1e-10);

    for i in 0..m {
        for j in (i + 1)..n {
            if tensor.get(&[i, j])?.abs() > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Check if a matrix is orthogonal (Q^T * Q ≈ I) within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for orthogonality (default: 1e-6)
///
/// # Returns
///
/// True if matrix is approximately orthogonal
pub fn is_orthogonal(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Ok(false);
    }

    let tolerance = tol.unwrap_or(1e-6);

    // Compute Q^T * Q
    let q_t = tensor.transpose(-2, -1)?;
    let qtq = q_t.matmul(tensor)?;

    // Check if result is identity
    is_identity(&qtq, Some(tolerance))
}

/// Extract diagonal elements from a matrix as a vector
///
/// # Arguments
///
/// * `tensor` - Matrix to extract diagonal from
///
/// # Returns
///
/// Vector containing diagonal elements
pub fn extract_diagonal(tensor: &Tensor) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Diagonal extraction requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let diag_len = m.min(n);

    let mut diag_data = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        diag_data.push(tensor.get(&[i, i])?);
    }

    Tensor::from_data(diag_data, vec![diag_len], tensor.device())
}

/// Compute the Frobenius inner product of two matrices: <A, B> = trace(A^T * B)
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// Frobenius inner product value
pub fn frobenius_inner_product(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    if a.shape() != b.shape() {
        return Err(TorshError::InvalidArgument(
            "Frobenius inner product requires matrices of same shape".to_string(),
        ));
    }

    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Frobenius inner product requires 2D tensors".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);

    let mut sum = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            sum += a.get(&[i, j])? * b.get(&[i, j])?;
        }
    }

    Ok(sum)
}

/// Create a block diagonal matrix from a list of matrices
///
/// # Arguments
///
/// * `blocks` - List of square matrices to place on diagonal
///
/// # Returns
///
/// Block diagonal matrix
pub fn block_diag(blocks: &[&Tensor]) -> TorshResult<Tensor> {
    if blocks.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Block diagonal requires at least one block".to_string(),
        ));
    }

    // Validate all blocks are 2D and square
    let mut block_sizes = Vec::new();
    for block in blocks {
        if block.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "All blocks must be 2D tensors".to_string(),
            ));
        }
        let (m, n) = (block.shape().dims()[0], block.shape().dims()[1]);
        if m != n {
            return Err(TorshError::InvalidArgument(
                "All blocks must be square matrices".to_string(),
            ));
        }
        block_sizes.push(m);
    }

    let total_size: usize = block_sizes.iter().sum();
    let mut result_data = vec![0.0f32; total_size * total_size];

    let mut offset = 0;
    for (block, &size) in blocks.iter().zip(block_sizes.iter()) {
        for i in 0..size {
            for j in 0..size {
                let row = offset + i;
                let col = offset + j;
                result_data[row * total_size + col] = block.get(&[i, j])?;
            }
        }
        offset += size;
    }

    Tensor::from_data(
        result_data,
        vec![total_size, total_size],
        blocks[0].device(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::eye;

    #[test]
    fn test_is_diagonal() -> TorshResult<()> {
        // Test diagonal matrix
        let diag_data = vec![1.0, 0.0, 0.0, 2.0];
        let diag = Tensor::from_data(diag_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        assert!(is_diagonal(&diag, None)?);

        // Test non-diagonal matrix
        let non_diag_data = vec![1.0, 0.5, 0.0, 2.0];
        let non_diag = Tensor::from_data(non_diag_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        assert!(!is_diagonal(&non_diag, None)?);

        Ok(())
    }

    #[test]
    fn test_is_identity() -> TorshResult<()> {
        let identity = eye::<f32>(3)?;
        assert!(is_identity(&identity, None)?);

        let non_identity_data = vec![1.0, 0.0, 0.0, 2.0];
        let non_identity =
            Tensor::from_data(non_identity_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        assert!(!is_identity(&non_identity, None)?);

        Ok(())
    }

    #[test]
    fn test_is_upper_triangular() -> TorshResult<()> {
        let upper_data = vec![1.0, 2.0, 0.0, 3.0];
        let upper = Tensor::from_data(upper_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        assert!(is_upper_triangular(&upper, None)?);

        let not_upper_data = vec![1.0, 2.0, 1.0, 3.0];
        let not_upper = Tensor::from_data(not_upper_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        assert!(!is_upper_triangular(&not_upper, None)?);

        Ok(())
    }

    #[test]
    fn test_extract_diagonal() -> TorshResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Tensor::from_data(data, vec![3, 3], torsh_core::DeviceType::Cpu)?;

        let diag = extract_diagonal(&matrix)?;

        assert_eq!(diag.shape().dims(), &[3]);
        assert_eq!(diag.get(&[0])?, 1.0);
        assert_eq!(diag.get(&[1])?, 5.0);
        assert_eq!(diag.get(&[2])?, 9.0);

        Ok(())
    }

    #[test]
    fn test_frobenius_inner_product() -> TorshResult<()> {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Tensor::from_data(a_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let b_data = vec![2.0, 0.0, 0.0, 2.0];
        let b = Tensor::from_data(b_data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let result = frobenius_inner_product(&a, &b)?;

        // <A, B> = 1*2 + 2*0 + 3*0 + 4*2 = 2 + 8 = 10
        assert_eq!(result, 10.0);

        Ok(())
    }
}
