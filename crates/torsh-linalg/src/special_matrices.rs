//! Special matrix constructors

#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_question_mark)]

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Create diagonal matrix from vector
///
/// If tensor is 1D, creates a 2D diagonal matrix with the values on the specified diagonal.
/// If tensor is 2D, extracts the specified diagonal.
pub fn diag(tensor: &Tensor, diagonal: i32) -> TorshResult<Tensor> {
    match tensor.shape().ndim() {
        1 => {
            // Create diagonal matrix from 1D tensor
            let n = tensor.shape().dims()[0];
            let size = n + diagonal.unsigned_abs() as usize;
            let mut data = vec![0.0f32; size * size];

            for i in 0..n {
                let row = if diagonal >= 0 {
                    i
                } else {
                    i + (-diagonal) as usize
                };
                let col = if diagonal >= 0 {
                    i + diagonal as usize
                } else {
                    i
                };

                if row < size && col < size {
                    data[row * size + col] = tensor.get(&[i])?;
                }
            }

            Ok(torsh_tensor::Tensor::from_data(
                data,
                vec![size, size],
                tensor.device(),
            )?)
        }
        2 => {
            // Extract diagonal from 2D tensor
            let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);

            let start_row = if diagonal < 0 {
                (-diagonal) as usize
            } else {
                0
            };
            let start_col = if diagonal > 0 { diagonal as usize } else { 0 };

            let diag_len =
                ((m as i32 - start_row as i32).min(n as i32 - start_col as i32)).max(0) as usize;
            let mut data = vec![0.0f32; diag_len];

            for i in 0..diag_len {
                data[i] = tensor.get(&[start_row + i, start_col + i])?;
            }

            Ok(torsh_tensor::Tensor::from_data(
                data,
                vec![diag_len],
                tensor.device(),
            )?)
        }
        _ => Err(TorshError::InvalidArgument(
            "diag requires 1D or 2D tensor".to_string(),
        )),
    }
}

/// Create identity matrix
pub fn eye(n: usize, m: Option<usize>) -> TorshResult<Tensor> {
    let m = m.unwrap_or(n);

    // Use existing eye function from torsh_tensor
    Ok(torsh_tensor::creation::eye::<f32>(n.max(m))?)
}

/// Create Vandermonde matrix
///
/// Generate a Vandermonde matrix where each column is a power of the input vector.
/// If increasing is true, powers go from 0 to N-1, otherwise from N-1 to 0.
pub fn vander(x: &Tensor, n: Option<usize>, increasing: bool) -> TorshResult<Tensor> {
    if x.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "vander requires 1D input tensor".to_string(),
        ));
    }

    let m = x.shape().dims()[0];
    let n = n.unwrap_or(m);
    let mut data = vec![0.0f32; m * n];

    for i in 0..m {
        let xi = x.get(&[i])?;
        for j in 0..n {
            let power = if increasing { j } else { n - 1 - j };
            data[i * n + j] = xi.powi(power as i32);
        }
    }

    Ok(torsh_tensor::Tensor::from_data(
        data,
        vec![m, n],
        x.device(),
    )?)
}

/// Create Toeplitz matrix
///
/// Construct a Toeplitz matrix where each descending diagonal from left to right is constant.
/// c: First column of the matrix
/// r: First row of the matrix (optional, defaults to conjugate of c)
pub fn toeplitz(c: &Tensor, r: Option<&Tensor>) -> TorshResult<Tensor> {
    if c.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "toeplitz requires 1D column tensor".to_string(),
        ));
    }

    let m = c.shape().dims()[0];
    let (n, r_tensor) = match r {
        Some(r_provided) => {
            if r_provided.shape().ndim() != 1 {
                return Err(TorshError::InvalidArgument(
                    "toeplitz requires 1D row tensor".to_string(),
                ));
            }
            // Check that c[0] == r[0] if both are provided
            let c0 = c.get(&[0])?;
            let r0 = r_provided.get(&[0])?;
            if (c0 - r0).abs() > 1e-6 {
                // Use c[0] and ignore r[0] (following NumPy behavior)
            }
            (r_provided.shape().dims()[0], r_provided.clone())
        }
        None => {
            // Use c as both column and row (symmetric Toeplitz)
            (m, c.clone())
        }
    };

    let mut data = vec![0.0f32; m * n];

    // Fill the matrix
    for i in 0..m {
        for j in 0..n {
            if i >= j {
                // Use column values
                data[i * n + j] = c.get(&[i - j])?;
            } else {
                // Use row values
                data[i * n + j] = r_tensor.get(&[j - i])?;
            }
        }
    }

    torsh_tensor::Tensor::from_data(data, vec![m, n], c.device())
}

/// Create Hankel matrix
///
/// Construct a Hankel matrix where each ascending anti-diagonal from left to right is constant.
/// c: First column of the matrix
/// r: Last row of the matrix (optional)
pub fn hankel(c: &Tensor, r: Option<&Tensor>) -> TorshResult<Tensor> {
    if c.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "hankel requires 1D column tensor".to_string(),
        ));
    }

    let m = c.shape().dims()[0];
    let (n, full_sequence) = match r {
        Some(r_provided) => {
            if r_provided.shape().ndim() != 1 {
                return Err(TorshError::InvalidArgument(
                    "hankel requires 1D row tensor".to_string(),
                ));
            }
            let n = r_provided.shape().dims()[0];

            // Check continuity: c[-1] should equal r[0]
            let c_last = c.get(&[m - 1])?;
            let r_first = r_provided.get(&[0])?;
            if (c_last - r_first).abs() > 1e-6 {
                // Use c[-1] and ignore r[0] (following NumPy behavior)
            }

            // Create full sequence: c[0..m-1] + r[0..n]
            let mut seq = vec![0.0f32; m + n - 1];
            for i in 0..m {
                seq[i] = c.get(&[i])?;
            }
            for i in 1..n {
                seq[m - 1 + i] = r_provided.get(&[i])?;
            }
            (n, seq)
        }
        None => {
            // Default: create square matrix with zeros for undefined elements
            let mut seq = vec![0.0f32; 2 * m - 1];
            for i in 0..m {
                seq[i] = c.get(&[i])?;
            }
            (m, seq)
        }
    };

    let mut data = vec![0.0f32; m * n];

    // Fill the matrix
    for i in 0..m {
        for j in 0..n {
            let idx = i + j;
            if idx < full_sequence.len() {
                data[i * n + j] = full_sequence[idx];
            }
        }
    }

    torsh_tensor::Tensor::from_data(data, vec![m, n], c.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_diag_create_from_vector() -> TorshResult<()> {
        // Create diagonal matrix from 1D vector
        let vec = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let diag_mat = diag(&vec, 0)?;

        // Should create 3x3 diagonal matrix
        assert_eq!(diag_mat.shape().dims(), &[3, 3]);

        // Verify diagonal elements
        assert_relative_eq!(diag_mat.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 2])?, 3.0, epsilon = 1e-6);

        // Verify off-diagonal elements are zero
        assert_relative_eq!(diag_mat.get(&[0, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[0, 2])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 2])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 1])?, 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_diag_create_superdiagonal() -> TorshResult<()> {
        // Create superdiagonal matrix (k=1)
        let vec = Tensor::from_data(vec![1.0f32, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let diag_mat = diag(&vec, 1)?;

        // Should create 3x3 matrix with values on superdiagonal
        assert_eq!(diag_mat.shape().dims(), &[3, 3]);

        // Verify superdiagonal elements
        assert_relative_eq!(diag_mat.get(&[0, 1])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 2])?, 2.0, epsilon = 1e-6);

        // Verify main diagonal is zero
        assert_relative_eq!(diag_mat.get(&[0, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 2])?, 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_diag_create_subdiagonal() -> TorshResult<()> {
        // Create subdiagonal matrix (k=-1)
        let vec = Tensor::from_data(vec![3.0f32, 4.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let diag_mat = diag(&vec, -1)?;

        // Should create 3x3 matrix with values on subdiagonal
        assert_eq!(diag_mat.shape().dims(), &[3, 3]);

        // Verify subdiagonal elements
        assert_relative_eq!(diag_mat.get(&[1, 0])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 1])?, 4.0, epsilon = 1e-6);

        // Verify main diagonal is zero
        assert_relative_eq!(diag_mat.get(&[0, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[1, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(diag_mat.get(&[2, 2])?, 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_diag_extract_from_matrix() -> TorshResult<()> {
        // Create a test matrix
        let mat = torsh_tensor::creation::zeros::<f32>(&[3, 3])?;
        mat.set(&[0, 0], 1.0)?;
        mat.set(&[0, 1], 2.0)?;
        mat.set(&[0, 2], 3.0)?;
        mat.set(&[1, 0], 4.0)?;
        mat.set(&[1, 1], 5.0)?;
        mat.set(&[1, 2], 6.0)?;
        mat.set(&[2, 0], 7.0)?;
        mat.set(&[2, 1], 8.0)?;
        mat.set(&[2, 2], 9.0)?;

        // Extract main diagonal
        let main_diag = diag(&mat, 0)?;
        assert_eq!(main_diag.shape().dims(), &[3]);
        assert_relative_eq!(main_diag.get(&[0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(main_diag.get(&[1])?, 5.0, epsilon = 1e-6);
        assert_relative_eq!(main_diag.get(&[2])?, 9.0, epsilon = 1e-6);

        // Extract superdiagonal
        let super_diag = diag(&mat, 1)?;
        assert_eq!(super_diag.shape().dims(), &[2]);
        assert_relative_eq!(super_diag.get(&[0])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(super_diag.get(&[1])?, 6.0, epsilon = 1e-6);

        // Extract subdiagonal
        let sub_diag = diag(&mat, -1)?;
        assert_eq!(sub_diag.shape().dims(), &[2]);
        assert_relative_eq!(sub_diag.get(&[0])?, 4.0, epsilon = 1e-6);
        assert_relative_eq!(sub_diag.get(&[1])?, 8.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_eye_square() -> TorshResult<()> {
        let identity = eye(3, None)?;

        assert_eq!(identity.shape().dims(), &[3, 3]);

        // Verify identity matrix properties
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(identity.get(&[i, j])?, expected, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_eye_rectangular() -> TorshResult<()> {
        let identity = eye(2, Some(3))?;

        // Should return 3x3 identity (max of dimensions)
        assert_eq!(identity.shape().dims(), &[3, 3]);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(identity.get(&[i, j])?, expected, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_vander_increasing() -> TorshResult<()> {
        let x = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let vand = vander(&x, Some(4), true)?;

        // Should create 3x4 Vandermonde matrix
        assert_eq!(vand.shape().dims(), &[3, 4]);

        // First row: [1^0, 1^1, 1^2, 1^3] = [1, 1, 1, 1]
        assert_relative_eq!(vand.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[0, 1])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[0, 2])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[0, 3])?, 1.0, epsilon = 1e-6);

        // Second row: [2^0, 2^1, 2^2, 2^3] = [1, 2, 4, 8]
        assert_relative_eq!(vand.get(&[1, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[1, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[1, 2])?, 4.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[1, 3])?, 8.0, epsilon = 1e-6);

        // Third row: [3^0, 3^1, 3^2, 3^3] = [1, 3, 9, 27]
        assert_relative_eq!(vand.get(&[2, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[2, 1])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[2, 2])?, 9.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[2, 3])?, 27.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_vander_decreasing() -> TorshResult<()> {
        let x = Tensor::from_data(vec![2.0f32, 3.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let vand = vander(&x, Some(3), false)?;

        // Should create 2x3 Vandermonde matrix with decreasing powers
        assert_eq!(vand.shape().dims(), &[2, 3]);

        // First row: [2^2, 2^1, 2^0] = [4, 2, 1]
        assert_relative_eq!(vand.get(&[0, 0])?, 4.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[0, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[0, 2])?, 1.0, epsilon = 1e-6);

        // Second row: [3^2, 3^1, 3^0] = [9, 3, 1]
        assert_relative_eq!(vand.get(&[1, 0])?, 9.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[1, 1])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(vand.get(&[1, 2])?, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_vander_default_size() -> TorshResult<()> {
        let x = Tensor::from_data(vec![1.0f32, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let vand = vander(&x, None, true)?;

        // Should create 2x2 matrix (default n = len(x))
        assert_eq!(vand.shape().dims(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_toeplitz_symmetric() -> TorshResult<()> {
        let c = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let toep = toeplitz(&c, None)?;

        // Should create symmetric 3x3 Toeplitz matrix
        assert_eq!(toep.shape().dims(), &[3, 3]);

        // Verify structure: [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
        assert_relative_eq!(toep.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[0, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[0, 2])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[1, 0])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[1, 1])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[1, 2])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[2, 0])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[2, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[2, 2])?, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_toeplitz_nonsymmetric() -> TorshResult<()> {
        let c = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let r = Tensor::from_data(
            vec![1.0f32, 4.0, 5.0, 6.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;
        let toep = toeplitz(&c, Some(&r))?;

        // Should create 3x4 Toeplitz matrix
        assert_eq!(toep.shape().dims(), &[3, 4]);

        // Verify structure
        // First row should be r: [1, 4, 5, 6]
        assert_relative_eq!(toep.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[0, 1])?, 4.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[0, 2])?, 5.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[0, 3])?, 6.0, epsilon = 1e-6);

        // First column should be c: [1, 2, 3]
        assert_relative_eq!(toep.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[1, 0])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(toep.get(&[2, 0])?, 3.0, epsilon = 1e-6);

        // Check some other elements
        assert_relative_eq!(toep.get(&[1, 1])?, 1.0, epsilon = 1e-6); // Same diagonal as [0,0]
        assert_relative_eq!(toep.get(&[2, 1])?, 2.0, epsilon = 1e-6); // Same diagonal as [1,0]

        Ok(())
    }

    #[test]
    fn test_hankel_default() -> TorshResult<()> {
        let c = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let hank = hankel(&c, None)?;

        // Should create 3x3 Hankel matrix
        assert_eq!(hank.shape().dims(), &[3, 3]);

        // Verify structure: [[1, 2, 3], [2, 3, 0], [3, 0, 0]]
        assert_relative_eq!(hank.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[0, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[0, 2])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 0])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 1])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 2])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[2, 0])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[2, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[2, 2])?, 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_hankel_with_last_row() -> TorshResult<()> {
        let c = Tensor::from_data(vec![1.0f32, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let r = Tensor::from_data(vec![2.0f32, 3.0, 4.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let hank = hankel(&c, Some(&r))?;

        // Should create 2x3 Hankel matrix
        assert_eq!(hank.shape().dims(), &[2, 3]);

        // Verify structure: [[1, 2, 3], [2, 3, 4]]
        assert_relative_eq!(hank.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[0, 1])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[0, 2])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 0])?, 2.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 1])?, 3.0, epsilon = 1e-6);
        assert_relative_eq!(hank.get(&[1, 2])?, 4.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_error_cases() -> TorshResult<()> {
        // Test 3D tensor for diag
        let tensor3d = torsh_tensor::creation::zeros::<f32>(&[2, 2, 2])?;
        assert!(diag(&tensor3d, 0).is_err());

        // Test 2D tensor for vander
        let tensor2d = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        assert!(vander(&tensor2d, None, true).is_err());

        // Test 2D tensor for toeplitz column
        assert!(toeplitz(&tensor2d, None).is_err());

        // Test 2D tensor for hankel column
        assert!(hankel(&tensor2d, None).is_err());

        // Test 2D tensor for toeplitz row
        let vec1d = torsh_tensor::creation::zeros::<f32>(&[3])?;
        assert!(toeplitz(&vec1d, Some(&tensor2d)).is_err());

        // Test 2D tensor for hankel row
        assert!(hankel(&vec1d, Some(&tensor2d)).is_err());

        Ok(())
    }

    #[test]
    fn test_diag_roundtrip() -> TorshResult<()> {
        // Test that creating a diagonal matrix and extracting its diagonal gives back original vector
        let vec = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let diag_mat = diag(&vec, 0)?;
        let extracted = diag(&diag_mat, 0)?;

        assert_eq!(extracted.shape().dims(), vec.shape().dims());
        for i in 0..3 {
            assert_relative_eq!(extracted.get(&[i])?, vec.get(&[i])?, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_empty_inputs() -> TorshResult<()> {
        // Test empty vector for diag
        let empty_vec = torsh_tensor::creation::zeros::<f32>(&[0])?;
        let diag_mat = diag(&empty_vec, 0)?;
        assert_eq!(diag_mat.shape().dims(), &[0, 0]);

        // Test empty vector for vander
        let vand = vander(&empty_vec, Some(2), true)?;
        assert_eq!(vand.shape().dims(), &[0, 2]);

        Ok(())
    }
}
