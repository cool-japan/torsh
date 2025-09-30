//! Sparse tensor operations

use crate::{CooTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::{creation::zeros, Tensor};

/// Consolidated helper functions for sparse operations
mod utils {
    use super::*;

    /// Validate that two tensors have the same shape
    pub fn validate_same_shape(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<()> {
        if a.shape() != b.shape() {
            return Err(TorshError::InvalidArgument(format!(
                "Shape mismatch: {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }
        Ok(())
    }

    /// Validate dimensions for matrix multiplication
    pub fn validate_matmul_dims(a: &dyn SparseTensor, b: &Tensor) -> TorshResult<()> {
        if b.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Right operand must be a 2D tensor".to_string(),
            ));
        }

        if a.shape().dims()[1] != b.shape().dims()[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension mismatch: [{} x {}] @ [{} x {}]",
                a.shape().dims()[0],
                a.shape().dims()[1],
                b.shape().dims()[0],
                b.shape().dims()[1]
            )));
        }
        Ok(())
    }

    /// Validate that a tensor is square
    pub fn validate_square(tensor: &dyn SparseTensor) -> TorshResult<()> {
        let shape = tensor.shape();
        if shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Matrix must be square".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate axis parameter for 2D operations
    pub fn validate_axis(axis: usize) -> TorshResult<()> {
        if axis > 1 {
            return Err(TorshError::InvalidArgument(
                "Axis must be 0 (rows) or 1 (columns)".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert tensor to COO with error handling
    pub fn to_coo_safe(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
        tensor.to_coo()
    }

    /// Create a HashMap from COO triplets for fast lookup
    pub fn create_position_map(coo: &CooTensor) -> HashMap<(usize, usize), f32> {
        let mut map = HashMap::new();
        for (row, col, val) in coo.triplets() {
            map.insert((row, col), val);
        }
        map
    }

    /// Extract triplets and filter by threshold
    pub fn extract_filtered_triplets(
        triplets: Vec<(usize, usize, f32)>,
        threshold: f32,
    ) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        triplets
            .into_iter()
            .filter(|(_, _, v)| v.abs() > threshold)
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            )
    }

    /// Common pattern for element-wise operations
    pub fn element_wise_operation<F>(
        a: &dyn SparseTensor,
        b: &dyn SparseTensor,
        op: F,
    ) -> TorshResult<CooTensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        validate_same_shape(a, b)?;

        let a_coo = to_coo_safe(a)?;
        let b_coo = to_coo_safe(b)?;
        let b_map = create_position_map(&b_coo);

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (row, col, a_val) in a_coo.triplets() {
            if let Some(&b_val) = b_map.get(&(row, col)) {
                let result = op(a_val, b_val);
                if result.abs() > f32::EPSILON {
                    row_indices.push(row);
                    col_indices.push(col);
                    values.push(result);
                }
            }
        }

        CooTensor::new(row_indices, col_indices, values, a.shape().clone())
    }

    /// Common pattern for reduction operations
    pub fn reduce_operation<F>(
        tensor: &dyn SparseTensor,
        axis: Option<usize>,
        op: F,
    ) -> TorshResult<Vec<f32>>
    where
        F: Fn(&mut Vec<f32>, usize, usize, f32),
    {
        let coo = to_coo_safe(tensor)?;
        let shape = tensor.shape();

        match axis {
            None => {
                // Global reduction
                let mut result = vec![0.0; 1];
                for (row, col, val) in coo.triplets() {
                    op(&mut result, row, col, val);
                }
                Ok(result)
            }
            Some(0) => {
                // Sum across rows (result has shape [cols])
                let cols = shape.dims()[1];
                let mut result = vec![0.0; cols];
                for (row, col, val) in coo.triplets() {
                    op(&mut result, row, col, val);
                }
                Ok(result)
            }
            Some(1) => {
                // Sum across columns (result has shape [rows])
                let rows = shape.dims()[0];
                let mut result = vec![0.0; rows];
                for (row, col, val) in coo.triplets() {
                    op(&mut result, row, col, val);
                }
                Ok(result)
            }
            Some(invalid_axis) => {
                validate_axis(invalid_axis)?;
                panic!("Invalid axis: {invalid_axis}. This should be unreachable after validate_axis check.")
            }
        }
    }
}

/// Sparse matrix multiplication
pub fn spmm(a: &dyn SparseTensor, b: &Tensor) -> TorshResult<Tensor> {
    // Validate dimensions
    utils::validate_matmul_dims(a, b)?;

    // Convert to CSR for efficient row access
    let a_csr = a.to_csr()?;

    // Perform sparse-dense matrix multiplication
    let m = a.shape().dims()[0];
    let n = b.shape().dims()[1];
    let result = zeros::<f32>(&[m, n])?;

    // For each row in sparse matrix
    for i in 0..m {
        let (cols, vals) = a_csr.get_row(i)?;

        // For each column in dense matrix
        for j in 0..n {
            let mut sum = 0.0;

            // Compute dot product of sparse row with dense column
            for (k, &col) in cols.iter().enumerate() {
                sum += vals[k] * b.get(&[col, j])?;
            }

            result.set(&[i, j], sum)?;
        }
    }

    Ok(result)
}

/// Sparse matrix addition
pub fn spadd(a: &dyn SparseTensor, b: &dyn SparseTensor, alpha: f32) -> TorshResult<CooTensor> {
    // Validate shapes
    utils::validate_same_shape(a, b)?;

    // Convert both to COO for easier manipulation
    let a_coo = utils::to_coo_safe(a)?;
    let b_coo = utils::to_coo_safe(b)?;

    // Get all triplets
    let mut triplets = a_coo.triplets();

    // Add scaled b triplets
    for (row, col, val) in b_coo.triplets() {
        // Check if this position already exists in a
        let mut found = false;
        for t in triplets.iter_mut() {
            if t.0 == row && t.1 == col {
                t.2 += alpha * val;
                found = true;
                break;
            }
        }

        // If not found, add new triplet
        if !found {
            triplets.push((row, col, alpha * val));
        }
    }

    // Extract indices and values
    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, a.shape().clone())
}

/// Element-wise multiplication of sparse tensors
pub fn sphadamard(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    utils::element_wise_operation(a, b, |a_val, b_val| a_val * b_val)
}

/// Transpose a sparse tensor
pub fn transpose(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;
    Ok(coo.transpose())
}

/// Sum all elements in a sparse tensor
pub fn sum(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let result = utils::reduce_operation(tensor, None, |result, _, _, val| {
        result[0] += val;
    })?;
    Ok(result[0])
}

/// Sum along axis (0 for rows, 1 for columns)
pub fn sum_axis(tensor: &dyn SparseTensor, axis: usize) -> TorshResult<Vec<f32>> {
    utils::validate_axis(axis)?;

    utils::reduce_operation(tensor, Some(axis), |result, row, col, val| {
        match axis {
            0 => result[col] += val, // Sum across rows -> index by col
            1 => result[row] += val, // Sum across columns -> index by row
            _ => panic!(
                "Invalid axis: {axis}. This should be unreachable after validate_axis check."
            ),
        }
    })
}

/// Compute L2 norm of sparse tensor
pub fn norm(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let result = utils::reduce_operation(tensor, None, |result, _, _, val| {
        result[0] += val * val;
    })?;
    Ok(result[0].sqrt())
}

/// Scale sparse tensor by scalar
pub fn scale(tensor: &dyn SparseTensor, scalar: f32) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;
    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| (r, c, v * scalar))
        .collect();

    let (row_indices, col_indices, values) = utils::extract_filtered_triplets(triplets, 0.0);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Get diagonal elements
pub fn diag(tensor: &dyn SparseTensor) -> TorshResult<Vec<f32>> {
    utils::validate_square(tensor)?;

    let n = tensor.shape().dims()[0];
    let mut diagonal = vec![0.0; n];

    let coo = utils::to_coo_safe(tensor)?;
    for (row, col, val) in coo.triplets() {
        if row == col {
            diagonal[row] = val;
        }
    }

    Ok(diagonal)
}

/// Additional consolidated operations enabled by the utilities
/// Element-wise addition of sparse tensors (uses consolidated element_wise_operation)
pub fn element_add(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    utils::element_wise_operation(a, b, |a_val, b_val| a_val + b_val)
}

/// Element-wise subtraction of sparse tensors
pub fn element_sub(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    utils::element_wise_operation(a, b, |a_val, b_val| a_val - b_val)
}

/// Element-wise division of sparse tensors
pub fn element_div(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    utils::element_wise_operation(a, b, |a_val, b_val| {
        if b_val.abs() < f32::EPSILON {
            0.0 // Handle division by zero
        } else {
            a_val / b_val
        }
    })
}

/// Compute mean along axis
pub fn mean_axis(tensor: &dyn SparseTensor, axis: usize) -> TorshResult<Vec<f32>> {
    let sums = sum_axis(tensor, axis)?;
    let shape = tensor.shape();
    let divisor = if axis == 0 {
        shape.dims()[0]
    } else {
        shape.dims()[1]
    };

    Ok(sums.into_iter().map(|s| s / divisor as f32).collect())
}

/// Compute variance along axis
pub fn var_axis(tensor: &dyn SparseTensor, axis: usize) -> TorshResult<Vec<f32>> {
    let means = mean_axis(tensor, axis)?;
    utils::validate_axis(axis)?;

    let result = utils::reduce_operation(tensor, Some(axis), |result, row, col, val| {
        let idx = if axis == 0 { col } else { row };
        let diff = val - means[idx];
        result[idx] += diff * diff;
    })?;

    let shape = tensor.shape();
    let divisor = if axis == 0 {
        shape.dims()[0]
    } else {
        shape.dims()[1]
    };

    Ok(result.into_iter().map(|v| v / divisor as f32).collect())
}

/// Compute max absolute value
pub fn max_abs(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let result = utils::reduce_operation(tensor, None, |result, _, _, val| {
        result[0] = result[0].max(val.abs());
    })?;
    Ok(result[0])
}

/// Sparse-sparse matrix multiplication (basic implementation)
pub fn sparse_matmul(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    // Use the optimized implementation
    sparse_matmul_optimized(a, b)
}

/// Optimized sparse-sparse matrix multiplication using CSR format
pub fn sparse_matmul_optimized(
    a: &dyn SparseTensor,
    b: &dyn SparseTensor,
) -> TorshResult<CooTensor> {
    // Validate dimensions
    if a.shape().dims()[1] != b.shape().dims()[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: [{} x {}] @ [{} x {}]",
            a.shape().dims()[0],
            a.shape().dims()[1],
            b.shape().dims()[0],
            b.shape().dims()[1]
        )));
    }

    // Convert A to CSR and B to CSC for optimal access patterns
    let a_csr = a.to_csr()?;
    let b_csc = b.to_csc()?;

    let m = a.shape().dims()[0]; // rows in result
    let n = b.shape().dims()[1]; // cols in result

    // Use HashMap for accumulating results efficiently
    let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

    // For each row in A
    for i in 0..m {
        let (a_cols, a_vals) = a_csr.get_row(i)?;

        // For each column in B
        for j in 0..n {
            let (b_rows, b_vals) = b_csc.get_col(j)?;

            // Compute dot product between A's row i and B's column j
            let mut dot_product = 0.0;
            let (mut a_idx, mut b_idx) = (0, 0);

            // Use two-pointer technique for sorted arrays
            while a_idx < a_cols.len() && b_idx < b_rows.len() {
                if a_cols[a_idx] == b_rows[b_idx] {
                    dot_product += a_vals[a_idx] * b_vals[b_idx];
                    a_idx += 1;
                    b_idx += 1;
                } else if a_cols[a_idx] < b_rows[b_idx] {
                    a_idx += 1;
                } else {
                    b_idx += 1;
                }
            }

            // Store non-zero results
            if dot_product.abs() > f32::EPSILON {
                result_map.insert((i, j), dot_product);
            }
        }
    }

    // Convert result map to COO format
    let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) = result_map.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut rows, mut cols, mut vals), ((r, c), v)| {
            rows.push(r);
            cols.push(c);
            vals.push(v);
            (rows, cols, vals)
        },
    );

    let result_shape = crate::Shape::new(vec![m, n]);
    CooTensor::new(row_indices, col_indices, values, result_shape)
}

/// Block-based sparse matrix multiplication for better cache performance
pub fn sparse_matmul_blocked(
    a: &dyn SparseTensor,
    b: &dyn SparseTensor,
    block_size: Option<usize>,
) -> TorshResult<CooTensor> {
    let block_size = block_size.unwrap_or(64);

    // Validate dimensions
    if a.shape().dims()[1] != b.shape().dims()[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: [{} x {}] @ [{} x {}]",
            a.shape().dims()[0],
            a.shape().dims()[1],
            b.shape().dims()[0],
            b.shape().dims()[1]
        )));
    }

    let m = a.shape().dims()[0];
    let n = b.shape().dims()[1];
    let k = a.shape().dims()[1];

    // Convert to efficient formats
    let a_csr = a.to_csr()?;
    let b_csc = b.to_csc()?;

    let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

    // Block-wise multiplication for better cache locality
    for i_block in (0..m).step_by(block_size) {
        for j_block in (0..n).step_by(block_size) {
            for k_block in (0..k).step_by(block_size) {
                // Define block boundaries
                let i_end = std::cmp::min(i_block + block_size, m);
                let j_end = std::cmp::min(j_block + block_size, n);
                let k_end = std::cmp::min(k_block + block_size, k);

                // Process block
                for i in i_block..i_end {
                    let (a_cols, a_vals) = a_csr.get_row(i)?;

                    for j in j_block..j_end {
                        let (b_rows, b_vals) = b_csc.get_col(j)?;

                        // Compute partial dot product for this block
                        let mut partial_sum = 0.0;
                        let (mut a_idx, mut b_idx) = (0, 0);

                        while a_idx < a_cols.len() && b_idx < b_rows.len() {
                            let a_col = a_cols[a_idx];
                            let b_row = b_rows[b_idx];

                            // Only consider elements in the current k-block
                            if a_col < k_block || b_row < k_block {
                                if a_col < b_row {
                                    a_idx += 1;
                                } else {
                                    b_idx += 1;
                                }
                                continue;
                            }

                            if a_col >= k_end || b_row >= k_end {
                                break;
                            }

                            if a_col == b_row {
                                partial_sum += a_vals[a_idx] * b_vals[b_idx];
                                a_idx += 1;
                                b_idx += 1;
                            } else if a_col < b_row {
                                a_idx += 1;
                            } else {
                                b_idx += 1;
                            }
                        }

                        // Accumulate partial results
                        if partial_sum.abs() > f32::EPSILON {
                            *result_map.entry((i, j)).or_insert(0.0) += partial_sum;
                        }
                    }
                }
            }
        }
    }

    // Filter out near-zero results and convert to COO
    let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) = result_map
        .into_iter()
        .filter(|(_, v)| v.abs() > f32::EPSILON)
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rows, mut cols, mut vals), ((r, c), v)| {
                rows.push(r);
                cols.push(c);
                vals.push(v);
                (rows, cols, vals)
            },
        );

    let result_shape = crate::Shape::new(vec![m, n]);
    CooTensor::new(row_indices, col_indices, values, result_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_spmm() {
        // Create sparse matrix
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 2.0, 3.0];
        let shape = crate::Shape::new(vec![3, 3]);
        let sparse = CooTensor::new(row_indices, col_indices, values, shape).unwrap();

        // Create dense matrix
        let dense = ones::<f32>(&[3, 2]).unwrap();

        // Multiply
        let result = spmm(&sparse as &dyn SparseTensor, &dense).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_spadd() {
        // Create two sparse matrices
        let a = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        let b = CooTensor::new(
            vec![0, 1],
            vec![1, 0],
            vec![3.0, 4.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // Add with alpha=0.5
        let result = spadd(&a as &dyn SparseTensor, &b as &dyn SparseTensor, 0.5).unwrap();

        assert_eq!(result.nnz(), 4); // All positions are filled
    }

    #[test]
    fn test_sum() {
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            crate::Shape::new(vec![3, 3]),
        )
        .unwrap();

        let total = sum(&coo as &dyn SparseTensor).unwrap();
        assert_eq!(total, 6.0);
    }

    #[test]
    fn test_sum_axis() {
        let coo = CooTensor::new(
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // Sum across rows (axis=0)
        let col_sums = sum_axis(&coo as &dyn SparseTensor, 0).unwrap();
        assert_eq!(col_sums, vec![4.0, 6.0]); // [1+3, 2+4]

        // Sum across columns (axis=1)
        let row_sums = sum_axis(&coo as &dyn SparseTensor, 1).unwrap();
        assert_eq!(row_sums, vec![3.0, 7.0]); // [1+2, 3+4]
    }

    #[test]
    fn test_norm() {
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![3.0, 4.0, 0.0],
            crate::Shape::new(vec![3, 3]),
        )
        .unwrap();

        let l2_norm = norm(&coo as &dyn SparseTensor).unwrap();
        assert_eq!(l2_norm, 5.0); // sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5
    }

    #[test]
    fn test_scale() {
        let coo = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![2.0, 4.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        let scaled = scale(&coo as &dyn SparseTensor, 0.5).unwrap();
        let triplets = scaled.triplets();

        assert_eq!(triplets.len(), 2);
        assert_eq!(triplets[0].2, 1.0); // 2.0 * 0.5
        assert_eq!(triplets[1].2, 2.0); // 4.0 * 0.5
    }

    #[test]
    fn test_diag() {
        let coo = CooTensor::new(
            vec![0, 0, 1, 1, 2],
            vec![0, 1, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            crate::Shape::new(vec![3, 3]),
        )
        .unwrap();

        let diagonal = diag(&coo as &dyn SparseTensor).unwrap();
        assert_eq!(diagonal, vec![1.0, 3.0, 5.0]);
    }
}
