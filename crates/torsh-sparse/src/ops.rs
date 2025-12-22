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

/// Compute minimum value in the sparse tensor (considering zeros)
pub fn min(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let coo = utils::to_coo_safe(tensor)?;
    let triplets = coo.triplets();

    if triplets.is_empty() {
        return Ok(0.0); // All zeros
    }

    // Check if there are any zero elements in the sparse tensor
    let total_elements = tensor.shape().dims().iter().product::<usize>();
    let nnz = triplets.len();

    let mut min_val = triplets[0].2;
    for (_, _, val) in &triplets {
        min_val = min_val.min(*val);
    }

    // If there are implicit zeros and min_val > 0, the minimum is 0
    if nnz < total_elements && min_val > 0.0 {
        Ok(0.0)
    } else {
        Ok(min_val)
    }
}

/// Compute maximum value in the sparse tensor (considering zeros)
pub fn max(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let coo = utils::to_coo_safe(tensor)?;
    let triplets = coo.triplets();

    if triplets.is_empty() {
        return Ok(0.0); // All zeros
    }

    // Check if there are any zero elements in the sparse tensor
    let total_elements = tensor.shape().dims().iter().product::<usize>();
    let nnz = triplets.len();

    let mut max_val = triplets[0].2;
    for (_, _, val) in &triplets {
        max_val = max_val.max(*val);
    }

    // If there are implicit zeros and max_val < 0, the maximum is 0
    if nnz < total_elements && max_val < 0.0 {
        Ok(0.0)
    } else {
        Ok(max_val)
    }
}

/// Compute overall mean of the sparse tensor (considering zeros)
pub fn mean(tensor: &dyn SparseTensor) -> TorshResult<f32> {
    let total_sum = sum(tensor)?;
    let total_elements = tensor.shape().dims().iter().product::<usize>();
    Ok(total_sum / total_elements as f32)
}

/// Compute standard deviation of the sparse tensor (considering zeros)
pub fn std(tensor: &dyn SparseTensor, ddof: usize) -> TorshResult<f32> {
    let mean_val = mean(tensor)?;
    let coo = utils::to_coo_safe(tensor)?;
    let triplets = coo.triplets();

    let total_elements = tensor.shape().dims().iter().product::<usize>();
    let nnz = triplets.len();

    // Calculate variance
    let mut sum_sq_diff = 0.0;

    // Sum of squared differences for non-zero elements
    for (_, _, val) in &triplets {
        let diff = val - mean_val;
        sum_sq_diff += diff * diff;
    }

    // Add contribution from implicit zeros
    let num_zeros = total_elements - nnz;
    sum_sq_diff += num_zeros as f32 * mean_val * mean_val;

    // Calculate standard deviation
    let variance = sum_sq_diff / (total_elements - ddof) as f32;
    Ok(variance.sqrt())
}

/// Compute standard deviation along an axis
pub fn std_axis(tensor: &dyn SparseTensor, axis: usize, ddof: usize) -> TorshResult<Vec<f32>> {
    utils::validate_axis(axis)?;

    let variances = var_axis(tensor, axis)?;

    // Adjust variance for degrees of freedom if needed
    let adjustment_factor = if ddof > 0 {
        let n = if axis == 0 {
            tensor.shape().dims()[0]
        } else {
            tensor.shape().dims()[1]
        };
        (n as f32) / ((n - ddof) as f32)
    } else {
        1.0
    };

    Ok(variances
        .iter()
        .map(|v| (v * adjustment_factor).sqrt())
        .collect())
}

/// Add two matrices and then multiply: result = alpha * (A @ B) + beta * C
/// This is a common operation in neural networks (similar to torch.addmm)
pub fn addmm(
    c: &dyn SparseTensor,
    a: &dyn SparseTensor,
    b: &dyn SparseTensor,
    alpha: f32,
    beta: f32,
) -> TorshResult<CooTensor> {
    // Compute A @ B
    let ab = sparse_matmul_optimized(a, b)?;

    // Scale by alpha
    let scaled_ab = scale(&ab as &dyn SparseTensor, alpha)?;

    // Scale C by beta
    let scaled_c = scale(c, beta)?;

    // Add the results
    element_add(
        &scaled_ab as &dyn SparseTensor,
        &scaled_c as &dyn SparseTensor,
    )
}

/// Compute sparse softmax along an axis (typically for neural networks)
/// For sparse tensors, this is tricky because softmax typically makes the result dense
/// This implementation computes row-wise softmax (axis=1)
pub fn sparse_softmax(tensor: &dyn SparseTensor, axis: usize) -> TorshResult<CooTensor> {
    utils::validate_axis(axis)?;

    let coo = utils::to_coo_safe(tensor)?;
    let triplets = coo.triplets();

    if axis == 1 {
        // Row-wise softmax
        let mut row_max: HashMap<usize, f32> = HashMap::new();
        let mut row_sum_exp: HashMap<usize, f32> = HashMap::new();

        // Find max value per row
        for (row, _, val) in &triplets {
            row_max
                .entry(*row)
                .and_modify(|m| *m = m.max(*val))
                .or_insert(*val);
        }

        // Compute sum of exp(val - max) per row (for numerical stability)
        let mut new_triplets = Vec::new();
        for (row, col, val) in &triplets {
            let max_val = row_max.get(row).unwrap_or(&0.0);
            let exp_val = (val - max_val).exp();
            new_triplets.push((*row, *col, exp_val));
            row_sum_exp
                .entry(*row)
                .and_modify(|s| *s += exp_val)
                .or_insert(exp_val);
        }

        // Normalize by sum
        let final_triplets: Vec<(usize, usize, f32)> = new_triplets
            .into_iter()
            .map(|(row, col, exp_val)| {
                let sum = row_sum_exp.get(&row).unwrap_or(&1.0);
                (row, col, exp_val / sum)
            })
            .collect();

        CooTensor::from_triplets(
            final_triplets,
            (tensor.shape().dims()[0], tensor.shape().dims()[1]),
        )
    } else {
        // Column-wise softmax
        let mut col_max: HashMap<usize, f32> = HashMap::new();
        let mut col_sum_exp: HashMap<usize, f32> = HashMap::new();

        // Find max value per column
        for (_, col, val) in &triplets {
            col_max
                .entry(*col)
                .and_modify(|m| *m = m.max(*val))
                .or_insert(*val);
        }

        // Compute sum of exp(val - max) per column
        let mut new_triplets = Vec::new();
        for (row, col, val) in &triplets {
            let max_val = col_max.get(col).unwrap_or(&0.0);
            let exp_val = (val - max_val).exp();
            new_triplets.push((*row, *col, exp_val));
            col_sum_exp
                .entry(*col)
                .and_modify(|s| *s += exp_val)
                .or_insert(exp_val);
        }

        // Normalize by sum
        let final_triplets: Vec<(usize, usize, f32)> = new_triplets
            .into_iter()
            .map(|(row, col, exp_val)| {
                let sum = col_sum_exp.get(&col).unwrap_or(&1.0);
                (row, col, exp_val / sum)
            })
            .collect();

        CooTensor::from_triplets(
            final_triplets,
            (tensor.shape().dims()[0], tensor.shape().dims()[1]),
        )
    }
}

/// Compute sparse log_softmax along an axis
/// This is more numerically stable than log(softmax(x))
pub fn sparse_log_softmax(tensor: &dyn SparseTensor, axis: usize) -> TorshResult<CooTensor> {
    utils::validate_axis(axis)?;

    let coo = utils::to_coo_safe(tensor)?;
    let triplets = coo.triplets();

    if axis == 1 {
        // Row-wise log_softmax
        let mut row_max: HashMap<usize, f32> = HashMap::new();
        let mut row_log_sum_exp: HashMap<usize, f32> = HashMap::new();

        // Find max value per row
        for (row, _, val) in &triplets {
            row_max
                .entry(*row)
                .and_modify(|m| *m = m.max(*val))
                .or_insert(*val);
        }

        // Compute log(sum(exp(val - max))) per row
        let mut exp_values = Vec::new();
        for (row, col, val) in &triplets {
            let max_val = row_max.get(row).unwrap_or(&0.0);
            let exp_val = (val - max_val).exp();
            exp_values.push((*row, *col, exp_val, *val, *max_val));
            row_log_sum_exp
                .entry(*row)
                .and_modify(|s| *s += exp_val)
                .or_insert(exp_val);
        }

        // Compute log_sum_exp for each row
        let mut row_lse: HashMap<usize, f32> = HashMap::new();
        for (row, sum_exp) in row_log_sum_exp {
            let max_val = row_max.get(&row).unwrap_or(&0.0);
            row_lse.insert(row, max_val + sum_exp.ln());
        }

        // Compute log_softmax: val - log_sum_exp
        let final_triplets: Vec<(usize, usize, f32)> = exp_values
            .into_iter()
            .map(|(row, col, _, val, _)| {
                let lse = row_lse.get(&row).unwrap_or(&0.0);
                (row, col, val - lse)
            })
            .collect();

        CooTensor::from_triplets(
            final_triplets,
            (tensor.shape().dims()[0], tensor.shape().dims()[1]),
        )
    } else {
        // Column-wise log_softmax
        let mut col_max: HashMap<usize, f32> = HashMap::new();
        let mut col_log_sum_exp: HashMap<usize, f32> = HashMap::new();

        // Find max value per column
        for (_, col, val) in &triplets {
            col_max
                .entry(*col)
                .and_modify(|m| *m = m.max(*val))
                .or_insert(*val);
        }

        // Compute log(sum(exp(val - max))) per column
        let mut exp_values = Vec::new();
        for (row, col, val) in &triplets {
            let max_val = col_max.get(col).unwrap_or(&0.0);
            let exp_val = (val - max_val).exp();
            exp_values.push((*row, *col, exp_val, *val, *max_val));
            col_log_sum_exp
                .entry(*col)
                .and_modify(|s| *s += exp_val)
                .or_insert(exp_val);
        }

        // Compute log_sum_exp for each column
        let mut col_lse: HashMap<usize, f32> = HashMap::new();
        for (col, sum_exp) in col_log_sum_exp {
            let max_val = col_max.get(&col).unwrap_or(&0.0);
            col_lse.insert(col, max_val + sum_exp.ln());
        }

        // Compute log_softmax: val - log_sum_exp
        let final_triplets: Vec<(usize, usize, f32)> = exp_values
            .into_iter()
            .map(|(row, col, _, val, _)| {
                let lse = col_lse.get(&col).unwrap_or(&0.0);
                (row, col, val - lse)
            })
            .collect();

        CooTensor::from_triplets(
            final_triplets,
            (tensor.shape().dims()[0], tensor.shape().dims()[1]),
        )
    }
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

/// Solve triangular system Ax = b where A is a sparse triangular matrix
///
/// # Arguments
/// * `a` - Triangular sparse matrix (either upper or lower triangular)
/// * `b` - Right-hand side vector or matrix
/// * `upper` - If true, A is upper triangular; if false, A is lower triangular
/// * `transpose` - If true, solve A^T x = b instead of Ax = b
///
/// # Returns
/// Solution x to the triangular system
pub fn triangular_solve(
    a: &dyn SparseTensor,
    b: &Tensor,
    upper: bool,
    transpose: bool,
) -> TorshResult<Tensor> {
    // Validate that matrix is square
    utils::validate_square(a)?;

    let n = a.shape().dims()[0];

    // Validate b dimensions
    if b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: matrix size {} but RHS size {}",
            n,
            b.shape().dims()[0]
        )));
    }

    // Check if b is a vector or matrix
    let is_vector = b.shape().ndim() == 1;
    let nrhs = if is_vector { 1 } else { b.shape().dims()[1] };

    // Convert to CSR for efficient row access
    let a_csr = a.to_csr()?;

    // Initialize solution
    let x = if is_vector {
        zeros::<f32>(&[n])?
    } else {
        zeros::<f32>(&[n, nrhs])?
    };

    // Solve based on triangular type and transpose flag
    match (upper, transpose) {
        (false, false) => {
            // Lower triangular: forward substitution
            for i in 0..n {
                for j in 0..nrhs {
                    let mut sum = if is_vector {
                        b.get(&[i])?
                    } else {
                        b.get(&[i, j])?
                    };

                    let (cols, vals) = a_csr.get_row(i)?;
                    for (k, &col) in cols.iter().enumerate() {
                        if col < i {
                            let x_val = if is_vector {
                                x.get(&[col])?
                            } else {
                                x.get(&[col, j])?
                            };
                            sum -= vals[k] * x_val;
                        } else if col == i {
                            // Diagonal element
                            if vals[k].abs() < f32::EPSILON {
                                return Err(TorshError::ComputeError(
                                    "Singular matrix: zero diagonal element".to_string(),
                                ));
                            }
                            sum /= vals[k];
                            break;
                        }
                    }

                    if is_vector {
                        x.set(&[i], sum)?;
                    } else {
                        x.set(&[i, j], sum)?;
                    }
                }
            }
        }
        (true, false) => {
            // Upper triangular: backward substitution
            for i in (0..n).rev() {
                for j in 0..nrhs {
                    let mut sum = if is_vector {
                        b.get(&[i])?
                    } else {
                        b.get(&[i, j])?
                    };

                    let (cols, vals) = a_csr.get_row(i)?;
                    for (k, &col) in cols.iter().enumerate() {
                        if col > i {
                            let x_val = if is_vector {
                                x.get(&[col])?
                            } else {
                                x.get(&[col, j])?
                            };
                            sum -= vals[k] * x_val;
                        } else if col == i {
                            // Diagonal element
                            if vals[k].abs() < f32::EPSILON {
                                return Err(TorshError::ComputeError(
                                    "Singular matrix: zero diagonal element".to_string(),
                                ));
                            }
                            sum /= vals[k];
                        }
                    }

                    if is_vector {
                        x.set(&[i], sum)?;
                    } else {
                        x.set(&[i, j], sum)?;
                    }
                }
            }
        }
        (false, true) => {
            // Lower triangular transposed (acts as upper): backward substitution on transpose
            // A^T x = b where A is lower triangular, so A^T is upper triangular
            let a_csc = a.to_csc()?; // Use CSC for efficient column access (rows of A^T)

            for i in (0..n).rev() {
                for j in 0..nrhs {
                    let mut sum = if is_vector {
                        b.get(&[i])?
                    } else {
                        b.get(&[i, j])?
                    };

                    let (rows, vals) = a_csc.get_col(i)?;
                    for (k, &row) in rows.iter().enumerate() {
                        if row > i {
                            let x_val = if is_vector {
                                x.get(&[row])?
                            } else {
                                x.get(&[row, j])?
                            };
                            sum -= vals[k] * x_val;
                        } else if row == i {
                            if vals[k].abs() < f32::EPSILON {
                                return Err(TorshError::ComputeError(
                                    "Singular matrix: zero diagonal element".to_string(),
                                ));
                            }
                            sum /= vals[k];
                        }
                    }

                    if is_vector {
                        x.set(&[i], sum)?;
                    } else {
                        x.set(&[i, j], sum)?;
                    }
                }
            }
        }
        (true, true) => {
            // Upper triangular transposed (acts as lower): forward substitution on transpose
            let a_csc = a.to_csc()?;

            for i in 0..n {
                for j in 0..nrhs {
                    let mut sum = if is_vector {
                        b.get(&[i])?
                    } else {
                        b.get(&[i, j])?
                    };

                    let (rows, vals) = a_csc.get_col(i)?;
                    for (k, &row) in rows.iter().enumerate() {
                        if row < i {
                            let x_val = if is_vector {
                                x.get(&[row])?
                            } else {
                                x.get(&[row, j])?
                            };
                            sum -= vals[k] * x_val;
                        } else if row == i {
                            if vals[k].abs() < f32::EPSILON {
                                return Err(TorshError::ComputeError(
                                    "Singular matrix: zero diagonal element".to_string(),
                                ));
                            }
                            sum /= vals[k];
                            break;
                        }
                    }

                    if is_vector {
                        x.set(&[i], sum)?;
                    } else {
                        x.set(&[i, j], sum)?;
                    }
                }
            }
        }
    }

    Ok(x)
}

/// Performs element-wise multiplication and addition: out = input + value * tensor1 * tensor2
/// PyTorch equivalent: torch.addcmul(input, tensor1, tensor2, value=1.0)
///
/// # Arguments
/// * `input` - Base sparse tensor
/// * `tensor1` - First multiplicand sparse tensor
/// * `tensor2` - Second multiplicand sparse tensor
/// * `value` - Scalar multiplier for the element-wise product
///
/// # Returns
/// Result of input + value * (tensor1 * tensor2) as COO tensor
pub fn addcmul(
    input: &dyn SparseTensor,
    tensor1: &dyn SparseTensor,
    tensor2: &dyn SparseTensor,
    value: f32,
) -> TorshResult<CooTensor> {
    // Validate all tensors have the same shape
    utils::validate_same_shape(input, tensor1)?;
    utils::validate_same_shape(input, tensor2)?;

    // Convert all to COO for element-wise operations
    let input_coo = utils::to_coo_safe(input)?;
    let tensor1_coo = utils::to_coo_safe(tensor1)?;
    let tensor2_coo = utils::to_coo_safe(tensor2)?;

    // Create position maps for efficient lookup
    let input_map = utils::create_position_map(&input_coo);
    let tensor1_map = utils::create_position_map(&tensor1_coo);
    let tensor2_map = utils::create_position_map(&tensor2_coo);

    // Compute result: input + value * tensor1 * tensor2
    let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

    // Add input values
    for ((row, col), val) in input_map {
        result_map.insert((row, col), val);
    }

    // Add value * tensor1 * tensor2 for positions where both tensor1 and tensor2 are non-zero
    for ((row, col), val1) in tensor1_map {
        if let Some(&val2) = tensor2_map.get(&(row, col)) {
            let product = value * val1 * val2;
            *result_map.entry((row, col)).or_insert(0.0) += product;
        }
    }

    // Convert result map to COO format
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

    CooTensor::new(row_indices, col_indices, values, input.shape().clone())
}

/// Performs element-wise division and addition: out = input + value * tensor1 / tensor2
/// PyTorch equivalent: torch.addcdiv(input, tensor1, tensor2, value=1.0)
///
/// # Arguments
/// * `input` - Base sparse tensor
/// * `tensor1` - Numerator sparse tensor
/// * `tensor2` - Denominator sparse tensor
/// * `value` - Scalar multiplier for the element-wise quotient
///
/// # Returns
/// Result of input + value * (tensor1 / tensor2) as COO tensor
///
/// # Notes
/// - Division by zero results in 0 (sparse semantics)
/// - Only positions where tensor2 is non-zero contribute to the result
pub fn addcdiv(
    input: &dyn SparseTensor,
    tensor1: &dyn SparseTensor,
    tensor2: &dyn SparseTensor,
    value: f32,
) -> TorshResult<CooTensor> {
    // Validate all tensors have the same shape
    utils::validate_same_shape(input, tensor1)?;
    utils::validate_same_shape(input, tensor2)?;

    // Convert all to COO for element-wise operations
    let input_coo = utils::to_coo_safe(input)?;
    let tensor1_coo = utils::to_coo_safe(tensor1)?;
    let tensor2_coo = utils::to_coo_safe(tensor2)?;

    // Create position maps for efficient lookup
    let input_map = utils::create_position_map(&input_coo);
    let tensor1_map = utils::create_position_map(&tensor1_coo);
    let tensor2_map = utils::create_position_map(&tensor2_coo);

    // Compute result: input + value * tensor1 / tensor2
    let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

    // Add input values
    for ((row, col), val) in input_map {
        result_map.insert((row, col), val);
    }

    // Add value * tensor1 / tensor2 for positions where both tensor1 and tensor2 are non-zero
    for ((row, col), val1) in tensor1_map {
        if let Some(&val2) = tensor2_map.get(&(row, col)) {
            if val2.abs() > f32::EPSILON {
                let quotient = value * val1 / val2;
                *result_map.entry((row, col)).or_insert(0.0) += quotient;
            }
        }
    }

    // Convert result map to COO format
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

    CooTensor::new(row_indices, col_indices, values, input.shape().clone())
}

/// Apply a conditional fill operation: replace values that satisfy a condition
/// PyTorch equivalent: torch.masked_fill(tensor, mask, value)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
/// * `condition` - Predicate function that returns true for values to be replaced
/// * `fill_value` - Value to use for replacement
///
/// # Returns
/// New sparse tensor with values replaced where condition is true
pub fn masked_fill<F>(
    tensor: &dyn SparseTensor,
    condition: F,
    fill_value: f32,
) -> TorshResult<CooTensor>
where
    F: Fn(f32) -> bool,
{
    let coo = utils::to_coo_safe(tensor)?;

    // Apply condition and replace matching values
    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| {
            if condition(v) {
                (r, c, fill_value)
            } else {
                (r, c, v)
            }
        })
        .collect();

    // Filter out zeros
    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Clamp sparse tensor values to a specified range
/// PyTorch equivalent: torch.clamp(tensor, min, max)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
/// * `min` - Optional minimum value (None means no lower bound)
/// * `max` - Optional maximum value (None means no upper bound)
///
/// # Returns
/// New sparse tensor with values clamped to [min, max]
pub fn clamp(
    tensor: &dyn SparseTensor,
    min: Option<f32>,
    max: Option<f32>,
) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;

    // Clamp values
    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, mut v)| {
            if let Some(min_val) = min {
                v = v.max(min_val);
            }
            if let Some(max_val) = max {
                v = v.min(max_val);
            }
            (r, c, v)
        })
        .collect();

    // Filter out zeros
    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Compute absolute value of sparse tensor elements
/// PyTorch equivalent: torch.abs(tensor)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
///
/// # Returns
/// New sparse tensor with absolute values
pub fn abs(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;

    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| (r, c, v.abs()))
        .collect();

    let (row_indices, col_indices, values) = utils::extract_filtered_triplets(triplets, 0.0);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Compute sign of sparse tensor elements
/// PyTorch equivalent: torch.sign(tensor)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
///
/// # Returns
/// New sparse tensor with signs (-1, 0, or 1)
///
/// # Notes
/// - sign(x) = -1 if x < 0
/// - sign(x) = 0 if x == 0
/// - sign(x) = 1 if x > 0
pub fn sign(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;

    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| {
            let sign_val = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            };
            (r, c, sign_val)
        })
        .collect();

    // Filter out zeros (sign(0) = 0)
    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Apply power operation element-wise: tensor^exponent
/// PyTorch equivalent: torch.pow(tensor, exponent)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
/// * `exponent` - Power to raise each element to
///
/// # Returns
/// New sparse tensor with each element raised to the power
pub fn pow(tensor: &dyn SparseTensor, exponent: f32) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;

    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| (r, c, v.powf(exponent)))
        .collect();

    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
}

/// Square each element of sparse tensor
/// PyTorch equivalent: torch.square(tensor) or tensor**2
///
/// # Arguments
/// * `tensor` - Input sparse tensor
///
/// # Returns
/// New sparse tensor with squared values
pub fn square(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    pow(tensor, 2.0)
}

/// Square root of each element of sparse tensor
/// PyTorch equivalent: torch.sqrt(tensor)
///
/// # Arguments
/// * `tensor` - Input sparse tensor
///
/// # Returns
/// New sparse tensor with square root values
///
/// # Notes
/// - Negative values will produce NaN (following PyTorch behavior)
pub fn sqrt(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = utils::to_coo_safe(tensor)?;

    let triplets: Vec<_> = coo
        .triplets()
        .into_iter()
        .map(|(r, c, v)| (r, c, v.sqrt()))
        .collect();

    let (row_indices, col_indices, values) =
        utils::extract_filtered_triplets(triplets, f32::EPSILON);

    CooTensor::new(row_indices, col_indices, values, tensor.shape().clone())
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

    #[test]
    fn test_triangular_solve_lower() {
        // Create a lower triangular matrix:
        // [2  0  0]
        // [1  3  0]
        // [2  1  4]
        let lower = CooTensor::new(
            vec![0, 1, 1, 2, 2, 2],
            vec![0, 0, 1, 0, 1, 2],
            vec![2.0, 1.0, 3.0, 2.0, 1.0, 4.0],
            crate::Shape::new(vec![3, 3]),
        )
        .unwrap();

        // Solve L x = b where b = [2, 7, 15]
        // Forward substitution:
        //   x[0] = 2/2 = 1
        //   x[1] = (7 - 1*1)/3 = 2
        //   x[2] = (15 - 2*1 - 1*2)/4 = 11/4 = 2.75
        let b = torsh_tensor::Tensor::from_vec(vec![2.0, 7.0, 15.0], &[3]).unwrap();

        let x = triangular_solve(&lower as &dyn SparseTensor, &b, false, false).unwrap();

        // Verify solution
        assert!((x.get(&[0]).unwrap() - 1.0).abs() < 1e-5);
        assert!((x.get(&[1]).unwrap() - 2.0).abs() < 1e-5);
        assert!((x.get(&[2]).unwrap() - 2.75).abs() < 1e-5);
    }

    #[test]
    fn test_triangular_solve_upper() {
        // Create an upper triangular matrix:
        // [2  1  2]
        // [0  3  1]
        // [0  0  4]
        let upper = CooTensor::new(
            vec![0, 0, 0, 1, 1, 2],
            vec![0, 1, 2, 1, 2, 2],
            vec![2.0, 1.0, 2.0, 3.0, 1.0, 4.0],
            crate::Shape::new(vec![3, 3]),
        )
        .unwrap();

        // Solve U x = b where b = [18, 15, 12]
        // Expected: x = [1, 2, 3]
        let b = torsh_tensor::Tensor::from_vec(vec![18.0, 15.0, 12.0], &[3]).unwrap();

        let x = triangular_solve(&upper as &dyn SparseTensor, &b, true, false).unwrap();

        // Verify solution
        assert!((x.get(&[0]).unwrap() - 1.0).abs() < 1e-5);
        assert!((x.get(&[1]).unwrap() - 2.0).abs() < 1e-5);
        assert!((x.get(&[2]).unwrap() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_addcmul() {
        // Create sparse tensors
        // input = [[1, 0], [0, 2]]
        let input = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // tensor1 = [[2, 3], [0, 0]]
        let tensor1 = CooTensor::new(
            vec![0, 0],
            vec![0, 1],
            vec![2.0, 3.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // tensor2 = [[4, 5], [0, 0]]
        let tensor2 = CooTensor::new(
            vec![0, 0],
            vec![0, 1],
            vec![4.0, 5.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // out = input + 0.5 * tensor1 * tensor2
        // out[0,0] = 1 + 0.5 * 2 * 4 = 1 + 4 = 5
        // out[0,1] = 0 + 0.5 * 3 * 5 = 7.5
        // out[1,1] = 2 + 0 = 2
        let result = addcmul(
            &input as &dyn SparseTensor,
            &tensor1 as &dyn SparseTensor,
            &tensor2 as &dyn SparseTensor,
            0.5,
        )
        .unwrap();

        // Check results
        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        assert!((result_map.get(&(0, 0)).unwrap() - 5.0).abs() < 1e-5);
        assert!((result_map.get(&(0, 1)).unwrap() - 7.5).abs() < 1e-5);
        assert!((result_map.get(&(1, 1)).unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_addcdiv() {
        // Create sparse tensors
        // input = [[1, 0], [0, 2]]
        let input = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // tensor1 = [[8, 10], [0, 0]]
        let tensor1 = CooTensor::new(
            vec![0, 0],
            vec![0, 1],
            vec![8.0, 10.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // tensor2 = [[2, 5], [0, 0]]
        let tensor2 = CooTensor::new(
            vec![0, 0],
            vec![0, 1],
            vec![2.0, 5.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // out = input + 0.5 * tensor1 / tensor2
        // out[0,0] = 1 + 0.5 * 8 / 2 = 1 + 2 = 3
        // out[0,1] = 0 + 0.5 * 10 / 5 = 1
        // out[1,1] = 2 + 0 = 2
        let result = addcdiv(
            &input as &dyn SparseTensor,
            &tensor1 as &dyn SparseTensor,
            &tensor2 as &dyn SparseTensor,
            0.5,
        )
        .unwrap();

        // Check results
        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        assert!((result_map.get(&(0, 0)).unwrap() - 3.0).abs() < 1e-5);
        assert!((result_map.get(&(0, 1)).unwrap() - 1.0).abs() < 1e-5);
        assert!((result_map.get(&(1, 1)).unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_masked_fill() {
        // Create sparse tensor
        let tensor = CooTensor::new(
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            vec![1.0, -2.0, 3.0, -4.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // Fill negative values with 0
        let result = masked_fill(&tensor as &dyn SparseTensor, |v| v < 0.0, 0.0).unwrap();

        // Check that negative values are replaced
        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        assert_eq!(result_map.get(&(0, 0)).unwrap(), &1.0); // unchanged
                                                            // (0, 1) should be filtered out as it becomes 0
        assert_eq!(result_map.get(&(1, 0)).unwrap(), &3.0); // unchanged
                                                            // (1, 1) should be filtered out as it becomes 0
        assert_eq!(result_map.len(), 2); // Only 2 non-zero elements remain
    }

    #[test]
    fn test_clamp() {
        let tensor = CooTensor::new(
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            vec![-2.0, 1.0, 5.0, 3.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        // Clamp values between 0 and 4
        let result = clamp(&tensor as &dyn SparseTensor, Some(0.0), Some(4.0)).unwrap();

        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        // (0, 0): -2.0 clamped to 0.0 -> filtered out as zero
        assert_eq!(result_map.get(&(0, 1)).unwrap(), &1.0); // unchanged
        assert_eq!(result_map.get(&(1, 0)).unwrap(), &4.0); // 5.0 clamped to 4.0
        assert_eq!(result_map.get(&(1, 1)).unwrap(), &3.0); // unchanged
    }

    #[test]
    fn test_abs_sparse() {
        let tensor = CooTensor::new(
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![-2.0, 3.0, -4.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        let result = abs(&tensor as &dyn SparseTensor).unwrap();

        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        assert_eq!(result_map.get(&(0, 0)).unwrap(), &2.0);
        assert_eq!(result_map.get(&(0, 1)).unwrap(), &3.0);
        assert_eq!(result_map.get(&(1, 1)).unwrap(), &4.0);
    }

    #[test]
    fn test_sign_sparse() {
        let tensor = CooTensor::new(
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![-2.0, 3.0, 0.0],
            crate::Shape::new(vec![2, 2]),
        )
        .unwrap();

        let result = sign(&tensor as &dyn SparseTensor).unwrap();

        let result_map: std::collections::HashMap<(usize, usize), f32> = result
            .triplets()
            .into_iter()
            .map(|(r, c, v)| ((r, c), v))
            .collect();

        assert_eq!(result_map.get(&(0, 0)).unwrap(), &-1.0);
        assert_eq!(result_map.get(&(0, 1)).unwrap(), &1.0);
        // (1, 1) is 0, so sign(0) = 0 -> filtered out
        assert_eq!(result_map.len(), 2);
    }
}
