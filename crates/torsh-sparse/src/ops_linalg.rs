//! Sparse linear algebra and advanced element-wise operations
//!
//! This module contains operations that require more advanced mathematical
//! constructs: triangular solvers, addcmul/addcdiv, masking, clamping,
//! and unary element-wise ops (abs, sign, pow, sqrt).

use super::*;

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
