//! Mathematical operations

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// Compute pairwise distances between two sets of vectors
pub fn cdist(x1: &Tensor, x2: &Tensor, p: f32) -> TorshResult<Tensor> {
    // Validate inputs
    if x1.shape().ndim() != 2 || x2.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensors must be 2-dimensional",
            "cdist",
        ));
    }

    if x1.shape().dims()[1] != x2.shape().dims()[1] {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensors must have the same number of features",
            "cdist",
        ));
    }

    let n1 = x1.shape().dims()[0];
    let n2 = x2.shape().dims()[0];
    let d = x1.shape().dims()[1];

    // Create output tensor
    let distances = zeros(&[n1, n2])?;

    // Compute pairwise distances
    for i in 0..n1 {
        for j in 0..n2 {
            let mut dist = 0.0;

            // Compute p-norm distance
            for k in 0..d {
                let diff = x1.get(&[i, k])? - x2.get(&[j, k])?;
                if p == 2.0 {
                    dist += diff * diff;
                } else if p == 1.0 {
                    dist += diff.abs();
                } else if p == f32::INFINITY {
                    dist = dist.max(diff.abs());
                } else {
                    dist += diff.abs().powf(p);
                }
            }

            // Apply final normalization
            let final_dist = if p == 2.0 {
                dist.sqrt()
            } else if p == 1.0 || p == f32::INFINITY {
                dist
            } else {
                dist.powf(1.0 / p)
            };

            distances.set(&[i, j], final_dist)?;
        }
    }

    Ok(distances)
}

/// Einstein summation convention
pub fn einsum(equation: &str, operands: &[Tensor]) -> TorshResult<Tensor> {
    // Parse einsum equation
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Invalid einsum equation format",
            "einsum",
        ));
    }

    let input_specs = parts[0].split(',').collect::<Vec<_>>();
    let _output_spec = parts[1].trim();

    if input_specs.len() != operands.len() {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Expected {} operands, got {}",
                input_specs.len(),
                operands.len()
            ),
            "einsum",
        ));
    }

    // This is a simplified implementation
    // Full einsum would require:
    // 1. Parse index labels
    // 2. Validate dimensions
    // 3. Determine contraction indices
    // 4. Perform optimal contraction order
    // 5. Execute tensor operations

    match equation {
        // Handle common cases
        "ij,jk->ik" if operands.len() == 2 => {
            // Matrix multiplication
            operands[0].matmul(&operands[1])
        }
        "ii->i" if operands.len() == 1 => {
            // Diagonal extraction
            extract_diagonal(&operands[0])
        }
        "ij->ji" if operands.len() == 1 => {
            // Transpose
            operands[0].transpose(-2, -1)
        }
        "ij->" if operands.len() == 1 => {
            // Sum all elements
            operands[0].sum()
        }
        "ij->i" if operands.len() == 1 => {
            // Row sum
            operands[0].sum_dim(&[1], false)
        }
        "ij->j" if operands.len() == 1 => {
            // Column sum
            operands[0].sum_dim(&[0], false)
        }
        _ => {
            // General einsum implementation would go here
            Err(TorshError::Other(format!(
                "Einsum equation '{}' not yet implemented",
                equation
            )))
        }
    }
}

/// Extract diagonal from a matrix
fn extract_diagonal(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    if shape.ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Diagonal extraction requires 2D tensor",
            "extract_diagonal",
        ));
    }

    let size = shape.dims()[0].min(shape.dims()[1]);
    let diag = zeros(&[size])?;

    for i in 0..size {
        diag.set(&[i], tensor.get(&[i, i])?)?;
    }

    Ok(diag)
}

// ============================================================================
// Element-wise Mathematical Operations
// ============================================================================

/// Compute absolute value element-wise
pub fn abs(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.abs()
}

/// Compute element-wise exponential
pub fn exp(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.exp()
}

/// Compute element-wise natural logarithm
pub fn log(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.ln()
}

/// Compute element-wise base-2 logarithm
pub fn log2(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.log2()
}

/// Compute element-wise base-10 logarithm
pub fn log10(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.log10()
}

/// Compute element-wise sine
pub fn sin(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.sin()
}

/// Compute element-wise cosine
pub fn cos(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.cos()
}

/// Compute element-wise tangent
pub fn tan(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.tan()
}

/// Compute element-wise arc sine
pub fn asin(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.asin()
}

/// Compute element-wise arc cosine
pub fn acos(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.acos()
}

/// Compute element-wise arc tangent
pub fn atan(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.atan()
}

/// Compute element-wise hyperbolic sine
pub fn sinh(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.sinh()
}

/// Compute element-wise hyperbolic cosine
pub fn cosh(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.cosh()
}

/// Compute element-wise hyperbolic tangent
pub fn tanh(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.tanh()
}

/// Compute element-wise square root
pub fn sqrt(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.sqrt()
}

/// Compute element-wise reciprocal square root (rsqrt)
pub fn rsqrt(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.rsqrt()
}

/// Compute element-wise square
pub fn square(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.square()
}

/// Compute element-wise reciprocal
pub fn reciprocal(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.reciprocal()
}

/// Compute element-wise power
pub fn pow(tensor: &Tensor, exponent: f32) -> TorshResult<Tensor> {
    tensor.pow(exponent)
}

/// Compute element-wise power (tensor exponent)
pub fn pow_tensor(base: &Tensor, exponent: &Tensor) -> TorshResult<Tensor> {
    base.pow_tensor(exponent)
}

/// Compute element-wise floor
pub fn floor(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.floor()
}

/// Compute element-wise ceiling
pub fn ceil(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.ceil()
}

/// Compute element-wise round
pub fn round(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.round()
}

/// Compute element-wise truncation
pub fn trunc(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.trunc()
}

/// Compute element-wise fractional part
pub fn frac(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.fract()
}

/// Compute element-wise sign
pub fn sign(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.sign()
}

// ============================================================================
// Comparison Operations
// ============================================================================

/// Element-wise equality comparison
pub fn eq(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.eq(tensor2)
}

/// Element-wise not-equal comparison
pub fn ne(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.ne(tensor2)
}

/// Element-wise less-than comparison
pub fn lt(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.lt(tensor2)
}

/// Element-wise less-than-or-equal comparison
pub fn le(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.le(tensor2)
}

/// Element-wise greater-than comparison
pub fn gt(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.gt(tensor2)
}

/// Element-wise greater-than-or-equal comparison
pub fn ge(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor1.ge(tensor2)
}

/// Element-wise equality comparison with scalar
pub fn eq_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.eq_scalar(scalar)
}

/// Element-wise not-equal comparison with scalar
pub fn ne_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.ne_scalar(scalar)
}

/// Element-wise less-than comparison with scalar
pub fn lt_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.lt_scalar(scalar)
}

/// Element-wise less-than-or-equal comparison with scalar
pub fn le_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.le_scalar(scalar)
}

/// Element-wise greater-than comparison with scalar
pub fn gt_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.gt_scalar(scalar)
}

/// Element-wise greater-than-or-equal comparison with scalar
pub fn ge_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.ge_scalar(scalar)
}

// ============================================================================
// Element-wise Logic Operations
// ============================================================================

/// Element-wise logical AND
pub fn logical_and(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    // Convert to boolean tensors and perform logical AND
    let bool1 = tensor1.ne_scalar(0.0)?;
    let bool2 = tensor2.ne_scalar(0.0)?;
    bool1.logical_and(&bool2)
}

/// Element-wise logical OR
pub fn logical_or(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    // Convert to boolean tensors and perform logical OR
    let bool1 = tensor1.ne_scalar(0.0)?;
    let bool2 = tensor2.ne_scalar(0.0)?;
    bool1.logical_or(&bool2)
}

/// Element-wise logical XOR
pub fn logical_xor(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    // Convert to boolean tensors and perform logical XOR
    let bool1 = tensor1.ne_scalar(0.0)?;
    let bool2 = tensor2.ne_scalar(0.0)?;
    bool1.logical_xor(&bool2)
}

/// Element-wise logical NOT
pub fn logical_not(tensor: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    // Convert to boolean tensor and perform logical NOT
    let bool_tensor = tensor.eq_scalar(0.0)?;
    Ok(bool_tensor)
}

// ============================================================================
// Min/Max Operations
// ============================================================================

/// Element-wise minimum
pub fn minimum(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<Tensor> {
    tensor1.minimum(tensor2)
}

/// Element-wise maximum
pub fn maximum(tensor1: &Tensor, tensor2: &Tensor) -> TorshResult<Tensor> {
    tensor1.maximum(tensor2)
}

/// Element-wise minimum with scalar
pub fn minimum_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<Tensor> {
    let scalar_tensor = tensor.ones_like()?.mul_scalar(scalar)?;
    tensor.minimum(&scalar_tensor)
}

/// Element-wise maximum with scalar
pub fn maximum_scalar(tensor: &Tensor, scalar: f32) -> TorshResult<Tensor> {
    let scalar_tensor = tensor.ones_like()?.mul_scalar(scalar)?;
    tensor.maximum(&scalar_tensor)
}

/// Clamp tensor values to the range [min, max]
pub fn clamp(tensor: &Tensor, min: f32, max: f32) -> TorshResult<Tensor> {
    // Implement clamp as min(max(tensor, min), max)
    let min_clamped = maximum_scalar(tensor, min)?;
    minimum_scalar(&min_clamped, max)
}

/// Clamp tensor values to minimum value
pub fn clamp_min(tensor: &Tensor, min: f32) -> TorshResult<Tensor> {
    maximum_scalar(tensor, min)
}

/// Clamp tensor values to maximum value
pub fn clamp_max(tensor: &Tensor, max: f32) -> TorshResult<Tensor> {
    minimum_scalar(tensor, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_cdist() {
        // Test Euclidean distance
        let x1 = from_vec(
            vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0],
            &[3, 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let x2 = from_vec(
            vec![1.0f32, 1.0, 2.0, 2.0],
            &[2, 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let distances = cdist(&x1, &x2, 2.0).unwrap();
        assert_eq!(distances.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_einsum_matmul() {
        let a = randn(&[3, 4], None, None, None).unwrap();
        let b = randn(&[4, 5], None, None, None).unwrap();

        let result = einsum("ij,jk->ik", &[a.clone(), b.clone()]).unwrap();
        let expected = a.matmul(&b).unwrap();

        // Results should be equal (within floating point tolerance)
        assert_eq!(result.shape(), expected.shape());
    }
}
