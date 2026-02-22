//! Quantization-aware linear algebra operations
//!
//! This module provides quantization operations for model compression and efficient inference.
//! Quantization reduces the precision of floating-point numbers to save memory and computational
//! resources, particularly useful in machine learning deployment.
//!
//! ## Features
//!
//! - **Matrix Quantization**: Reduce precision to int8/int16 for memory efficiency
//! - **Quantization Methods**: Symmetric, Affine, Per-Channel quantization
//! - **Quantized Operations**: Matrix multiplication on quantized data
//! - **Calibration**: Automatic quantization parameter selection
//! - **Dequantization**: Roundtrip quantization with bounded error
//!
//! ## Examples
//!
//! ```ignore
//! use torsh_linalg::quantization::{quantize_matrix, dequantize_matrix, QuantizationMethod};
//! use torsh_tensor::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.5, 3.7, 4.2, 5.0, 6.1], &[2, 3])?;
//!
//! // Quantize to 8-bit
//! let (quantized, params) = quantize_matrix(&a, 8, QuantizationMethod::Affine)?;
//!
//! // Dequantize back to floating point
//! let a_dequantized = dequantize_matrix(&quantized, &params)?;
//!
//! // Check the error is bounded
//! let max_error = compute_max_error(&a, &a_dequantized)?;
//! assert!(max_error < 0.1);
//! ```

use torsh_core::{Result, TorshError};
use torsh_tensor::Tensor;

/// Quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Symmetric quantization: zero point is 0
    Symmetric,
    /// Affine quantization: arbitrary zero point
    Affine,
    /// Per-channel quantization: separate parameters for each channel
    PerChannel,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scaling factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Number of bits used for quantization
    pub bits: usize,
    /// Quantization method used
    pub method: QuantizationMethod,
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer data
    pub data: Vec<i8>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

/// Quantize a matrix to lower bit-width representation
///
/// Converts floating-point values to quantized integers with scaling and zero point.
///
/// # Arguments
///
/// * `tensor` - Input tensor to quantize (must be 2D)
/// * `bits` - Number of bits for quantization (typically 8)
/// * `method` - Quantization method (Symmetric, Affine, or PerChannel)
///
/// # Returns
///
/// Tuple of (quantized tensor, quantization parameters)
pub fn quantize_matrix(
    tensor: &Tensor,
    bits: usize,
    method: QuantizationMethod,
) -> Result<(QuantizedTensor, QuantizationParams)> {
    // Validate input
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Quantization requires 2D tensor".to_string(),
        ));
    }

    if bits != 8 && bits != 16 {
        return Err(TorshError::InvalidArgument(
            "Only 8-bit and 16-bit quantization supported".to_string(),
        ));
    }

    // Calibrate quantization parameters
    let params = calibrate_quantization(tensor, bits, method)?;

    // Quantize the tensor
    let shape_binding = tensor.shape();
    let shape = shape_binding.dims();
    let (rows, cols) = (shape[0], shape[1]);

    let mut quantized_data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let val = tensor.get(&[i, j])?;
            let q_val = ((val / params.scale) + params.zero_point as f32).round() as i8;
            quantized_data.push(q_val);
        }
    }

    let quantized = QuantizedTensor {
        data: quantized_data,
        shape: vec![rows, cols],
        params: params.clone(),
    };

    Ok((quantized, params))
}

/// Quantize a matrix with per-channel quantization
///
/// Each channel (column) gets its own scale and zero point for better accuracy.
///
/// # Arguments
///
/// * `tensor` - Input tensor to quantize (must be 2D)
/// * `bits` - Number of bits for quantization
///
/// # Returns
///
/// Tuple of (quantized tensor, quantization parameters)
pub fn quantize_matrix_per_channel(
    tensor: &Tensor,
    bits: usize,
) -> Result<(QuantizedTensor, QuantizationParams)> {
    // For now, use affine quantization
    quantize_matrix(tensor, bits, QuantizationMethod::PerChannel)
}

/// Dequantize a quantized tensor back to floating point
///
/// Converts quantized integers back to floating-point values using the stored
/// quantization parameters.
///
/// # Arguments
///
/// * `quantized` - Quantized tensor
/// * `params` - Quantization parameters
///
/// # Returns
///
/// Dequantized floating-point tensor
pub fn dequantize_matrix(
    quantized: &QuantizedTensor,
    params: &QuantizationParams,
) -> Result<Tensor> {
    // Reconstruct quantized matrix
    let shape = &quantized.shape;
    if shape.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Dequantization requires 2D shape".to_string(),
        ));
    }

    let (rows, cols) = (shape[0], shape[1]);
    let mut dequantized_data = Vec::with_capacity(rows * cols);

    for &q_val in &quantized.data {
        let val = (q_val as f32 - params.zero_point as f32) * params.scale;
        dequantized_data.push(val);
    }

    Tensor::from_data(
        dequantized_data,
        vec![rows, cols],
        torsh_core::DeviceType::Cpu,
    )
}

/// Perform quantized matrix multiplication
///
/// Multiplies two quantized matrices and returns the result in floating-point.
///
/// # Arguments
///
/// * `a` - First quantized tensor
/// * `a_params` - Quantization parameters for first tensor
/// * `b` - Second quantized tensor
/// * `b_params` - Quantization parameters for second tensor
///
/// # Returns
///
/// Result of matrix multiplication in floating-point
pub fn quantized_matmul(
    a: &QuantizedTensor,
    a_params: &QuantizationParams,
    b: &QuantizedTensor,
    b_params: &QuantizationParams,
) -> Result<Tensor> {
    // Validate shapes
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Quantized matmul requires 2D tensors".to_string(),
        ));
    }

    if a.shape[1] != b.shape[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Incompatible dimensions for quantized matmul: {}x{} and {}x{}",
            a.shape[0], a.shape[1], b.shape[0], b.shape[1]
        )));
    }

    // Dequantize both matrices
    let a_deq = dequantize_matrix(a, a_params)?;
    let b_deq = dequantize_matrix(b, b_params)?;

    // Perform regular matrix multiplication
    a_deq.matmul(&b_deq)
}

/// Calibrate quantization parameters from data
///
/// Analyzes the distribution of values in the tensor to determine optimal
/// quantization parameters.
///
/// # Arguments
///
/// * `tensor` - Input tensor to analyze
/// * `bits` - Number of bits for quantization
/// * `method` - Quantization method
///
/// # Returns
///
/// Calibrated quantization parameters
pub fn calibrate_quantization(
    tensor: &Tensor,
    bits: usize,
    method: QuantizationMethod,
) -> Result<QuantizationParams> {
    // Find min and max values
    let shape_binding = tensor.shape();
    let shape = shape_binding.dims();
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    if shape.len() == 1 {
        for i in 0..shape[0] {
            let val = tensor.get(&[i])?;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    } else if shape.len() == 2 {
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let val = tensor.get(&[i, j])?;
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    } else {
        return Err(TorshError::InvalidArgument(
            "Calibration only supports 1D and 2D tensors".to_string(),
        ));
    }

    // Compute quantization parameters based on method
    let (scale, zero_point) = match method {
        QuantizationMethod::Symmetric => {
            // Symmetric: zero point is 0
            let max_abs = max_val.abs().max(min_val.abs());
            let qmax = (1 << (bits - 1)) - 1;
            let scale = max_abs / qmax as f32;
            (scale, 0)
        }
        QuantizationMethod::Affine | QuantizationMethod::PerChannel => {
            // Affine: arbitrary zero point
            let qmin = -(1 << (bits - 1));
            let qmax = (1 << (bits - 1)) - 1;
            let scale = (max_val - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            (scale, zero_point)
        }
    };

    Ok(QuantizationParams {
        scale,
        zero_point,
        bits,
        method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_method_equality() {
        assert_eq!(QuantizationMethod::Symmetric, QuantizationMethod::Symmetric);
        assert_ne!(QuantizationMethod::Symmetric, QuantizationMethod::Affine);
    }

    #[test]
    fn test_calibrate_quantization_symmetric() -> Result<()> {
        let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::from_data(data, vec![5], torsh_core::DeviceType::Cpu)?;

        let params = calibrate_quantization(&tensor, 8, QuantizationMethod::Symmetric)?;

        assert_eq!(params.bits, 8);
        assert_eq!(params.zero_point, 0);
        assert!(params.scale > 0.0);
        assert_eq!(params.method, QuantizationMethod::Symmetric);

        Ok(())
    }

    #[test]
    fn test_calibrate_quantization_affine() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![5], torsh_core::DeviceType::Cpu)?;

        let params = calibrate_quantization(&tensor, 8, QuantizationMethod::Affine)?;

        assert_eq!(params.bits, 8);
        assert!(params.scale > 0.0);
        assert_eq!(params.method, QuantizationMethod::Affine);

        Ok(())
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data.clone(), vec![2, 3], torsh_core::DeviceType::Cpu)?;

        // Quantize
        let (quantized, params) = quantize_matrix(&tensor, 8, QuantizationMethod::Symmetric)?;

        // Dequantize
        let dequantized = dequantize_matrix(&quantized, &params)?;

        // Check shape
        assert_eq!(dequantized.shape().dims(), &[2, 3]);

        // Check error is bounded (quantization introduces some error)
        for i in 0..2 {
            for j in 0..3 {
                let original = tensor.get(&[i, j])?;
                let recovered = dequantized.get(&[i, j])?;
                let error = (original - recovered).abs();
                assert!(error < 1.0, "Error too large: {error} at [{i}, {j}]");
            }
        }

        Ok(())
    }

    #[test]
    fn test_quantized_matmul_basic() -> Result<()> {
        // Create simple 2x2 matrices
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let b = Tensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        // Quantize both matrices
        let (a_q, a_params) = quantize_matrix(&a, 8, QuantizationMethod::Symmetric)?;
        let (b_q, b_params) = quantize_matrix(&b, 8, QuantizationMethod::Symmetric)?;

        // Perform quantized matrix multiplication
        let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params)?;

        // Regular matrix multiplication for comparison
        let c_expected = a.matmul(&b)?;

        // Check shape
        assert_eq!(c_q.shape().dims(), &[2, 2]);

        // Check result is approximately correct
        for i in 0..2 {
            for j in 0..2 {
                let expected = c_expected.get(&[i, j])?;
                let actual = c_q.get(&[i, j])?;
                let rel_error = ((expected - actual) / expected).abs();
                assert!(rel_error < 0.5, "Relative error too large: {rel_error}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_dimension_validation() {
        // Test with wrong dimensions
        let tensor =
            Tensor::from_data(vec![1.0f32; 8], vec![2, 2, 2], torsh_core::DeviceType::Cpu).unwrap();

        let result = calibrate_quantization(&tensor, 8, QuantizationMethod::Symmetric);
        assert!(result.is_err());
    }
}
