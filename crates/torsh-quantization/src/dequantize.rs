//! Dequantization operations

use crate::TorshResult;
use scirs2_core::parallel_ops::*;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Dequantize a quantized tensor using affine quantization
/// Formula: x = scale * (q - zero_point)
pub fn dequantize_per_tensor_affine(
    quantized_tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<Tensor> {
    let data = quantized_tensor.data()?;

    // Dequantize each element: x = scale * (q - zero_point)
    // Use parallel processing for large tensors (>1000 elements)
    let dequantized_data: Vec<f32> = if data.len() > 1000 {
        data.par_iter()
            .map(|&q| scale * (q - zero_point as f32))
            .collect()
    } else {
        data.iter()
            .map(|&q| scale * (q - zero_point as f32))
            .collect()
    };

    let dequantized_tensor = Tensor::from_data(
        dequantized_data,
        quantized_tensor.shape().dims().to_vec(),
        quantized_tensor.device(),
    )?;
    Ok(dequantized_tensor)
}

/// Dequantize a tensor using symmetric quantization (zero_point = 0)
pub fn dequantize_per_tensor_symmetric(
    quantized_tensor: &Tensor,
    scale: f32,
) -> TorshResult<Tensor> {
    dequantize_per_tensor_affine(quantized_tensor, scale, 0)
}

/// Per-channel dequantization
pub fn dequantize_per_channel_affine(
    quantized_tensor: &Tensor,
    scales: &[f32],
    zero_points: &[i32],
    axis: usize,
) -> TorshResult<Tensor> {
    let data = quantized_tensor.data()?;
    let binding = quantized_tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let channel_size = shape[axis];
    if scales.len() != channel_size || zero_points.len() != channel_size {
        return Err(TorshError::InvalidArgument(
            "Scales and zero_points length must match channel size".to_string(),
        ));
    }

    // Calculate strides for the given axis
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut dequantized_data = vec![0.0; data.len()];

    for (idx, &q) in data.iter().enumerate() {
        // Calculate which channel this element belongs to
        let channel_idx = (idx / strides[axis]) % shape[axis];
        let scale = scales[channel_idx];
        let zero_point = zero_points[channel_idx];

        dequantized_data[idx] = scale * (q - zero_point as f32);
    }

    let dequantized_tensor =
        Tensor::from_data(dequantized_data, shape.to_vec(), quantized_tensor.device())?;
    Ok(dequantized_tensor)
}

/// Auto-dequantize a tensor (tries to extract quantization parameters)
pub fn dequantize_auto(
    quantized_tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<Tensor> {
    // Determine the best dequantization method based on tensor properties
    let numel = quantized_tensor.numel();

    // For small tensors, use simple per-tensor affine dequantization
    if numel <= 1000 {
        return dequantize_per_tensor_affine(quantized_tensor, scale, zero_point);
    }

    // For larger tensors, check if we can use symmetric dequantization
    // (when zero_point is 0, we can use faster symmetric dequantization)
    if zero_point == 0 {
        return dequantize_per_tensor_symmetric(quantized_tensor, scale);
    }

    // Default to per-tensor affine dequantization
    dequantize_per_tensor_affine(quantized_tensor, scale, zero_point)
}

/// Dequantize module parameters
// Temporarily disabled: pub fn dequantize_module(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn dequantize_module(module: &mut dyn crate::qat::Module) -> TorshResult<()> {
    // Iterate through quantized module parameters and dequantize them
    // This is a simplified implementation that demonstrates the concept

    // Get named parameters to track which parameters need dequantization
    let named_params = module.named_parameters();

    // For each parameter, check if it needs dequantization
    for (name, param) in named_params {
        // Check if the parameter appears to be quantized (simplified heuristic)
        // In practice, you'd store quantization metadata with the parameters
        let data = param.data()?;

        // Simple heuristic: if all values are integers in [-128, 127], it might be quantized
        let is_quantized = data
            .iter()
            .all(|&x| x.fract() == 0.0 && x >= -128.0 && x <= 127.0);

        if is_quantized && !data.is_empty() {
            // Use default scale and zero_point for dequantization
            // In practice, these would be stored with the quantized parameters
            let scale = 1.0 / 127.0; // Default scale for symmetric quantization
            let zero_point = 0; // Default zero point for symmetric quantization

            let _dequantized = dequantize_auto(param, scale, zero_point)?;

            // Note: In a real implementation, we would update the module's parameters
            // with the dequantized versions, but this requires more complex module handling

            println!("Would dequantize parameter: {}", name);
        }
    }

    Ok(())
}

/// Dequantize a batch of tensors
pub fn dequantize_batch(
    quantized_tensors: &[Tensor],
    scales: &[f32],
    zero_points: &[i32],
) -> TorshResult<Vec<Tensor>> {
    if quantized_tensors.len() != scales.len() || quantized_tensors.len() != zero_points.len() {
        return Err(TorshError::InvalidArgument(
            "Tensor count must match scales and zero_points length".to_string(),
        ));
    }

    let mut dequantized = Vec::with_capacity(quantized_tensors.len());

    for (i, tensor) in quantized_tensors.iter().enumerate() {
        let dequant = dequantize_per_tensor_affine(tensor, scales[i], zero_points[i])?;
        dequantized.push(dequant);
    }

    Ok(dequantized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::quantize_per_tensor_affine;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_dequantize_per_tensor_affine() {
        // Test round-trip quantization/dequantization
        let original_data = vec![0.0, 1.0, 2.0, 3.0];
        let original_tensor = tensor_1d(&original_data).unwrap();

        let scale = 0.1;
        let zero_point = 0;

        // Quantize
        let (quantized, _, _) =
            quantize_per_tensor_affine(&original_tensor, scale, zero_point).unwrap();

        // Dequantize
        let dequantized = dequantize_per_tensor_affine(&quantized, scale, zero_point).unwrap();

        // Check that dequantized values are close to original (within quantization error)
        let dequant_data = dequantized.to_vec().unwrap();
        for (i, &original_val) in original_data.iter().enumerate() {
            assert_relative_eq!(dequant_data[i], original_val, epsilon = scale);
        }
    }

    #[test]
    fn test_dequantize_per_tensor_symmetric() {
        let quantized_data = vec![0.0, 10.0, 20.0, 30.0];
        let quantized_tensor = tensor_1d(&quantized_data).unwrap();

        let scale = 0.1;

        let dequantized = dequantize_per_tensor_symmetric(&quantized_tensor, scale).unwrap();
        let dequant_data = dequantized.to_vec().unwrap();

        // Expected: [0.0, 1.0, 2.0, 3.0]
        assert_relative_eq!(dequant_data[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(dequant_data[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(dequant_data[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(dequant_data[3], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dequantize_batch() {
        let tensor1 = tensor_1d(&[0.0, 10.0]).unwrap();
        let tensor2 = tensor_1d(&[0.0, 20.0]).unwrap();
        let tensors = vec![tensor1, tensor2];

        let scales = vec![0.1, 0.2];
        let zero_points = vec![0, 0];

        let dequantized = dequantize_batch(&tensors, &scales, &zero_points).unwrap();

        assert_eq!(dequantized.len(), 2);

        let data1 = dequantized[0].to_vec().unwrap();
        let data2 = dequantized[1].to_vec().unwrap();

        assert_relative_eq!(data1[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(data1[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(data2[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(data2[1], 4.0, epsilon = 1e-5);
    }
}
