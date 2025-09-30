//! Quantization operations for tensor conversion and computation

use crate::quantization::{QuantizationParams, QuantizationScheme};
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

/// Quantize a tensor using the given parameters
pub fn quantize_tensor(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    match params.dst_dtype {
        DType::I8 => quantize_to_int8(tensor, params),
        DType::U8 => quantize_to_uint8(tensor, params),
        DType::I16 => quantize_to_int16(tensor, params),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported quantization target dtype: {:?}",
            params.dst_dtype
        ))),
    }
}

/// Dequantize a tensor using the given parameters
pub fn dequantize_tensor(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    match params.dst_dtype {
        DType::I8 => dequantize_from_int8(tensor, params),
        DType::U8 => dequantize_from_uint8(tensor, params),
        DType::I16 => dequantize_from_int16(tensor, params),
        _ => {
            // If we have F32 tensors with quantized values, treat them as dequantized already
            if params.src_dtype == DType::F32 {
                Ok(tensor.clone())
            } else {
                Err(TorshError::InvalidArgument(format!(
                    "Unsupported dequantization source dtype: {:?}",
                    params.src_dtype
                )))
            }
        }
    }
}

/// Quantize tensor to INT8
fn quantize_to_int8(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut quantized = Vec::with_capacity(data.len());

    for &value in &data {
        let q_value = quantize_value_symmetric(value, params.scale, params.qmin, params.qmax);
        quantized.push(q_value as f32);
    }

    Ok(Tensor::from_data(
        quantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )?)
}

/// Quantize tensor to UINT8
fn quantize_to_uint8(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut quantized = Vec::with_capacity(data.len());

    for &value in &data {
        let q_value = quantize_value_asymmetric(
            value,
            params.scale,
            params.zero_point,
            params.qmin,
            params.qmax,
        );
        quantized.push(q_value as f32);
    }

    Ok(Tensor::from_data(
        quantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )?)
}

/// Quantize tensor to INT16
fn quantize_to_int16(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut quantized = Vec::with_capacity(data.len());

    for &value in &data {
        let q_value = quantize_value_symmetric(value, params.scale, params.qmin, params.qmax);
        quantized.push(q_value as f32);
    }

    Ok(Tensor::from_data(
        quantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )?)
}

/// Dequantize tensor from INT8
fn dequantize_from_int8(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut dequantized = Vec::with_capacity(data.len());

    for &value in &data {
        let deq_value = dequantize_value_symmetric(value.round() as i32, params.scale);
        dequantized.push(deq_value);
    }

    Tensor::from_data(
        dequantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Dequantize tensor from UINT8
fn dequantize_from_uint8(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut dequantized = Vec::with_capacity(data.len());

    for &value in &data {
        let deq_value =
            dequantize_value_asymmetric(value.round() as i32, params.scale, params.zero_point);
        dequantized.push(deq_value);
    }

    Tensor::from_data(
        dequantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Dequantize tensor from INT16
fn dequantize_from_int16(tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let mut dequantized = Vec::with_capacity(data.len());

    for &value in &data {
        let deq_value = dequantize_value_symmetric(value.round() as i32, params.scale);
        dequantized.push(deq_value);
    }

    Tensor::from_data(
        dequantized,
        tensor.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Quantize a single value using symmetric quantization
fn quantize_value_symmetric(value: f32, scale: f32, qmin: i32, qmax: i32) -> i32 {
    let quantized = (value / scale).round() as i32;
    quantized.clamp(qmin, qmax)
}

/// Quantize a single value using asymmetric quantization
fn quantize_value_asymmetric(value: f32, scale: f32, zero_point: i32, qmin: i32, qmax: i32) -> i32 {
    let quantized = ((value / scale).round() as i32) + zero_point;
    quantized.clamp(qmin, qmax)
}

/// Dequantize a single value using symmetric quantization
fn dequantize_value_symmetric(quantized: i32, scale: f32) -> f32 {
    quantized as f32 * scale
}

/// Dequantize a single value using asymmetric quantization
fn dequantize_value_asymmetric(quantized: i32, scale: f32, zero_point: i32) -> f32 {
    (quantized - zero_point) as f32 * scale
}

/// Quantized matrix multiplication for INT8
pub fn quantized_matmul_int8(
    a: &Tensor,
    b: &Tensor,
    a_params: &QuantizationParams,
    b_params: &QuantizationParams,
    output_params: &QuantizationParams,
) -> Result<Tensor> {
    // This is a simplified implementation
    // Real INT8 GEMM would use optimized kernels

    // Dequantize inputs for computation
    let a_fp32 = dequantize_tensor(a, a_params)?;
    let b_fp32 = dequantize_tensor(b, b_params)?;

    // Perform FP32 multiplication
    let result_fp32 = a_fp32.matmul(&b_fp32)?;

    // Quantize result
    quantize_tensor(&result_fp32, output_params)
}

/// Quantized convolution for INT8
pub fn quantized_conv2d_int8(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    input_params: &QuantizationParams,
    weight_params: &QuantizationParams,
    bias_params: Option<&QuantizationParams>,
    output_params: &QuantizationParams,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    // Simplified implementation - real version would use optimized INT8 convolution

    // Dequantize inputs
    let _input_fp32 = dequantize_tensor(input, input_params)?;
    let _weight_fp32 = dequantize_tensor(weight, weight_params)?;

    let bias_fp32 = if let (Some(bias), Some(bias_params)) = (bias, bias_params) {
        Some(dequantize_tensor(bias, bias_params)?)
    } else {
        None
    };

    // Implement actual convolution operation
    let input_fp32 = dequantize_tensor(input, input_params)?;
    let weight_fp32 = dequantize_tensor(weight, weight_params)?;

    // Use the Conv2d functional operation
    let conv_output = crate::functional::conv2d(
        &input_fp32,
        &weight_fp32,
        bias_fp32.as_ref(),
        stride,
        padding,
        (1, 1), // dilation
        1,      // groups
    )?;

    // Quantize result
    quantize_tensor(&conv_output, output_params)
}

/// Calculate output shape for 2D convolution
#[allow(dead_code)]
fn calculate_conv2d_output_shape(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
) -> Vec<usize> {
    let batch_size = input_shape[0];
    let out_channels = kernel_shape[0];
    let input_height = input_shape[2];
    let input_width = input_shape[3];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];

    let output_height = (input_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    vec![batch_size, out_channels, output_height, output_width]
}

/// Fused quantized ReLU operation
pub fn quantized_relu_int8(
    input: &Tensor,
    input_params: &QuantizationParams,
    _output_params: &QuantizationParams,
) -> Result<Tensor> {
    // For INT8 ReLU, we can directly operate on quantized values
    let data = input.to_vec()?;
    let mut output = Vec::with_capacity(data.len());

    // Calculate zero point in quantized space
    let zero_quantized = quantize_value_asymmetric(
        0.0,
        input_params.scale,
        input_params.zero_point,
        input_params.qmin,
        input_params.qmax,
    );

    for &value in &data {
        let result = if (value as i32) > zero_quantized {
            value
        } else {
            zero_quantized as f32
        };
        output.push(result);
    }

    Tensor::from_data(
        output,
        input.shape().dims().to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Per-channel quantization for weights
pub fn per_channel_quantize_weights(
    weights: &Tensor,
    scheme: &QuantizationScheme,
    target_dtype: DType,
) -> Result<(Tensor, Vec<QuantizationParams>)> {
    let shape = weights.shape();
    let num_channels = shape.dims()[0]; // Assuming first dimension is output channels

    let mut channel_params = Vec::with_capacity(num_channels);
    let mut quantized_data = Vec::new();

    for channel in 0..num_channels {
        // Extract channel data
        let channel_tensor = weights.slice(0, channel, channel + 1)?;
        let channel_data = channel_tensor.to_vec()?;

        // Calculate quantization parameters for this channel
        let flattened_data: Vec<f32> = channel_data;
        let (min_val, max_val) = flattened_data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let params = match scheme {
            QuantizationScheme::Symmetric => {
                let scale = max_val.abs().max(min_val.abs()) / 127.0;
                QuantizationParams::symmetric(scale, DType::F32, target_dtype)
            }
            QuantizationScheme::Asymmetric => {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32;
                QuantizationParams::asymmetric(scale, zero_point, DType::F32, target_dtype)
            }
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported quantization scheme for per-channel quantization".to_string(),
                ))
            }
        };

        // Quantize channel data
        for &value in &flattened_data {
            let q_value = match target_dtype {
                DType::I8 => {
                    quantize_value_symmetric(value, params.scale, params.qmin, params.qmax) as i8
                        as u8
                }
                DType::U8 => quantize_value_asymmetric(
                    value,
                    params.scale,
                    params.zero_point,
                    params.qmin,
                    params.qmax,
                ) as u8,
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "Unsupported target dtype".to_string(),
                    ))
                }
            };
            quantized_data.push(q_value);
        }

        channel_params.push(params);
    }

    let quantized_tensor = {
        // For simplicity, convert all to f32 for consistency
        let float_data: Vec<f32> = quantized_data.into_iter().map(|x| x as f32).collect();
        Tensor::from_data(
            float_data,
            weights.shape().dims().to_vec(),
            torsh_core::device::DeviceType::Cpu,
        )?
    };
    Ok((quantized_tensor, channel_params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_symmetric_quantization() -> Result<()> {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let tensor = Tensor::from_data(data, vec![6], torsh_core::device::DeviceType::Cpu)?;

        let params = QuantizationParams::symmetric(2.0 / 127.0, DType::F32, DType::I8);

        let quantized = quantize_tensor(&tensor, &params)?;
        let dequantized = dequantize_tensor(&quantized, &params)?;

        // Check that quantization/dequantization is approximately correct
        let original_data = tensor.to_vec()?;
        let recovered_data = dequantized.to_vec()?;

        for (orig, recovered) in original_data.iter().zip(recovered_data.iter()) {
            assert!(
                (orig - recovered).abs() < 0.1,
                "Original: {}, Recovered: {}",
                orig,
                recovered
            );
        }
        Ok(())
    }

    #[test]
    fn test_asymmetric_quantization() -> Result<()> {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![6], torsh_core::device::DeviceType::Cpu)?;

        let scale = 5.0 / 255.0;
        let zero_point = 0;
        let params = QuantizationParams::asymmetric(scale, zero_point, DType::F32, DType::U8);

        let quantized = quantize_tensor(&tensor, &params)?;
        let dequantized = dequantize_tensor(&quantized, &params)?;

        let original_data = tensor.to_vec()?;
        let recovered_data = dequantized.to_vec()?;

        for (orig, recovered) in original_data.iter().zip(recovered_data.iter()) {
            assert!(
                (orig - recovered).abs() < 0.1,
                "Original: {}, Recovered: {}",
                orig,
                recovered
            );
        }
        Ok(())
    }

    #[test]
    fn test_quantized_relu() {
        let data = vec![-2.0f32, -1.0f32, 0.0f32, 1.0f32, 2.0f32];
        let tensor = Tensor::from_data(data, vec![5], torsh_core::device::DeviceType::Cpu).unwrap();

        let params = QuantizationParams::symmetric(1.0 / 127.0, DType::F32, DType::I8);

        let result = quantized_relu_int8(&tensor, &params, &params).unwrap();
        let result_data = result.to_vec().unwrap();

        // ReLU should clamp negative values to zero
        for (i, &value) in result_data.iter().enumerate() {
            if i < 2 {
                assert_eq!(value, 0.0, "Negative values should be clamped to zero");
            } else {
                assert!(
                    value >= 0.0,
                    "Non-negative values should remain non-negative"
                );
            }
        }
    }
}
