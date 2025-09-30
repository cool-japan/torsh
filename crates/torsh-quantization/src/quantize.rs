//! Quantization operations

use crate::{QScheme, TorshResult};
use rayon::prelude::*;
use torsh_core::{DType, TorshError};
use torsh_tensor::Tensor;

/// Calculate strides for tensor dimensions (optimized, reusable function)
#[inline]
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// SIMD-optimized quantization for large arrays (when AVX2 is available)
#[cfg(target_feature = "avx2")]
#[inline]
fn quantize_simd_f32_to_i8(data: &[f32], scale: f32, zero_point: i32, output: &mut [i8]) {
    use std::arch::x86_64::*;

    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;

    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    unsafe {
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let zero_point_vec = _mm256_set1_ps(zero_point_f32);
        let min_val = _mm256_set1_ps(-128.0);
        let max_val = _mm256_set1_ps(127.0);

        for (i, chunk) in chunks.enumerate() {
            let input = _mm256_loadu_ps(chunk.as_ptr());
            let scaled = _mm256_fmadd_ps(input, inv_scale_vec, zero_point_vec);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
            let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_val), min_val);

            let as_i32 = _mm256_cvtps_epi32(clamped);
            let as_i16_lo = _mm256_extracti128_si256(as_i32, 0);
            let as_i16_hi = _mm256_extracti128_si256(as_i32, 1);
            let as_i16 = _mm_packs_epi32(as_i16_lo, as_i16_hi);
            let as_i8 = _mm_packs_epi16(as_i16, as_i16);

            _mm_storel_epi64(output[i * 8..].as_mut_ptr() as *mut __m128i, as_i8);
        }
    }

    // Process remainder
    for (i, &val) in remainder.iter().enumerate() {
        let quantized = (val * inv_scale + zero_point_f32).round();
        output[chunks.len() * 8 + i] = quantized.max(-128.0).min(127.0) as i8;
    }
}

/// Ultra-high-performance AVX-512 VNNI quantization for latest Intel processors
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512vnni",
    target_feature = "avx512bw"
))]
#[inline]
fn quantize_avx512_vnni_f32_to_i8(data: &[f32], scale: f32, zero_point: i32, output: &mut [i8]) {
    use std::arch::x86_64::*;

    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;

    let chunks = data.chunks_exact(16); // AVX-512 processes 16 f32 at once
    let remainder = chunks.remainder();

    unsafe {
        let inv_scale_vec = _mm512_set1_ps(inv_scale);
        let zero_point_vec = _mm512_set1_ps(zero_point_f32);
        let min_val = _mm512_set1_ps(-128.0);
        let max_val = _mm512_set1_ps(127.0);

        for (i, chunk) in chunks.enumerate() {
            // Load 16 f32 values
            let input = _mm512_loadu_ps(chunk.as_ptr());

            // Scale and add zero point with FMA
            let scaled = _mm512_fmadd_ps(input, inv_scale_vec, zero_point_vec);

            // Round to nearest integer
            let rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);

            // Clamp to quantization range
            let clamped = _mm512_max_ps(_mm512_min_ps(rounded, max_val), min_val);

            // Convert to i32 then pack to i8
            let as_i32 = _mm512_cvtps_epi32(clamped);

            // Pack i32 -> i16 -> i8 with saturation (AVX-512BW)
            let as_i16 = _mm512_packs_epi32(as_i32, as_i32);
            let as_i8_512 = _mm512_packs_epi16(as_i16, as_i16);

            // Extract lower 128 bits containing our 16 i8 values
            let as_i8_128 = _mm512_extracti32x4_epi32(as_i8_512, 0);
            _mm_storeu_si128(output[i * 16..].as_mut_ptr() as *mut __m128i, as_i8_128);
        }
    }

    // Process remainder with scalar code
    for (i, &val) in remainder.iter().enumerate() {
        let quantized = (val * inv_scale + zero_point_f32).round();
        output[chunks.len() * 16 + i] = quantized.clamp(-128.0, 127.0) as i8;
    }
}

/// Fallback scalar quantization
#[inline]
fn quantize_scalar_f32_to_i8(data: &[f32], scale: f32, zero_point: i32, output: &mut [i8]) {
    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;

    for (i, &val) in data.iter().enumerate() {
        let quantized = (val * inv_scale + zero_point_f32).round();
        output[i] = quantized.clamp(-128.0, 127.0) as i8;
    }
}

/// Optimized quantization with automatic SIMD detection and runtime feature detection
#[inline]
fn quantize_optimized(data: &[f32], scale: f32, zero_point: i32) -> Vec<f32> {
    let mut output_i8 = vec![0i8; data.len()];

    // Use the most advanced SIMD available at runtime
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512vnni",
        target_feature = "avx512bw"
    ))]
    {
        if data.len() >= 16
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
        {
            quantize_avx512_vnni_f32_to_i8(data, scale, zero_point, &mut output_i8);
        } else if data.len() >= 8 && is_x86_feature_detected!("avx2") {
            quantize_simd_f32_to_i8(data, scale, zero_point, &mut output_i8);
        } else {
            quantize_scalar_f32_to_i8(data, scale, zero_point, &mut output_i8);
        }
    }

    #[cfg(all(
        target_feature = "avx2",
        not(all(
            target_feature = "avx512f",
            target_feature = "avx512vnni",
            target_feature = "avx512bw"
        ))
    ))]
    {
        if data.len() >= 8 && is_x86_feature_detected!("avx2") {
            quantize_simd_f32_to_i8(data, scale, zero_point, &mut output_i8);
        } else {
            quantize_scalar_f32_to_i8(data, scale, zero_point, &mut output_i8);
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        quantize_scalar_f32_to_i8(data, scale, zero_point, &mut output_i8);
    }

    // Convert back to f32 for compatibility
    output_i8.into_iter().map(|x| x as f32).collect()
}

/// Quantize a tensor to INT8 using per-tensor affine quantization
pub fn quantize_per_tensor_affine(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    // Validate inputs
    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Quantization scale must be positive".to_string(),
        ));
    }

    if !(-128..=127).contains(&zero_point) {
        return Err(TorshError::InvalidArgument(
            "Zero point must be in range [-128, 127]".to_string(),
        ));
    }

    // Use optimized quantization with SIMD when available
    let quantized_f32: Vec<f32> = if data.len() > 1000 {
        // Use parallel processing for large tensors
        data.par_chunks(4096) // Process in cache-friendly chunks
            .flat_map(|chunk| quantize_optimized(chunk, scale, zero_point))
            .collect()
    } else {
        quantize_optimized(&data, scale, zero_point)
    };

    let quantized_tensor = Tensor::from_data(
        quantized_f32,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    );

    Ok((quantized_tensor?, scale, zero_point))
}

/// Quantize a tensor using symmetric quantization (zero_point = 0)
pub fn quantize_per_tensor_symmetric(tensor: &Tensor, scale: f32) -> TorshResult<(Tensor, f32)> {
    let (quantized_tensor, computed_scale, _) = quantize_per_tensor_affine(tensor, scale, 0)?;
    Ok((quantized_tensor, computed_scale))
}

/// Calculate quantization parameters (scale and zero_point) from tensor statistics
pub fn calculate_qparams(
    tensor: &Tensor,
    qmin: i32,
    qmax: i32,
    _dtype: DType,
) -> TorshResult<(f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot calculate quantization parameters for empty tensor".to_string(),
        ));
    }

    if qmin >= qmax {
        return Err(TorshError::InvalidArgument(
            "qmin must be less than qmax".to_string(),
        ));
    }

    // Optimized min/max calculation with better numerical stability
    let (min_val, max_val) = if data.len() > 10000 {
        // Use parallel processing for very large tensors
        data.par_iter().map(|&val| (val, val)).reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
        )
    } else {
        data.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            })
    };

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(TorshError::InvalidArgument(
            "Tensor contains non-finite values (NaN or infinity)".to_string(),
        ));
    }

    // Ensure the range includes zero for better numerical stability
    let min_val = min_val.min(0.0);
    let max_val = max_val.max(0.0);

    // Add small epsilon to prevent zero range
    let range = max_val - min_val;
    let adjusted_range = if range < 1e-7 {
        1e-7 // Minimum meaningful range
    } else {
        range
    };

    // Calculate scale with improved numerical stability
    let scale = adjusted_range / (qmax - qmin) as f32;

    // Use more precise zero point calculation
    let zero_point_exact = qmin as f64 - (min_val as f64) / (scale as f64);
    let zero_point = zero_point_exact.round().max(qmin as f64).min(qmax as f64) as i32;

    Ok((scale, zero_point))
}

/// Quantize using per-channel affine quantization
pub fn quantize_per_channel_affine(
    tensor: &Tensor,
    scales: &[f32],
    zero_points: &[i32],
    axis: usize,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    let data = tensor.data()?;
    let binding = tensor.shape();
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

    // Calculate strides for the given axis (using optimized helper)
    let strides = calculate_strides(shape);

    let mut quantized_data = vec![0i8; data.len()];

    for (idx, &x) in data.iter().enumerate() {
        // Calculate which channel this element belongs to
        let channel_idx = (idx / strides[axis]) % shape[axis];
        let scale = scales[channel_idx];
        let zero_point = zero_points[channel_idx];

        // Quantize: q = round(x / scale) + zero_point
        let quantized = (x / scale).round() + zero_point as f32;
        quantized_data[idx] = quantized.clamp(-128.0, 127.0) as i8;
    }

    // Convert to f32 tensor for compatibility
    let quantized_f32: Vec<f32> = quantized_data.iter().map(|&x| x as f32).collect();
    let quantized_tensor = Tensor::from_data(quantized_f32, shape.to_vec(), tensor.device());

    Ok((quantized_tensor?, scales.to_vec(), zero_points.to_vec()))
}

/// Calculate per-channel quantization parameters
pub fn calculate_per_channel_qparams(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
) -> TorshResult<(Vec<f32>, Vec<i32>)> {
    let data = tensor.data()?;
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let (qmin, qmax) = match dtype {
        DType::I8 => (-128, 127),
        DType::U8 => (0, 255),
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported quantization dtype".to_string(),
            ))
        }
    };

    let channel_size = shape[axis];
    let mut channel_mins = vec![f32::INFINITY; channel_size];
    let mut channel_maxs = vec![f32::NEG_INFINITY; channel_size];

    // Calculate strides for the given axis
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Find min/max for each channel
    for (idx, &val) in data.iter().enumerate() {
        let channel_idx = (idx / strides[axis]) % shape[axis];
        channel_mins[channel_idx] = channel_mins[channel_idx].min(val);
        channel_maxs[channel_idx] = channel_maxs[channel_idx].max(val);
    }

    let mut scales = Vec::with_capacity(channel_size);
    let mut zero_points = Vec::with_capacity(channel_size);

    for ch in 0..channel_size {
        let min_val = channel_mins[ch].min(0.0);
        let max_val = channel_maxs[ch].max(0.0);

        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let zero_point = (qmin as f32 - min_val / scale)
            .round()
            .max(qmin as f32)
            .min(qmax as f32) as i32;

        scales.push(scale);
        zero_points.push(zero_point);
    }

    Ok((scales, zero_points))
}

/// Quantize tensor with automatic parameter calculation
pub fn quantize_tensor_auto(
    tensor: &Tensor,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, f32, i32)> {
    let (qmin, qmax) = match dtype {
        DType::I8 => (-128, 127),
        DType::U8 => (0, 255),
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported quantization dtype".to_string(),
            ))
        }
    };

    let (scale, zero_point) = calculate_qparams(tensor, qmin, qmax, dtype)?;

    match scheme {
        QScheme::PerTensorAffine => quantize_per_tensor_affine(tensor, scale, zero_point),
        QScheme::PerTensorSymmetric => {
            let (quantized, computed_scale) = quantize_per_tensor_symmetric(tensor, scale)?;
            Ok((quantized, computed_scale, 0))
        }
        QScheme::PerChannelAffine => {
            // For per-channel, we need to specify the axis (default to 0 for weights)
            let axis = 0;
            let (scales, zero_points) = calculate_per_channel_qparams(tensor, axis, dtype)?;
            let (quantized, _, _) =
                quantize_per_channel_affine(tensor, &scales, &zero_points, axis)?;
            // Return the first channel's parameters for compatibility
            Ok((quantized, scales[0], zero_points[0]))
        }
        QScheme::PerChannelSymmetric => {
            let axis = 0;
            let (scales, _) = calculate_per_channel_qparams(tensor, axis, dtype)?;
            let zero_points = vec![0; scales.len()];
            let (quantized, _, _) =
                quantize_per_channel_affine(tensor, &scales, &zero_points, axis)?;
            Ok((quantized, scales[0], 0))
        }
        QScheme::Int4PerTensor => {
            // Use the quantize_int4_per_tensor function from lib.rs
            crate::quantize_int4_per_tensor(tensor, &crate::QuantConfig::int4())
        }
        QScheme::Int4PerChannel => {
            // Use the quantize_int4_per_channel function from lib.rs
            let axis = 0;
            crate::quantize_int4_per_channel(tensor, axis, &crate::QuantConfig::int4())
        }
        QScheme::Binary => {
            // Use the quantize_binary function from lib.rs
            crate::quantize_binary(tensor)
        }
        QScheme::Ternary => {
            // Use the quantize_ternary function from lib.rs
            crate::quantize_ternary(tensor)
        }
        QScheme::GroupWise => {
            // Use the quantize_group_wise function from lib.rs with default parameters
            let axis = 0;
            let group_size = 32;
            crate::quantize_group_wise(
                tensor,
                axis,
                group_size,
                &crate::QuantConfig::group_wise(axis, group_size),
            )
        }
        QScheme::MixedPrecision => {
            // Mixed precision requires different handling
            Err(TorshError::InvalidArgument(
                "Mixed precision quantization requires specialized API".to_string(),
            ))
        }
    }
}

/// Quantize tensor with per-channel scheme
pub fn quantize_per_channel_auto(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    match scheme {
        QScheme::PerChannelAffine => {
            let (scales, zero_points) = calculate_per_channel_qparams(tensor, axis, dtype)?;
            quantize_per_channel_affine(tensor, &scales, &zero_points, axis)
        }
        QScheme::PerChannelSymmetric => {
            let (scales, _) = calculate_per_channel_qparams(tensor, axis, dtype)?;
            let zero_points = vec![0; scales.len()];
            quantize_per_channel_affine(tensor, &scales, &zero_points, axis)
        }
        _ => Err(TorshError::InvalidArgument(
            "Scheme not supported for per-channel quantization".to_string(),
        )),
    }
}

/// Convenience function for automatic quantization based on configuration
/// This function takes a tensor and a QuantConfig and returns a quantized tensor
/// along with the scale and zero point parameters
pub fn quantize_auto(
    tensor: &Tensor,
    config: &crate::QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
    quantize_tensor_auto(tensor, config.dtype, config.scheme)
}

/// Dynamic quantization for modules
// Temporarily disabled: pub fn quantize_dynamic(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn quantize_dynamic(module: &mut dyn crate::TemporaryModule) -> TorshResult<()> {
    // Iterate through module parameters and quantize them dynamically
    let mut quantized_params = Vec::new();

    // Get mutable parameters
    let mut_params = module.parameters_mut();

    for param in mut_params {
        // Use INT8 quantization for each parameter
        let config = crate::QuantConfig::int8();
        let (quantized, _scale, _zero_point) = quantize_auto(param, &config)?;

        // Store quantized parameter (in practice, this would update the module's parameters)
        quantized_params.push(quantized);
    }

    // Note: In a real implementation, we would update the module's parameters
    // with the quantized versions, but this requires more complex module handling

    Ok(())
}

/// Static quantization preparation
// Temporarily disabled: pub fn prepare_qat(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn prepare_qat(module: &mut dyn crate::TemporaryModule) -> TorshResult<()> {
    // Insert fake quantization operations into the module for QAT
    // This is a simplified implementation that sets up the module for QAT

    // Switch module to training mode for QAT
    module.train(true);

    // Get named parameters to track which parameters need fake quantization
    let named_params = module.named_parameters();

    // For each parameter, we would typically insert fake quantization nodes
    // In this simplified implementation, we just validate that parameters exist
    for (name, param) in named_params {
        // Validate parameter can be quantized
        if param.numel() == 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Parameter {} is empty and cannot be quantized",
                name
            )));
        }

        // In a real implementation, we would insert fake quantization observers
        // and prepare the parameter for quantization-aware training
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{tensor_1d, tensor_2d};

    #[test]
    fn test_calculate_qparams() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let (scale, zero_point) = calculate_qparams(&tensor, -128, 127, DType::I8).unwrap();

        // Scale should be approximately (2.0 - (-2.0)) / (127 - (-128)) = 4.0 / 255
        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
    }

    #[test]
    fn test_quantize_per_tensor_affine() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        let scale = 0.1;
        let zero_point = 0;

        let (quantized, ret_scale, ret_zero_point) =
            quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();

        assert_eq!(ret_scale, scale);
        assert_eq!(ret_zero_point, zero_point);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_quantize_tensor_auto() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let (quantized, scale, zero_point) =
            quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorAffine).unwrap();

        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_quantization() {
        // Create a 2x3 tensor where each row has different scales
        let tensor = tensor_2d(&[
            &[0.0, 1.0, 2.0],  // Channel 0: range [0, 2]
            &[0.0, 5.0, 10.0], // Channel 1: range [0, 10]
        ])
        .unwrap();

        let axis = 0; // Quantize along the first dimension (channels)
        let (scales, zero_points) =
            calculate_per_channel_qparams(&tensor, axis, DType::I8).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);

        // Channel 1 should have a larger scale than channel 0
        assert!(scales[1] > scales[0]);

        let (quantized, ret_scales, ret_zero_points) =
            quantize_per_channel_affine(&tensor, &scales, &zero_points, axis).unwrap();

        assert_eq!(ret_scales, scales);
        assert_eq!(ret_zero_points, zero_points);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_auto() {
        let tensor = tensor_2d(&[&[-2.0, 0.0, 2.0], &[-10.0, 0.0, 10.0]]).unwrap();

        let (quantized, scales, zero_points) =
            quantize_per_channel_auto(&tensor, 0, DType::I8, QScheme::PerChannelAffine).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);
        assert!(scales[1] > scales[0]); // Channel 1 has larger range
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_symmetric() {
        let tensor = tensor_2d(&[&[-1.0, 0.0, 1.0], &[-5.0, 0.0, 5.0]]).unwrap();

        let (_quantized, scales, zero_points) =
            quantize_per_channel_auto(&tensor, 0, DType::I8, QScheme::PerChannelSymmetric).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);

        // All zero points should be 0 for symmetric quantization
        for &zp in &zero_points {
            assert_eq!(zp, 0);
        }

        assert!(scales[1] > scales[0]); // Channel 1 has larger range
    }
}
