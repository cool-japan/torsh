//! Core quantization algorithms and tensor operations
//!
//! This module provides the fundamental quantization and dequantization algorithms
//! for tensor operations, including per-tensor and per-channel quantization schemes.
//!
//! # Features
//!
//! - **Per-tensor quantization**: Single scale/zero-point for entire tensor
//! - **Per-channel quantization**: Individual scale/zero-point per channel
//! - **Dequantization**: Reverse quantization to restore floating point values
//! - **Multiple schemes**: Affine and symmetric quantization support
//! - **Configuration-driven**: Integration with QuantConfig for flexible usage

use crate::config::{QScheme, QuantConfig};
use torsh_core::{
    dtype::DType,
    error::{Result as TorshResult, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use scirs2_core::parallel_ops::*;

/// Quantize a tensor using specified configuration
pub fn quantize_with_config(
    tensor: &Tensor,
    config: &QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
    config.validate()?;

    match config.scheme {
        QScheme::PerTensorAffine | QScheme::PerTensorSymmetric => {
            quantize_tensor_auto(tensor, config.dtype, config.scheme)
        }
        QScheme::PerChannelAffine | QScheme::PerChannelSymmetric => {
            let axis = config.ch_axis.unwrap_or(0);
            let (quantized, scales, zero_points) =
                quantize_per_channel_auto(tensor, axis, config.dtype, config.scheme)?;
            // Return first channel's parameters for compatibility
            Ok((quantized, scales[0], zero_points[0]))
        }
        QScheme::GroupWise => {
            let axis = config.ch_axis.unwrap_or(0);
            let group_size = config.group_size.unwrap_or(32);
            crate::specialized::quantize_group_wise(tensor, axis, group_size, config)
        }
        QScheme::Int4PerTensor => crate::specialized::quantize_int4_per_tensor(tensor, config),
        QScheme::Int4PerChannel => {
            let axis = config.ch_axis.unwrap_or(0);
            crate::specialized::quantize_int4_per_channel(tensor, axis, config)
        }
        QScheme::Binary => crate::specialized::quantize_binary(tensor),
        QScheme::Ternary => crate::specialized::quantize_ternary(tensor),
        QScheme::MixedPrecision => {
            // Mixed precision requires different handling
            Err(TorshError::InvalidArgument(
                "Mixed precision quantization requires specialized API".to_string(),
            ))
        }
    }
}

/// Quantize a tensor to INT8 using specified scale and zero point
pub fn quantize_per_tensor(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
    _dtype: DType,
) -> TorshResult<Tensor> {
    let (quantized, _, _) = quantize_per_tensor_affine(tensor, scale, zero_point)?;
    Ok(quantized)
}

/// Dequantize a quantized tensor using scale and zero_point
pub fn dequantize(tensor: &Tensor, scale: f32, zero_point: i32) -> TorshResult<Tensor> {
    dequantize_per_tensor_affine(tensor, scale, zero_point)
}

/// Auto-quantize a tensor using per-tensor scheme
pub fn quantize_tensor_auto(
    tensor: &Tensor,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot quantize empty tensor".to_string(),
        ));
    }

    // Calculate min and max values using SIMD acceleration when beneficial
    let (min_val, max_val) = if data.len() > 64 && crate::simd_ops::is_simd_available() {
        crate::simd_ops::find_min_max_simd(&data)?
    } else {
        // Fallback for small tensors
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        (min_val, max_val)
    };

    // Determine quantization parameters based on scheme
    let (scale, zero_point) = match scheme {
        QScheme::PerTensorAffine => calculate_affine_quantization_params(min_val, max_val, dtype)?,
        QScheme::PerTensorSymmetric => {
            calculate_symmetric_quantization_params(min_val, max_val, dtype)?
        }
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unsupported scheme for auto quantization: {:?}",
                scheme
            )));
        }
    };

    quantize_per_tensor_affine(tensor, scale, zero_point)
}

/// Auto-quantize a tensor using per-channel scheme
pub fn quantize_per_channel_auto(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let num_channels = shape[axis];
    let data = tensor.data()?;

    // Calculate strides for efficient channel access
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);
    let mut quantized_data = vec![0.0f32; data.len()];

    // Process each channel
    for ch in 0..num_channels {
        let mut channel_min = f32::INFINITY;
        let mut channel_max = f32::NEG_INFINITY;

        // Calculate channel statistics
        let mut indices = vec![0; shape.len()];
        let channel_size = data.len() / num_channels;

        for i in 0..channel_size {
            // Calculate multi-dimensional index for this channel
            let mut temp_i = i;
            for dim in (0..shape.len()).rev() {
                if dim == axis {
                    indices[dim] = ch;
                } else {
                    let other_dim_size = if dim == axis { 1 } else { shape[dim] };
                    indices[dim] = temp_i % other_dim_size;
                    temp_i /= other_dim_size;
                }
            }

            // Convert multi-dimensional index to flat index
            let flat_idx = indices
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>();

            if flat_idx < data.len() {
                let val = data[flat_idx];
                channel_min = channel_min.min(val);
                channel_max = channel_max.max(val);
            }
        }

        // Calculate quantization parameters for this channel
        let (scale, zero_point) = match scheme {
            QScheme::PerChannelAffine => {
                calculate_affine_quantization_params(channel_min, channel_max, dtype)?
            }
            QScheme::PerChannelSymmetric => {
                calculate_symmetric_quantization_params(channel_min, channel_max, dtype)?
            }
            _ => {
                return Err(TorshError::InvalidArgument(format!(
                    "Unsupported scheme for per-channel quantization: {:?}",
                    scheme
                )));
            }
        };

        scales.push(scale);
        zero_points.push(zero_point);

        // Quantize channel data
        for i in 0..channel_size {
            let mut temp_i = i;
            for dim in (0..shape.len()).rev() {
                if dim == axis {
                    indices[dim] = ch;
                } else {
                    let other_dim_size = if dim == axis { 1 } else { shape[dim] };
                    indices[dim] = temp_i % other_dim_size;
                    temp_i /= other_dim_size;
                }
            }

            let flat_idx = indices
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>();

            if flat_idx < data.len() {
                let val = data[flat_idx];
                let quantized = ((val / scale).round() + zero_point as f32).clamp(
                    get_dtype_range(dtype).0 as f32,
                    get_dtype_range(dtype).1 as f32,
                );
                quantized_data[flat_idx] = quantized;
            }
        }
    }

    let quantized_tensor = Tensor::from_data(quantized_data, shape.to_vec(), tensor.device())?;

    Ok((quantized_tensor, scales, zero_points))
}

/// Quantize a tensor using per-tensor affine quantization (returns I8 tensor)
pub fn quantize_per_tensor_affine_i8(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<(Tensor<i8>, f32, i32)> {
    let data = tensor.data()?;

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    let quantized_data: Vec<i8> = data
        .iter()
        .map(|&x| {
            let quantized = (x / scale).round() + zero_point as f32;
            // Clamp to int8 range and convert to i8
            quantized.clamp(-128.0, 127.0) as i8
        })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, zero_point))
}

/// Quantize a tensor using per-tensor affine quantization
pub fn quantize_per_tensor_affine(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    // Use SIMD-accelerated quantization when available and beneficial
    let mut quantized_data = vec![0.0f32; data.len()];
    if data.len() > 64 && crate::simd_ops::is_simd_available() {
        // Use SIMD for larger tensors
        crate::simd_ops::quantize_per_tensor_affine_simd(
            &data,
            scale,
            zero_point,
            &mut quantized_data,
        )?;
    } else {
        // Fallback to scalar implementation for small tensors
        for (i, &x) in data.iter().enumerate() {
            let quantized = (x / scale).round() + zero_point as f32;
            quantized_data[i] = quantized.clamp(-128.0, 127.0);
        }
    }

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, zero_point))
}

/// Dequantize a tensor using per-tensor affine dequantization
pub fn dequantize_per_tensor_affine(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<Tensor> {
    let data = tensor.data()?;

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    // Use SIMD-accelerated dequantization when available and beneficial
    let mut dequantized_data = vec![0.0f32; data.len()];
    if data.len() > 64 && crate::simd_ops::is_simd_available() {
        // Use SIMD for larger tensors
        crate::simd_ops::dequantize_per_tensor_affine_simd(
            &data,
            scale,
            zero_point,
            &mut dequantized_data,
        )?;
    } else {
        // Fallback to scalar implementation for small tensors
        for (i, &x) in data.iter().enumerate() {
            dequantized_data[i] = (x - zero_point as f32) * scale;
        }
    }

    let dequantized_tensor = Tensor::from_data(
        dequantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok(dequantized_tensor)
}

/// Calculate affine quantization parameters (scale and zero_point)
pub fn calculate_affine_quantization_params(
    min_val: f32,
    max_val: f32,
    dtype: DType,
) -> TorshResult<(f32, i32)> {
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(TorshError::InvalidArgument(
            "Min and max values must be finite".to_string(),
        ));
    }

    if min_val > max_val {
        return Err(TorshError::InvalidArgument(
            "Min value must be <= max value".to_string(),
        ));
    }

    let (qmin, qmax) = get_dtype_range(dtype);
    let qmin = qmin as f32;
    let qmax = qmax as f32;

    // Handle edge case where min == max
    if (max_val - min_val).abs() < f32::EPSILON {
        let scale = 1.0;
        let zero_point = qmin as i32;
        return Ok((scale, zero_point));
    }

    // Calculate scale
    let scale = (max_val - min_val) / (qmax - qmin);

    // Calculate zero_point
    let zero_point_fp = qmin - min_val / scale;
    let zero_point = zero_point_fp.round().clamp(qmin, qmax) as i32;

    Ok((scale, zero_point))
}

/// Calculate symmetric quantization parameters (scale only, zero_point = 0)
pub fn calculate_symmetric_quantization_params(
    min_val: f32,
    max_val: f32,
    dtype: DType,
) -> TorshResult<(f32, i32)> {
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(TorshError::InvalidArgument(
            "Min and max values must be finite".to_string(),
        ));
    }

    let (_qmin, qmax) = get_dtype_range(dtype);
    let abs_max = min_val.abs().max(max_val.abs());

    // Handle edge case where range is zero
    if abs_max < f32::EPSILON {
        return Ok((1.0, 0));
    }

    // For symmetric quantization, we use the maximum absolute value
    // and map it to the maximum quantized range
    let scale = abs_max / qmax as f32;
    let zero_point = 0; // Symmetric quantization always has zero_point = 0

    Ok((scale, zero_point))
}

/// Get the quantization range for a given data type
pub fn get_dtype_range(dtype: DType) -> (i32, i32) {
    match dtype {
        DType::I8 => (-128, 127),
        DType::U8 => (0, 255),
        DType::I16 => (-32768, 32767),
        DType::I32 => (i32::MIN, i32::MAX),
        _ => (-128, 127), // Default to int8 range
    }
}

/// Convenience function to quantize with automatic parameter calculation
pub fn quantize_auto(tensor: &Tensor, config: &QuantConfig) -> TorshResult<(Tensor, f32, i32)> {
    quantize_with_config(tensor, config)
}

// ===== Cache-Aware Algorithm Enhancements =====

/// Cache-aware quantization parameters for optimal memory access patterns
#[derive(Debug, Clone)]
pub struct CacheAwareParams {
    /// Cache line size in bytes (typically 64 bytes)
    pub cache_line_size: usize,
    /// L1 cache size in bytes (typically 32KB)
    pub l1_cache_size: usize,
    /// L2 cache size in bytes (typically 256KB)
    pub l2_cache_size: usize,
    /// L3 cache size in bytes (typically 8MB)
    pub l3_cache_size: usize,
    /// Prefetch distance (elements ahead to prefetch)
    pub prefetch_distance: usize,
    /// Enable cache-optimized chunking
    pub enable_chunking: bool,
}

impl Default for CacheAwareParams {
    fn default() -> Self {
        Self {
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,       // 32KB L1
            l2_cache_size: 256 * 1024,      // 256KB L2
            l3_cache_size: 8 * 1024 * 1024, // 8MB L3
            prefetch_distance: 16,
            enable_chunking: true,
        }
    }
}

/// Cache-aware per-tensor quantization optimized for memory hierarchy
pub fn quantize_per_tensor_affine_cache_aware(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
    cache_params: &CacheAwareParams,
) -> TorshResult<()> {
    if input.len() != output.len() {
        return Err(TorshError::InvalidArgument(
            "Input and output length mismatch".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;

    if !cache_params.enable_chunking || input.len() < cache_params.cache_line_size {
        // Direct processing for small arrays
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            let quantized = (*inp * inv_scale).round() + zero_point_f32;
            *out = quantized.clamp(-128.0, 127.0);
        }
        return Ok(());
    }

    // Calculate optimal chunk size based on cache hierarchy
    let _elements_per_cache_line = cache_params.cache_line_size / std::mem::size_of::<f32>();
    let optimal_chunk_size =
        (cache_params.l2_cache_size / std::mem::size_of::<f32>() / 4).min(input.len());

    // Process in cache-friendly chunks
    input
        .par_chunks(optimal_chunk_size)
        .zip(output.par_chunks_mut(optimal_chunk_size))
        .for_each(|(input_chunk, output_chunk)| {
            // Process chunk with cache-friendly pattern
            for (inp, out) in input_chunk.iter().zip(output_chunk.iter_mut()) {
                let quantized = (*inp * inv_scale).round() + zero_point_f32;
                *out = quantized.clamp(-128.0, 127.0);
            }
        });

    Ok(())
}

/// Cache-optimized tensor statistics calculation with blocking
pub fn calculate_tensor_stats_cache_optimized(
    data: &[f32],
    cache_params: &CacheAwareParams,
) -> TorshResult<(f32, f32, f32, f32)> {
    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot calculate stats of empty tensor".to_string(),
        ));
    }

    let optimal_block_size = cache_params.l2_cache_size / std::mem::size_of::<f32>();
    let block_size = optimal_block_size.min(data.len());

    // Use blocked algorithm for better cache performance
    let results: Vec<(f32, f32, f64, f64)> = data
        .par_chunks(block_size)
        .map(|chunk| {
            let mut local_min = f32::INFINITY;
            let mut local_max = f32::NEG_INFINITY;
            let mut local_sum = 0.0f64;
            let mut local_sum_sq = 0.0f64;

            // Process block with good cache locality
            for &val in chunk {
                local_min = local_min.min(val);
                local_max = local_max.max(val);
                let val_f64 = val as f64;
                local_sum += val_f64;
                local_sum_sq += val_f64 * val_f64;
            }

            (local_min, local_max, local_sum, local_sum_sq)
        })
        .collect();

    // Combine results
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut total_sum = 0.0f64;
    let mut total_sum_sq = 0.0f64;

    for (local_min, local_max, local_sum, local_sum_sq) in results {
        min_val = min_val.min(local_min);
        max_val = max_val.max(local_max);
        total_sum += local_sum;
        total_sum_sq += local_sum_sq;
    }

    let n = data.len() as f64;
    let mean = (total_sum / n) as f32;
    let variance = ((total_sum_sq / n) - (mean as f64).powi(2)) as f32;

    Ok((min_val, max_val, mean, variance.sqrt()))
}

/// Cache-friendly matrix quantization using tiling for 2D tensors
pub fn quantize_matrix_cache_friendly(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
    cache_params: &CacheAwareParams,
) -> TorshResult<()> {
    if matrix.len() != rows * cols || output.len() != rows * cols {
        return Err(TorshError::InvalidArgument(
            "Matrix dimensions don't match buffer sizes".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;

    // Calculate optimal tile sizes based on cache hierarchy
    let elements_per_cache_line = cache_params.cache_line_size / std::mem::size_of::<f32>();
    let l2_elements = cache_params.l2_cache_size / std::mem::size_of::<f32>();

    // Find good tile dimensions that fit in L2 cache
    let max_tile_size = (l2_elements / 4).min(1024); // Reserve space for other data
    let tile_rows = (max_tile_size / cols).max(1).min(rows);
    let tile_cols = (max_tile_size / tile_rows)
        .max(elements_per_cache_line)
        .min(cols);

    // Process matrix in cache-friendly tiles
    for row_start in (0..rows).step_by(tile_rows) {
        let row_end = (row_start + tile_rows).min(rows);

        for col_start in (0..cols).step_by(tile_cols) {
            let col_end = (col_start + tile_cols).min(cols);

            // Process tile with good spatial locality
            for row in row_start..row_end {
                for col in col_start..col_end {
                    let idx = row * cols + col;
                    let quantized = (matrix[idx] * inv_scale).round() + zero_point_f32;
                    output[idx] = quantized.clamp(-128.0, 127.0);
                }
            }
        }
    }

    Ok(())
}

/// Prefetch-aware sequential quantization for streaming data
pub fn quantize_streaming_with_prefetch(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
    cache_params: &CacheAwareParams,
) -> TorshResult<()> {
    if input.len() != output.len() {
        return Err(TorshError::InvalidArgument(
            "Input and output length mismatch".to_string(),
        ));
    }

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scale must be positive".to_string(),
        ));
    }

    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;
    let prefetch_distance = cache_params.prefetch_distance;

    // Sequential processing with software prefetching
    for i in 0..input.len() {
        // Software prefetch hint for future data (compiler may optimize this)
        if i + prefetch_distance < input.len() {
            // This is a hint to the processor - actual prefetch intrinsics would be platform-specific
            let _prefetch_addr = &input[i + prefetch_distance];
        }

        let quantized = (input[i] * inv_scale).round() + zero_point_f32;
        output[i] = quantized.clamp(-128.0, 127.0);
    }

    Ok(())
}

/// Get cache-aware optimization recommendations for tensor operations
pub fn get_cache_optimization_recommendations(
    tensor_size: usize,
    element_size: usize,
    cache_params: &CacheAwareParams,
) -> Vec<String> {
    let mut recommendations = Vec::new();
    let total_bytes = tensor_size * element_size;

    if total_bytes <= cache_params.l1_cache_size {
        recommendations.push("Tensor fits in L1 cache - use simple sequential access".to_string());
    } else if total_bytes <= cache_params.l2_cache_size {
        recommendations.push("Tensor fits in L2 cache - consider blocked algorithms".to_string());
    } else if total_bytes <= cache_params.l3_cache_size {
        recommendations
            .push("Tensor fits in L3 cache - use tiled processing with medium blocks".to_string());
    } else {
        recommendations
            .push("Large tensor - use streaming algorithms with prefetching".to_string());
        recommendations
            .push("Consider parallel processing to utilize multiple cache hierarchies".to_string());
    }

    let elements_per_cache_line = cache_params.cache_line_size / element_size;
    if tensor_size % elements_per_cache_line != 0 {
        recommendations.push(format!(
            "Consider padding to align with cache lines ({}B boundaries)",
            cache_params.cache_line_size
        ));
    }

    recommendations
}

/// Auto-select optimal quantization algorithm based on cache analysis
pub fn quantize_with_cache_optimization(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
    cache_params: Option<&CacheAwareParams>,
) -> TorshResult<()> {
    let default_params = CacheAwareParams::default();
    let params = cache_params.unwrap_or(&default_params);
    let total_bytes = std::mem::size_of_val(input);

    if total_bytes <= params.l1_cache_size {
        // Small data - use simple sequential processing
        quantize_streaming_with_prefetch(input, scale, zero_point, output, params)
    } else if total_bytes <= params.l2_cache_size {
        // Medium data - use cache-aware chunking
        quantize_per_tensor_affine_cache_aware(input, scale, zero_point, output, params)
    } else {
        // Large data - use parallel processing with cache-friendly chunks
        quantize_per_tensor_affine_cache_aware(input, scale, zero_point, output, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{QScheme, QuantConfig};

    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_calculate_affine_quantization_params() {
        // Test normal case
        let (scale, zero_point) =
            calculate_affine_quantization_params(-1.0, 1.0, DType::I8).unwrap();

        assert!(scale > 0.0);
        assert!(zero_point >= -128 && zero_point <= 127);

        // Test edge case: min == max
        let (scale, zero_point) =
            calculate_affine_quantization_params(1.0, 1.0, DType::I8).unwrap();

        assert_eq!(scale, 1.0);
        assert_eq!(zero_point, -128);

        // Test invalid case: min > max
        let result = calculate_affine_quantization_params(2.0, 1.0, DType::I8);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_symmetric_quantization_params() {
        // Test normal case
        let (scale, zero_point) =
            calculate_symmetric_quantization_params(-2.0, 1.0, DType::I8).unwrap();

        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Symmetric always has zero_point = 0

        // Test edge case: zero range
        let (scale, zero_point) =
            calculate_symmetric_quantization_params(0.0, 0.0, DType::I8).unwrap();

        assert_eq!(scale, 1.0);
        assert_eq!(zero_point, 0);
    }

    #[test]
    fn test_get_dtype_range() {
        assert_eq!(get_dtype_range(DType::I8), (-128, 127));
        assert_eq!(get_dtype_range(DType::U8), (0, 255));
        assert_eq!(get_dtype_range(DType::I16), (-32768, 32767));
    }

    #[test]
    fn test_quantize_per_tensor_affine() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = tensor_1d(&data).unwrap();

        let (quantized, scale, zero_point) = quantize_per_tensor_affine(&tensor, 0.1, 0).unwrap();

        let quantized_data = quantized.data().unwrap();

        // Verify quantization: (value / scale) + zero_point
        assert_eq!(quantized_data[0], 10.0); // (1.0 / 0.1) + 0 = 10
        assert_eq!(quantized_data[1], 20.0); // (2.0 / 0.1) + 0 = 20
        assert_eq!(scale, 0.1);
        assert_eq!(zero_point, 0);
    }

    #[test]
    fn test_dequantize_per_tensor_affine() {
        let quantized_data = vec![10.0, 20.0, 30.0, 40.0];
        let quantized_tensor = tensor_1d(&quantized_data).unwrap();

        let dequantized = dequantize_per_tensor_affine(&quantized_tensor, 0.1, 0).unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Verify dequantization: (quantized_value - zero_point) * scale
        assert!((dequantized_data[0] - 1.0).abs() < 1e-6); // (10 - 0) * 0.1 = 1.0
        assert!((dequantized_data[1] - 2.0).abs() < 1e-6); // (20 - 0) * 0.1 = 2.0
    }

    #[test]
    fn test_quantize_tensor_auto() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let (quantized, scale, zero_point) =
            quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorAffine).unwrap();

        assert!(scale > 0.0);
        assert!(zero_point >= -128 && zero_point <= 127);

        // Verify tensor was quantized
        let quantized_data = quantized.data().unwrap();
        assert_eq!(quantized_data.len(), data.len());
    }

    #[test]
    fn test_quantize_with_config() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        let result = quantize_with_config(&tensor, &config);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert!(zero_point >= -128 && zero_point <= 127);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_dequantize() {
        let quantized_data = vec![64.0, 128.0, -64.0, 0.0];
        let quantized_tensor = tensor_1d(&quantized_data).unwrap();

        let dequantized = dequantize(&quantized_tensor, 0.5, 0).unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Test dequantization with scale 0.5
        assert!((dequantized_data[0] - 32.0).abs() < 1e-6);
        assert!((dequantized_data[1] - 64.0).abs() < 1e-6);
        assert!((dequantized_data[2] + 32.0).abs() < 1e-6);
        assert!((dequantized_data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_auto() {
        let data = vec![0.5, 1.0, 1.5, 2.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        let result = quantize_auto(&tensor, &config);
        assert!(result.is_ok());

        let (quantized, scale, _zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_error_cases() {
        // Test invalid scale
        let data = vec![1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = quantize_per_tensor_affine(&tensor, -1.0, 0);
        assert!(result.is_err());

        let result = dequantize_per_tensor_affine(&tensor, 0.0, 0);
        assert!(result.is_err());

        // Test empty tensor
        let empty_data: Vec<f32> = vec![];
        let empty_tensor = tensor_1d(&empty_data).unwrap();

        let result = quantize_tensor_auto(&empty_tensor, DType::I8, QScheme::PerTensorAffine);
        assert!(result.is_err());
    }

    // ===== Cache-Aware Algorithm Tests =====

    #[test]
    fn test_cache_aware_quantization() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        let cache_params = CacheAwareParams::default();

        let result =
            quantize_per_tensor_affine_cache_aware(&input, 0.1, 0, &mut output, &cache_params);

        assert!(result.is_ok());
        assert_eq!(output[0], 10.0);
        assert_eq!(output[7], 80.0);
    }

    #[test]
    fn test_cache_optimized_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cache_params = CacheAwareParams::default();

        let result = calculate_tensor_stats_cache_optimized(&data, &cache_params);
        assert!(result.is_ok());

        let (min_val, max_val, mean, std_dev) = result.unwrap();
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 10.0);
        assert!((mean - 5.5).abs() < 0.001);
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_matrix_cache_friendly_quantization() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = vec![0.0; 9];
        let cache_params = CacheAwareParams::default();

        let result =
            quantize_matrix_cache_friendly(&matrix, 3, 3, 0.1, 0, &mut output, &cache_params);

        assert!(result.is_ok());
        assert_eq!(output[0], 10.0); // 1.0 / 0.1 = 10
        assert_eq!(output[8], 90.0); // 9.0 / 0.1 = 90
    }

    #[test]
    fn test_streaming_with_prefetch() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut output = vec![0.0; 5];
        let cache_params = CacheAwareParams::default();

        let result = quantize_streaming_with_prefetch(&input, 0.01, 10, &mut output, &cache_params);

        assert!(result.is_ok());
        // 0.1 / 0.01 + 10 = 10 + 10 = 20
        assert_eq!(output[0], 20.0);
    }

    #[test]
    fn test_cache_optimization_recommendations() {
        let cache_params = CacheAwareParams::default();

        // Small tensor (L1 cache)
        let recommendations = get_cache_optimization_recommendations(1000, 4, &cache_params);
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].contains("L1 cache"));

        // Large tensor (beyond L3)
        let large_recommendations =
            get_cache_optimization_recommendations(10_000_000, 4, &cache_params);
        assert!(large_recommendations
            .iter()
            .any(|r| r.contains("streaming")));
    }

    #[test]
    fn test_auto_cache_optimization() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        // Test with default cache parameters
        let result = quantize_with_cache_optimization(&input, 0.1, 0, &mut output, None);

        assert!(result.is_ok());
        assert_eq!(output[0], 10.0);
        assert_eq!(output[3], 40.0);
    }

    #[test]
    fn test_cache_params_default() {
        let params = CacheAwareParams::default();

        assert_eq!(params.cache_line_size, 64);
        assert_eq!(params.l1_cache_size, 32 * 1024);
        assert_eq!(params.l2_cache_size, 256 * 1024);
        assert_eq!(params.l3_cache_size, 8 * 1024 * 1024);
        assert_eq!(params.prefetch_distance, 16);
        assert!(params.enable_chunking);
    }

    #[test]
    fn test_cache_aware_error_cases() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 3]; // Wrong size
        let cache_params = CacheAwareParams::default();

        let result =
            quantize_per_tensor_affine_cache_aware(&input, 0.1, 0, &mut output, &cache_params);
        assert!(result.is_err());

        // Test invalid scale
        let mut output_correct = vec![0.0; 2];
        let result = quantize_per_tensor_affine_cache_aware(
            &input,
            -0.1,
            0,
            &mut output_correct,
            &cache_params,
        );
        assert!(result.is_err());

        // Test empty data for stats
        let empty_data: Vec<f32> = vec![];
        let result = calculate_tensor_stats_cache_optimized(&empty_data, &cache_params);
        assert!(result.is_err());
    }
}
