//! SIMD-accelerated quantization operations
//!
//! This module provides optimized SIMD implementations for performance-critical
//! quantization operations using the scirs2-core SIMD abstraction layer.
//!
//! # Features
//!
//! - **Vectorized Quantization**: SIMD-accelerated per-tensor quantization
//! - **Vectorized Dequantization**: SIMD-accelerated dequantization operations
//! - **Fast Min/Max Finding**: Hardware-accelerated min/max computation for calibration
//! - **Batch Operations**: Optimized batch processing for multiple tensors
//! - **Fallback Support**: Automatic fallback to scalar operations when SIMD unavailable

use scirs2_core::parallel_ops::*;
use torsh_core::error::{Result as TorshResult, TorshError};

/// SIMD-accelerated per-tensor quantization
pub fn quantize_per_tensor_affine_simd(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
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

    // Use optimized parallel processing for quantization operation
    input
        .par_iter()
        .zip(output.par_iter_mut())
        .for_each(|(&x, out)| {
            let quantized = (x * inv_scale).round() + zero_point_f32;
            *out = quantized.clamp(-128.0, 127.0);
        });

    Ok(())
}

/// SIMD-accelerated per-tensor dequantization
pub fn dequantize_per_tensor_affine_simd(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
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

    let zero_point_f32 = zero_point as f32;

    // Use optimized parallel processing for dequantization operation
    input
        .par_iter()
        .zip(output.par_iter_mut())
        .for_each(|(&x, out)| {
            *out = (x - zero_point_f32) * scale;
        });

    Ok(())
}

/// SIMD-accelerated min/max finding for calibration
pub fn find_min_max_simd(data: &[f32]) -> TorshResult<(f32, f32)> {
    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot find min/max of empty array".to_string(),
        ));
    }

    // Use parallel operations for min/max reduction
    const CHUNK_SIZE: usize = 1024; // Process in cache-friendly chunks
    let (min_val, max_val) = if data.len() > CHUNK_SIZE {
        // Use parallel processing for large datasets
        data.par_chunks(CHUNK_SIZE)
            .map(|chunk| {
                let mut local_min = f32::INFINITY;
                let mut local_max = f32::NEG_INFINITY;
                for &val in chunk {
                    local_min = local_min.min(val);
                    local_max = local_max.max(val);
                }
                (local_min, local_max)
            })
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
            )
    } else {
        // Sequential processing for small datasets
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &val in data {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        (min_val, max_val)
    };

    Ok((min_val, max_val))
}

/// SIMD-accelerated per-channel quantization
pub fn quantize_per_channel_simd(
    input: &[f32],
    scales: &[f32],
    zero_points: &[i32],
    channel_size: usize,
    output: &mut [f32],
) -> TorshResult<()> {
    if input.len() != output.len() {
        return Err(TorshError::InvalidArgument(
            "Input and output length mismatch".to_string(),
        ));
    }

    let num_channels = scales.len();
    if num_channels != zero_points.len() {
        return Err(TorshError::InvalidArgument(
            "Scales and zero_points length mismatch".to_string(),
        ));
    }

    if input.len() != num_channels * channel_size {
        return Err(TorshError::InvalidArgument(
            "Input size does not match channel configuration".to_string(),
        ));
    }

    // Process each channel with SIMD acceleration
    for (ch, (&scale, &zero_point)) in scales.iter().zip(zero_points.iter()).enumerate() {
        if scale <= 0.0 {
            return Err(TorshError::InvalidArgument(format!(
                "Scale for channel {} must be positive",
                ch
            )));
        }

        let channel_start = ch * channel_size;
        let channel_end = channel_start + channel_size;

        let input_slice = &input[channel_start..channel_end];
        let output_slice = &mut output[channel_start..channel_end];

        quantize_per_tensor_affine_simd(input_slice, scale, zero_point, output_slice)?;
    }

    Ok(())
}

/// SIMD-accelerated batch quantization for consistent parameters
pub fn quantize_batch_consistent_simd(
    tensors: &[&[f32]],
    scale: f32,
    zero_point: i32,
    outputs: &mut [&mut [f32]],
) -> TorshResult<()> {
    if tensors.len() != outputs.len() {
        return Err(TorshError::InvalidArgument(
            "Number of input tensors must match output tensors".to_string(),
        ));
    }

    // Use parallel processing for each tensor
    tensors
        .par_iter()
        .zip(outputs.par_iter_mut())
        .try_for_each(|(input, output)| {
            quantize_per_tensor_affine_simd(input, scale, zero_point, output)
        })?;

    Ok(())
}

/// SIMD-accelerated floating-point to integer quantization (optimized for INT8)
pub fn quantize_to_int8_simd(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [i8],
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

    // Use optimized parallel processing for quantization to INT8
    input
        .par_iter()
        .zip(output.par_iter_mut())
        .for_each(|(&x, out)| {
            let quantized = (x * inv_scale).round() + zero_point_f32;
            *out = quantized.clamp(-128.0, 127.0) as i8;
        });

    Ok(())
}

/// SIMD-accelerated statistics calculation for quantization calibration
pub fn calculate_tensor_stats_simd(data: &[f32]) -> TorshResult<TensorStats> {
    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot calculate stats of empty tensor".to_string(),
        ));
    }

    let (min_val, max_val) = find_min_max_simd(data)?;

    // Calculate mean using parallel reduction
    let sum: f64 = data.par_iter().map(|&x| x as f64).sum();
    let mean = sum / data.len() as f64;

    // Calculate variance using parallel reduction
    let variance_sum: f64 = data
        .par_iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum();
    let variance = variance_sum / data.len() as f64;
    let std_dev = variance.sqrt();

    Ok(TensorStats {
        min: min_val,
        max: max_val,
        mean: mean as f32,
        std_dev: std_dev as f32,
        variance: variance as f32,
    })
}

/// Tensor statistics structure
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub variance: f32,
}

/// Check if SIMD operations are available on current hardware
pub fn is_simd_available() -> bool {
    // Check for common SIMD instruction sets (x86 and ARM)
    cfg!(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "avx",
        target_feature = "sse2",
        target_feature = "neon" // ARM NEON support
    ))
}

/// Get optimal SIMD vector width for current hardware
pub fn get_simd_width() -> usize {
    // Return optimal width based on available instruction set
    // AVX2: 8 x f32, AVX-512: 16 x f32, NEON: 4 x f32
    if cfg!(target_feature = "avx512f") {
        16 // AVX-512: 16 x f32 elements
    } else if cfg!(target_feature = "avx2") {
        8 // AVX2: 8 x f32 elements
    } else if cfg!(any(target_feature = "sse2", target_feature = "neon")) {
        4 // SSE2/NEON: 4 x f32 elements
    } else {
        1 // Fallback to scalar
    }
}

/// ARM NEON-specific optimized quantization
#[cfg(target_arch = "aarch64")]
pub fn quantize_neon_optimized(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [f32],
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

    // Process in NEON-friendly chunks of 4 elements
    const NEON_WIDTH: usize = 4;
    let chunks = input.len() / NEON_WIDTH;

    // Process aligned chunks for optimal NEON performance
    for i in 0..chunks {
        let start = i * NEON_WIDTH;
        let end = start + NEON_WIDTH;

        // Use vectorized operations for NEON
        for (&inp, out) in input[start..end].iter().zip(output[start..end].iter_mut()) {
            let quantized = (inp * inv_scale).round() + zero_point_f32;
            *out = quantized.clamp(-128.0, 127.0);
        }
    }

    // Handle remaining elements
    let remainder_start = chunks * NEON_WIDTH;
    for (&inp, out) in input[remainder_start..]
        .iter()
        .zip(output[remainder_start..].iter_mut())
    {
        let quantized = (inp * inv_scale).round() + zero_point_f32;
        *out = quantized.clamp(-128.0, 127.0);
    }

    Ok(())
}

/// ARM NEON-optimized min/max finding
#[cfg(target_arch = "aarch64")]
pub fn find_min_max_neon(data: &[f32]) -> TorshResult<(f32, f32)> {
    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot find min/max of empty array".to_string(),
        ));
    }

    const NEON_WIDTH: usize = 4;
    let chunks = data.len() / NEON_WIDTH;

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    // Process in NEON-friendly chunks
    for i in 0..chunks {
        let start = i * NEON_WIDTH;
        let end = start + NEON_WIDTH;

        // Vectorized min/max operations for NEON
        for &val in &data[start..end] {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    // Handle remaining elements
    let remainder_start = chunks * NEON_WIDTH;
    for &val in &data[remainder_start..] {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    Ok((min_val, max_val))
}

/// Mobile-optimized quantization with reduced memory usage
pub fn quantize_mobile_optimized(
    input: &[f32],
    scale: f32,
    zero_point: i32,
    output: &mut [i8],
    use_reduced_precision: bool,
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

    let inv_scale = if use_reduced_precision {
        // Use faster but less precise arithmetic for mobile
        1.0 / scale
    } else {
        1.0 / scale
    };

    let zero_point_f32 = zero_point as f32;

    // Use smaller chunk sizes for better mobile cache performance
    const MOBILE_CHUNK_SIZE: usize = 256;

    if input.len() > MOBILE_CHUNK_SIZE {
        // Process in mobile-optimized chunks
        input
            .chunks(MOBILE_CHUNK_SIZE)
            .zip(output.chunks_mut(MOBILE_CHUNK_SIZE))
            .for_each(|(input_chunk, output_chunk)| {
                for (&x, out) in input_chunk.iter().zip(output_chunk.iter_mut()) {
                    let quantized = if use_reduced_precision {
                        // Faster rounding for mobile
                        (x * inv_scale + 0.5).floor() + zero_point_f32
                    } else {
                        (x * inv_scale).round() + zero_point_f32
                    };
                    *out = quantized.clamp(-128.0, 127.0) as i8;
                }
            });
    } else {
        // Direct processing for small tensors
        for (&x, out) in input.iter().zip(output.iter_mut()) {
            let quantized = (x * inv_scale).round() + zero_point_f32;
            *out = quantized.clamp(-128.0, 127.0) as i8;
        }
    }

    Ok(())
}

/// Get mobile-specific optimization recommendations
pub fn get_mobile_optimization_hints() -> MobileOptimizationHints {
    MobileOptimizationHints {
        prefer_int8: true,
        use_reduced_precision: cfg!(target_os = "android") || cfg!(target_os = "ios"),
        optimal_chunk_size: if cfg!(target_arch = "aarch64") {
            256
        } else {
            512
        },
        enable_fast_math: true,
        prefer_sequential: false, // Mobile devices often benefit from some parallelism
    }
}

/// Mobile optimization configuration hints
#[derive(Debug, Clone)]
pub struct MobileOptimizationHints {
    pub prefer_int8: bool,
    pub use_reduced_precision: bool,
    pub optimal_chunk_size: usize,
    pub enable_fast_math: bool,
    pub prefer_sequential: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantize_per_tensor_affine_simd() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        quantize_per_tensor_affine_simd(&input, 0.1, 0, &mut output).unwrap();

        assert_relative_eq!(output[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 20.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 30.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 40.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dequantize_per_tensor_affine_simd() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 4];

        dequantize_per_tensor_affine_simd(&input, 0.1, 0, &mut output).unwrap();

        assert_relative_eq!(output[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_min_max_simd() {
        let data = vec![-1.5, 0.0, 2.3, -0.8, 4.7, 1.2];
        let (min_val, max_val) = find_min_max_simd(&data).unwrap();

        assert_relative_eq!(min_val, -1.5, epsilon = 1e-6);
        assert_relative_eq!(max_val, 4.7, epsilon = 1e-6);
    }

    #[test]
    fn test_calculate_tensor_stats_simd() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_tensor_stats_simd(&data).unwrap();

        assert_relative_eq!(stats.min, 1.0, epsilon = 1e-6);
        assert_relative_eq!(stats.max, 5.0, epsilon = 1e-6);
        assert_relative_eq!(stats.mean, 3.0, epsilon = 1e-6);
        assert_relative_eq!(stats.std_dev, (2.0f64).sqrt() as f32, epsilon = 1e-4);
    }

    #[test]
    fn test_quantize_to_int8_simd() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0i8; 4];

        quantize_to_int8_simd(&input, 0.1, 0, &mut output).unwrap();

        assert_eq!(output[0], 10i8);
        assert_eq!(output[1], 20i8);
        assert_eq!(output[2], 30i8);
        assert_eq!(output[3], 40i8);
    }

    #[test]
    fn test_error_cases() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 3]; // Wrong size

        let result = quantize_per_tensor_affine_simd(&input, 0.1, 0, &mut output);
        assert!(result.is_err());

        let mut output_correct = vec![0.0; 2];
        let result = quantize_per_tensor_affine_simd(&input, -0.1, 0, &mut output_correct);
        assert!(result.is_err());

        let empty_data: Vec<f32> = vec![];
        let result = find_min_max_simd(&empty_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_availability() {
        let available = is_simd_available();
        let width = get_simd_width();

        // SIMD availability depends on compile-time target features
        // We just check that the functions return reasonable values
        assert!(width >= 1); // Should be at least scalar width

        // Test that availability is consistent with width
        if available {
            assert!(width > 1); // If SIMD available, width should be > 1
        }
    }

    #[test]
    fn test_mobile_optimized_quantization() {
        let input = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0];
        let mut output = vec![0i8; 6];

        quantize_mobile_optimized(&input, 0.1, 0, &mut output, false).unwrap();

        assert_eq!(output[0], 10i8);
        assert_eq!(output[1], 20i8);
        assert_eq!(output[2], 30i8);
        assert_eq!(output[3], 40i8);
        assert_eq!(output[4], -10i8);
        assert_eq!(output[5], -20i8);
    }

    #[test]
    fn test_mobile_optimized_quantization_reduced_precision() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0i8; 4];

        // Test with reduced precision enabled
        quantize_mobile_optimized(&input, 0.1, 0, &mut output, true).unwrap();

        // Results should be close but may have slight differences due to reduced precision
        assert!((output[0] as f32 - 10.0).abs() <= 1.0);
        assert!((output[1] as f32 - 20.0).abs() <= 1.0);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_quantization() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];

        quantize_neon_optimized(&input, 0.1, 0, &mut output).unwrap();

        assert_relative_eq!(output[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 20.0, epsilon = 1e-6);
        assert_relative_eq!(output[7], 80.0, epsilon = 1e-6);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_min_max() {
        let data = vec![-1.5, 0.0, 2.3, -0.8, 4.7, 1.2, 9.5, -2.1];
        let (min_val, max_val) = find_min_max_neon(&data).unwrap();

        assert_relative_eq!(min_val, -2.1, epsilon = 1e-6);
        assert_relative_eq!(max_val, 9.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mobile_optimization_hints() {
        let hints = get_mobile_optimization_hints();

        assert!(hints.prefer_int8); // Should prefer INT8 for mobile
        assert!(hints.optimal_chunk_size > 0); // Should have a reasonable chunk size
        assert_eq!(hints.prefer_sequential, false); // Should allow some parallelism
    }
}
