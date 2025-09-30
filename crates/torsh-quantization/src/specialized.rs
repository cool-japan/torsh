//! Specialized quantization algorithms for advanced use cases
//!
//! This module provides advanced quantization techniques beyond standard INT8 quantization,
//! including low-bit quantization, extreme compression methods, and adaptive precision.
//!
//! # Features
//!
//! - **INT4 Quantization**: 4-bit quantization for extreme compression
//! - **Binary Quantization**: 1-bit quantization using {-1, +1} values
//! - **Ternary Quantization**: 2-bit quantization using {-1, 0, +1} values
//! - **Group-wise Quantization**: Channel grouping for improved accuracy
//! - **Mixed Precision**: Layer-specific precision assignment
//! - **Adaptive Thresholding**: Smart threshold selection for extreme quantization

use crate::config::{MixedPrecisionConfig, QuantConfig};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, string::String, vec::Vec};

use torsh_core::{
    dtype::DType,
    error::{Result as TorshResult, TorshError},
};
use torsh_tensor::Tensor;

/// INT4 quantization (4-bit per tensor)
pub fn quantize_int4_per_tensor(
    tensor: &Tensor,
    _config: &QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b)).min(0.0);
    let max_val = data
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        .max(0.0);

    // INT4 range: -8 to 7
    let scale = (max_val - min_val) / 15.0; // 15 = 7 - (-8)
    let scale = if scale == 0.0 { 1.0 } else { scale };

    let zero_point = (-8.0 - min_val / scale).round().clamp(-8.0, 7.0) as i32;

    let quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let quantized = (x / scale).round() + zero_point as f32;
            quantized.clamp(-8.0, 7.0) // Store as f32 for compatibility
        })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, zero_point))
}

/// INT4 per-channel quantization
pub fn quantize_int4_per_channel(
    tensor: &Tensor,
    axis: usize,
    _config: &QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
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
    for i in (0..shape.len().saturating_sub(1)).rev() {
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
        for (i, &val) in data.iter().enumerate() {
            let mut ch_idx = 0;
            let mut remaining = i;

            // Calculate channel index for this element
            for (dim, &stride) in strides.iter().enumerate() {
                let coord = remaining / stride;
                remaining %= stride;
                if dim == axis {
                    ch_idx = coord;
                }
            }

            if ch_idx == ch {
                channel_min = channel_min.min(val);
                channel_max = channel_max.max(val);
            }
        }

        // Ensure min <= max
        channel_min = channel_min.min(0.0);
        channel_max = channel_max.max(0.0);

        // Calculate INT4 quantization parameters for this channel
        let scale = (channel_max - channel_min) / 15.0; // INT4 range: -8 to 7
        let scale = if scale == 0.0 { 1.0 } else { scale };
        let zero_point = (-8.0 - channel_min / scale).round().clamp(-8.0, 7.0) as i32;

        scales.push(scale);
        zero_points.push(zero_point);

        // Quantize channel data
        for (i, &val) in data.iter().enumerate() {
            let mut ch_idx = 0;
            let mut remaining = i;

            for (dim, &stride) in strides.iter().enumerate() {
                let coord = remaining / stride;
                remaining %= stride;
                if dim == axis {
                    ch_idx = coord;
                }
            }

            if ch_idx == ch {
                let quantized = (val / scale).round() + zero_point as f32;
                quantized_data[i] = quantized.clamp(-8.0, 7.0);
            }
        }
    }

    let quantized_tensor = Tensor::from_data(quantized_data, shape.to_vec(), tensor.device())?;

    // Return average parameters for compatibility
    let avg_scale = scales.iter().sum::<f32>() / scales.len() as f32;
    let avg_zero_point =
        (zero_points.iter().sum::<i32>() as f32 / zero_points.len() as f32).round() as i32;

    Ok((quantized_tensor, avg_scale, avg_zero_point))
}

/// Binary quantization (-1, +1)
pub fn quantize_binary(tensor: &Tensor) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot quantize empty tensor".to_string(),
        ));
    }

    // Calculate scale as the mean of absolute values
    let scale = data.iter().map(|&x| x.abs()).sum::<f32>() / data.len() as f32;
    let scale = if scale == 0.0 { 1.0 } else { scale };

    let quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, 0)) // Binary is symmetric, so zero_point = 0
}

/// Ternary quantization (-1, 0, +1)
pub fn quantize_ternary(tensor: &Tensor) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot quantize empty tensor".to_string(),
        ));
    }

    // Calculate threshold as fraction of max absolute value
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let threshold = max_abs * 0.7; // Threshold parameter

    // Calculate scale from non-zero values
    let non_zero_sum: f32 = data
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .map(|&x| x.abs())
        .sum();
    let non_zero_count = data.iter().filter(|&&x| x.abs() > threshold).count();

    let scale = if non_zero_count > 0 {
        non_zero_sum / non_zero_count as f32
    } else {
        1.0
    };

    let quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x.abs() <= threshold {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, 0)) // Ternary is symmetric, so zero_point = 0
}

/// Group-wise quantization (divide channels into groups and quantize per-group)
pub fn quantize_group_wise(
    tensor: &Tensor,
    axis: usize,
    group_size: usize,
    config: &QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    if group_size == 0 {
        return Err(TorshError::InvalidArgument(
            "Group size must be greater than 0".to_string(),
        ));
    }

    let num_channels = shape[axis];
    let num_groups = num_channels.div_ceil(group_size); // Ceiling division

    let data = tensor.data()?;
    let mut quantized_data = vec![0.0f32; data.len()];

    // Calculate strides for indexing (optimized version)
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut group_scales = Vec::new();
    let mut group_zero_points = Vec::new();

    // Process each group
    for group_idx in 0..num_groups {
        let start_ch = group_idx * group_size;
        let end_ch = (start_ch + group_size).min(num_channels);

        // Collect data for this group
        let mut group_data = Vec::new();
        for ch in start_ch..end_ch {
            // Extract data for this channel
            for (i, _) in data.iter().enumerate() {
                let idx = i;
                let mut ch_idx = 0;
                let mut remaining = idx;

                // Calculate channel index for this element
                for (dim, &stride) in strides.iter().enumerate() {
                    let coord = remaining / stride;
                    remaining %= stride;
                    if dim == axis {
                        ch_idx = coord;
                    }
                }

                if ch_idx == ch {
                    group_data.push(data[i]);
                }
            }
        }

        if group_data.is_empty() {
            continue;
        }

        // Calculate quantization parameters for this group
        let min_val = group_data
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b))
            .min(0.0);
        let max_val = group_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            .max(0.0);

        let (qmin, qmax) = config.get_qint_range();
        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let zero_point = (qmin as f32 - min_val / scale)
            .round()
            .max(qmin as f32)
            .min(qmax as f32) as i32;

        group_scales.push(scale);
        group_zero_points.push(zero_point);

        // Quantize this group's data
        for ch in start_ch..end_ch {
            for i in 0..data.len() {
                let idx = i;
                let mut ch_idx = 0;
                let mut remaining = idx;

                // Calculate channel index for this element
                for (dim, &stride) in strides.iter().enumerate() {
                    let coord = remaining / stride;
                    remaining %= stride;
                    if dim == axis {
                        ch_idx = coord;
                    }
                }

                if ch_idx == ch {
                    let quantized = (data[i] / scale).round() + zero_point as f32;
                    quantized_data[i] = quantized.max(qmin as f32).min(qmax as f32);
                }
            }
        }
    }

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    // Return average scale and zero_point for compatibility
    let avg_scale = if group_scales.is_empty() {
        1.0
    } else {
        group_scales.iter().sum::<f32>() / group_scales.len() as f32
    };
    let avg_zero_point = if group_zero_points.is_empty() {
        0
    } else {
        (group_zero_points.iter().sum::<i32>() as f32 / group_zero_points.len() as f32).round()
            as i32
    };

    Ok((quantized_tensor, avg_scale, avg_zero_point))
}

/// Mixed precision quantization for different layers
pub fn quantize_mixed_precision(
    tensors: &HashMap<String, Tensor>,
    config: &MixedPrecisionConfig,
) -> TorshResult<HashMap<String, (Tensor, f32, i32)>> {
    let mut results = HashMap::new();

    for (layer_name, tensor) in tensors {
        // Determine precision for this layer
        let precision = determine_layer_precision(layer_name, config);

        // Create quantization config for this precision
        let layer_config = create_precision_config(precision);

        // Quantize using appropriate scheme
        let result = crate::algorithms::quantize_with_config(tensor, &layer_config)?;
        results.insert(layer_name.clone(), result);
    }

    Ok(results)
}

/// Determine precision for a layer based on mixed precision config
pub fn determine_layer_precision(layer_name: &str, config: &MixedPrecisionConfig) -> DType {
    // Check exact matches first
    for (pattern, precision) in &config.layer_precision {
        if layer_name.contains(pattern) {
            return *precision;
        }
    }

    // Return default precision
    config.default_precision
}

/// Create quantization config for specific precision
pub fn create_precision_config(precision: DType) -> QuantConfig {
    match precision {
        DType::I8 => QuantConfig::int8(),
        DType::U8 => QuantConfig::uint8(),
        DType::F16 => {
            // For FP16, we don't quantize but use reduced precision
            QuantConfig {
                dtype: DType::F16,
                enable_fake_quant: false,
                ..Default::default()
            }
        }
        DType::F32 => {
            // Keep full precision
            QuantConfig {
                dtype: DType::F32,
                enable_fake_quant: false,
                ..Default::default()
            }
        }
        _ => QuantConfig::int8(), // Default fallback
    }
}

/// Advanced binary quantization with learned threshold
pub fn quantize_binary_learned_threshold(
    tensor: &Tensor,
    threshold: Option<f32>,
) -> TorshResult<(Tensor, f32, i32, f32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot quantize empty tensor".to_string(),
        ));
    }

    // Use provided threshold or learn it
    let threshold = threshold.unwrap_or_else(|| {
        // Simple threshold learning: mean of absolute values
        let abs_sum: f32 = data.iter().map(|&x| x.abs()).sum();
        abs_sum / data.len() as f32
    });

    // Calculate scale from values above threshold
    let above_threshold: Vec<f32> = data
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .cloned()
        .collect();

    let scale = if above_threshold.is_empty() {
        1.0
    } else {
        above_threshold.iter().map(|&x| x.abs()).sum::<f32>() / above_threshold.len() as f32
    };

    let quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x.abs() <= threshold {
                0.0
            } else if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, 0, threshold))
}

/// Adaptive ternary quantization with optimal threshold selection
pub fn quantize_ternary_adaptive(tensor: &Tensor) -> TorshResult<(Tensor, f32, i32, f32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot quantize empty tensor".to_string(),
        ));
    }

    // Find optimal threshold using search
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let mut best_threshold = 0.0;
    let mut best_error = f32::INFINITY;

    // Search for optimal threshold
    for i in 1..=10 {
        let threshold = max_abs * (i as f32 * 0.1);
        let error = calculate_ternary_error(&data, threshold);
        if error < best_error {
            best_error = error;
            best_threshold = threshold;
        }
    }

    // Apply quantization with best threshold
    let non_zero_sum: f32 = data
        .iter()
        .filter(|&&x| x.abs() > best_threshold)
        .map(|&x| x.abs())
        .sum();
    let non_zero_count = data.iter().filter(|&&x| x.abs() > best_threshold).count();

    let scale = if non_zero_count > 0 {
        non_zero_sum / non_zero_count as f32
    } else {
        1.0
    };

    let quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x.abs() <= best_threshold {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    let quantized_tensor = Tensor::from_data(
        quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;

    Ok((quantized_tensor, scale, 0, best_threshold))
}

/// Calculate quantization error for ternary quantization with given threshold
fn calculate_ternary_error(data: &[f32], threshold: f32) -> f32 {
    data.iter()
        .map(|&x| {
            let quantized = if x.abs() <= threshold {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                -1.0
            };
            (x - quantized).powi(2)
        })
        .sum::<f32>()
        / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_quantize_int4_per_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int4();

        let result = quantize_int4_per_tensor(&tensor, &config);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert!(zero_point >= -8 && zero_point <= 7);

        let quantized_data = quantized.data().unwrap();
        assert_eq!(quantized_data.len(), data.len());

        // Check that values are in INT4 range
        for &val in &quantized_data {
            assert!(val >= -8.0 && val <= 7.0);
        }
    }

    #[test]
    fn test_quantize_binary() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = quantize_binary(&tensor);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Binary is symmetric

        let quantized_data = quantized.data().unwrap();
        assert_eq!(quantized_data.len(), data.len());

        // Check that all values are either -1.0 or 1.0
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 1.0);
        }
    }

    #[test]
    fn test_quantize_ternary() {
        let data = vec![-3.0, -1.0, 0.1, 1.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = quantize_ternary(&tensor);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Ternary is symmetric

        let quantized_data = quantized.data().unwrap();
        assert_eq!(quantized_data.len(), data.len());

        // Check that all values are -1.0, 0.0, or 1.0
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_quantize_group_wise() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();
        let config = QuantConfig::group_wise(1, 2);

        let result = quantize_group_wise(&tensor, 1, 2, &config);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_mixed_precision() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "embedding".to_string(),
            tensor_1d(&[1.0, 2.0, 3.0]).unwrap(),
        );
        tensors.insert(
            "attention".to_string(),
            tensor_1d(&[4.0, 5.0, 6.0]).unwrap(),
        );

        let config = MixedPrecisionConfig::default();

        let result = quantize_mixed_precision(&tensors, &config);
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("embedding"));
        assert!(results.contains_key("attention"));
    }

    #[test]
    fn test_determine_layer_precision() {
        let config = MixedPrecisionConfig::default();

        let embedding_precision = determine_layer_precision("layer.embedding.weight", &config);
        assert_eq!(embedding_precision, DType::I8);

        let attention_precision = determine_layer_precision("layer.attention.query", &config);
        assert_eq!(attention_precision, DType::F16);

        let unknown_precision = determine_layer_precision("layer.unknown.weight", &config);
        assert_eq!(unknown_precision, DType::I8); // Default
    }

    #[test]
    fn test_binary_learned_threshold() {
        let data = vec![-2.0, -0.1, 0.1, 0.5, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = quantize_binary_learned_threshold(&tensor, Some(0.3));
        assert!(result.is_ok());

        let (quantized, scale, zero_point, threshold) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0);
        assert_eq!(threshold, 0.3);

        let quantized_data = quantized.data().unwrap();

        // Values below threshold should become 0, others ±1
        for (i, &original) in data.iter().enumerate() {
            let expected = if original.abs() <= 0.3 {
                0.0
            } else if original >= 0.0 {
                1.0
            } else {
                -1.0
            };
            assert_eq!(quantized_data[i], expected);
        }
    }

    #[test]
    fn test_ternary_adaptive() {
        let data = vec![-3.0, -0.5, 0.0, 0.5, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = quantize_ternary_adaptive(&tensor);
        assert!(result.is_ok());

        let (quantized, scale, zero_point, threshold) = result.unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0);
        assert!(threshold > 0.0);

        let quantized_data = quantized.data().unwrap();
        assert_eq!(quantized_data.len(), data.len());

        // All values should be -1, 0, or 1
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_error_cases() {
        // Test empty tensor
        let empty_data: Vec<f32> = vec![];
        let empty_tensor = tensor_1d(&empty_data).unwrap();

        assert!(quantize_binary(&empty_tensor).is_err());
        assert!(quantize_ternary(&empty_tensor).is_err());

        // Test invalid axis
        let data = vec![1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::group_wise(0, 2);

        let result = quantize_group_wise(&tensor, 5, 2, &config);
        assert!(result.is_err());

        // Test zero group size
        let result = quantize_group_wise(&tensor, 0, 0, &config);
        assert!(result.is_err());
    }
}
