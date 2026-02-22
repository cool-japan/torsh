//! Quantization utility functions and helpers

use crate::quantization::{
    BackendQuantConfig, CalibrationConfig, CalibrationMethod, DeploymentPlatform,
    QuantizationConfig, QuantizationParams, QuantizationScheme,
};
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::{BTreeMap, HashMap};

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Calculate quantization error metrics
pub fn calculate_quantization_error(
    original: &Tensor,
    quantized: &Tensor,
    params: &QuantizationParams,
) -> Result<QuantizationError> {
    // Dequantize for comparison
    let dequantized = params.dequantize(quantized)?;

    let orig_data = original.to_vec()?;
    let deq_data = dequantized.to_vec()?;

    if orig_data.len() != deq_data.len() {
        return Err(TorshError::InvalidArgument(
            "Tensor sizes do not match for error calculation".to_string(),
        ));
    }

    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut max_error: f32 = 0.0;
    let mut dot_product = 0.0;
    let mut norm_orig = 0.0;
    let mut norm_deq = 0.0;

    for (&orig, &deq) in orig_data.iter().zip(deq_data.iter()) {
        let error = orig - deq;
        let abs_error = error.abs();

        mse += error * error;
        mae += abs_error;
        max_error = max_error.max(abs_error);

        dot_product += orig * deq;
        norm_orig += orig * orig;
        norm_deq += deq * deq;
    }

    let n = orig_data.len() as f32;
    mse /= n;
    mae /= n;

    let cosine_similarity = if norm_orig > 0.0 && norm_deq > 0.0 {
        dot_product / (norm_orig.sqrt() * norm_deq.sqrt())
    } else {
        0.0
    };

    let snr = if mse > 0.0 {
        let signal_power = norm_orig / n;
        10.0 * (signal_power / mse).log10()
    } else {
        f32::INFINITY
    };

    Ok(QuantizationError {
        mse,
        mae,
        max_error,
        cosine_similarity,
        snr,
    })
}

/// Quantization error metrics
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Mean squared error
    pub mse: f32,
    /// Mean absolute error
    pub mae: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Cosine similarity
    pub cosine_similarity: f32,
    /// Signal-to-noise ratio in dB
    pub snr: f32,
}

impl QuantizationError {
    /// Check if quantization error is within acceptable bounds
    pub fn is_acceptable(&self, max_mse: f32, min_cosine: f32, min_snr: f32) -> bool {
        self.mse <= max_mse && self.cosine_similarity >= min_cosine && self.snr >= min_snr
    }
}

/// Create quantization config for different deployment scenarios
pub fn create_deployment_config(platform: DeploymentPlatform) -> QuantizationConfig {
    match platform {
        DeploymentPlatform::Mobile => {
            QuantizationConfig {
                dtype: DType::U8,
                scheme: QuantizationScheme::Asymmetric,
                backend_config: BackendQuantConfig {
                    use_hardware_acceleration: true,
                    enable_kernel_fusion: true,
                    optimize_memory_layout: true,
                    target_platform: DeploymentPlatform::Mobile,
                },
                calibration: CalibrationConfig {
                    num_samples: 50, // Fewer samples for mobile
                    method: CalibrationMethod::MinMax,
                    outlier_percentile: 99.9,
                    use_moving_average: true,
                    momentum: 0.9,
                },
                per_channel: true,
                quantize_weights: true,
                quantize_activations: true,
            }
        }
        DeploymentPlatform::Edge => {
            QuantizationConfig {
                dtype: DType::I8,
                scheme: QuantizationScheme::Symmetric,
                backend_config: BackendQuantConfig {
                    use_hardware_acceleration: false, // Edge devices may not have specialized hardware
                    enable_kernel_fusion: false,
                    optimize_memory_layout: true,
                    target_platform: DeploymentPlatform::Edge,
                },
                calibration: CalibrationConfig {
                    num_samples: 25, // Very few samples for edge
                    method: CalibrationMethod::MinMax,
                    outlier_percentile: 99.5,
                    use_moving_average: false,
                    momentum: 0.9,
                },
                per_channel: false, // Simpler quantization
                quantize_weights: true,
                quantize_activations: false, // Keep activations in FP32 for accuracy
            }
        }
        DeploymentPlatform::Server => {
            QuantizationConfig {
                dtype: DType::I8,
                scheme: QuantizationScheme::KLDivergence,
                backend_config: BackendQuantConfig {
                    use_hardware_acceleration: true,
                    enable_kernel_fusion: true,
                    optimize_memory_layout: true,
                    target_platform: DeploymentPlatform::Server,
                },
                calibration: CalibrationConfig {
                    num_samples: 500, // More samples for better accuracy
                    method: CalibrationMethod::Entropy,
                    outlier_percentile: 99.99,
                    use_moving_average: true,
                    momentum: 0.95,
                },
                per_channel: true,
                quantize_weights: true,
                quantize_activations: true,
            }
        }
        DeploymentPlatform::WASM => {
            QuantizationConfig {
                dtype: DType::U8,
                scheme: QuantizationScheme::Asymmetric,
                backend_config: BackendQuantConfig {
                    use_hardware_acceleration: false, // WASM limitations
                    enable_kernel_fusion: false,
                    optimize_memory_layout: true,
                    target_platform: DeploymentPlatform::WASM,
                },
                calibration: CalibrationConfig {
                    num_samples: 100,
                    method: CalibrationMethod::MinMax,
                    outlier_percentile: 99.9,
                    use_moving_average: true,
                    momentum: 0.9,
                },
                per_channel: false, // Keep it simple for WASM
                quantize_weights: true,
                quantize_activations: true,
            }
        }
        _ => QuantizationConfig::default(),
    }
}

/// Analyze tensor for optimal quantization parameters
pub fn analyze_tensor_distribution(tensor: &Tensor) -> Result<TensorDistributionStats> {
    let data = tensor.to_vec()?;

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sum_squares = 0.0;

    for &value in &data {
        min_val = min_val.min(value);
        max_val = max_val.max(value);
        sum += value;
        sum_squares += value * value;
    }

    let n = data.len() as f32;
    let mean = sum / n;
    let variance = (sum_squares / n) - (mean * mean);
    let std_dev = variance.sqrt();

    // Calculate percentiles
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("data comparison should not involve NaN")
    });

    let percentiles = calculate_percentiles(&sorted_data);

    // Check for sparsity
    let zero_count = data.iter().filter(|&&x| x.abs() < 1e-7).count();
    let sparsity = zero_count as f32 / n;

    // Detect outliers using IQR method
    let q1 = percentiles.get(&25).copied().unwrap_or(min_val);
    let q3 = percentiles.get(&75).copied().unwrap_or(max_val);
    let iqr = q3 - q1;
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    let outlier_count = data
        .iter()
        .filter(|&&x| x < lower_bound || x > upper_bound)
        .count();
    let outlier_ratio = outlier_count as f32 / n;

    Ok(TensorDistributionStats {
        min_val,
        max_val,
        mean,
        std_dev,
        percentiles,
        sparsity,
        outlier_ratio,
        num_elements: data.len(),
    })
}

/// Calculate percentiles for a sorted array
fn calculate_percentiles(sorted_data: &[f32]) -> BTreeMap<i32, f32> {
    let mut percentiles = BTreeMap::new();
    let percentile_points = vec![1, 5, 10, 25, 50, 75, 90, 95, 99];

    for p in percentile_points {
        let p_frac = p as f32 / 100.0;
        let index = (p_frac * (sorted_data.len() - 1) as f32) as usize;
        let index = index.min(sorted_data.len() - 1);
        percentiles.insert(p, sorted_data[index]);
    }

    percentiles
}

/// Tensor distribution statistics
#[derive(Debug, Clone)]
pub struct TensorDistributionStats {
    pub min_val: f32,
    pub max_val: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub percentiles: BTreeMap<i32, f32>,
    pub sparsity: f32,
    pub outlier_ratio: f32,
    pub num_elements: usize,
}

impl TensorDistributionStats {
    /// Recommend optimal quantization scheme based on distribution
    pub fn recommend_quantization_scheme(&self) -> QuantizationScheme {
        // If tensor has many outliers, use percentile-based quantization
        if self.outlier_ratio > 0.05 {
            return QuantizationScheme::Percentile(99.9);
        }

        // If tensor is sparse, dynamic quantization might be better
        if self.sparsity > 0.7 {
            return QuantizationScheme::Dynamic;
        }

        // If tensor has both positive and negative values around zero, use symmetric
        if self.min_val < 0.0
            && self.max_val > 0.0
            && (self.min_val.abs() - self.max_val.abs()).abs() / self.max_val.abs() < 0.2
        {
            return QuantizationScheme::Symmetric;
        }

        // Otherwise, use asymmetric for better range utilization
        QuantizationScheme::Asymmetric
    }

    /// Check if tensor is suitable for quantization
    pub fn is_quantizable(&self, min_dynamic_range: f32) -> bool {
        let dynamic_range = self.max_val - self.min_val;
        dynamic_range > min_dynamic_range && self.std_dev > 1e-6
    }
}

/// Batch process multiple tensors for quantization analysis
pub fn batch_analyze_tensors(tensors: &[&Tensor]) -> Result<Vec<TensorDistributionStats>> {
    tensors
        .iter()
        .map(|tensor| analyze_tensor_distribution(tensor))
        .collect()
}

/// Find optimal bit-width for quantization given error constraints
pub fn find_optimal_bitwidth(
    tensor: &Tensor,
    max_error: f32,
    test_bitwidths: &[usize],
) -> Result<Option<usize>> {
    let mut best_bitwidth = None;

    for &bitwidth in test_bitwidths {
        let dtype = match bitwidth {
            8 => DType::I8,
            16 => DType::I16,
            _ => continue, // Skip unsupported bit-widths
        };

        // Create quantization parameters
        let data = tensor.to_vec()?;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let scale = match dtype {
            DType::I8 => max_val.abs().max(min_val.abs()) / 127.0,
            DType::I16 => max_val.abs().max(min_val.abs()) / 32767.0,
            _ => continue,
        };

        let params = QuantizationParams::symmetric(scale, DType::F32, dtype);

        // Test quantization
        let quantized = params.quantize(tensor)?;
        let error = calculate_quantization_error(tensor, &quantized, &params)?;

        if error.mse <= max_error * max_error {
            best_bitwidth = Some(bitwidth);
            break; // Found the smallest acceptable bit-width
        }
    }

    Ok(best_bitwidth)
}

/// Create mixed precision configuration based on layer sensitivity analysis
pub fn create_mixed_precision_config(
    layer_sensitivities: &HashMap<String, f32>,
    error_threshold: f32,
) -> Vec<(String, DType)> {
    let mut config = Vec::new();

    for (layer_name, &sensitivity) in layer_sensitivities {
        let dtype = if sensitivity > error_threshold * 2.0 {
            DType::F16 // Keep high precision for sensitive layers
        } else if sensitivity > error_threshold {
            DType::I16 // Medium precision
        } else {
            DType::I8 // Low precision for robust layers
        };

        config.push((layer_name.clone(), dtype));
    }

    config
}

/// Estimate model size reduction from quantization
pub fn estimate_size_reduction(
    original_dtype: DType,
    quantized_dtype: DType,
    num_parameters: usize,
) -> SizeReduction {
    let original_bits = match original_dtype {
        DType::F32 => 32,
        DType::F16 => 16,
        _ => 32,
    };

    let quantized_bits = match quantized_dtype {
        DType::I8 | DType::U8 => 8,
        DType::I16 => 16,
        DType::F16 => 16,
        DType::F32 => 32,
        _ => 32,
    };

    let original_size = (num_parameters * original_bits) / 8; // bytes
    let quantized_size = (num_parameters * quantized_bits) / 8; // bytes
    let reduction_bytes = original_size - quantized_size;
    let reduction_ratio = original_size as f32 / quantized_size as f32;

    SizeReduction {
        original_size_bytes: original_size,
        quantized_size_bytes: quantized_size,
        reduction_bytes,
        reduction_ratio,
        compression_percentage: (1.0 - (quantized_size as f32 / original_size as f32)) * 100.0,
    }
}

/// Model size reduction statistics
#[derive(Debug, Clone)]
pub struct SizeReduction {
    pub original_size_bytes: usize,
    pub quantized_size_bytes: usize,
    pub reduction_bytes: usize,
    pub reduction_ratio: f32,
    pub compression_percentage: f32,
}

impl SizeReduction {
    /// Format size in human-readable units
    pub fn format_size(&self) -> (String, String) {
        let original = format_bytes(self.original_size_bytes);
        let quantized = format_bytes(self.quantized_size_bytes);
        (original, quantized)
    }
}

/// Format bytes in human-readable units
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_tensor_distribution_analysis() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.1, -0.1];
        let tensor = Tensor::from_data(data, vec![9], torsh_core::device::DeviceType::Cpu).unwrap();

        let stats = analyze_tensor_distribution(&tensor).unwrap();

        assert!(stats.min_val <= -2.0);
        assert!(stats.max_val >= 2.0);
        assert!(stats.mean.abs() < 0.5); // Should be close to zero
        assert!(stats.num_elements == 9);
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
    }

    #[test]
    fn test_quantization_error_calculation() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0];
        let original =
            Tensor::from_data(original_data, vec![4], torsh_core::device::DeviceType::Cpu).unwrap();

        let params = QuantizationParams::symmetric(4.0 / 127.0, DType::F32, DType::I8);
        let quantized = params.quantize(&original).unwrap();

        let error = calculate_quantization_error(&original, &quantized, &params).unwrap();

        assert!(error.mse >= 0.0);
        assert!(error.mae >= 0.0);
        assert!(error.max_error >= 0.0);
        assert!(error.cosine_similarity >= 0.0 && error.cosine_similarity <= 1.0);
    }

    #[test]
    fn test_deployment_config_creation() {
        let mobile_config = create_deployment_config(DeploymentPlatform::Mobile);
        assert_eq!(mobile_config.dtype, DType::U8);
        assert_eq!(mobile_config.scheme, QuantizationScheme::Asymmetric);
        assert!(mobile_config.per_channel);

        let edge_config = create_deployment_config(DeploymentPlatform::Edge);
        assert_eq!(edge_config.dtype, DType::I8);
        assert_eq!(edge_config.scheme, QuantizationScheme::Symmetric);
        assert!(!edge_config.per_channel);
    }

    #[test]
    fn test_size_reduction_estimation() {
        let reduction = estimate_size_reduction(DType::F32, DType::I8, 1000);

        assert_eq!(reduction.original_size_bytes, 4000); // 1000 * 32 bits / 8
        assert_eq!(reduction.quantized_size_bytes, 1000); // 1000 * 8 bits / 8
        assert_eq!(reduction.reduction_bytes, 3000);
        assert_eq!(reduction.reduction_ratio, 4.0);
        assert_eq!(reduction.compression_percentage, 75.0);
    }

    #[test]
    fn test_optimal_bitwidth_finding() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![5], torsh_core::device::DeviceType::Cpu).unwrap();

        let bitwidths = vec![8, 16];
        let result = find_optimal_bitwidth(&tensor, 0.1, &bitwidths).unwrap();

        // Should find some acceptable bit-width
        assert!(result.is_some());
    }
}
