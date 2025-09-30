//! Quantization utilities and helper functions
//!
//! This module provides a comprehensive set of utility functions for quantization operations
//! including configuration validation, batch processing, error diagnostics, performance
//! optimization, benchmarking, and reporting tools.

use crate::{config::QuantConfig, observers::Observer};
use torsh_core::{error::Result as TorshResult, TorshError};
use torsh_tensor::Tensor;
// Note: ops module with dequantize and quantize_per_tensor not found, will implement locally

/// Implementation of quantize_per_tensor using algorithms module
fn quantize_per_tensor(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
    _dtype: torsh_core::DType,
) -> TorshResult<Tensor> {
    let (quantized, _, _) =
        crate::algorithms::quantize_per_tensor_affine(tensor, scale, zero_point)?;
    Ok(quantized)
}

/// Implementation of dequantize using algorithms module
#[allow(dead_code)]
fn dequantize(tensor: &Tensor, scale: f32, zero_point: i32) -> TorshResult<Tensor> {
    crate::algorithms::dequantize_per_tensor_affine(tensor, scale, zero_point)
}

/// Enhanced configuration validator with helpful suggestions
///
/// Validates a quantization configuration and provides performance and accuracy suggestions
/// based on the configuration parameters.
///
/// # Arguments
/// * `config` - The quantization configuration to validate
///
/// # Returns
/// A vector of suggestion strings for optimization
pub fn validate_config_with_suggestions(config: &QuantConfig) -> TorshResult<Vec<String>> {
    use crate::config::{ObserverType, QScheme, QuantBackend};

    let mut suggestions = Vec::new();

    // Run basic validation first
    config.validate()?;

    // Add performance suggestions
    match config.scheme {
        QScheme::PerChannelAffine | QScheme::PerChannelSymmetric => {
            if config.observer_type == ObserverType::MinMax {
                suggestions.push("Consider using Histogram observer for per-channel quantization for better accuracy".to_string());
            }
        }
        QScheme::GroupWise => {
            if let Some(group_size) = config.group_size {
                if group_size < 8 {
                    suggestions.push("Very small group sizes may not provide significant benefits over per-channel quantization".to_string());
                } else if group_size > 128 {
                    suggestions.push(
                        "Large group sizes may reduce the benefits of group-wise quantization"
                            .to_string(),
                    );
                }
            }
        }
        QScheme::Int4PerTensor | QScheme::Int4PerChannel => {
            if config.observer_type == ObserverType::MinMax {
                suggestions.push("Consider using Histogram observer for INT4 quantization to handle outliers better".to_string());
            }
        }
        QScheme::Binary | QScheme::Ternary => {
            if config.observer_type != ObserverType::MinMax {
                suggestions.push(
                    "MinMax observer is typically sufficient for binary/ternary quantization"
                        .to_string(),
                );
            }
        }
        _ => {}
    }

    // Backend suggestions
    if config.backend == QuantBackend::Native {
        suggestions.push(
            "Consider using FBGEMM or QNNPACK backends for better performance in production"
                .to_string(),
        );
    }

    // Observer suggestions
    if config.enable_fake_quant && config.observer_type != ObserverType::MovingAverage {
        suggestions
            .push("MovingAverage observer is recommended for QAT (fake quantization)".to_string());
    }

    Ok(suggestions)
}

/// Create optimized configuration for common use cases
///
/// Generates optimized quantization configurations for specific use cases and target platforms.
///
/// # Arguments
/// * `use_case` - The target use case ("inference_cpu", "inference_mobile", "training", etc.)
/// * `target_platform` - The target platform ("x86", "arm", "gpu", etc.)
///
/// # Returns
/// An optimized quantization configuration
pub fn create_optimized_config(use_case: &str, target_platform: &str) -> TorshResult<QuantConfig> {
    use crate::config::{ObserverType, QuantBackend, ReduceRange};

    let base_config = match use_case.to_lowercase().as_str() {
        "inference_cpu" => QuantConfig::int8()
            .with_backend(QuantBackend::Fbgemm)
            .with_observer(ObserverType::Histogram),
        "inference_mobile" => QuantConfig::int8()
            .with_backend(QuantBackend::Qnnpack)
            .with_observer(ObserverType::MinMax)
            .with_reduce_range(ReduceRange::Reduce),
        "training" => QuantConfig::qat().with_observer(ObserverType::MovingAverage),
        "extreme_compression" => QuantConfig::int4().with_observer(ObserverType::Histogram),
        "transformers" => QuantConfig::group_wise(0, 64).with_observer(ObserverType::Histogram),
        "edge_device" => QuantConfig::binary().with_observer(ObserverType::MinMax),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unknown use case: {use_case}"
            )))
        }
    };

    let optimized_config = match target_platform.to_lowercase().as_str() {
        "x86" | "x64" => base_config.with_backend(QuantBackend::Fbgemm),
        "arm" | "mobile" => base_config.with_backend(QuantBackend::Qnnpack),
        "gpu" => base_config.with_backend(QuantBackend::Native),
        _ => base_config,
    };

    Ok(optimized_config)
}

/// Batch quantization utility for multiple tensors with consistent parameters
///
/// Quantizes multiple tensors using globally consistent parameters calculated across all tensors.
/// This ensures that all tensors use the same scale and zero point for consistent quantization.
///
/// # Arguments
/// * `tensors` - Slice of tensor references to quantize
/// * `config` - Quantization configuration to use
///
/// # Returns
/// Vector of quantized tensors with their scale and zero point parameters
pub fn quantize_batch_consistent(
    tensors: &[&Tensor],
    config: &QuantConfig,
) -> TorshResult<Vec<(Tensor, f32, i32)>> {
    if tensors.is_empty() {
        return Ok(Vec::new());
    }

    // Calculate global statistics across all tensors for consistency
    let mut global_observer = Observer::new(config.observer_type);

    for tensor in tensors {
        global_observer.update(tensor)?;
    }

    let (global_scale, global_zero_point) = global_observer.calculate_qparams(config.dtype)?;

    // Quantize all tensors using the same parameters
    let mut results = Vec::new();
    for tensor in tensors {
        let quantized = quantize_per_tensor(tensor, global_scale, global_zero_point, config.dtype)?;
        results.push((quantized, global_scale, global_zero_point));
    }

    Ok(results)
}

/// Error recovery utility that provides detailed diagnostics
///
/// Analyzes a failed quantization operation and provides detailed diagnostics
/// about the tensor properties, configuration issues, and recovery suggestions.
///
/// # Arguments
/// * `tensor` - The tensor that failed to quantize
/// * `config` - The quantization configuration that was used
/// * `error` - The error that occurred during quantization
///
/// # Returns
/// A detailed diagnostic string with analysis and recovery suggestions
pub fn diagnose_quantization_failure(
    tensor: &Tensor,
    config: &QuantConfig,
    error: &TorshError,
) -> String {
    let mut diagnosis = format!("Quantization failed with error: {error}\n\n");

    // Analyze tensor properties
    let shape = tensor.shape();
    let data_result = tensor.data();

    diagnosis.push_str("Tensor Analysis:\n");
    diagnosis.push_str(&format!("  Shape: {:?}\n", shape.dims()));
    diagnosis.push_str(&format!("  Total elements: {}\n", shape.numel()));

    if let Ok(data) = data_result {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let has_nan = data.iter().any(|&x| x.is_nan());
        let has_inf = data.iter().any(|&x| x.is_infinite());

        diagnosis.push_str(&format!("  Data range: [{min_val:.6}, {max_val:.6}]\n"));
        diagnosis.push_str(&format!("  Contains NaN: {has_nan}\n"));
        diagnosis.push_str(&format!("  Contains Inf: {has_inf}\n"));

        if has_nan || has_inf {
            diagnosis.push_str(
                "\nSuggestion: Clean tensor data to remove NaN/Inf values before quantization.\n",
            );
        }

        if max_val - min_val < 1e-6 {
            diagnosis.push_str("\nSuggestion: Tensor has very small dynamic range. Consider using a different tensor or adjusting the quantization scheme.\n");
        }
    }

    // Analyze configuration
    diagnosis.push_str("\nConfiguration Analysis:\n");
    diagnosis.push_str(&format!("  Scheme: {:?}\n", config.scheme));
    diagnosis.push_str(&format!("  Observer: {:?}\n", config.observer_type));
    diagnosis.push_str(&format!("  Backend: {:?}\n", config.backend));

    match config.validate() {
        Ok(_) => diagnosis.push_str("  Configuration is valid\n"),
        Err(e) => diagnosis.push_str(&format!("  Configuration error: {e}\n")),
    }

    // Provide recovery suggestions
    diagnosis.push_str("\nRecovery Suggestions:\n");
    diagnosis.push_str(
        "1. Try a simpler quantization scheme (e.g., PerTensorAffine with MinMax observer)\n",
    );
    diagnosis.push_str("2. Use quantize_with_fallback() for automatic error recovery\n");
    diagnosis.push_str("3. Check tensor data for NaN/Inf values\n");
    diagnosis.push_str("4. Ensure tensor has sufficient dynamic range\n");
    diagnosis
        .push_str("5. Try a different observer type (Histogram for outlier-robust quantization)\n");

    diagnosis
}

/// Performance optimization hints based on tensor characteristics
///
/// Analyzes tensor properties and provides optimization hints for better quantization performance.
///
/// # Arguments
/// * `tensor` - The tensor to analyze
/// * `config` - The quantization configuration
///
/// # Returns
/// Vector of optimization hint strings
pub fn get_optimization_hints(tensor: &Tensor, config: &QuantConfig) -> Vec<String> {
    use crate::config::{ObserverType, QScheme};

    let mut hints = Vec::new();
    let shape = tensor.shape();
    let numel = shape.numel();

    // Size-based hints
    if numel > 1_000_000 {
        hints.push("Large tensor detected. Consider using parallel processing with Rayon for better performance.".to_string());
        if config.observer_type == ObserverType::Percentile {
            hints.push("For large tensors, Histogram observer may be more memory-efficient than Percentile observer.".to_string());
        }
    }

    // Shape-based hints
    if shape.dims().len() >= 2 && shape.dims().iter().any(|&dim| dim > 16) {
        hints.push("Multi-channel tensor detected. Per-channel or group-wise quantization may provide better accuracy.".to_string());
    }

    // Scheme-specific hints
    match config.scheme {
        QScheme::PerTensorAffine | QScheme::PerTensorSymmetric => {
            if shape.dims().len() > 2 {
                hints.push("Consider per-channel quantization for better accuracy with multi-dimensional tensors.".to_string());
            }
        }
        QScheme::GroupWise => {
            if let Some(group_size) = config.group_size {
                let total_elements = shape.dims().iter().product::<usize>();
                if total_elements / group_size < 4 {
                    hints.push("Too few groups for group-wise quantization. Consider per-tensor quantization instead.".to_string());
                }
            }
        }
        QScheme::Int4PerTensor | QScheme::Int4PerChannel => {
            hints.push("INT4 quantization detected. Ensure your inference backend supports INT4 operations.".to_string());
        }
        QScheme::Binary | QScheme::Ternary => {
            hints.push(
                "Extreme quantization scheme detected. Verify accuracy requirements are met."
                    .to_string(),
            );
        }
        _ => {}
    }

    hints
}

/// Export quantization configuration to JSON string
///
/// Serializes a quantization configuration to a JSON string for persistence or transfer.
///
/// # Arguments
/// * `config` - The quantization configuration to export
///
/// # Returns
/// A JSON string representation of the configuration
pub fn export_config_to_json(config: &QuantConfig) -> TorshResult<String> {
    match serde_json::to_string_pretty(config) {
        Ok(json) => Ok(json),
        Err(e) => Err(TorshError::InvalidArgument(format!(
            "Failed to serialize config: {e}"
        ))),
    }
}

/// Import quantization configuration from JSON string
///
/// Deserializes a quantization configuration from a JSON string.
///
/// # Arguments
/// * `json` - The JSON string to deserialize
///
/// # Returns
/// The deserialized quantization configuration
pub fn import_config_from_json(json: &str) -> TorshResult<QuantConfig> {
    match serde_json::from_str(json) {
        Ok(config) => Ok(config),
        Err(e) => Err(TorshError::InvalidArgument(format!(
            "Failed to deserialize config: {e}"
        ))),
    }
}
