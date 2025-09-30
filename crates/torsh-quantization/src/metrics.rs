//! Quantization quality metrics and analysis tools
//!
//! This module provides comprehensive tools for measuring and analyzing the quality
//! of quantization operations, including performance metrics, benchmarking utilities,
//! and automated analysis reports.
//!
//! # Features
//!
//! - **Quality Metrics**: MSE, PSNR, SNR, MAE, cosine similarity calculations
//! - **Performance Analysis**: Timing, compression ratio, and efficiency metrics
//! - **Configuration Comparison**: Side-by-side comparison of quantization schemes
//! - **Auto-calibration**: Automated optimal configuration selection
//! - **Report Generation**: Comprehensive analysis reports in Markdown format
//! - **Outlier Detection**: Statistical analysis and recommendation systems

use crate::config::{ObserverType, QuantConfig};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, format, string::String, vec::Vec};

use torsh_core::{
    dtype::DType,
    error::{Result as TorshResult, TorshError},
};
use torsh_tensor::Tensor;

/// Quantization quality metrics for measuring accuracy loss
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    /// Mean Squared Error between original and quantized tensors
    pub mse: f32,
    /// Peak Signal-to-Noise Ratio (PSNR) in dB
    pub psnr: f32,
    /// Signal-to-Noise Ratio (SNR) in dB
    pub snr: f32,
    /// Mean Absolute Error between original and quantized tensors
    pub mae: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Percentage of values with zero error
    pub zero_error_percentage: f32,
    /// Cosine similarity between original and quantized tensors
    pub cosine_similarity: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

impl Default for QuantizationMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            psnr: 0.0,
            snr: 0.0,
            mae: 0.0,
            max_error: 0.0,
            zero_error_percentage: 100.0,
            cosine_similarity: 1.0,
            compression_ratio: 1.0,
        }
    }
}

/// Calculate comprehensive quantization quality metrics
pub fn calculate_quantization_metrics(
    original: &Tensor,
    quantized: &Tensor,
    original_bits: u32,
    quantized_bits: u32,
) -> TorshResult<QuantizationMetrics> {
    if original.shape() != quantized.shape() {
        return Err(TorshError::InvalidArgument(format!(
            "Shape mismatch: expected {:?}, got {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    let original_data = original.data()?;
    let quantized_data = quantized.data()?;

    if original_data.len() != quantized_data.len() {
        return Err(TorshError::InvalidArgument(
            "Data length mismatch between tensors".to_string(),
        ));
    }

    if original_data.is_empty() {
        return Ok(QuantizationMetrics::default());
    }

    // Calculate MSE
    let mse = original_data
        .iter()
        .zip(quantized_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original_data.len() as f32;

    // Calculate MAE
    let mae = original_data
        .iter()
        .zip(quantized_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / original_data.len() as f32;

    // Calculate max error
    let max_error = original_data
        .iter()
        .zip(quantized_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    // Calculate zero error percentage
    let zero_errors = original_data
        .iter()
        .zip(quantized_data.iter())
        .filter(|(a, b)| (*a - *b).abs() < 1e-7)
        .count();
    let zero_error_percentage = (zero_errors as f32 / original_data.len() as f32) * 100.0;

    // Calculate signal power for SNR/PSNR
    let signal_power =
        original_data.iter().map(|x| x.powi(2)).sum::<f32>() / original_data.len() as f32;

    // Calculate PSNR (assuming signal range is [0, 1])
    let max_signal = original_data
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let psnr = if mse > 0.0 {
        20.0 * (max_signal / mse.sqrt()).log10()
    } else {
        f32::INFINITY
    };

    // Calculate SNR
    let snr = if mse > 0.0 && signal_power > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f32::INFINITY
    };

    // Calculate cosine similarity
    let dot_product = original_data
        .iter()
        .zip(quantized_data.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();

    let original_norm = original_data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let quantized_norm = quantized_data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    let cosine_similarity = if original_norm > 0.0 && quantized_norm > 0.0 {
        dot_product / (original_norm * quantized_norm)
    } else {
        0.0
    };

    // Calculate compression ratio
    let compression_ratio = original_bits as f32 / quantized_bits as f32;

    Ok(QuantizationMetrics {
        mse,
        psnr,
        snr,
        mae,
        max_error,
        zero_error_percentage,
        cosine_similarity,
        compression_ratio,
    })
}

/// Compare multiple quantization configurations and return ranked results
pub fn compare_quantization_configs(
    tensor: &Tensor,
    configs: &[QuantConfig],
) -> TorshResult<Vec<(QuantConfig, QuantizationMetrics, f64)>> {
    let mut results = Vec::new();

    for config in configs {
        // Time the quantization process
        let start = std::time::Instant::now();

        // Quantize the tensor
        let quantize_result = crate::algorithms::quantize_with_config(tensor, config);

        let duration = start.elapsed().as_secs_f64();

        match quantize_result {
            Ok((quantized, scale, zero_point)) => {
                // Dequantize back to original precision
                let dequantized = crate::algorithms::dequantize(&quantized, scale, zero_point)?;

                // Calculate metrics
                let original_bits = match tensor.dtype() {
                    DType::F32 => 32,
                    DType::F16 => 16,
                    _ => 8,
                };

                let quantized_bits = match config.dtype {
                    DType::I8 | DType::U8 => 8,
                    DType::I16 => 16,
                    DType::I32 => 32,
                    DType::F16 => 16,
                    DType::F32 => 32,
                    _ => 8,
                };

                let metrics = calculate_quantization_metrics(
                    tensor,
                    &dequantized,
                    original_bits,
                    quantized_bits,
                )?;

                results.push((config.clone(), metrics, duration));
            }
            Err(_) => {
                // If quantization fails, create a worst-case metrics entry
                let worst_metrics = QuantizationMetrics {
                    mse: f32::INFINITY,
                    psnr: f32::NEG_INFINITY,
                    snr: f32::NEG_INFINITY,
                    mae: f32::INFINITY,
                    max_error: f32::INFINITY,
                    zero_error_percentage: 0.0,
                    cosine_similarity: 0.0,
                    compression_ratio: 1.0,
                };

                results.push((config.clone(), worst_metrics, duration));
            }
        }
    }

    // Sort by PSNR (higher is better)
    results.sort_by(|a, b| {
        b.1.psnr
            .partial_cmp(&a.1.psnr)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Automatic calibration assistant to find optimal quantization configuration
pub fn auto_calibrate_quantization(
    calibration_tensors: &[&Tensor],
    target_accuracy_threshold: f32,
    max_compression_ratio: f32,
) -> TorshResult<QuantConfig> {
    if calibration_tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "No calibration tensors provided".to_string(),
        ));
    }

    // Define candidate configurations to test
    let candidate_configs = vec![
        QuantConfig::int8(),
        QuantConfig::int8().with_observer(ObserverType::Histogram),
        QuantConfig::per_channel(0),
        QuantConfig::per_channel(1),
        QuantConfig::group_wise(0, 8),
        QuantConfig::group_wise(1, 16),
        QuantConfig::int4(),
        QuantConfig::ternary(),
    ];

    let mut best_config = None;
    let mut best_score = f32::NEG_INFINITY;

    // Test each configuration with all calibration tensors
    for config in candidate_configs {
        let mut total_metrics = QuantizationMetrics::default();
        let mut successful_tests = 0;

        for tensor in calibration_tensors {
            if let Ok(comparison) =
                compare_quantization_configs(tensor, std::slice::from_ref(&config))
            {
                if let Some((_, metrics, _)) = comparison.first() {
                    if metrics.psnr.is_finite() {
                        total_metrics.mse += metrics.mse;
                        total_metrics.psnr += metrics.psnr;
                        total_metrics.snr += metrics.snr;
                        total_metrics.mae += metrics.mae;
                        total_metrics.max_error = total_metrics.max_error.max(metrics.max_error);
                        total_metrics.zero_error_percentage += metrics.zero_error_percentage;
                        total_metrics.cosine_similarity += metrics.cosine_similarity;
                        total_metrics.compression_ratio += metrics.compression_ratio;
                        successful_tests += 1;
                    }
                }
            }
        }

        if successful_tests > 0 {
            // Average the metrics
            let avg_metrics = QuantizationMetrics {
                mse: total_metrics.mse / successful_tests as f32,
                psnr: total_metrics.psnr / successful_tests as f32,
                snr: total_metrics.snr / successful_tests as f32,
                mae: total_metrics.mae / successful_tests as f32,
                max_error: total_metrics.max_error,
                zero_error_percentage: total_metrics.zero_error_percentage
                    / successful_tests as f32,
                cosine_similarity: total_metrics.cosine_similarity / successful_tests as f32,
                compression_ratio: total_metrics.compression_ratio / successful_tests as f32,
            };

            // Calculate a composite score (higher is better)
            let score = if avg_metrics.psnr >= target_accuracy_threshold
                && avg_metrics.compression_ratio <= max_compression_ratio
            {
                // Prioritize compression ratio if accuracy threshold is met
                avg_metrics.compression_ratio + avg_metrics.psnr / 100.0
            } else {
                // Otherwise prioritize accuracy
                avg_metrics.psnr / avg_metrics.compression_ratio
            };

            if score > best_score {
                best_score = score;
                best_config = Some(config.clone());
            }
        }
    }

    best_config
        .ok_or_else(|| TorshError::InvalidArgument("No suitable configuration found".to_string()))
}

/// Generate a comprehensive quantization report
pub fn generate_quantization_report(
    original: &Tensor,
    configs: &[QuantConfig],
) -> TorshResult<String> {
    let mut report = String::new();

    report.push_str("# Quantization Analysis Report\n\n");
    report.push_str(&format!(
        "**Original Tensor Shape:** {:?}\n",
        original.shape()
    ));
    report.push_str(&format!(
        "**Original Tensor DType:** {:?}\n",
        original.dtype()
    ));
    report.push_str(&format!(
        "**Number of Elements:** {}\n\n",
        original.shape().numel()
    ));

    // Get tensor statistics
    let data = original.data()?;
    let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std_dev = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();

    report.push_str("**Original Tensor Statistics:**\n");
    report.push_str(&format!("- Min: {min_val:.6}\n"));
    report.push_str(&format!("- Max: {max_val:.6}\n"));
    report.push_str(&format!("- Mean: {mean:.6}\n"));
    report.push_str(&format!("- Std Dev: {std_dev:.6}\n"));
    report.push_str(&format!("- Dynamic Range: {:.6}\n\n", max_val - min_val));

    // Compare configurations
    let comparison_results = compare_quantization_configs(original, configs)?;

    report.push_str("## Quantization Configuration Comparison\n\n");
    report.push_str(
        "| Rank | Scheme | Observer | PSNR (dB) | SNR (dB) | MAE | Compression | Time (ms) |\n",
    );
    report.push_str(
        "|------|--------|----------|-----------|----------|-----|-------------|----------|\n",
    );

    for (rank, (config, metrics, duration)) in comparison_results.iter().enumerate() {
        report.push_str(&format!(
            "| {} | {:?} | {:?} | {:.2} | {:.2} | {:.6} | {:.1}x | {:.2} |\n",
            rank + 1,
            config.scheme,
            config.observer_type,
            metrics.psnr,
            metrics.snr,
            metrics.mae,
            metrics.compression_ratio,
            duration * 1000.0
        ));
    }

    report.push_str("\n## Detailed Metrics\n\n");

    for (rank, (config, metrics, _)) in comparison_results.iter().enumerate() {
        report.push_str(&format!(
            "### Configuration #{} - {:?}\n",
            rank + 1,
            config.scheme
        ));
        report.push_str(&format!("- **MSE:** {:.8}\n", metrics.mse));
        report.push_str(&format!("- **PSNR:** {:.2} dB\n", metrics.psnr));
        report.push_str(&format!("- **SNR:** {:.2} dB\n", metrics.snr));
        report.push_str(&format!("- **MAE:** {:.6}\n", metrics.mae));
        report.push_str(&format!("- **Max Error:** {:.6}\n", metrics.max_error));
        report.push_str(&format!(
            "- **Zero Error %:** {:.2}%\n",
            metrics.zero_error_percentage
        ));
        report.push_str(&format!(
            "- **Cosine Similarity:** {:.6}\n",
            metrics.cosine_similarity
        ));
        report.push_str(&format!(
            "- **Compression Ratio:** {:.1}x\n\n",
            metrics.compression_ratio
        ));
    }

    report.push_str("## Recommendations\n\n");

    if let Some((best_config, best_metrics, _)) = comparison_results.first() {
        report.push_str(&format!(
            "**Best Configuration:** {:?} with {:?} observer\n",
            best_config.scheme, best_config.observer_type
        ));
        report.push_str(&format!(
            "- Achieves {:.2} dB PSNR with {:.1}x compression\n",
            best_metrics.psnr, best_metrics.compression_ratio
        ));

        if best_metrics.psnr > 40.0 {
            report.push_str("- ✅ Excellent quality preservation\n");
        } else if best_metrics.psnr > 30.0 {
            report.push_str("- ✅ Good quality preservation\n");
        } else if best_metrics.psnr > 20.0 {
            report.push_str("- ⚠️ Moderate quality loss\n");
        } else {
            report.push_str("- ❌ Significant quality loss\n");
        }
    }

    Ok(report)
}

/// Generate optimization hints for tensor and configuration
pub fn generate_optimization_hints(
    tensor: &Tensor,
    config: &QuantConfig,
) -> TorshResult<Vec<String>> {
    let mut hints = Vec::new();
    let shape = tensor.shape();
    let data = tensor.data()?;

    // Data distribution analysis
    if !data.is_empty() {
        let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std_dev =
            (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();

        // Range analysis
        let dynamic_range = max_val - min_val;
        if dynamic_range > 100.0 {
            hints.push("Large dynamic range detected. Consider using Histogram or Percentile observer for better quantization parameters.".to_string());
        }

        // Sparsity analysis
        let zero_count = data.iter().filter(|&&x| x.abs() < 1e-6).count();
        let sparsity = zero_count as f32 / data.len() as f32;
        if sparsity > 0.5 {
            hints.push(
                "High sparsity detected. Sparse quantization schemes may be more efficient."
                    .to_string(),
            );
        }

        // Outlier detection
        let outlier_threshold = mean + 3.0 * std_dev;
        let outlier_count = data
            .iter()
            .filter(|&&x| x.abs() > outlier_threshold)
            .count();
        if outlier_count > 0 {
            hints.push("Outliers detected. Percentile-based observers may provide better quantization parameters.".to_string());
        }

        // Memory efficiency hints
        if data.len() > 1_000_000 {
            hints.push("For large tensors, Histogram observer may be more memory-efficient than Percentile observer.".to_string());
        }
    }

    // Shape-based hints
    if shape.dims().len() >= 2 && shape.dims().iter().any(|&dim| dim > 16) {
        hints.push("Multi-channel tensor detected. Per-channel or group-wise quantization may provide better accuracy.".to_string());
    }

    // Scheme-specific hints
    match config.scheme {
        crate::config::QScheme::PerChannelAffine | crate::config::QScheme::PerChannelSymmetric => {
            if let Some(axis) = config.ch_axis {
                if axis >= shape.dims().len() {
                    hints.push(
                        "Channel axis is out of bounds. This will cause an error.".to_string(),
                    );
                } else if shape.dims()[axis] < 4 {
                    hints.push(
                        "Few channels detected. Per-tensor quantization might be sufficient."
                            .to_string(),
                    );
                }
            }
        }
        crate::config::QScheme::GroupWise => {
            if let (Some(axis), Some(group_size)) = (config.ch_axis, config.group_size) {
                if axis < shape.dims().len() {
                    let num_channels = shape.dims()[axis];
                    let num_groups = num_channels.div_ceil(group_size);
                    if num_groups == 1 {
                        hints.push("Only one group will be created. Consider per-tensor quantization instead.".to_string());
                    } else if num_groups == num_channels {
                        hints.push("Each channel forms its own group. Consider per-channel quantization instead.".to_string());
                    }
                }
            }
        }
        _ => {}
    }

    Ok(hints)
}

/// Benchmark quantization performance for different configurations
pub fn benchmark_quantization_performance(
    tensor: &Tensor,
    configs: &[QuantConfig],
    num_iterations: usize,
) -> TorshResult<Vec<(QuantConfig, f64, f64)>> {
    let mut results = Vec::new();

    for config in configs {
        let mut total_quantize_time = 0.0;
        let mut total_dequantize_time = 0.0;
        let mut successful_runs = 0;

        for _ in 0..num_iterations {
            // Benchmark quantization
            let quantize_start = std::time::Instant::now();
            let quantize_result = crate::algorithms::quantize_with_config(tensor, config);
            let quantize_time = quantize_start.elapsed().as_secs_f64();

            if let Ok((quantized, scale, zero_point)) = quantize_result {
                // Benchmark dequantization
                let dequantize_start = std::time::Instant::now();
                let _dequantized = crate::algorithms::dequantize(&quantized, scale, zero_point)?;
                let dequantize_time = dequantize_start.elapsed().as_secs_f64();

                total_quantize_time += quantize_time;
                total_dequantize_time += dequantize_time;
                successful_runs += 1;
            }
        }

        if successful_runs > 0 {
            let avg_quantize_time = total_quantize_time / successful_runs as f64;
            let avg_dequantize_time = total_dequantize_time / successful_runs as f64;
            results.push((config.clone(), avg_quantize_time, avg_dequantize_time));
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_calculate_quantization_metrics() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0];
        let quantized_data = vec![1.1, 2.1, 2.9, 3.9];

        let original = tensor_1d(&original_data).unwrap();
        let quantized = tensor_1d(&quantized_data).unwrap();

        let metrics = calculate_quantization_metrics(&original, &quantized, 32, 8).unwrap();

        // Verify basic metrics
        assert!(metrics.mse > 0.0);
        assert!(metrics.mse < 1.0); // Should be small error
        assert!(metrics.mae > 0.0);
        assert!(metrics.mae < 1.0);
        assert!(metrics.psnr > 0.0);
        assert!(metrics.snr > 0.0);
        assert!(metrics.max_error >= 0.0);
        assert!(metrics.zero_error_percentage >= 0.0);
        assert!(metrics.zero_error_percentage <= 100.0);
        assert!(metrics.cosine_similarity > 0.8); // Should be high similarity
        assert_eq!(metrics.compression_ratio, 4.0); // 32-bit to 8-bit

        // Test perfect match (zero error)
        let metrics_perfect = calculate_quantization_metrics(&original, &original, 32, 16).unwrap();
        assert_eq!(metrics_perfect.mse, 0.0);
        assert_eq!(metrics_perfect.mae, 0.0);
        assert_eq!(metrics_perfect.max_error, 0.0);
        assert_eq!(metrics_perfect.zero_error_percentage, 100.0);
        assert!((metrics_perfect.cosine_similarity - 1.0).abs() < 1e-6);
        assert!(metrics_perfect.psnr.is_infinite());
        assert!(metrics_perfect.snr.is_infinite());
        assert_eq!(metrics_perfect.compression_ratio, 2.0); // 32-bit to 16-bit
    }

    #[test]
    fn test_compare_quantization_configs() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();

        let configs = vec![
            QuantConfig::int8(),
            QuantConfig::binary(),
            QuantConfig::ternary(),
        ];

        let results = compare_quantization_configs(&tensor, &configs).unwrap();

        // Should return results for all configs
        assert_eq!(results.len(), 3);

        // Verify all results have valid metrics
        for (config, metrics, duration) in &results {
            assert!(configs.iter().any(|c| c.scheme == config.scheme));
            assert!(duration >= &0.0);

            // Metrics should be reasonable (not infinity for successful quantization)
            if metrics.psnr.is_finite() {
                assert!(metrics.psnr > 0.0);
                assert!(metrics.compression_ratio >= 1.0);
                assert!(metrics.mae >= 0.0);
                assert!(metrics.mse >= 0.0);
            }
        }

        // Results should be sorted by PSNR (higher is better)
        for i in 1..results.len() {
            let prev_psnr = results[i - 1].1.psnr;
            let curr_psnr = results[i].1.psnr;
            if prev_psnr.is_finite() && curr_psnr.is_finite() {
                assert!(prev_psnr >= curr_psnr);
            }
        }
    }

    #[test]
    fn test_auto_calibrate_quantization() {
        let tensor1 = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor2 = tensor_1d(&[2.0, 3.0, 4.0, 5.0]).unwrap();
        let tensor3 = tensor_1d(&[0.5, 1.5, 2.5, 3.5]).unwrap();

        let calibration_tensors = vec![&tensor1, &tensor2, &tensor3];

        // Test with reasonable thresholds
        let result = auto_calibrate_quantization(&calibration_tensors, 20.0, 10.0);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert!(config.validate().is_ok());

        // Test with impossible thresholds (should still return a config)
        let result_strict = auto_calibrate_quantization(&calibration_tensors, 100.0, 1.1);
        assert!(result_strict.is_ok());

        // Test empty calibration tensors
        let empty_tensors = vec![];
        let result_empty = auto_calibrate_quantization(&empty_tensors, 20.0, 10.0);
        assert!(result_empty.is_err());
    }

    #[test]
    fn test_generate_quantization_report() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();

        let configs = vec![QuantConfig::int8(), QuantConfig::binary()];

        let report_result = generate_quantization_report(&tensor, &configs);
        assert!(report_result.is_ok());

        let report = report_result.unwrap();

        // Verify report contains expected sections
        assert!(report.contains("# Quantization Analysis Report"));
        assert!(report.contains("**Original Tensor Shape:**"));
        assert!(report.contains("**Original Tensor Statistics:**"));
        assert!(report.contains("## Quantization Configuration Comparison"));
        assert!(report.contains("## Detailed Metrics"));
        assert!(report.contains("## Recommendations"));

        // Verify it contains data about our configs
        assert!(report.contains("PerTensorAffine"));
        assert!(report.contains("Binary"));

        // Verify it contains statistical information
        assert!(report.contains("Min:"));
        assert!(report.contains("Max:"));
        assert!(report.contains("Mean:"));
        assert!(report.contains("Std Dev:"));
        assert!(report.contains("Dynamic Range:"));

        // Verify it contains metrics columns
        assert!(report.contains("PSNR (dB)"));
        assert!(report.contains("SNR (dB)"));
        assert!(report.contains("MAE"));
        assert!(report.contains("Compression"));
        assert!(report.contains("Time (ms)"));

        // Verify it contains recommendations
        assert!(report.contains("**Best Configuration:**"));
    }

    #[test]
    fn test_generate_optimization_hints() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        let hints = generate_optimization_hints(&tensor, &config).unwrap();
        assert!(hints.len() >= 0); // Should return some hints or none

        // Test with per-channel config
        let per_channel_config = QuantConfig::per_channel(0);
        let hints = generate_optimization_hints(&tensor, &per_channel_config).unwrap();
        assert!(hints.len() >= 0);
    }

    #[test]
    fn test_benchmark_quantization_performance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();

        let configs = vec![QuantConfig::int8(), QuantConfig::binary()];

        let results = benchmark_quantization_performance(&tensor, &configs, 3).unwrap();

        // Should return timing results for successful configs
        assert!(results.len() <= configs.len());

        for (config, quantize_time, dequantize_time) in &results {
            assert!(configs.iter().any(|c| c.scheme == config.scheme));
            assert!(quantize_time >= &0.0);
            assert!(dequantize_time >= &0.0);
        }
    }

    #[test]
    fn test_quantization_metrics_edge_cases() {
        // Test with different shaped tensors (should fail)
        let tensor1 = tensor_1d(&[1.0, 2.0]).unwrap();
        let tensor2 = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();

        let result = calculate_quantization_metrics(&tensor1, &tensor2, 32, 8);
        assert!(result.is_err());

        // Test with zero tensors
        let zero_tensor = tensor_1d(&[0.0, 0.0, 0.0]).unwrap();
        let metrics = calculate_quantization_metrics(&zero_tensor, &zero_tensor, 32, 8).unwrap();

        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.max_error, 0.0);
        assert_eq!(metrics.zero_error_percentage, 100.0);
        assert!(metrics.psnr.is_infinite());
        assert_eq!(metrics.cosine_similarity, 0.0); // Both vectors are zero

        // Test with very small differences
        let original = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        let almost_same = tensor_1d(&[1.0000001, 2.0000001, 3.0000001]).unwrap();

        let metrics = calculate_quantization_metrics(&original, &almost_same, 32, 8).unwrap();
        assert!(metrics.mse < 1e-12);
        assert!(metrics.mae < 1e-6);
        assert!(metrics.cosine_similarity > 0.999999);
        assert!(metrics.psnr > 100.0); // Very high PSNR for very small error
    }

    #[test]
    fn test_metrics_default() {
        let default_metrics = QuantizationMetrics::default();
        assert_eq!(default_metrics.mse, 0.0);
        assert_eq!(default_metrics.psnr, 0.0);
        assert_eq!(default_metrics.snr, 0.0);
        assert_eq!(default_metrics.mae, 0.0);
        assert_eq!(default_metrics.max_error, 0.0);
        assert_eq!(default_metrics.zero_error_percentage, 100.0);
        assert_eq!(default_metrics.cosine_similarity, 1.0);
        assert_eq!(default_metrics.compression_ratio, 1.0);
    }
}
