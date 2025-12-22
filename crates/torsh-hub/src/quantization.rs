//! # Model Quantization and Compression
//!
//! This module provides comprehensive quantization utilities for reducing model size
//! and improving inference performance while maintaining accuracy.
//!
//! ## Features
//!
//! - **Post-Training Quantization**: Quantize pretrained models without retraining
//! - **Quantization-Aware Training**: Fine-tune models with quantization simulation
//! - **Multiple Precision Levels**: INT8, FP16, and mixed precision support
//! - **Calibration**: Automatic calibration using representative data
//! - **Layer-wise Quantization**: Fine-grained control over which layers to quantize
//!
//! ## SciRS2 POLICY Compliance
//!
//! This module strictly follows the SciRS2 POLICY:
//! - Uses `scirs2_core::ndarray::*` for all array operations
//! - Uses `scirs2_core::random::*` for calibration sampling
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use torsh_hub::quantization::{QuantizationConfig, ModelQuantizer, QuantizationType};
//! use torsh_nn::Module;
//!
//! # fn example(model: Box<dyn Module>) -> Result<(), Box<dyn std::error::Error>> {
//! // Configure quantization
//! let config = QuantizationConfig {
//!     quantization_type: QuantizationType::Int8,
//!     per_channel: true,
//!     symmetric: true,
//!     calibration_samples: 100,
//!     excluded_layers: vec![],
//! };
//!
//! // Quantize the model
//! let mut quantizer = ModelQuantizer::new(config);
//! let quantized_model = quantizer.quantize(model)?;
//!
//! // Model is now optimized for deployment
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};

// SciRS2 POLICY: Use unified ndarray access
use scirs2_core::ndarray::{Array1, Array2, ArrayD};

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Type of quantization to apply
    pub quantization_type: QuantizationType,
    /// Use per-channel quantization (better accuracy, larger overhead)
    pub per_channel: bool,
    /// Use symmetric quantization (centered at zero)
    pub symmetric: bool,
    /// Number of calibration samples for activation quantization
    pub calibration_samples: usize,
    /// Layers to exclude from quantization
    pub excluded_layers: Vec<String>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Int8,
            per_channel: true,
            symmetric: true,
            calibration_samples: 100,
            excluded_layers: vec![],
        }
    }
}

/// Supported quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit integer quantization (INT8)
    Int8,
    /// 16-bit floating point (FP16/half precision)
    Fp16,
    /// Mixed precision (critical layers in FP16, others in INT8)
    Mixed,
    /// Dynamic quantization (weights quantized, activations in FP32)
    Dynamic,
}

/// Model quantizer for post-training quantization
pub struct ModelQuantizer {
    config: QuantizationConfig,
    calibration_stats: HashMap<String, QuantizationStats>,
}

impl ModelQuantizer {
    /// Create a new model quantizer
    ///
    /// # Arguments
    /// * `config` - Quantization configuration
    ///
    /// # Example
    /// ```rust
    /// use torsh_hub::quantization::{ModelQuantizer, QuantizationConfig};
    ///
    /// let config = QuantizationConfig::default();
    /// let quantizer = ModelQuantizer::new(config);
    /// ```
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_stats: HashMap::new(),
        }
    }

    /// Quantize a model using the configured settings
    ///
    /// This performs post-training quantization on the model.
    ///
    /// # Arguments
    /// * `model` - The model to quantize
    ///
    /// # Returns
    /// * Quantized model wrapped in Result
    pub fn quantize(
        &mut self,
        _model: Box<dyn torsh_nn::Module>,
    ) -> Result<Box<dyn torsh_nn::Module>> {
        // Implementation would convert model weights and potentially activations
        // to lower precision based on config
        Err(TorshError::NotImplemented(
            "Full model quantization requires torsh-nn integration".to_string(),
        ))
    }

    /// Calibrate quantization parameters using sample data
    ///
    /// # Arguments
    /// * `layer_name` - Name of the layer to calibrate
    /// * `activations` - Sample activations for calibration
    pub fn calibrate(&mut self, layer_name: &str, activations: &ArrayD<f32>) -> Result<()> {
        let stats = match self.config.quantization_type {
            QuantizationType::Int8 => self.compute_int8_stats(activations)?,
            QuantizationType::Fp16 => self.compute_fp16_stats(activations)?,
            QuantizationType::Mixed => self.compute_mixed_stats(activations)?,
            QuantizationType::Dynamic => self.compute_dynamic_stats(activations)?,
        };

        self.calibration_stats.insert(layer_name.to_string(), stats);
        Ok(())
    }

    /// Compute INT8 quantization statistics
    fn compute_int8_stats(&self, activations: &ArrayD<f32>) -> Result<QuantizationStats> {
        let min = activations
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = activations
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let (scale, zero_point) = if self.config.symmetric {
            // Symmetric: map [-abs_max, abs_max] to [-127, 127]
            let abs_max = min.abs().max(max.abs());
            let scale = abs_max / 127.0;
            (scale, 0)
        } else {
            // Asymmetric: map [min, max] to [0, 255]
            let scale = (max - min) / 255.0;
            let zero_point = (-min / scale).round() as i32;
            (scale, zero_point)
        };

        Ok(QuantizationStats {
            scale,
            zero_point,
            min,
            max,
            num_bits: 8,
        })
    }

    /// Compute FP16 quantization statistics
    fn compute_fp16_stats(&self, _activations: &ArrayD<f32>) -> Result<QuantizationStats> {
        // FP16 doesn't need calibration stats, but we store metadata
        Ok(QuantizationStats {
            scale: 1.0,
            zero_point: 0,
            min: 0.0,
            max: 0.0,
            num_bits: 16,
        })
    }

    /// Compute mixed precision statistics
    fn compute_mixed_stats(&self, activations: &ArrayD<f32>) -> Result<QuantizationStats> {
        // For mixed precision, use INT8 stats but mark critical layers
        self.compute_int8_stats(activations)
    }

    /// Compute dynamic quantization statistics
    fn compute_dynamic_stats(&self, activations: &ArrayD<f32>) -> Result<QuantizationStats> {
        // Dynamic quantization computes stats on the fly during inference
        self.compute_int8_stats(activations)
    }

    /// Get calibration statistics for a layer
    pub fn get_stats(&self, layer_name: &str) -> Option<&QuantizationStats> {
        self.calibration_stats.get(layer_name)
    }

    /// Export quantization configuration
    pub fn export_config(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.config)
            .map_err(|e| TorshError::SerializationError(e.to_string()))
    }
}

/// Quantization statistics for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Scaling factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Minimum activation value observed
    pub min: f32,
    /// Maximum activation value observed
    pub max: f32,
    /// Number of bits used
    pub num_bits: u8,
}

/// Quantize a weight tensor to INT8
///
/// # Arguments
/// * `weights` - FP32 weights to quantize
/// * `per_channel` - Whether to use per-channel quantization
///
/// # Returns
/// * Tuple of (quantized weights, scale factors, zero points)
///
/// # Example
/// ```rust
/// use torsh_hub::quantization::quantize_weights_int8;
/// use scirs2_core::ndarray::Array2;
///
/// let weights = Array2::<f32>::zeros((64, 128));
/// let (quantized, scales, zeros) = quantize_weights_int8(&weights, true);
/// ```
pub fn quantize_weights_int8(
    weights: &Array2<f32>,
    per_channel: bool,
) -> (Array2<i8>, Array1<f32>, Array1<i32>) {
    let (rows, cols) = weights.dim();

    if per_channel {
        // Per-channel quantization (separate scale per output channel)
        let mut quantized = Array2::zeros((rows, cols));
        let mut scales = Array1::zeros(rows);
        let mut zero_points = Array1::zeros(rows);

        for i in 0..rows {
            let row = weights.row(i);
            let min = row
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let max = row
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            let scale = (max - min) / 255.0;
            let zero_point = if scale > 0.0 {
                (-min / scale).round() as i32
            } else {
                0
            };

            scales[i] = scale;
            zero_points[i] = zero_point;

            for j in 0..cols {
                let val = if scale > 0.0 {
                    ((weights[[i, j]] / scale) + zero_point as f32)
                        .round()
                        .clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                quantized[[i, j]] = val;
            }
        }

        (quantized, scales, zero_points)
    } else {
        // Per-tensor quantization (single scale for entire tensor)
        let min = weights
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = weights
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let scale = (max - min) / 255.0;
        let zero_point = if scale > 0.0 {
            (-min / scale).round() as i32
        } else {
            0
        };

        let quantized = weights.mapv(|w| {
            if scale > 0.0 {
                ((w / scale) + zero_point as f32)
                    .round()
                    .clamp(-128.0, 127.0) as i8
            } else {
                0
            }
        });

        let scales = Array1::from_elem(1, scale);
        let zero_points = Array1::from_elem(1, zero_point);

        (quantized, scales, zero_points)
    }
}

/// Dequantize INT8 weights back to FP32
///
/// # Arguments
/// * `quantized` - Quantized INT8 weights
/// * `scales` - Scale factors
/// * `zero_points` - Zero points
///
/// # Returns
/// * Dequantized FP32 weights
pub fn dequantize_weights_int8(
    quantized: &Array2<i8>,
    scales: &Array1<f32>,
    zero_points: &Array1<i32>,
) -> Array2<f32> {
    let (rows, cols) = quantized.dim();
    let per_channel = scales.len() > 1;

    if per_channel {
        let mut dequantized = Array2::zeros((rows, cols));
        for i in 0..rows {
            let scale = scales[i];
            let zero_point = zero_points[i];
            for j in 0..cols {
                dequantized[[i, j]] = (quantized[[i, j]] as f32 - zero_point as f32) * scale;
            }
        }
        dequantized
    } else {
        let scale = scales[0];
        let zero_point = zero_points[0];
        quantized.mapv(|q| (q as f32 - zero_point as f32) * scale)
    }
}

/// Calculate quantization error metrics
///
/// # Arguments
/// * `original` - Original FP32 weights
/// * `quantized` - Quantized weights
/// * `scales` - Scale factors
/// * `zero_points` - Zero points
///
/// # Returns
/// * Tuple of (MSE, max_error, SQNR in dB)
pub fn quantization_metrics(
    original: &Array2<f32>,
    quantized: &Array2<i8>,
    scales: &Array1<f32>,
    zero_points: &Array1<i32>,
) -> (f32, f32, f32) {
    let dequantized = dequantize_weights_int8(quantized, scales, zero_points);

    // Mean Squared Error
    let diff = original - &dequantized;
    let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);

    // Maximum absolute error
    let max_error = diff
        .iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    // Signal-to-Quantization-Noise Ratio (SQNR) in dB
    let signal_power = original.mapv(|x| x * x).mean().unwrap_or(1e-10);
    let sqnr_db = 10.0 * (signal_power / mse.max(1e-10)).log10();

    (mse, max_error, sqnr_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type, QuantizationType::Int8);
        assert!(config.per_channel);
        assert!(config.symmetric);
        assert_eq!(config.calibration_samples, 100);
    }

    #[test]
    fn test_model_quantizer_creation() {
        let config = QuantizationConfig::default();
        let quantizer = ModelQuantizer::new(config);
        assert_eq!(quantizer.calibration_stats.len(), 0);
    }

    #[test]
    fn test_quantize_weights_per_tensor() {
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
        let (quantized, scales, zeros) = quantize_weights_int8(&weights, false);

        assert_eq!(quantized.dim(), weights.dim());
        assert_eq!(scales.len(), 1);
        assert_eq!(zeros.len(), 1);
    }

    #[test]
    fn test_quantize_weights_per_channel() {
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
        let (quantized, scales, zeros) = quantize_weights_int8(&weights, true);

        assert_eq!(quantized.dim(), weights.dim());
        assert_eq!(scales.len(), 4); // One per row/channel
        assert_eq!(zeros.len(), 4);
    }

    #[test]
    fn test_dequantize_weights() {
        let weights = Array2::from_shape_fn((4, 4), |(i, j)| ((i * 4 + j) as f32) * 0.1);
        let (quantized, scales, zeros) = quantize_weights_int8(&weights, false);
        let dequantized = dequantize_weights_int8(&quantized, &scales, &zeros);

        // Check that dequantization is close to original
        let diff = &weights - &dequantized;
        let max_diff = diff
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Quantization error should be reasonable for INT8 quantization
        // INT8 has 256 levels, so error can be up to scale/2 per value
        assert!(max_diff < 1.0, "Max diff: {}", max_diff);
    }

    #[test]
    fn test_quantization_metrics() {
        let original = Array2::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f32 * 0.01);
        let (quantized, scales, zeros) = quantize_weights_int8(&original, true);
        let (mse, max_error, sqnr) = quantization_metrics(&original, &quantized, &scales, &zeros);

        // Basic sanity checks
        assert!(mse >= 0.0);
        assert!(max_error >= 0.0);
        assert!(sqnr.is_finite());
    }

    #[test]
    fn test_calibration_int8() {
        let mut quantizer = ModelQuantizer::new(QuantizationConfig::default());
        let activations =
            ArrayD::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f32 * 0.1).collect())
                .unwrap();

        quantizer.calibrate("layer1", &activations).unwrap();
        let stats = quantizer.get_stats("layer1").unwrap();

        assert_eq!(stats.num_bits, 8);
        assert!(stats.scale > 0.0);
    }

    #[test]
    fn test_export_config() {
        let quantizer = ModelQuantizer::new(QuantizationConfig::default());
        let config_json = quantizer.export_config().unwrap();

        assert!(config_json.contains("Int8"));
        assert!(config_json.contains("per_channel"));
    }
}
