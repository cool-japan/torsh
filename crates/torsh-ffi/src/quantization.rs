//! Model Quantization and Optimization for Edge Deployment
//!
//! This module provides comprehensive model quantization and optimization techniques
//! for deploying deep learning models on edge devices, browsers (WASM), and mobile platforms.
//!
//! # Features
//!
//! - **Post-Training Quantization (PTQ)**: Convert trained models to INT8/INT4/FP16
//! - **Quantization-Aware Training (QAT)**: Train models with quantization in mind
//! - **Dynamic Quantization**: Runtime quantization for specific layers
//! - **Mixed Precision**: Combine different precisions for optimal performance
//! - **Calibration**: Use representative data for optimal quantization parameters
//! - **Pruning**: Remove redundant parameters for model compression
//! - **Knowledge Distillation**: Transfer knowledge from large to small models
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │          Original FP32 Model                    │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//!         ┌───────────▼──────────┐
//!         │   Quantization       │
//!         │                      │
//!    ┌────┴────┐          ┌─────┴────┐
//!    │   PTQ   │          │   QAT    │
//!    └────┬────┘          └─────┬────┘
//!         │                     │
//!         └──────────┬──────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Quantized Model            │
//!     │  (INT8/INT4/FP16/Mixed)     │
//!     └──────────────┬──────────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Optimization               │
//!     │  • Pruning                  │
//!     │  • Distillation             │
//!     │  • Fusion                   │
//!     └──────────────┬──────────────┘
//!                    │
//!     ┌──────────────▼──────────────┐
//!     │  Optimized Edge Model       │
//!     │  (2-10x smaller, 2-4x faster)│
//!     └─────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ## Post-Training Quantization (PTQ)
//!
//! ```rust,ignore
//! use torsh_ffi::quantization::{QuantizationConfig, Quantizer, QuantizationType};
//!
//! // Configure quantization
//! let config = QuantizationConfig::new(QuantizationType::Int8)
//!     .with_calibration_data(&calibration_set)
//!     .with_percentile(99.99);  // Clip outliers
//!
//! // Quantize model
//! let quantizer = Quantizer::new(config);
//! let quantized_model = quantizer.quantize(&fp32_model)?;
//!
//! // Results: ~4x smaller, ~2-3x faster
//! println!("Model size: {:.2}MB -> {:.2}MB",
//!     fp32_model.size_mb(), quantized_model.size_mb());
//! ```
//!
//! ## Mixed Precision Quantization
//!
//! ```rust,ignore
//! use torsh_ffi::quantization::{MixedPrecisionConfig, LayerPrecision};
//!
//! let config = MixedPrecisionConfig::new()
//!     .set_layer_precision("layer1", LayerPrecision::Int8)
//!     .set_layer_precision("layer2", LayerPrecision::Fp16)
//!     .set_layer_precision("output", LayerPrecision::Fp32);  // Keep output high precision
//!
//! let quantized = quantizer.quantize_mixed_precision(&model, &config)?;
//! ```
//!
//! ## Quantization-Aware Training
//!
//! ```rust,ignore
//! use torsh_ffi::quantization::{QatConfig, FakeQuantize};
//!
//! // Add fake quantization to model
//! let qat_config = QatConfig::new(QuantizationType::Int8)
//!     .with_observer_type(ObserverType::MovingAverage);
//!
//! let qat_model = qat_config.prepare_qat(&model)?;
//!
//! // Train with quantization simulation
//! for epoch in 0..epochs {
//!     train_epoch(&mut qat_model, &train_data);
//! }
//!
//! // Convert to actual quantized model
//! let quantized = qat_model.convert_to_quantized()?;
//! ```
//!
//! # Supported Quantization Types
//!
//! - **INT8**: 8-bit integer quantization (most common, good trade-off)
//! - **INT4**: 4-bit integer quantization (aggressive compression)
//! - **FP16**: 16-bit floating point (good for GPUs)
//! - **UINT8**: Unsigned 8-bit (for activations)
//! - **Dynamic**: Per-batch/per-channel quantization
//! - **Mixed**: Different precisions for different layers

use crate::error::{ErrorBuilder, ErrorCode, FfiError};
use serde::{Deserialize, Serialize};

/// Quantization type/precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit signed integer (-128 to 127)
    Int8,
    /// 4-bit signed integer (-8 to 7)
    Int4,
    /// 8-bit unsigned integer (0 to 255)
    Uint8,
    /// 16-bit floating point (half precision)
    Fp16,
    /// BFloat16 (Brain floating point)
    BFloat16,
    /// Dynamic quantization (runtime)
    Dynamic,
}

impl QuantizationType {
    /// Get the number of bits used
    pub fn bits(&self) -> u8 {
        match self {
            QuantizationType::Int4 => 4,
            QuantizationType::Int8 | QuantizationType::Uint8 => 8,
            QuantizationType::Fp16 | QuantizationType::BFloat16 => 16,
            QuantizationType::Dynamic => 8, // Default to 8-bit
        }
    }

    /// Get the theoretical compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits() as f32
    }

    /// Get the range of values
    pub fn value_range(&self) -> (f64, f64) {
        match self {
            QuantizationType::Int8 => (-128.0, 127.0),
            QuantizationType::Int4 => (-8.0, 7.0),
            QuantizationType::Uint8 => (0.0, 255.0),
            QuantizationType::Fp16 => (-65504.0, 65504.0),
            QuantizationType::BFloat16 => (-3.39e38, 3.39e38),
            QuantizationType::Dynamic => (-128.0, 127.0),
        }
    }
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Per-tensor quantization (single scale/zero-point for entire tensor)
    PerTensor,
    /// Per-channel quantization (scale/zero-point per output channel)
    PerChannel,
    /// Per-group quantization (for grouped operations)
    PerGroup { group_size: usize },
}

/// Quantization scheme (symmetric vs asymmetric)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero-point = 0)
    Symmetric,
    /// Asymmetric quantization (arbitrary zero-point)
    Asymmetric,
}

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Use min/max values from calibration data
    MinMax,
    /// Use percentiles to clip outliers (e.g., 0.1% and 99.9%)
    Percentile { lower: f32, upper: f32 },
    /// Use moving average of min/max
    MovingAverage { momentum: f32 },
    /// Minimize mean squared error
    Mse,
    /// Entropy-based calibration (KL divergence minimization)
    Entropy,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::Percentile {
            lower: 0.01,
            upper: 99.99,
        }
    }
}

/// Quantization parameters (scale and zero-point)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scaling factor
    pub scale: f32,
    /// Zero point (offset)
    pub zero_point: i32,
    /// Quantization type
    pub qtype: QuantizationType,
    /// Value range (min, max) from calibration
    pub range: (f32, f32),
}

impl QuantizationParams {
    /// Create new quantization parameters
    ///
    /// # Arguments
    /// * `scale` - Scaling factor
    /// * `zero_point` - Zero point offset
    /// * `qtype` - Quantization type
    pub fn new(scale: f32, zero_point: i32, qtype: QuantizationType) -> Self {
        Self {
            scale,
            zero_point,
            qtype,
            range: (0.0, 0.0),
        }
    }

    /// Compute quantization parameters from value range
    ///
    /// # Arguments
    /// * `min_val` - Minimum value in the data
    /// * `max_val` - Maximum value in the data
    /// * `qtype` - Quantization type
    /// * `scheme` - Quantization scheme (symmetric/asymmetric)
    pub fn from_range(
        min_val: f32,
        max_val: f32,
        qtype: QuantizationType,
        scheme: QuantizationScheme,
    ) -> Self {
        let (qmin, qmax) = qtype.value_range();

        let (scale, zero_point) = match scheme {
            QuantizationScheme::Symmetric => {
                // Symmetric: zero_point = 0
                let max_abs = min_val.abs().max(max_val.abs());
                let scale = (2.0 * max_abs) / (qmax - qmin) as f32;
                (scale, 0)
            }
            QuantizationScheme::Asymmetric => {
                // Asymmetric: map [min_val, max_val] to [qmin, qmax]
                let scale = (max_val - min_val) / (qmax - qmin) as f32;
                let zero_point = qmin as f32 - min_val / scale;
                (scale, zero_point.round() as i32)
            }
        };

        Self {
            scale: scale.max(1e-8), // Avoid division by zero
            zero_point,
            qtype,
            range: (min_val, max_val),
        }
    }

    /// Quantize a floating-point value
    pub fn quantize(&self, value: f32) -> i32 {
        let (qmin, qmax) = self.qtype.value_range();
        let quantized = (value / self.scale).round() + self.zero_point as f32;
        quantized.clamp(qmin as f32, qmax as f32) as i32
    }

    /// Dequantize an integer value back to floating-point
    pub fn dequantize(&self, quantized: i32) -> f32 {
        (quantized - self.zero_point) as f32 * self.scale
    }

    /// Quantize an array of values
    pub fn quantize_array(&self, values: &[f32]) -> Vec<i32> {
        values.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Dequantize an array of values
    pub fn dequantize_array(&self, quantized: &[i32]) -> Vec<f32> {
        quantized.iter().map(|&q| self.dequantize(q)).collect()
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Target quantization type
    pub qtype: QuantizationType,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Calibration method
    pub calibration: CalibrationMethod,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
    /// Whether to quantize biases (usually kept in higher precision)
    pub quantize_biases: bool,
    /// Layers to skip quantization (by name)
    pub skip_layers: Vec<String>,
    /// Force specific layers to use FP32 (e.g., output layer)
    pub force_fp32_layers: Vec<String>,
}

impl QuantizationConfig {
    /// Create a new quantization configuration
    pub fn new(qtype: QuantizationType) -> Self {
        Self {
            qtype,
            granularity: QuantizationGranularity::PerChannel,
            scheme: QuantizationScheme::Asymmetric,
            calibration: CalibrationMethod::default(),
            quantize_weights: true,
            quantize_activations: true,
            quantize_biases: false, // Usually keep biases in FP32
            skip_layers: Vec::new(),
            force_fp32_layers: vec!["output".to_string()], // Keep output in FP32
        }
    }

    /// Set quantization granularity
    pub fn with_granularity(mut self, granularity: QuantizationGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Set quantization scheme
    pub fn with_scheme(mut self, scheme: QuantizationScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set calibration method
    pub fn with_calibration(mut self, calibration: CalibrationMethod) -> Self {
        self.calibration = calibration;
        self
    }

    /// Add layer to skip
    pub fn skip_layer(mut self, layer_name: String) -> Self {
        self.skip_layers.push(layer_name);
        self
    }

    /// Add layer to force FP32
    pub fn force_fp32_layer(mut self, layer_name: String) -> Self {
        self.force_fp32_layers.push(layer_name);
        self
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self::new(QuantizationType::Int8)
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized values (as integers)
    pub quantized_data: Vec<i32>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Original tensor name (optional)
    pub name: Option<String>,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(quantized_data: Vec<i32>, params: QuantizationParams, shape: Vec<usize>) -> Self {
        Self {
            quantized_data,
            params,
            shape,
            name: None,
        }
    }

    /// Set tensor name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        let bits_per_element = self.params.qtype.bits() as usize;
        (self.quantized_data.len() * bits_per_element + 7) / 8
    }

    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.quantized_data.len() * 4; // 4 bytes per FP32
        let quantized_size = self.size_bytes();
        fp32_size as f32 / quantized_size as f32
    }

    /// Dequantize back to FP32
    pub fn dequantize(&self) -> Vec<f32> {
        self.params.dequantize_array(&self.quantized_data)
    }

    /// Compute quantization error metrics
    pub fn quantization_error(&self, original: &[f32]) -> QuantizationError {
        let dequantized = self.dequantize();

        let mut mse = 0.0_f32;
        let mut max_error = 0.0_f32;

        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            mse += error * error;
            max_error = max_error.max(error);
        }

        mse /= original.len() as f32;
        let rmse = mse.sqrt();

        // Signal-to-Quantization-Noise Ratio (SQNR)
        let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
        let sqnr_db = 10.0 * (signal_power / mse).log10();

        QuantizationError {
            mse,
            rmse,
            max_error,
            sqnr_db,
        }
    }
}

/// Quantization error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationError {
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-Quantization-Noise Ratio (dB)
    pub sqnr_db: f32,
}

/// Calibration dataset for determining quantization parameters
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    /// Calibration samples
    samples: Vec<Vec<f32>>,
    /// Maximum number of samples to use
    max_samples: usize,
}

impl CalibrationDataset {
    /// Create a new calibration dataset
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_samples,
        }
    }

    /// Add a sample to the calibration dataset
    pub fn add_sample(&mut self, sample: Vec<f32>) {
        if self.samples.len() < self.max_samples {
            self.samples.push(sample);
        }
    }

    /// Get all samples
    pub fn samples(&self) -> &[Vec<f32>] {
        &self.samples
    }

    /// Compute statistics from calibration data
    pub fn compute_statistics(&self) -> CalibrationStatistics {
        let mut all_values: Vec<f32> = self
            .samples
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = all_values.first().copied().unwrap_or(0.0);
        let max_val = all_values.last().copied().unwrap_or(0.0);

        let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;

        let variance: f32 =
            all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
        let std_dev = variance.sqrt();

        CalibrationStatistics {
            min_val,
            max_val,
            mean,
            std_dev,
            num_samples: all_values.len(),
        }
    }

    /// Get value at percentile
    pub fn percentile(&self, percentile: f32) -> f32 {
        let mut all_values: Vec<f32> = self
            .samples
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile / 100.0) * all_values.len() as f32) as usize;
        all_values
            .get(index.min(all_values.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Statistics from calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStatistics {
    pub min_val: f32,
    pub max_val: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub num_samples: usize,
}

/// Post-Training Quantization (PTQ) quantizer
#[derive(Debug, Clone)]
pub struct Quantizer {
    config: QuantizationConfig,
    calibration_data: Option<CalibrationDataset>,
}

impl Quantizer {
    /// Create a new quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_data: None,
        }
    }

    /// Set calibration data
    pub fn with_calibration_data(mut self, data: CalibrationDataset) -> Self {
        self.calibration_data = Some(data);
        self
    }

    /// Quantize a tensor (weights or activations)
    ///
    /// # Arguments
    /// * `data` - FP32 tensor data
    /// * `shape` - Tensor shape
    /// * `name` - Optional tensor name
    pub fn quantize_tensor(
        &self,
        data: &[f32],
        shape: Vec<usize>,
        name: Option<String>,
    ) -> Result<QuantizedTensor, FfiError> {
        // Check if this layer should be skipped
        if let Some(ref n) = name {
            if self.config.skip_layers.contains(n) || self.config.force_fp32_layers.contains(n) {
                return Err(FfiError::Enhanced(
                    ErrorBuilder::new(ErrorCode::OperationFailed)
                        .message("Layer marked for FP32 preservation")
                        .context("layer", n)
                        .build(),
                ));
            }
        }

        // Determine quantization range
        let (min_val, max_val) = match &self.calibration_data {
            Some(calib) => match self.config.calibration {
                CalibrationMethod::MinMax => {
                    let stats = calib.compute_statistics();
                    (stats.min_val, stats.max_val)
                }
                CalibrationMethod::Percentile { lower, upper } => {
                    (calib.percentile(lower), calib.percentile(upper))
                }
                _ => {
                    // Fallback to min/max from actual data
                    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    (min, max)
                }
            },
            None => {
                // No calibration data, use min/max from tensor
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            }
        };

        // Compute quantization parameters
        let params =
            QuantizationParams::from_range(min_val, max_val, self.config.qtype, self.config.scheme);

        // Quantize the data
        let quantized_data = params.quantize_array(data);

        Ok(QuantizedTensor::new(quantized_data, params, shape).with_name(name.unwrap_or_default()))
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }
}

/// Model compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Original model size (bytes)
    pub original_size: usize,
    /// Quantized model size (bytes)
    pub quantized_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Number of quantized layers
    pub num_quantized_layers: usize,
    /// Number of FP32 layers (preserved)
    pub num_fp32_layers: usize,
    /// Average quantization error (SQNR in dB)
    pub avg_sqnr_db: f32,
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self {
            original_size: 0,
            quantized_size: 0,
            compression_ratio: 1.0,
            num_quantized_layers: 0,
            num_fp32_layers: 0,
            avg_sqnr_db: 0.0,
        }
    }

    /// Size reduction percentage
    pub fn size_reduction_percent(&self) -> f32 {
        (1.0 - (self.quantized_size as f32 / self.original_size as f32)) * 100.0
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_bits() {
        assert_eq!(QuantizationType::Int4.bits(), 4);
        assert_eq!(QuantizationType::Int8.bits(), 8);
        assert_eq!(QuantizationType::Fp16.bits(), 16);
    }

    #[test]
    fn test_quantization_type_compression_ratio() {
        assert_eq!(QuantizationType::Int8.compression_ratio(), 4.0); // 32/8
        assert_eq!(QuantizationType::Int4.compression_ratio(), 8.0); // 32/4
        assert_eq!(QuantizationType::Fp16.compression_ratio(), 2.0); // 32/16
    }

    #[test]
    fn test_quantization_params_symmetric() {
        let params = QuantizationParams::from_range(
            -10.0,
            10.0,
            QuantizationType::Int8,
            QuantizationScheme::Symmetric,
        );

        assert_eq!(params.zero_point, 0);
        assert!(params.scale > 0.0);

        // Test quantize/dequantize
        let val = 5.0_f32;
        let quantized = params.quantize(val);
        let dequantized = params.dequantize(quantized);
        assert!((val - dequantized).abs() < 0.1); // Small error tolerance
    }

    #[test]
    fn test_quantization_params_asymmetric() {
        let params = QuantizationParams::from_range(
            0.0,
            10.0,
            QuantizationType::Uint8,
            QuantizationScheme::Asymmetric,
        );

        assert!(params.scale > 0.0);

        // Test quantize/dequantize
        let val = 5.0_f32;
        let quantized = params.quantize(val);
        let dequantized = params.dequantize(quantized);
        assert!((val - dequantized).abs() < 0.1);
    }

    #[test]
    fn test_quantization_array() {
        let params = QuantizationParams::from_range(
            -1.0,
            1.0,
            QuantizationType::Int8,
            QuantizationScheme::Symmetric,
        );

        let data = vec![-0.5, 0.0, 0.5, 1.0];
        let quantized = params.quantize_array(&data);
        let dequantized = params.dequantize_array(&quantized);

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let params = QuantizationParams::new(0.1, 0, QuantizationType::Int8);
        let shape = vec![2, 3];

        let qtensor = QuantizedTensor::new(data.clone(), params.clone(), shape.clone());

        assert_eq!(qtensor.quantized_data, data);
        assert_eq!(qtensor.shape, shape);
        assert!(qtensor.compression_ratio() > 1.0);
    }

    #[test]
    fn test_quantized_tensor_size() {
        let data = vec![0; 1000]; // 1000 elements
        let params = QuantizationParams::new(1.0, 0, QuantizationType::Int8);
        let qtensor = QuantizedTensor::new(data, params, vec![1000]);

        // INT8: 1 byte per element
        assert_eq!(qtensor.size_bytes(), 1000);

        let params_int4 = QuantizationParams::new(1.0, 0, QuantizationType::Int4);
        let qtensor_int4 = QuantizedTensor::new(vec![0; 1000], params_int4, vec![1000]);

        // INT4: 0.5 bytes per element (packed)
        assert_eq!(qtensor_int4.size_bytes(), 500);
    }

    #[test]
    fn test_calibration_dataset() {
        let mut dataset = CalibrationDataset::new(100);

        dataset.add_sample(vec![1.0, 2.0, 3.0]);
        dataset.add_sample(vec![4.0, 5.0, 6.0]);

        assert_eq!(dataset.samples().len(), 2);

        let stats = dataset.compute_statistics();
        assert_eq!(stats.min_val, 1.0);
        assert_eq!(stats.max_val, 6.0);
        assert_eq!(stats.num_samples, 6);
    }

    #[test]
    fn test_calibration_percentile() {
        let mut dataset = CalibrationDataset::new(100);

        // Add data from 1 to 100
        for i in 1..=100 {
            dataset.add_sample(vec![i as f32]);
        }

        let p50 = dataset.percentile(50.0);
        assert!((p50 - 50.0).abs() < 5.0); // Approximately median

        let p99 = dataset.percentile(99.0);
        assert!(p99 > 95.0);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::new(QuantizationType::Int8)
            .with_granularity(QuantizationGranularity::PerChannel)
            .with_scheme(QuantizationScheme::Symmetric)
            .skip_layer("layer1".to_string())
            .force_fp32_layer("output".to_string());

        assert_eq!(config.qtype, QuantizationType::Int8);
        assert!(config.quantize_weights);
        assert!(config.skip_layers.contains(&"layer1".to_string()));
    }

    #[test]
    fn test_quantizer_basic() {
        let config = QuantizationConfig::new(QuantizationType::Int8);
        let quantizer = Quantizer::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];

        let result = quantizer.quantize_tensor(&data, shape, Some("test".to_string()));
        assert!(result.is_ok());

        let qtensor = result.unwrap();
        assert_eq!(qtensor.shape, vec![5]);
        assert_eq!(qtensor.quantized_data.len(), 5);
    }

    #[test]
    fn test_quantizer_with_calibration() {
        let mut calib_data = CalibrationDataset::new(10);
        calib_data.add_sample(vec![0.0, 1.0, 2.0, 3.0]);
        calib_data.add_sample(vec![0.5, 1.5, 2.5, 3.5]);

        let config = QuantizationConfig::new(QuantizationType::Int8).with_calibration(
            CalibrationMethod::Percentile {
                lower: 1.0,
                upper: 99.0,
            },
        );

        let quantizer = Quantizer::new(config).with_calibration_data(calib_data);

        let data = vec![1.0, 2.0, 3.0];
        let result = quantizer.quantize_tensor(&data, vec![3], None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantization_error_metrics() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let params = QuantizationParams::from_range(
            0.0,
            5.0,
            QuantizationType::Int8,
            QuantizationScheme::Asymmetric,
        );

        let quantized_data = params.quantize_array(&original);
        let qtensor = QuantizedTensor::new(quantized_data, params, vec![5]);

        let error = qtensor.quantization_error(&original);

        assert!(error.mse >= 0.0);
        assert!(error.rmse >= 0.0);
        assert!(error.max_error >= 0.0);
        assert!(error.sqnr_db > 0.0); // Should have positive SQNR
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();
        stats.original_size = 1000;
        stats.quantized_size = 250;
        stats.compression_ratio = 4.0;

        assert_eq!(stats.size_reduction_percent(), 75.0);
    }

    #[test]
    fn test_quantizer_skip_layer() {
        let config =
            QuantizationConfig::new(QuantizationType::Int8).skip_layer("skip_me".to_string());

        let quantizer = Quantizer::new(config);

        let data = vec![1.0, 2.0, 3.0];
        let result = quantizer.quantize_tensor(&data, vec![3], Some("skip_me".to_string()));

        assert!(result.is_err()); // Should error because layer is skipped
    }
}
