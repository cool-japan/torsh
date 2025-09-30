//! Model quantization utilities for reducing precision and model size
//!
//! This module provides comprehensive quantization support including:
//! - Post-training quantization (PTQ)
//! - Quantization-aware training (QAT)
//! - Dynamic quantization
//! - Mixed-precision quantization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Quantization data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationDType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    Uint8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit floating point
    Float16,
    /// Brain floating point (bfloat16)
    BFloat16,
}

/// Quantization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Post-training quantization
    PostTraining {
        /// Calibration dataset size
        calibration_samples: usize,
        /// Whether to use percentile-based calibration
        percentile_calibration: bool,
    },
    /// Quantization-aware training
    QuantizationAware {
        /// Fake quantization during training
        fake_quantize: bool,
        /// Gradual quantization schedule
        schedule: Option<QuantizationSchedule>,
    },
    /// Dynamic quantization (weights only)
    Dynamic,
}

/// Quantization schedule for gradual QAT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSchedule {
    /// Start epoch for quantization
    pub start_epoch: usize,
    /// Number of epochs to reach full quantization
    pub ramp_epochs: usize,
    /// Quantization strength progression
    pub progression: QuantizationProgression,
}

/// Quantization progression methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationProgression {
    /// Linear progression
    Linear,
    /// Exponential progression
    Exponential { rate: f64 },
    /// Step-wise progression
    StepWise { steps: Vec<(usize, f64)> },
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Target data type for quantization
    pub dtype: QuantizationDType,
    /// Quantization strategy
    pub strategy: QuantizationStrategy,
    /// Layers to quantize
    pub layer_filter: LayerQuantizationFilter,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Observer configuration for calibration
    pub observer_config: ObserverConfig,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            dtype: QuantizationDType::Int8,
            strategy: QuantizationStrategy::PostTraining {
                calibration_samples: 1000,
                percentile_calibration: false,
            },
            layer_filter: LayerQuantizationFilter::All,
            quantize_weights: true,
            quantize_activations: true,
            granularity: QuantizationGranularity::PerTensor,
            observer_config: ObserverConfig::default(),
        }
    }
}

/// Layer filtering for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerQuantizationFilter {
    /// Quantize all layers
    All,
    /// Include specific layers
    Include(Vec<String>),
    /// Exclude specific layers
    Exclude(Vec<String>),
    /// Quantize only certain layer types
    LayerTypes(Vec<QuantizableLayerType>),
}

/// Quantizable layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizableLayerType {
    Linear,
    Convolution,
    Embedding,
    Attention,
}

/// Quantization granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Per-tensor quantization (single scale/zero-point per tensor)
    PerTensor,
    /// Per-channel quantization (scale/zero-point per output channel)
    PerChannel,
    /// Per-group quantization (scale/zero-point per group of channels)
    PerGroup { group_size: usize },
}

/// Observer configuration for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    /// Observer type
    pub observer_type: ObserverType,
    /// Moving average factor for exponential moving average
    pub averaging_constant: f64,
    /// Percentiles for MinMax observer
    pub percentiles: (f64, f64),
    /// Number of bins for histogram observer
    pub histogram_bins: usize,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            observer_type: ObserverType::MinMax,
            averaging_constant: 0.01,
            percentiles: (0.01, 99.99),
            histogram_bins: 2048,
        }
    }
}

/// Observer types for activation range estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObserverType {
    /// Min-max observer
    MinMax,
    /// Moving average min-max observer
    MovingAverageMinMax,
    /// Percentile-based observer
    Percentile,
    /// Histogram observer
    Histogram,
    /// Entropy-based observer
    Entropy,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Min value
    pub min_val: f32,
    /// Max value
    pub max_val: f32,
    /// Quantization data type
    pub dtype: QuantizationDType,
}

/// Quantization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Original model size (bytes)
    pub original_size: usize,
    /// Quantized model size (bytes)
    pub quantized_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Per-layer quantization info
    pub layer_stats: HashMap<String, LayerQuantizationStats>,
    /// Theoretical speedup
    pub theoretical_speedup: f64,
}

/// Per-layer quantization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantizationStats {
    /// Original parameter count
    pub original_params: usize,
    /// Quantized parameter count
    pub quantized_params: usize,
    /// Bits per parameter
    pub bits_per_param: u8,
    /// Quantization error (MSE)
    pub quantization_error: f64,
}

/// Main quantization engine
pub struct ModelQuantizer {
    config: QuantizationConfig,
    quantization_params: HashMap<String, QuantizationParams>,
    observers: HashMap<String, Box<dyn ActivationObserver>>,
    stats: Option<QuantizationStats>,
}

impl ModelQuantizer {
    /// Create a new model quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            quantization_params: HashMap::new(),
            observers: HashMap::new(),
            stats: None,
        }
    }

    /// Quantize a model using the configured strategy
    pub fn quantize_model<M: Module>(&mut self, model: &mut M) -> Result<QuantizationStats> {
        match &self.config.strategy {
            QuantizationStrategy::PostTraining { .. } => self.post_training_quantization(model),
            QuantizationStrategy::QuantizationAware { .. } => {
                self.quantization_aware_training_setup(model)
            }
            QuantizationStrategy::Dynamic => self.dynamic_quantization(model),
        }
    }

    /// Post-training quantization
    fn post_training_quantization<M: Module>(
        &mut self,
        model: &mut M,
    ) -> Result<QuantizationStats> {
        let parameters = model.named_parameters();
        let filtered_params = self.filter_parameters(&parameters)?;

        // Calibrate quantization parameters
        self.calibrate_quantization_params(&filtered_params)?;

        // Quantize weights
        if self.config.quantize_weights {
            self.quantize_weights(&filtered_params)?;
        }

        // Calculate statistics
        let stats = self.calculate_quantization_stats(&parameters)?;
        self.stats = Some(stats.clone());

        Ok(stats)
    }

    /// Setup for quantization-aware training
    fn quantization_aware_training_setup<M: Module>(
        &mut self,
        model: &mut M,
    ) -> Result<QuantizationStats> {
        let parameters = model.named_parameters();
        let filtered_params = self.filter_parameters(&parameters)?;

        // Initialize fake quantization
        self.setup_fake_quantization(&filtered_params)?;

        // Calculate initial statistics
        let stats = self.calculate_quantization_stats(&parameters)?;
        self.stats = Some(stats.clone());

        Ok(stats)
    }

    /// Dynamic quantization (weights only)
    fn dynamic_quantization<M: Module>(&mut self, model: &mut M) -> Result<QuantizationStats> {
        let parameters = model.named_parameters();
        let filtered_params = self.filter_parameters(&parameters)?;

        // Quantize weights only
        self.dynamic_quantize_weights(&filtered_params)?;

        // Calculate statistics
        let stats = self.calculate_quantization_stats(&parameters)?;
        self.stats = Some(stats.clone());

        Ok(stats)
    }

    /// Quantize tensor using specified parameters
    pub fn quantize_tensor(&self, tensor: &Tensor, params: &QuantizationParams) -> Result<Tensor> {
        let float_data = tensor.to_vec()?;

        // Quantize values
        let quantized_data: Vec<i32> = float_data
            .iter()
            .map(|&x| self.quantize_value(x, params))
            .collect();

        // Convert back to float for tensor creation (simulating quantized values)
        let dequantized_data: Vec<f32> = quantized_data
            .iter()
            .map(|&q| self.dequantize_value(q, params))
            .collect();

        Tensor::from_data(
            dequantized_data,
            tensor.shape().dims().to_vec(),
            tensor.device(),
        )
    }

    /// Quantize a single value
    fn quantize_value(&self, value: f32, params: &QuantizationParams) -> i32 {
        let quantized = (value / params.scale).round() + params.zero_point as f32;

        // Clamp to valid range for the data type
        let (min_val, max_val) = self.get_quantization_range(params.dtype);
        quantized.clamp(min_val as f32, max_val as f32) as i32
    }

    /// Dequantize a single value
    fn dequantize_value(&self, quantized: i32, params: &QuantizationParams) -> f32 {
        (quantized - params.zero_point) as f32 * params.scale
    }

    /// Get quantization range for a data type
    fn get_quantization_range(&self, dtype: QuantizationDType) -> (i32, i32) {
        match dtype {
            QuantizationDType::Int8 => (-128, 127),
            QuantizationDType::Uint8 => (0, 255),
            QuantizationDType::Int16 => (-32768, 32767),
            QuantizationDType::Float16 | QuantizationDType::BFloat16 => {
                // For FP16/BF16, we still use integer range for quantization
                (-32768, 32767)
            }
        }
    }

    /// Calculate quantization parameters from tensor statistics
    pub fn calculate_quantization_params(
        &self,
        min_val: f32,
        max_val: f32,
        dtype: QuantizationDType,
    ) -> QuantizationParams {
        let (qmin, qmax) = self.get_quantization_range(dtype);

        // Ensure min and max are not equal
        let (adjusted_min, adjusted_max) = if (max_val - min_val).abs() < 1e-8 {
            (min_val - 1e-4, max_val + 1e-4)
        } else {
            (min_val, max_val)
        };

        // Calculate scale and zero point
        let scale = (adjusted_max - adjusted_min) / (qmax - qmin) as f32;
        let zero_point_float = qmin as f32 - adjusted_min / scale;
        let zero_point = zero_point_float.round().clamp(qmin as f32, qmax as f32) as i32;

        QuantizationParams {
            scale,
            zero_point,
            min_val: adjusted_min,
            max_val: adjusted_max,
            dtype,
        }
    }

    /// Calculate tensor statistics for quantization
    pub fn calculate_tensor_stats(&self, tensor: &Tensor) -> Result<(f32, f32)> {
        let float_data = tensor.to_vec()?;

        if float_data.is_empty() {
            return Ok((0.0, 0.0));
        }

        let min_val = float_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = float_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Ok((min_val, max_val))
    }

    // Helper methods
    fn filter_parameters(
        &self,
        parameters: &HashMap<String, Parameter>,
    ) -> Result<HashMap<String, Parameter>> {
        match &self.config.layer_filter {
            LayerQuantizationFilter::All => Ok(parameters.clone()),
            LayerQuantizationFilter::Include(names) => Ok(parameters
                .iter()
                .filter(|(name, _)| names.iter().any(|n| name.contains(n)))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()),
            LayerQuantizationFilter::Exclude(names) => Ok(parameters
                .iter()
                .filter(|(name, _)| !names.iter().any(|n| name.contains(n)))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()),
            LayerQuantizationFilter::LayerTypes(types) => Ok(parameters
                .iter()
                .filter(|(name, _)| {
                    types
                        .iter()
                        .any(|t| self.matches_quantizable_layer_type(name, t))
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()),
        }
    }

    fn matches_quantizable_layer_type(
        &self,
        layer_name: &str,
        layer_type: &QuantizableLayerType,
    ) -> bool {
        match layer_type {
            QuantizableLayerType::Linear => {
                layer_name.contains("linear") || layer_name.contains("fc")
            }
            QuantizableLayerType::Convolution => layer_name.contains("conv"),
            QuantizableLayerType::Embedding => layer_name.contains("embed"),
            QuantizableLayerType::Attention => {
                layer_name.contains("attn") || layer_name.contains("attention")
            }
        }
    }

    fn calibrate_quantization_params(
        &mut self,
        parameters: &HashMap<String, Parameter>,
    ) -> Result<()> {
        for (name, param) in parameters {
            let (min_val, max_val) = self.calculate_tensor_stats(&*param.tensor().read())?;
            let params = self.calculate_quantization_params(min_val, max_val, self.config.dtype);
            self.quantization_params.insert(name.clone(), params);
        }
        Ok(())
    }

    fn quantize_weights(&mut self, parameters: &HashMap<String, Parameter>) -> Result<()> {
        // In a real implementation, this would modify the actual model weights
        // For now, we'll just verify that we can quantize them
        for (name, param) in parameters {
            if let Some(params) = self.quantization_params.get(name) {
                let _quantized = self.quantize_tensor(&*param.tensor().read(), params)?;
                // In practice, you would replace the parameter with the quantized version
            }
        }
        Ok(())
    }

    fn setup_fake_quantization(&mut self, _parameters: &HashMap<String, Parameter>) -> Result<()> {
        // Setup fake quantization for QAT
        // This would involve adding fake quantization nodes to the computation graph
        Ok(())
    }

    fn dynamic_quantize_weights(&mut self, parameters: &HashMap<String, Parameter>) -> Result<()> {
        // Dynamic quantization - quantize weights on-the-fly
        self.calibrate_quantization_params(parameters)?;
        self.quantize_weights(parameters)
    }

    fn calculate_quantization_stats(
        &self,
        original_params: &HashMap<String, Parameter>,
    ) -> Result<QuantizationStats> {
        let mut original_size = 0;
        let mut quantized_size = 0;
        let mut layer_stats = HashMap::new();

        for (name, param) in original_params {
            let param_count = param.tensor().read().numel();
            let original_bytes = param_count * 4; // Assuming 32-bit floats
            original_size += original_bytes;

            let bits_per_param = match self.config.dtype {
                QuantizationDType::Int8 | QuantizationDType::Uint8 => 8,
                QuantizationDType::Int16
                | QuantizationDType::Float16
                | QuantizationDType::BFloat16 => 16,
            };

            let quantized_bytes = param_count * (bits_per_param as usize / 8);
            quantized_size += quantized_bytes;

            // Calculate quantization error (simplified)
            let quantization_error = if let Some(params) = self.quantization_params.get(name) {
                let param_tensor = param.tensor();
                let tensor_guard = param_tensor.read();
                let original_tensor = &*tensor_guard;
                let quantized_tensor = self.quantize_tensor(original_tensor, params)?;
                self.calculate_mse_error(original_tensor, &quantized_tensor)?
            } else {
                0.0
            };

            layer_stats.insert(
                name.clone(),
                LayerQuantizationStats {
                    original_params: param_count,
                    quantized_params: param_count,
                    bits_per_param,
                    quantization_error,
                },
            );
        }

        let compression_ratio = original_size as f64 / quantized_size as f64;
        let theoretical_speedup = self.estimate_speedup(self.config.dtype);

        Ok(QuantizationStats {
            original_size,
            quantized_size,
            compression_ratio,
            layer_stats,
            theoretical_speedup,
        })
    }

    fn calculate_mse_error(&self, original: &Tensor, quantized: &Tensor) -> Result<f64> {
        let orig_f32 = original.to_vec()?;
        let quant_f32 = quantized.to_vec()?;

        if orig_f32.len() != quant_f32.len() {
            return Err(TorshError::ComputeError(
                "Tensor size mismatch for MSE calculation".to_string(),
            ));
        }

        let mse = orig_f32
            .iter()
            .zip(quant_f32.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f32>()
            / orig_f32.len() as f32;

        Ok(mse as f64)
    }

    fn estimate_speedup(&self, dtype: QuantizationDType) -> f64 {
        // Theoretical speedup estimates based on data type
        match dtype {
            QuantizationDType::Int8 | QuantizationDType::Uint8 => 2.0,
            QuantizationDType::Int16 => 1.5,
            QuantizationDType::Float16 | QuantizationDType::BFloat16 => 1.3,
        }
    }

    /// Get quantization statistics
    pub fn get_stats(&self) -> Option<&QuantizationStats> {
        self.stats.as_ref()
    }

    /// Save quantization parameters
    /// TODO: Implement proper QuantizationParams serialization
    pub fn save_quantization_params(&self, _path: &str) -> Result<()> {
        // let params_data = serde_json::to_string_pretty(&self.quantization_params)?;
        // std::fs::write(path, params_data)?;
        unimplemented!("QuantizationParams serialization not yet implemented")
    }

    /// Load quantization parameters
    pub fn load_quantization_params(&mut self, path: &str) -> Result<()> {
        let params_data = std::fs::read_to_string(path)?;
        // Note: This would need proper deserialization for QuantizationParams
        // For now, we'll just acknowledge the load attempt
        Ok(())
    }
}

/// Trait for activation observers
pub trait ActivationObserver: Send + Sync {
    /// Observe activation values
    fn observe(&mut self, tensor: &Tensor) -> Result<()>;

    /// Get the observed min and max values
    fn get_min_max(&self) -> (f32, f32);

    /// Reset the observer
    fn reset(&mut self);
}

/// Min-max observer implementation
pub struct MinMaxObserver {
    min_val: f32,
    max_val: f32,
}

impl MinMaxObserver {
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
        }
    }
}

impl ActivationObserver for MinMaxObserver {
    fn observe(&mut self, tensor: &Tensor) -> Result<()> {
        let float_data = tensor.to_vec()?;

        for &val in &float_data {
            self.min_val = self.min_val.min(val);
            self.max_val = self.max_val.max(val);
        }

        Ok(())
    }

    fn get_min_max(&self) -> (f32, f32) {
        (self.min_val, self.max_val)
    }

    fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
    }
}

/// Utility functions for quantization
pub mod utils {
    use super::*;

    /// Create a standard post-training quantization config
    pub fn post_training_quantization_config(dtype: QuantizationDType) -> QuantizationConfig {
        QuantizationConfig {
            dtype,
            strategy: QuantizationStrategy::PostTraining {
                calibration_samples: 1000,
                percentile_calibration: false,
            },
            ..Default::default()
        }
    }

    /// Create a quantization-aware training config
    pub fn quantization_aware_training_config(dtype: QuantizationDType) -> QuantizationConfig {
        QuantizationConfig {
            dtype,
            strategy: QuantizationStrategy::QuantizationAware {
                fake_quantize: true,
                schedule: Some(QuantizationSchedule {
                    start_epoch: 5,
                    ramp_epochs: 10,
                    progression: QuantizationProgression::Linear,
                }),
            },
            ..Default::default()
        }
    }

    /// Create a dynamic quantization config
    pub fn dynamic_quantization_config(dtype: QuantizationDType) -> QuantizationConfig {
        QuantizationConfig {
            dtype,
            strategy: QuantizationStrategy::Dynamic,
            quantize_activations: false, // Dynamic quantization is weights-only
            ..Default::default()
        }
    }

    /// Calculate theoretical memory savings
    pub fn calculate_memory_savings(original_size: usize, quantized_size: usize) -> (f64, f64) {
        let compression_ratio = original_size as f64 / quantized_size as f64;
        let memory_savings = (1.0 - (quantized_size as f64 / original_size as f64)) * 100.0;
        (compression_ratio, memory_savings)
    }

    /// Estimate inference speedup for different quantization types
    pub fn estimate_inference_speedup(dtype: QuantizationDType, hardware_type: &str) -> f64 {
        match (dtype, hardware_type) {
            (QuantizationDType::Int8, "cpu") => 2.5,
            (QuantizationDType::Int8, "gpu") => 1.8,
            (QuantizationDType::Uint8, "cpu") => 2.3,
            (QuantizationDType::Uint8, "gpu") => 1.7,
            (QuantizationDType::Int16, "cpu") => 1.6,
            (QuantizationDType::Int16, "gpu") => 1.4,
            (QuantizationDType::Float16, "gpu") => 1.8,
            (QuantizationDType::BFloat16, "gpu") => 1.7,
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;
    use torsh_tensor::Tensor;

    #[test]
    fn test_quantization_config_creation() {
        let config = QuantizationConfig::default();
        assert_eq!(config.dtype, QuantizationDType::Int8);
        assert!(config.quantize_weights);
        assert!(config.quantize_activations);
    }

    #[test]
    fn test_quantization_params_calculation() {
        let quantizer = ModelQuantizer::new(QuantizationConfig::default());
        let params = quantizer.calculate_quantization_params(-1.0, 1.0, QuantizationDType::Int8);

        assert!(params.scale > 0.0);
        assert!(params.zero_point >= -128 && params.zero_point <= 127);
    }

    #[test]
    fn test_tensor_quantization() {
        let device = DeviceType::Cpu;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let tensor = Tensor::from_data(data, vec![5], device).unwrap();

        let quantizer = ModelQuantizer::new(QuantizationConfig::default());
        let params = quantizer.calculate_quantization_params(-1.0, 1.0, QuantizationDType::Int8);
        let quantized = quantizer.quantize_tensor(&tensor, &params).unwrap();

        assert_eq!(quantized.shape(), tensor.shape());
    }

    #[test]
    fn test_min_max_observer() {
        let device = DeviceType::Cpu;
        let tensor1 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], device).unwrap();
        let tensor2 = Tensor::from_data(vec![-1.0, 0.0, 4.0], vec![3], device).unwrap();

        let mut observer = MinMaxObserver::new();
        observer.observe(&tensor1).unwrap();
        observer.observe(&tensor2).unwrap();

        let (min_val, max_val) = observer.get_min_max();
        assert_eq!(min_val, -1.0);
        assert_eq!(max_val, 4.0);
    }

    #[test]
    fn test_quantization_range() {
        let quantizer = ModelQuantizer::new(QuantizationConfig::default());

        let (min, max) = quantizer.get_quantization_range(QuantizationDType::Int8);
        assert_eq!(min, -128);
        assert_eq!(max, 127);

        let (min, max) = quantizer.get_quantization_range(QuantizationDType::Uint8);
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn test_utility_functions() {
        let config = utils::post_training_quantization_config(QuantizationDType::Int8);
        assert!(matches!(
            config.strategy,
            QuantizationStrategy::PostTraining { .. }
        ));

        let (compression, savings) = utils::calculate_memory_savings(1000, 250);
        assert_eq!(compression, 4.0);
        assert_eq!(savings, 75.0);

        let speedup = utils::estimate_inference_speedup(QuantizationDType::Int8, "cpu");
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = QuantizationConfig {
            dtype: QuantizationDType::Int16,
            strategy: QuantizationStrategy::Dynamic,
            quantize_weights: true,
            quantize_activations: false,
            ..Default::default()
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: QuantizationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.dtype, deserialized.dtype);
        assert!(matches!(
            deserialized.strategy,
            QuantizationStrategy::Dynamic
        ));
        assert!(!deserialized.quantize_activations);
    }
}
