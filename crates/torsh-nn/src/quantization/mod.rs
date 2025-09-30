//! Quantization support for model compression and deployment optimization
//!
//! This module provides comprehensive quantization capabilities including:
//! - INT8 quantization for inference acceleration
//! - Quantization-aware training (QAT)
//! - Post-training quantization (PTQ)
//! - Dynamic quantization
//! - Model compression utilities

pub mod calibration;
pub mod ops;
pub mod qat;
pub mod schemes;
pub mod utils;

use crate::{Module, Parameter};
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Target data type for quantization
    pub dtype: DType,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Backend-specific optimizations
    pub backend_config: BackendQuantConfig,
    /// Calibration configuration
    pub calibration: CalibrationConfig,
    /// Whether to use per-channel quantization
    pub per_channel: bool,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
}

/// Quantization schemes
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization around zero
    Symmetric,
    /// Asymmetric quantization with zero-point
    Asymmetric,
    /// Dynamic quantization (runtime calibration)
    Dynamic,
    /// KL divergence-based quantization
    KLDivergence,
    /// Percentile-based quantization
    Percentile(f32),
}

/// Backend-specific quantization configuration
#[derive(Debug, Clone)]
pub struct BackendQuantConfig {
    /// Use hardware acceleration when available
    pub use_hardware_acceleration: bool,
    /// Kernel fusion optimizations
    pub enable_kernel_fusion: bool,
    /// Memory layout optimizations
    pub optimize_memory_layout: bool,
    /// Target deployment platform
    pub target_platform: DeploymentPlatform,
}

/// Deployment platform targets
#[derive(Debug, Clone, PartialEq)]
pub enum DeploymentPlatform {
    /// General CPU deployment
    CPU,
    /// GPU deployment
    GPU,
    /// Mobile devices (ARM, NEON)
    Mobile,
    /// Edge devices with specific constraints
    Edge,
    /// Server deployment
    Server,
    /// WebAssembly deployment
    WASM,
}

/// Calibration configuration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of calibration samples
    pub num_samples: usize,
    /// Calibration method
    pub method: CalibrationMethod,
    /// Percentile for outlier handling
    pub outlier_percentile: f32,
    /// Whether to use moving averages
    pub use_moving_average: bool,
    /// Moving average momentum
    pub momentum: f32,
}

/// Calibration methods
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Min-max calibration
    MinMax,
    /// Entropy-based calibration
    Entropy,
    /// MSE-based calibration
    MSE,
    /// Cosine similarity-based calibration
    CosineSimilarity,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point (for asymmetric quantization)
    pub zero_point: i32,
    /// Minimum quantized value
    pub qmin: i32,
    /// Maximum quantized value
    pub qmax: i32,
    /// Source dtype
    pub src_dtype: DType,
    /// Target dtype
    pub dst_dtype: DType,
}

impl QuantizationParams {
    /// Create symmetric quantization parameters
    pub fn symmetric(scale: f32, src_dtype: DType, dst_dtype: DType) -> Self {
        let (qmin, qmax) = match dst_dtype {
            DType::I8 => (-128i32, 127i32),
            DType::U8 => (0i32, 255i32),
            DType::I16 => (-32768i32, 32767i32),
            _ => panic!("Unsupported quantization dtype: {:?}", dst_dtype),
        };

        Self {
            scale,
            zero_point: 0,
            qmin,
            qmax,
            src_dtype,
            dst_dtype,
        }
    }

    /// Create asymmetric quantization parameters
    pub fn asymmetric(scale: f32, zero_point: i32, src_dtype: DType, dst_dtype: DType) -> Self {
        let (qmin, qmax) = match dst_dtype {
            DType::I8 => (-128i32, 127i32),
            DType::U8 => (0i32, 255i32),
            DType::I16 => (-32768i32, 32767i32),
            _ => panic!("Unsupported quantization dtype: {:?}", dst_dtype),
        };

        Self {
            scale,
            zero_point,
            qmin,
            qmax,
            src_dtype,
            dst_dtype,
        }
    }

    /// Quantize a tensor using these parameters
    pub fn quantize(&self, tensor: &Tensor) -> Result<Tensor> {
        ops::quantize_tensor(tensor, self)
    }

    /// Dequantize a tensor using these parameters
    pub fn dequantize(&self, tensor: &Tensor) -> Result<Tensor> {
        ops::dequantize_tensor(tensor, self)
    }
}

/// Quantized model wrapper
#[derive(Debug)]
pub struct QuantizedModel<M: Module> {
    /// Original model
    pub model: M,
    /// Quantization configuration
    pub config: QuantizationConfig,
    /// Per-layer quantization parameters
    pub layer_params: HashMap<String, QuantizationParams>,
    /// Calibration statistics
    pub calibration_stats: Option<CalibrationStats>,
}

impl<M: Module> QuantizedModel<M> {
    /// Create a new quantized model
    pub fn new(model: M, config: QuantizationConfig) -> Self {
        Self {
            model,
            config,
            layer_params: HashMap::new(),
            calibration_stats: None,
        }
    }

    /// Calibrate the model using sample data
    pub fn calibrate<I>(&mut self, calibration_data: I) -> Result<()>
    where
        I: Iterator<Item = Tensor>,
    {
        let mut calibrator = calibration::Calibrator::new(&self.config.calibration);
        calibrator.calibrate(&mut self.model, calibration_data)?;

        self.calibration_stats = Some(calibrator.stats());
        self.layer_params = calibrator.quantization_params();

        Ok(())
    }

    /// Convert model to quantized format
    pub fn quantize(&mut self) -> Result<()> {
        if self.layer_params.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Model must be calibrated before quantization".to_string(),
            ));
        }

        // Apply quantization to each layer
        for (layer_name, params) in &self.layer_params {
            // This would be implemented to actually quantize the model layers
            // For now, we just store the parameters
            println!(
                "Quantizing layer {} with scale={}, zero_point={}",
                layer_name, params.scale, params.zero_point
            );
        }

        Ok(())
    }

    /// Get model size reduction
    pub fn compression_ratio(&self) -> f32 {
        if self.layer_params.is_empty() {
            return 1.0;
        }

        // Calculate compression based on dtype size reduction
        let original_bits = match DType::F32 {
            DType::F32 => 32,
            DType::F16 => 16,
            _ => 32,
        };

        let quantized_bits = match self.config.dtype {
            DType::I8 | DType::U8 => 8,
            DType::I16 => 16,
            _ => 32,
        };

        original_bits as f32 / quantized_bits as f32
    }
}

impl<M: Module> Module for QuantizedModel<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // In a full implementation, this would:
        // 1. Quantize inputs if needed
        // 2. Run quantized inference
        // 3. Dequantize outputs if needed
        self.model.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.model.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.model.named_parameters()
    }

    fn training(&self) -> bool {
        self.model.training()
    }

    fn train(&mut self) {
        self.model.train()
    }

    fn eval(&mut self) {
        self.model.eval()
    }

    fn set_training(&mut self, training: bool) {
        self.model.set_training(training);
    }

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        self.model.to_device(device)
    }
}

/// Calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Number of samples processed
    pub num_samples: usize,
    /// Per-layer activation ranges
    pub activation_ranges: HashMap<String, (f32, f32)>,
    /// Per-layer weight ranges
    pub weight_ranges: HashMap<String, (f32, f32)>,
    /// Calibration loss/error metrics
    pub metrics: CalibrationMetrics,
}

/// Calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Mean squared error
    pub mse: f32,
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Cosine similarity
    pub cosine_similarity: f32,
    /// KL divergence
    pub kl_divergence: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            dtype: DType::I8,
            scheme: QuantizationScheme::Symmetric,
            backend_config: BackendQuantConfig::default(),
            calibration: CalibrationConfig::default(),
            per_channel: false,
            quantize_weights: true,
            quantize_activations: true,
        }
    }
}

impl Default for BackendQuantConfig {
    fn default() -> Self {
        Self {
            use_hardware_acceleration: true,
            enable_kernel_fusion: true,
            optimize_memory_layout: true,
            target_platform: DeploymentPlatform::CPU,
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            method: CalibrationMethod::MinMax,
            outlier_percentile: 99.99,
            use_moving_average: true,
            momentum: 0.9,
        }
    }
}

/// Convenience functions for common quantization tasks
pub mod prelude {
    pub use super::qat::utils::{calibrate_qat_model, prepare_qat_model, progressive_qat_training};
    pub use super::qat::{
        FakeQuantize, QATConfig, QATLinear, QATModel, QATScheduler, QuantizedInferenceModel,
    };
    pub use super::{
        BackendQuantConfig, CalibrationConfig, CalibrationMethod, DeploymentPlatform,
        QuantizationConfig, QuantizationParams, QuantizationScheme, QuantizedModel,
    };

    /// Create INT8 symmetric quantization config
    pub fn int8_symmetric() -> QuantizationConfig {
        QuantizationConfig {
            dtype: torsh_core::dtype::DType::I8,
            scheme: QuantizationScheme::Symmetric,
            ..Default::default()
        }
    }

    /// Create INT8 asymmetric quantization config
    pub fn int8_asymmetric() -> QuantizationConfig {
        QuantizationConfig {
            dtype: torsh_core::dtype::DType::I8,
            scheme: QuantizationScheme::Asymmetric,
            ..Default::default()
        }
    }

    /// Create dynamic quantization config
    pub fn dynamic_quantization() -> QuantizationConfig {
        QuantizationConfig {
            scheme: QuantizationScheme::Dynamic,
            ..Default::default()
        }
    }

    /// Create QAT config for INT8 training
    pub fn qat_int8_config() -> QATConfig {
        QATConfig {
            weight_bits: 8,
            activation_bits: 8,
            scheme: QuantizationScheme::Symmetric,
            ..Default::default()
        }
    }

    /// Create conservative QAT config with longer warmup
    pub fn qat_conservative_config() -> QATConfig {
        QATConfig {
            warmup_epochs: 5,
            qparam_lr: 0.005,
            observer_momentum: 0.05,
            ..Default::default()
        }
    }

    /// Create aggressive QAT config for fast convergence
    pub fn qat_aggressive_config() -> QATConfig {
        QATConfig {
            warmup_epochs: 1,
            qparam_lr: 0.02,
            observer_momentum: 0.2,
            ..Default::default()
        }
    }
}
