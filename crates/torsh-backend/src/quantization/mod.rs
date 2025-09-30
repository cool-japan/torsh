//! Comprehensive quantization module for ToRSh backend
//!
//! This module provides a complete quantization framework for reducing model size
//! and improving inference performance while maintaining acceptable accuracy.
//! It supports various quantization schemes, hardware acceleration, and calibration
//! methods suitable for production deep learning systems.
//!
//! # Features
//!
//! - **Multiple Data Types**: Support for INT8, UINT8, INT4, UINT4, Binary, and Mixed precision
//! - **Quantization Schemes**: Linear, Symmetric, Asymmetric, Channel-wise, Block-wise, and Logarithmic
//! - **Hardware Acceleration**: CPU SIMD, Intel VNNI, NVIDIA DP4A, and Tensor Core support
//! - **Calibration Methods**: Min-max, Percentile, Entropy-based, MSE, and Adaptive calibration
//! - **Performance Tools**: Comprehensive benchmarking and auto-tuning capabilities
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_backend::quantization::{
//!     QuantizationParams, QuantizedDType, QuantizationScheme,
//!     ops::CpuQuantizationOps, QuantizationOps,
//! };
//! use torsh_core::Device;
//!
//! // Create quantization parameters
//! let params = QuantizationParams::int8_symmetric();
//!
//! // Create quantization operations
//! let device = Device::cpu().unwrap();
//! let ops = CpuQuantizationOps::new(device);
//!
//! // Quantize data
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let quantized = ops.quantize_f32(&data, &params).unwrap();
//!
//! // Dequantize back
//! let dequantized = ops.dequantize_f32(&quantized, &params).unwrap();
//! ```
//!
//! # Module Organization
//!
//! The quantization module is organized into several specialized sub-modules:
//!
//! - [`types`]: Core quantization data types and schemes
//! - [`params`]: Quantization parameter structures and utilities
//! - [`tensor`]: Quantized tensor representation and operations
//! - [`ops`]: Quantization operation traits and implementations
//! - [`hardware`]: Hardware acceleration features and detection
//! - [`specialized`]: Specialized hardware-specific operations (VNNI, DP4A, Tensor Cores)
//! - [`accelerator`]: Advanced acceleration and auto-tuning capabilities
//! - [`calibration`]: Calibration methods for optimal parameter selection
//! - [`benchmarks`]: Performance measurement and analysis tools

// Core modules
pub mod core;
pub mod ops;
pub mod params;
pub mod tensor;
pub mod types;

// Hardware acceleration modules
pub mod accelerator;
pub mod hardware;
pub mod specialized;

// Utility modules
pub mod benchmarks;
pub mod calibration;

// Re-export core types for convenience
pub use core::{dequantize_from_int8, quantize_to_int8};
pub use ops::{CpuQuantizationOps, QuantizationOps};
pub use params::QuantizationParams;
pub use tensor::QuantizedTensor;
pub use types::{QuantizationScheme, QuantizedDType};

// Re-export hardware features
pub use hardware::{
    QuantizationHardwareFeatures, QuantizationPerformanceHints, QuantizedMemoryLayout,
    SimdQuantizationOps,
};

// Re-export specialized operations
pub use specialized::{
    Dp4aQuantizationOps, SpecializedQuantizationOps, TensorCoreFormat, TensorCoreQuantizationOps,
    VnniQuantizationOps,
};

// Re-export accelerator types
pub use accelerator::{
    AdvancedQuantizationAccelerator, AutoTuningConfig, BenchmarkResults, OptimalQuantizationConfig,
    PerformanceRequirements, QuantizationOperationType, QuantizationRecommendations,
    QuantizationWorkload,
};

// Re-export calibration types
pub use calibration::{
    CalibrationFunction, CalibrationMethod, CalibrationStatistics, PercentileCalibrator,
    QuantizationCalibrator,
};

// Re-export benchmark types
pub use benchmarks::{
    BenchmarkConfig, BenchmarkResult, BenchmarkSummary, ComparativeBenchmarkResult,
    MemoryBenchmarkResults, MemoryUsage, QuantizationBenchmarkSuite,
};

use crate::{BackendResult, Device};
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

/// Create a quantization system for the specified device with auto-detected features
///
/// This is a convenience function that creates a complete quantization system
/// with hardware acceleration and calibration capabilities for the given device.
///
/// # Arguments
///
/// * `device` - Target device for quantization operations
///
/// # Returns
///
/// A configured quantization system ready for use
///
/// # Examples
///
/// ```rust
/// use torsh_backend::quantization;
/// use torsh_core::Device;
///
/// let device = Device::cpu().unwrap();
/// let system = quantization::create_quantization_system(device).unwrap();
/// ```
pub fn create_quantization_system(device: Device) -> BackendResult<QuantizationSystem> {
    QuantizationSystem::new(device)
}

/// Create quantization parameters optimized for the given data type and accuracy requirements
///
/// This function provides intelligent defaults for quantization parameters based on
/// the target data type and desired accuracy trade-offs.
///
/// # Arguments
///
/// * `dtype` - Target quantization data type
/// * `accuracy_priority` - Whether to prioritize accuracy (true) or speed (false)
///
/// # Returns
///
/// Optimized quantization parameters
pub fn create_optimal_params(dtype: QuantizedDType, accuracy_priority: bool) -> QuantizationParams {
    if accuracy_priority {
        // Prioritize accuracy with conservative settings
        match dtype {
            QuantizedDType::Int8 => QuantizationParams::int8_symmetric(),
            QuantizedDType::UInt8 => QuantizationParams::uint8_asymmetric(),
            QuantizedDType::Int4 => QuantizationParams::int4_symmetric(),
            _ => {
                let mut params = QuantizationParams::default();
                params.dtype = dtype;
                params
            }
        }
    } else {
        // Prioritize speed with more aggressive settings
        let mut params = QuantizationParams {
            dtype: dtype.clone(),
            scheme: QuantizationScheme::Symmetric, // Faster than asymmetric
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        };

        // Use asymmetric only for unsigned types
        if matches!(
            dtype,
            QuantizedDType::UInt8 | QuantizedDType::UInt4 | QuantizedDType::UInt16
        ) {
            params.scheme = QuantizationScheme::Asymmetric;
        }

        params
    }
}

/// Comprehensive quantization system that coordinates all quantization functionality
///
/// This struct provides a high-level interface to the quantization subsystem,
/// coordinating hardware detection, operation selection, calibration, and benchmarking.
#[derive(Debug)]
pub struct QuantizationSystem {
    /// Device for quantization operations
    device: Device,
    /// Hardware features available
    hw_features: QuantizationHardwareFeatures,
    /// Base quantization operations
    base_ops: CpuQuantizationOps,
    /// Advanced accelerator (if available)
    accelerator: Option<AdvancedQuantizationAccelerator>,
    /// Calibrator for parameter optimization
    calibrator: QuantizationCalibrator,
    /// Benchmark suite for performance measurement
    benchmark_suite: QuantizationBenchmarkSuite,
}

impl QuantizationSystem {
    /// Create a new quantization system for the specified device
    pub fn new(device: Device) -> BackendResult<Self> {
        let hw_features = QuantizationHardwareFeatures::detect_for_device(&device);
        let base_ops = CpuQuantizationOps::new();

        // Create advanced accelerator if hardware features are available
        let accelerator = if hw_features.supports_int8_simd || hw_features.supports_tensor_cores {
            Some(AdvancedQuantizationAccelerator::new(
                device.clone(),
                Arc::new(base_ops.clone()),
            ))
        } else {
            None
        };

        let calibrator = QuantizationCalibrator::new(CalibrationMethod::Adaptive, device.clone());
        let benchmark_suite =
            QuantizationBenchmarkSuite::new(device.clone(), BenchmarkConfig::default());

        Ok(Self {
            device,
            hw_features,
            base_ops,
            accelerator,
            calibrator,
            benchmark_suite,
        })
    }

    /// Get hardware features for this system
    pub fn hardware_features(&self) -> &QuantizationHardwareFeatures {
        &self.hw_features
    }

    /// Get the device this system is configured for
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if advanced acceleration is available
    pub fn has_acceleration(&self) -> bool {
        self.accelerator.is_some()
    }

    /// Get performance recommendations for the given workload
    pub fn get_recommendations(
        &self,
        workload: &QuantizationWorkload,
    ) -> QuantizationRecommendations {
        if let Some(ref accelerator) = self.accelerator {
            accelerator.get_recommendations(workload)
        } else {
            // Provide basic recommendations without acceleration
            QuantizationRecommendations::default()
        }
    }

    /// Calibrate quantization parameters from sample data
    pub fn calibrate_from_samples(
        &mut self,
        samples: Vec<Vec<f32>>,
        dtype: QuantizedDType,
        method: CalibrationMethod,
    ) -> BackendResult<QuantizationParams> {
        self.calibrator.set_method(method);
        self.calibrator.clear_samples();
        self.calibrator.add_samples(samples);
        self.calibrator.calibrate(dtype)
    }

    /// Auto-tune quantization parameters for optimal performance
    pub fn auto_tune(
        &mut self,
        workload: &QuantizationWorkload,
    ) -> BackendResult<OptimalQuantizationConfig> {
        if let Some(ref mut accelerator) = self.accelerator {
            accelerator.auto_tune(workload)
        } else {
            // Provide basic auto-tuning without acceleration
            Ok(OptimalQuantizationConfig::default())
        }
    }

    /// Benchmark quantization operations
    pub fn benchmark_operations(&mut self) -> BackendResult<BenchmarkSummary> {
        self.benchmark_suite
            .benchmark_quantization_ops(&self.base_ops)
    }

    /// Perform quantization using the optimal operations for this hardware
    pub fn quantize_f32(
        &self,
        input: &[f32],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        // Use accelerated operations if available and beneficial
        if self.should_use_acceleration(&params.dtype) {
            if let Some(ref accelerator) = self.accelerator {
                // For now, delegate to base ops
                // In a full implementation, would use accelerated paths
                return self.base_ops.quantize_f32(input, params);
            }
        }

        self.base_ops.quantize_f32(input, params)
    }

    /// Perform dequantization using the optimal operations for this hardware
    pub fn dequantize_f32(
        &self,
        input: &[u8],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        // Use accelerated operations if available and beneficial
        if self.should_use_acceleration(&params.dtype) {
            if let Some(ref accelerator) = self.accelerator {
                // For now, delegate to base ops
                // In a full implementation, would use accelerated paths
                return self.base_ops.dequantize_f32(input, params);
            }
        }

        self.base_ops.dequantize_f32(input, params)
    }

    /// Perform quantized matrix multiplication
    pub fn qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        self.base_ops.qmatmul(a, b)
    }

    /// Determine if acceleration should be used for the given data type
    fn should_use_acceleration(&self, dtype: &QuantizedDType) -> bool {
        self.hw_features.supports_dtype_efficiently(dtype)
    }

    /// Create a quantized tensor with optimal memory layout
    pub fn create_quantized_tensor(
        &self,
        shape: Vec<usize>,
        params: QuantizationParams,
    ) -> QuantizedTensor {
        QuantizedTensor::new(shape, params, self.device.clone())
    }

    /// Get optimal block size for operations on this hardware
    pub fn optimal_block_size(&self) -> usize {
        self.hw_features.optimal_block_size()
    }

    /// Get performance hints for this hardware
    pub fn performance_hints(&self) -> QuantizationPerformanceHints {
        QuantizationPerformanceHints::for_hardware(&self.hw_features)
    }
}

/// Utility functions for common quantization tasks
/// Convert floating-point data to a quantized tensor with automatic parameter selection
///
/// This function automatically selects optimal quantization parameters based on
/// the data characteristics and hardware capabilities.
pub fn auto_quantize_tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: Device,
    target_dtype: QuantizedDType,
) -> BackendResult<QuantizedTensor> {
    // Create calibrator to determine optimal parameters
    let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::Adaptive, device.clone());
    calibrator.add_sample(data.to_vec());

    let params = calibrator.calibrate(target_dtype)?;

    // Create quantization system
    let system = QuantizationSystem::new(device)?;
    let quantized_data = system.quantize_f32(data, &params)?;

    Ok(QuantizedTensor {
        data: quantized_data,
        shape,
        params,
        device: system.device.clone(),
    })
}

/// Estimate memory savings from quantization
///
/// Returns the memory savings ratio (0.0 to 1.0) when quantizing from FP32
/// to the specified quantized data type.
pub fn estimate_memory_savings(dtype: &QuantizedDType) -> f64 {
    let fp32_bits = 32.0;
    let quantized_bits = dtype.bits() as f64;
    1.0 - (quantized_bits / fp32_bits)
}

/// Estimate accuracy impact from quantization
///
/// Returns an estimated accuracy retention ratio (0.0 to 1.0) for the
/// specified quantization configuration.
pub fn estimate_accuracy_impact(dtype: &QuantizedDType, scheme: QuantizationScheme) -> f64 {
    // Base accuracy based on data type bit width
    let base_accuracy: f64 = match dtype {
        QuantizedDType::Int16 | QuantizedDType::UInt16 => 0.99,
        QuantizedDType::Int8 | QuantizedDType::UInt8 => 0.95,
        QuantizedDType::Int4 | QuantizedDType::UInt4 => 0.85,
        QuantizedDType::Binary => 0.70,
        QuantizedDType::Mixed(_) => 0.90,
    };

    // Scheme-specific adjustments
    let scheme_factor: f64 = match scheme {
        QuantizationScheme::Symmetric => 1.0,
        QuantizationScheme::Linear => 0.98,
        QuantizationScheme::Asymmetric => 0.96,
        QuantizationScheme::ChannelWise => 1.02, // Better accuracy
        QuantizationScheme::BlockWise => 1.01,   // Slightly better
        QuantizationScheme::Logarithmic => 0.90, // More aggressive
    };

    (base_accuracy * scheme_factor).min(1.0f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_quantization_system() {
        let device = Device::cpu().unwrap();
        let system = create_quantization_system(device);
        assert!(system.is_ok());

        let system = system.unwrap();
        assert!(!system.device().device_type().to_string().is_empty());
    }

    #[test]
    fn test_create_optimal_params() {
        // Test accuracy-prioritized parameters
        let params_acc = create_optimal_params(QuantizedDType::Int8, true);
        assert_eq!(params_acc.dtype, QuantizedDType::Int8);
        assert_eq!(params_acc.scheme, QuantizationScheme::Symmetric);

        // Test speed-prioritized parameters
        let params_speed = create_optimal_params(QuantizedDType::UInt8, false);
        assert_eq!(params_speed.dtype, QuantizedDType::UInt8);
        assert_eq!(params_speed.scheme, QuantizationScheme::Asymmetric);
    }

    #[test]
    fn test_quantization_system_creation() {
        let device = Device::cpu().unwrap();
        let system = QuantizationSystem::new(device);
        assert!(system.is_ok());

        let system = system.unwrap();
        assert!(system.hardware_features().max_parallel_ops >= 1);
    }

    #[test]
    fn test_quantization_system_operations() {
        let device = Device::cpu().unwrap();
        let system = QuantizationSystem::new(device).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let params = QuantizationParams::int8_symmetric();

        // Test quantization
        let quantized = system.quantize_f32(&data, &params);
        assert!(quantized.is_ok());

        // Test dequantization
        let quantized_data = quantized.unwrap();
        let dequantized = system.dequantize_f32(&quantized_data, &params);
        assert!(dequantized.is_ok());

        let dequantized_data = dequantized.unwrap();
        assert_eq!(dequantized_data.len(), data.len());
    }

    #[test]
    fn test_auto_quantize_tensor() {
        let device = Device::cpu().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let result = auto_quantize_tensor(&data, shape.clone(), device, QuantizedDType::Int8);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.params.dtype, QuantizedDType::Int8);
    }

    #[test]
    fn test_memory_savings_estimation() {
        // INT8 should save 75% memory compared to FP32
        let savings_int8 = estimate_memory_savings(&QuantizedDType::Int8);
        assert!((savings_int8 - 0.75).abs() < 0.01);

        // INT4 should save 87.5% memory
        let savings_int4 = estimate_memory_savings(&QuantizedDType::Int4);
        assert!((savings_int4 - 0.875).abs() < 0.01);

        // Binary should save 96.875% memory
        let savings_binary = estimate_memory_savings(&QuantizedDType::Binary);
        assert!((savings_binary - 0.96875).abs() < 0.01);
    }

    #[test]
    fn test_accuracy_impact_estimation() {
        // INT8 with symmetric scheme should have high accuracy
        let accuracy_int8 =
            estimate_accuracy_impact(&QuantizedDType::Int8, QuantizationScheme::Symmetric);
        assert!(accuracy_int8 >= 0.90);

        // INT4 should have lower accuracy than INT8
        let accuracy_int4 =
            estimate_accuracy_impact(&QuantizedDType::Int4, QuantizationScheme::Symmetric);
        assert!(accuracy_int4 < accuracy_int8);

        // Channel-wise should have better accuracy than linear
        let accuracy_channelwise =
            estimate_accuracy_impact(&QuantizedDType::Int8, QuantizationScheme::ChannelWise);
        let accuracy_linear =
            estimate_accuracy_impact(&QuantizedDType::Int8, QuantizationScheme::Linear);
        assert!(accuracy_channelwise >= accuracy_linear);
    }

    #[test]
    fn test_quantization_system_calibration() {
        let device = Device::cpu().unwrap();
        let mut system = QuantizationSystem::new(device).unwrap();

        let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result =
            system.calibrate_from_samples(samples, QuantizedDType::Int8, CalibrationMethod::MinMax);

        assert!(result.is_ok());
        let params = result.unwrap();
        assert_eq!(params.dtype, QuantizedDType::Int8);
    }

    #[test]
    fn test_quantization_system_benchmarking() {
        let device = Device::cpu().unwrap();
        let mut system = QuantizationSystem::new(device).unwrap();

        let result = system.benchmark_operations();
        assert!(result.is_ok());

        let summary = result.unwrap();
        assert!(!summary.results.is_empty());
    }

    #[test]
    fn test_quantization_system_tensor_creation() {
        let device = Device::cpu().unwrap();
        let system = QuantizationSystem::new(device).unwrap();

        let shape = vec![2, 3, 4];
        let params = QuantizationParams::int8_symmetric();

        let tensor = system.create_quantized_tensor(shape.clone(), params.clone());
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.params.dtype, params.dtype);
    }

    #[test]
    fn test_quantization_system_performance_hints() {
        let device = Device::cpu().unwrap();
        let system = QuantizationSystem::new(device).unwrap();

        let hints = system.performance_hints();
        assert!(!hints.preferred_dtypes.is_empty());
        assert!(!hints.preferred_schemes.is_empty());
        assert!(hints.optimal_batch_size > 0);
    }
}
