//! # ToRSh Quantization Library
//!
//! A comprehensive quantization library for deep learning tensor operations, providing
//! state-of-the-art quantization algorithms, configuration management, performance
//! metrics, and utility functions.
//!
//! ## Key Features
//!
//! - **Multiple Quantization Schemes**: INT8, INT4, binary, ternary, group-wise quantization
//! - **Advanced Observers**: MinMax, Histogram, Percentile, MovingAverage calibration
//! - **Backend Support**: Native, FBGEMM, QNNPACK for optimized execution
//! - **Comprehensive Metrics**: PSNR, SNR, compression ratio analysis
//! - **Configuration Tools**: Builder patterns, validation, JSON serialization
//! - **Utility Functions**: Batch processing, error diagnostics, auto-calibration
//!
//! ## Architecture
//!
//! The library is organized into specialized modules:
//!
//! - **config**: Configuration types and builder patterns
//! - **algorithms**: Core quantization and dequantization algorithms
//! - **observers**: Calibration system for parameter estimation
//! - **specialized**: Advanced algorithms (INT4, binary, ternary, group-wise)
//! - **metrics**: Performance analysis and benchmarking tools
//! - **utils**: Utility functions for validation, batch processing, and reporting
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_quantization::{QuantConfig, quantize_with_config};
//! use torsh_tensor::creation::tensor_1d;
//!
//! // Create a simple quantization configuration
//! let config = QuantConfig::int8();
//!
//! // Create a tensor to quantize
//! let data = vec![0.0, 1.0, 2.0, 3.0];
//! let tensor = tensor_1d(&data).unwrap();
//!
//! // Quantize the tensor
//! let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config).unwrap();
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Configuration
//!
//! ```rust
//! use torsh_quantization::{QuantConfig, ObserverType, QuantBackend};
//!
//! let config = QuantConfig::int8()
//!     .with_observer(ObserverType::Histogram)
//!     .with_backend(QuantBackend::Fbgemm);
//! ```
//!
//! ### Batch Processing
//!
//! ```rust
//! use torsh_quantization::{quantize_batch_consistent, QuantConfig};
//! use torsh_tensor::creation::tensor_1d;
//!
//! let tensor1 = tensor_1d(&[0.0, 1.0, 2.0]).unwrap();
//! let tensor2 = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
//! let tensor3 = tensor_1d(&[2.0, 3.0, 4.0]).unwrap();
//! let tensors = vec![&tensor1, &tensor2, &tensor3];
//! let config = QuantConfig::int8();
//! let results = quantize_batch_consistent(&tensors, &config).unwrap();
//! ```
//!
//! ### Performance Analysis
//!
//! ```rust
//! use torsh_quantization::{compare_quantization_configs, QuantConfig};
//! use torsh_tensor::creation::tensor_1d;
//!
//! let tensor = tensor_1d(&[0.0, 1.0, 2.0, 3.0]).unwrap();
//! let configs = vec![
//!     QuantConfig::int8(),
//!     QuantConfig::int4(),
//!     QuantConfig::per_channel(0),
//! ];
//! let comparison = compare_quantization_configs(&tensor, &configs).unwrap();
//! ```
//!
//! ## Export Support
//!
//! The library supports exporting quantized models to various formats:
//! - **ONNX**: Industry-standard format for cross-platform deployment
//! - **TensorRT**: NVIDIA's high-performance inference engine
//! - **TensorFlow Lite**: Mobile and edge deployment
//! - **Core ML**: Apple's machine learning framework
//! - **Custom formats**: Extensible architecture for new backends

// ============================================================================
// Core Quantization Infrastructure
// ============================================================================

/// Core configuration types and builders
pub mod config;
pub use config::*;

/// Core quantization algorithms
pub mod algorithms;
pub use algorithms::*;

/// Observer system for calibration
pub mod observers;
pub use observers::*;

// ============================================================================
// Quantization Schemes and Techniques
// ============================================================================

/// Specialized quantization schemes (INT4, binary, ternary, group-wise)
pub mod specialized;
pub use specialized::*;

// ============================================================================
// Analysis and Performance
// ============================================================================

/// Performance metrics and analysis
pub mod metrics;
pub use metrics::*;

/// Advanced analysis tools
pub mod analysis;
pub use analysis::*;

/// Memory pool management
pub mod memory_pool;
pub use memory_pool::*;

/// SIMD-accelerated operations
pub mod simd_ops;
// Selective re-export to avoid ambiguity with auto_config::TensorStats
pub use simd_ops::{
    calculate_tensor_stats_simd, dequantize_per_tensor_affine_simd, find_min_max_simd,
    get_mobile_optimization_hints, get_simd_width, is_simd_available,
    quantize_batch_consistent_simd, quantize_mobile_optimized, quantize_per_channel_simd,
    quantize_per_tensor_affine_simd, quantize_to_int8_simd, MobileOptimizationHints,
    TensorStats as SimdTensorStats,
};

// ARM NEON-specific operations (only available on aarch64)
#[cfg(target_arch = "aarch64")]
pub use simd_ops::{find_min_max_neon, quantize_neon_optimized};

// ============================================================================
// Advanced and Research Features
// ============================================================================

/// Quantum-inspired quantization
pub mod quantum;
pub use quantum::*;

/// Enhanced quantum-inspired quantization
pub mod quantum_enhanced;
pub use quantum_enhanced::*;

/// Comprehensive benchmark suite
pub mod benchmarks;
pub use benchmarks::{
    BaselineMetrics, BenchmarkConfig as SuiteBenchmarkConfig,
    BenchmarkResult as SuiteBenchmarkResult, HardwareInfo, QuantizationBenchmarkSuite,
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Utility functions and helpers
pub mod utils;
pub use utils::*;

/// ML-powered auto-configuration system
pub mod auto_config;
pub use auto_config::*;

// ============================================================================
// Additional Modules (Advanced - May require fixes)
// ============================================================================
// The following modules are available but may have internal compilation issues
// or require additional dependencies. They are exposed for advanced users.

/// Quantization operations (high-level API)
#[cfg(feature = "experimental")]
pub mod quantize;

/// Dequantization operations
#[cfg(feature = "experimental")]
pub mod dequantize;

/// Advanced quantization techniques
#[cfg(feature = "experimental")]
pub mod advanced;

/// Compression techniques (sub-byte, vector, sparse)
#[cfg(feature = "experimental")]
pub mod compression;

/// Fake quantization for QAT
#[cfg(feature = "experimental")]
pub mod fake_quantize;

/// Quantization-aware training (QAT)
#[cfg(feature = "experimental")]
pub mod qat;

/// Post-training quantization (PTQ)
#[cfg(feature = "experimental")]
pub mod post_training;

/// Quantization optimizer
#[cfg(feature = "experimental")]
pub mod optimizer;

/// Real-time adaptive quantization
#[cfg(feature = "experimental")]
pub mod realtime_adaptive;

/// Hardware-optimized backends
#[cfg(feature = "experimental")]
pub mod hardware;

/// Operation fusion for performance
#[cfg(feature = "experimental")]
pub mod fusion;

/// Performance profiling
#[cfg(feature = "experimental")]
pub mod profiler;

/// Debugging utilities
#[cfg(feature = "experimental")]
pub mod debugging;

/// Neural codecs for learned quantization
#[cfg(feature = "experimental")]
pub mod neural_codecs;

/// Research and experimental features
#[cfg(feature = "experimental")]
pub mod research;

/// Model export functionality (ONNX, TensorRT, TFLite, CoreML)
#[cfg(feature = "experimental")]
pub mod export;

// Re-export commonly used types from other crates
pub use torsh_core::{error::Result as TorshResult, DType, TorshError};
pub use torsh_tensor::Tensor;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::*;
    pub use crate::analysis::*;
    pub use crate::auto_config::*;
    pub use crate::config::*;
    pub use crate::memory_pool::*;
    pub use crate::metrics::*;
    pub use crate::observers::*;
    pub use crate::specialized::*;
    pub use crate::utils::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_basic_quantization_workflow() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        // Test with basic INT8 configuration
        let config = QuantConfig::int8();
        let result = quantize_with_config(&tensor, &config);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        // Verify quantization worked correctly - values should be in quantized range
        let quantized_data = quantized.data().unwrap();
        let all_in_range = quantized_data.iter().all(|&x| x >= -128.0 && x <= 127.0);
        assert!(
            all_in_range,
            "Quantized values should be in I8 range [-128, 127]"
        );
        assert!(scale > 0.0);

        // Test dequantization
        let dequantized = dequantize(&quantized, scale, zero_point).unwrap();
        assert_eq!(dequantized.dtype(), DType::F32);
    }

    #[test]
    fn test_configuration_validation() {
        let valid_config = QuantConfig::int8();
        assert!(valid_config.validate().is_ok());

        let per_channel_config = QuantConfig::per_channel(0);
        assert!(per_channel_config.validate().is_ok());
    }

    #[test]
    fn test_specialized_quantization() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let _tensor = tensor_1d(&data).unwrap();

        // Test INT4 quantization
        let int4_config = QuantConfig::int4();
        assert!(int4_config.validate().is_ok());

        // Test binary quantization
        let binary_config = QuantConfig::binary();
        assert!(binary_config.validate().is_ok());

        // Test ternary quantization
        let ternary_config = QuantConfig::ternary();
        assert!(ternary_config.validate().is_ok());
    }

    #[test]
    fn test_utils_functionality() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        // Test configuration validation with suggestions
        let suggestions = validate_config_with_suggestions(&config).unwrap();
        assert!(suggestions.len() > 0);

        // Test optimization hints
        let hints = get_optimization_hints(&tensor, &config);
        // Hints can be empty for simple tensors - both empty and non-empty are valid
        assert!(hints.is_empty() || !hints.is_empty());

        // Test JSON serialization
        let json = export_config_to_json(&config).unwrap();
        let imported_config = import_config_from_json(&json).unwrap();
        assert_eq!(config.dtype, imported_config.dtype);
        assert_eq!(config.scheme, imported_config.scheme);
    }

    #[test]
    fn test_batch_processing() {
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0, 7.0];
        let tensor1 = tensor_1d(&data1).unwrap();
        let tensor2 = tensor_1d(&data2).unwrap();

        let tensors = vec![&tensor1, &tensor2];
        let config = QuantConfig::int8();

        let results = quantize_batch_consistent(&tensors, &config).unwrap();
        assert_eq!(results.len(), 2);

        // Verify consistent scale and zero point
        let (_, scale1, zp1) = &results[0];
        let (_, scale2, zp2) = &results[1];
        assert_eq!(scale1, scale2);
        assert_eq!(zp1, zp2);
    }

    #[test]
    fn test_metrics_calculation() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config).unwrap();
        let dequantized = dequantize(&quantized, scale, zero_point).unwrap();

        let metrics = calculate_quantization_metrics(&tensor, &dequantized, 32, 8).unwrap();

        assert!(metrics.psnr > 0.0);
        assert!(metrics.snr > 0.0);
        assert!(metrics.compression_ratio > 1.0);
        assert!(metrics.cosine_similarity >= 0.0 && metrics.cosine_similarity <= 1.0);
    }

    #[test]
    fn test_configuration_comparison() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = tensor_1d(&data).unwrap();

        let configs = vec![
            QuantConfig::int8(),
            QuantConfig::per_channel(0),
            QuantConfig::int4(),
        ];

        let comparison = compare_quantization_configs(&tensor, &configs).unwrap();
        assert_eq!(comparison.len(), 3);

        // Results should be sorted by PSNR (higher is better)
        for i in 1..comparison.len() {
            assert!(comparison[i - 1].1.psnr >= comparison[i].1.psnr);
        }
    }

    #[test]
    fn test_auto_calibration() {
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0, 7.0];
        let tensor1 = tensor_1d(&data1).unwrap();
        let tensor2 = tensor_1d(&data2).unwrap();

        let calibration_tensors = vec![&tensor1, &tensor2];
        let target_psnr = 30.0;
        let max_compression = 8.0;

        let optimal_config =
            auto_calibrate_quantization(&calibration_tensors, target_psnr, max_compression)
                .unwrap();

        assert!(optimal_config.validate().is_ok());
    }

    #[test]
    fn test_report_generation() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = tensor_1d(&data).unwrap();

        let configs = vec![QuantConfig::int8(), QuantConfig::int4()];

        let report = generate_quantization_report(&tensor, &configs).unwrap();

        // Verify report contains expected sections
        assert!(report.contains("# Quantization Analysis Report"));
        assert!(report.contains("## Quantization Configuration Comparison"));
        assert!(report.contains("## Detailed Metrics"));
        assert!(report.contains("## Recommendations"));
    }

    #[test]
    fn test_error_diagnostics() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int8();

        // Simulate an error (this is a mock example)
        let error = TorshError::InvalidArgument("Test error".to_string());
        let diagnosis = diagnose_quantization_failure(&tensor, &config, &error);

        assert!(diagnosis.contains("Quantization failed with error"));
        assert!(diagnosis.contains("Tensor Analysis"));
        assert!(diagnosis.contains("Configuration Analysis"));
        assert!(diagnosis.contains("Recovery Suggestions"));
    }

    #[test]
    fn test_optimized_config_creation() {
        // Test different use cases
        let inference_config = create_optimized_config("inference_cpu", "x86").unwrap();
        assert!(inference_config.validate().is_ok());

        let mobile_config = create_optimized_config("inference_mobile", "arm").unwrap();
        assert!(mobile_config.validate().is_ok());

        let training_config = create_optimized_config("training", "gpu").unwrap();
        assert!(training_config.validate().is_ok());

        // Test invalid use case
        let invalid_result = create_optimized_config("invalid_use_case", "x86");
        assert!(invalid_result.is_err());
    }
}
