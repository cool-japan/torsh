//! Configuration types and defaults for adaptive quantization
//!
//! This module contains all configuration-related types and their default implementations
//! for the real-time adaptive quantization system.

use std::collections::HashMap;

/// Configuration for adaptive quantization
#[derive(Debug, Clone)]
pub struct AdaptiveQuantConfig {
    /// Enable ML-based parameter prediction
    pub enable_ml_prediction: bool,
    /// Enable real-time quality assessment
    pub enable_quality_assessment: bool,
    /// Enable workload pattern recognition
    pub enable_pattern_recognition: bool,
    /// Adaptation update frequency (operations)
    pub update_frequency: usize,
    /// Quality tolerance threshold
    pub quality_tolerance: f32,
    /// Performance priority weight (0.0-1.0)
    pub performance_weight: f32,
    /// Energy efficiency priority weight (0.0-1.0)
    pub energy_weight: f32,
    /// Accuracy priority weight (0.0-1.0)
    pub accuracy_weight: f32,
    /// Maximum adaptation rate per update
    pub max_adaptation_rate: f32,
}

impl Default for AdaptiveQuantConfig {
    fn default() -> Self {
        Self {
            enable_ml_prediction: true,
            enable_quality_assessment: true,
            enable_pattern_recognition: true,
            update_frequency: 100,
            quality_tolerance: 0.02,
            performance_weight: 0.3,
            energy_weight: 0.3,
            accuracy_weight: 0.4,
            max_adaptation_rate: 0.1,
        }
    }
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParameters {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Bit width
    pub bit_width: u8,
    /// Quantization scheme
    pub scheme: String,
}

impl Default for QuantizationParameters {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
            bit_width: 8,
            scheme: "symmetric".to_string(),
        }
    }
}

/// Optimization target specification
#[derive(Debug, Clone)]
pub struct OptimizationTarget {
    /// Target accuracy
    pub target_accuracy: f32,
    /// Target performance (ops/sec)
    pub target_performance: f32,
    /// Target energy efficiency
    pub target_energy_efficiency: f32,
    /// Priority weights
    pub weights: [f32; 3],
}

impl Default for OptimizationTarget {
    fn default() -> Self {
        Self {
            target_accuracy: 0.95,
            target_performance: 1000.0,
            target_energy_efficiency: 0.8,
            weights: [0.4, 0.3, 0.3], // [accuracy, performance, energy]
        }
    }
}

/// Constraint handler for optimization
#[derive(Debug, Clone)]
pub struct ConstraintHandler {
    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
    /// Quality constraints
    pub quality_constraints: QualityConstraints,
    /// Performance constraints
    pub performance_constraints: PerformanceConstraints,
}

impl Default for ConstraintHandler {
    fn default() -> Self {
        Self {
            hardware_constraints: HardwareConstraints::default(),
            quality_constraints: QualityConstraints::default(),
            performance_constraints: PerformanceConstraints::default(),
        }
    }
}

/// Hardware-specific constraints
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Supported bit widths
    pub supported_bit_widths: Vec<u8>,
    /// Maximum memory bandwidth
    pub max_memory_bandwidth: f32,
    /// SIMD capabilities
    pub simd_width: usize,
    /// Cache sizes
    pub cache_sizes: [usize; 3], // L1, L2, L3
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            supported_bit_widths: vec![4, 8, 16],
            max_memory_bandwidth: 1000.0, // GB/s
            simd_width: 256,              // bits
            cache_sizes: [32 * 1024, 256 * 1024, 8 * 1024 * 1024], // 32KB, 256KB, 8MB
        }
    }
}

/// Quality-related constraints
#[derive(Debug, Clone)]
pub struct QualityConstraints {
    /// Minimum acceptable SNR
    pub min_snr: f32,
    /// Maximum acceptable MSE
    pub max_mse: f32,
    /// Minimum perceptual quality
    pub min_perceptual_quality: f32,
}

impl Default for QualityConstraints {
    fn default() -> Self {
        Self {
            min_snr: 20.0,
            max_mse: 0.01,
            min_perceptual_quality: 0.8,
        }
    }
}

/// Performance-related constraints
#[derive(Debug, Clone)]
pub struct PerformanceConstraints {
    /// Maximum latency (ms)
    pub max_latency: f32,
    /// Minimum throughput (ops/sec)
    pub min_throughput: f32,
    /// Maximum energy consumption (watts)
    pub max_energy: f32,
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_latency: 10.0,     // 10ms
            min_throughput: 100.0, // 100 ops/sec
            max_energy: 50.0,      // 50W
        }
    }
}

/// Anomaly detection thresholds
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    /// SNR degradation threshold
    pub snr_threshold: f32,
    /// MSE increase threshold
    pub mse_threshold: f32,
    /// Perceptual quality drop threshold
    pub perceptual_threshold: f32,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            snr_threshold: -5.0,        // 5dB drop
            mse_threshold: 2.0,         // 2x increase
            perceptual_threshold: -0.1, // 10% drop
        }
    }
}

/// Performance profile for patterns
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Average execution time
    pub avg_execution_time: f32,
    /// Memory usage pattern
    pub memory_usage: f32,
    /// Energy consumption pattern
    pub energy_consumption: f32,
    /// Cache behavior
    pub cache_efficiency: f32,
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            avg_execution_time: 1.0,  // 1ms
            memory_usage: 100.0,      // 100MB
            energy_consumption: 10.0, // 10W
            cache_efficiency: 0.8,    // 80%
        }
    }
}

/// Feature extractor configuration
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Statistical features enabled
    pub enable_statistical: bool,
    /// Spectral features enabled
    pub enable_spectral: bool,
    /// Spatial features enabled
    pub enable_spatial: bool,
    /// Cached feature computation
    #[allow(dead_code)]
    pub(crate) feature_cache: HashMap<String, Vec<f32>>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            enable_statistical: true,
            enable_spectral: true,
            enable_spatial: true,
            feature_cache: HashMap::new(),
        }
    }
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self::default()
    }
}
