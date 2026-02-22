//! Size analysis for quantized models

use crate::QScheme;
use std::collections::HashMap;

/// Size analysis for quantized models
pub struct SizeAnalyzer;

impl SizeAnalyzer {
    /// Calculate theoretical model size for different quantization schemes
    pub fn calculate_model_size(num_parameters: usize, scheme: QScheme) -> f32 {
        let bytes_per_param = match scheme {
            QScheme::Binary => 0.125,                                // 1 bit
            QScheme::Ternary => 0.25,                                // 2 bits (with some overhead)
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 0.5, // 4 bits
            QScheme::PerTensorAffine
            | QScheme::PerChannelAffine
            | QScheme::PerTensorSymmetric
            | QScheme::PerChannelSymmetric => 1.0, // 8 bits
            QScheme::MixedPrecision => 2.0,                          // Assuming average of FP16
            QScheme::GroupWise => 1.0,                               // 8 bits
        };

        num_parameters as f32 * bytes_per_param
    }

    /// Calculate size reduction ratio compared to FP32
    pub fn calculate_size_reduction_ratio(num_parameters: usize, scheme: QScheme) -> f32 {
        let fp32_size = num_parameters as f32 * 4.0; // 4 bytes per FP32 parameter
        let quantized_size = Self::calculate_model_size(num_parameters, scheme);

        if quantized_size == 0.0 {
            return 1.0;
        }

        fp32_size / quantized_size
    }

    /// Calculate memory footprint including activations
    pub fn calculate_total_memory_footprint(
        num_parameters: usize,
        num_activations: usize,
        param_scheme: QScheme,
        activation_scheme: QScheme,
    ) -> f32 {
        let param_size = Self::calculate_model_size(num_parameters, param_scheme);
        let activation_size = Self::calculate_model_size(num_activations, activation_scheme);

        param_size + activation_size
    }

    /// Estimate disk storage requirements with compression
    pub fn estimate_compressed_size(base_size_mb: f32, scheme: QScheme) -> f32 {
        let compression_ratio = match scheme {
            QScheme::Binary => 0.7,   // Binary data compresses well
            QScheme::Ternary => 0.75, // Some redundancy
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 0.8,
            QScheme::PerTensorAffine | QScheme::PerChannelAffine => 0.85,
            QScheme::PerTensorSymmetric | QScheme::PerChannelSymmetric => 0.82,
            QScheme::MixedPrecision => 0.9, // Less compression
            QScheme::GroupWise => 0.83,
        };

        base_size_mb * compression_ratio
    }

    /// Calculate size reduction factor (legacy compatibility)
    pub fn size_reduction_factor(
        original_scheme: QScheme,
        quantized_scheme: QScheme,
        num_parameters: usize,
    ) -> f32 {
        let original_size = Self::calculate_model_size(num_parameters, original_scheme);
        let quantized_size = Self::calculate_model_size(num_parameters, quantized_scheme);

        if quantized_size == 0.0 {
            return 1.0;
        }

        original_size / quantized_size
    }

    /// Analyze size impact for different quantization schemes
    pub fn analyze_size_impact(num_parameters: usize) -> HashMap<QScheme, f32> {
        let mut size_analysis = HashMap::new();

        let schemes = vec![
            QScheme::Binary,
            QScheme::Ternary,
            QScheme::Int4PerTensor,
            QScheme::PerTensorAffine,
            QScheme::PerChannelAffine,
            QScheme::MixedPrecision,
            QScheme::GroupWise,
        ];

        for scheme in schemes {
            let size_mb = Self::model_size_mb(num_parameters, scheme);
            size_analysis.insert(scheme, size_mb);
        }

        size_analysis
    }

    /// Calculate model size in megabytes
    pub fn model_size_mb(num_parameters: usize, scheme: QScheme) -> f32 {
        Self::calculate_model_size(num_parameters, scheme) / (1024.0 * 1024.0)
    }

    /// Generate comprehensive size analysis report
    pub fn generate_size_report(
        num_parameters: usize,
        schemes: &[QScheme],
    ) -> HashMap<QScheme, SizeReport> {
        let mut report = HashMap::new();
        let fp32_size_mb = (num_parameters as f32 * 4.0) / (1024.0 * 1024.0);

        for &scheme in schemes {
            let quantized_size_mb = Self::model_size_mb(num_parameters, scheme);
            let reduction_ratio = Self::calculate_size_reduction_ratio(num_parameters, scheme);
            let compressed_size_mb = Self::estimate_compressed_size(quantized_size_mb, scheme);

            report.insert(
                scheme,
                SizeReport {
                    original_size_mb: fp32_size_mb,
                    quantized_size_mb,
                    compressed_size_mb,
                    reduction_ratio,
                    space_saved_mb: fp32_size_mb - quantized_size_mb,
                    compression_efficiency: (fp32_size_mb - compressed_size_mb) / fp32_size_mb,
                },
            );
        }

        report
    }
}

/// Detailed size analysis report
#[derive(Debug, Clone)]
pub struct SizeReport {
    /// Original model size in MB (FP32)
    pub original_size_mb: f32,
    /// Quantized model size in MB
    pub quantized_size_mb: f32,
    /// Compressed quantized model size in MB
    pub compressed_size_mb: f32,
    /// Size reduction ratio (original/quantized)
    pub reduction_ratio: f32,
    /// Space saved in MB
    pub space_saved_mb: f32,
    /// Compression efficiency (0.0 to 1.0)
    pub compression_efficiency: f32,
}

impl SizeReport {
    /// Check if the size reduction meets a minimum threshold
    pub fn meets_reduction_threshold(&self, min_ratio: f32) -> bool {
        self.reduction_ratio >= min_ratio
    }

    /// Get space savings as percentage
    pub fn space_savings_percentage(&self) -> f32 {
        (self.space_saved_mb / self.original_size_mb) * 100.0
    }
}
