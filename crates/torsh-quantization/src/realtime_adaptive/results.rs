//! Result types and report generation for adaptive quantization
//!
//! This module contains all result types, adaptation information,
//! runtime statistics, and comprehensive report generation functionality.

use super::{config::QuantizationParameters, quality_assessment::QualityMetrics};
use std::time::Duration;
use torsh_tensor::Tensor;

/// Result of adaptive quantization
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizationResult {
    /// Quantized tensor
    pub quantized_tensor: Tensor,
    /// Quantization parameters used
    pub parameters: QuantizationParameters,
    /// Quality metrics achieved
    pub quality_metrics: QualityMetrics,
    /// Detected workload pattern
    pub pattern_info: Option<String>,
    /// Adaptation information
    pub adaptation_info: Option<AdaptationInfo>,
    /// Runtime statistics
    pub runtime_stats: RuntimeStatistics,
}

/// Information about parameter adaptation
#[derive(Debug, Clone)]
pub struct AdaptationInfo {
    /// Original parameters
    pub original_params: QuantizationParameters,
    /// Adapted parameters
    pub adapted_params: QuantizationParameters,
    /// Quality improvement achieved
    pub quality_improvement: f32,
    /// Time taken for adaptation
    pub adaptation_time: Duration,
}

/// Simple quantization result
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Quantized tensor
    pub quantized_tensor: Tensor,
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
}

/// Runtime statistics collection
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    /// Total operations processed
    pub total_operations: usize,
    /// Total adaptation events
    pub adaptation_events: usize,
    /// Average quality score
    pub avg_quality: f32,
    /// Performance improvements
    pub performance_improvements: Vec<f32>,
    /// Energy savings achieved
    pub energy_savings: Vec<f32>,
    /// Prediction accuracy
    pub prediction_accuracy: f32,
}

impl Default for RuntimeStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            adaptation_events: 0,
            avg_quality: 1.0,
            performance_improvements: Vec::new(),
            energy_savings: Vec::new(),
            prediction_accuracy: 0.0,
        }
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: String,
    /// Suggestion text
    pub suggestion: String,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Expected improvement
    pub expected_improvement: f32,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl AdaptiveQuantizationResult {
    /// Generate comprehensive adaptive quantization report
    pub fn generate_report(&self) -> String {
        format!(
            "🤖 Adaptive Quantization Report\n\
             ================================\n\
             \n\
             🔧 Quantization Parameters:\n\
             • Scale: {:.6}\n\
             • Zero Point: {}\n\
             • Bit Width: {}\n\
             • Scheme: {}\n\
             \n\
             📊 Quality Metrics:\n\
             • SNR: {:.2} dB\n\
             • MSE: {:.6}\n\
             • PSNR: {:.2} dB\n\
             • Perceptual Score: {:.4}\n\
             • SSIM: {:.4}\n\
             \n\
             🎯 Pattern Information:\n\
             • Detected Pattern: {}\n\
             \n\
             ⚡ Adaptation Details:\n\
             {}\n\
             📈 Runtime Statistics:\n\
             • Total Operations: {}\n\
             • Adaptation Events: {}\n\
             • Average Quality: {:.4}\n\
             • Prediction Accuracy: {:.4}\n\
             \n\
             💡 Performance Insights:\n\
             • Adaptation Rate: {:.2}%\n\
             • Quality Consistency: {}\n\
             • Pattern Recognition: {}\n",
            self.parameters.scale,
            self.parameters.zero_point,
            self.parameters.bit_width,
            self.parameters.scheme,
            self.quality_metrics.snr,
            self.quality_metrics.mse,
            self.quality_metrics.psnr,
            self.quality_metrics.perceptual_score,
            self.quality_metrics.ssim,
            self.pattern_info.as_ref().unwrap_or(&"Unknown".to_string()),
            self.format_adaptation_info(),
            self.runtime_stats.total_operations,
            self.runtime_stats.adaptation_events,
            self.runtime_stats.avg_quality,
            self.runtime_stats.prediction_accuracy,
            self.calculate_adaptation_rate(),
            self.assess_quality_consistency(),
            self.assess_pattern_recognition_effectiveness()
        )
    }

    /// Format adaptation information
    fn format_adaptation_info(&self) -> String {
        match &self.adaptation_info {
            Some(info) => {
                format!(
                    "• Adaptation Performed: Yes\n\
                     • Quality Improvement: {:.4}\n\
                     • Adaptation Time: {:.2}ms\n\
                     • Parameter Changes:\n\
                       - Scale: {:.6} → {:.6}\n\
                       - Zero Point: {} → {}\n\
                       - Bit Width: {} → {}",
                    info.quality_improvement,
                    info.adaptation_time.as_secs_f32() * 1000.0,
                    info.original_params.scale,
                    info.adapted_params.scale,
                    info.original_params.zero_point,
                    info.adapted_params.zero_point,
                    info.original_params.bit_width,
                    info.adapted_params.bit_width
                )
            }
            None => "• Adaptation Performed: No".to_string(),
        }
    }

    /// Calculate adaptation rate percentage
    fn calculate_adaptation_rate(&self) -> f32 {
        if self.runtime_stats.total_operations == 0 {
            0.0
        } else {
            (self.runtime_stats.adaptation_events as f32
                / self.runtime_stats.total_operations as f32)
                * 100.0
        }
    }

    /// Assess quality consistency
    fn assess_quality_consistency(&self) -> String {
        match self.runtime_stats.avg_quality {
            q if q >= 0.95 => "Excellent".to_string(),
            q if q >= 0.85 => "Good".to_string(),
            q if q >= 0.70 => "Fair".to_string(),
            _ => "Needs Improvement".to_string(),
        }
    }

    /// Assess pattern recognition effectiveness
    fn assess_pattern_recognition_effectiveness(&self) -> String {
        match &self.pattern_info {
            Some(pattern) if pattern != "unknown" => "Effective".to_string(),
            Some(_) => "Limited".to_string(),
            None => "Disabled".to_string(),
        }
    }

    /// Generate detailed JSON report
    pub fn generate_json_report(&self) -> String {
        format!(
            r#"{{
  "adaptive_quantization_report": {{
    "parameters": {{
      "scale": {:.6},
      "zero_point": {},
      "bit_width": {},
      "scheme": "{}"
    }},
    "quality_metrics": {{
      "snr": {:.2},
      "mse": {:.6},
      "psnr": {:.2},
      "perceptual_score": {:.4},
      "ssim": {:.4}
    }},
    "pattern_info": {{
      "detected_pattern": "{}"
    }},
    "adaptation_info": {},
    "runtime_stats": {{
      "total_operations": {},
      "adaptation_events": {},
      "avg_quality": {:.4},
      "prediction_accuracy": {:.4}
    }},
    "performance_metrics": {{
      "adaptation_rate_percent": {:.2},
      "quality_consistency": "{}",
      "pattern_recognition": "{}"
    }}
  }}
}}"#,
            self.parameters.scale,
            self.parameters.zero_point,
            self.parameters.bit_width,
            self.parameters.scheme,
            self.quality_metrics.snr,
            self.quality_metrics.mse,
            self.quality_metrics.psnr,
            self.quality_metrics.perceptual_score,
            self.quality_metrics.ssim,
            self.pattern_info.as_ref().unwrap_or(&"unknown".to_string()),
            self.format_adaptation_info_json(),
            self.runtime_stats.total_operations,
            self.runtime_stats.adaptation_events,
            self.runtime_stats.avg_quality,
            self.runtime_stats.prediction_accuracy,
            self.calculate_adaptation_rate(),
            self.assess_quality_consistency(),
            self.assess_pattern_recognition_effectiveness()
        )
    }

    /// Format adaptation information for JSON
    fn format_adaptation_info_json(&self) -> String {
        match &self.adaptation_info {
            Some(info) => {
                format!(
                    r#"{{
      "performed": true,
      "quality_improvement": {:.4},
      "adaptation_time_ms": {:.2},
      "original_params": {{
        "scale": {:.6},
        "zero_point": {},
        "bit_width": {}
      }},
      "adapted_params": {{
        "scale": {:.6},
        "zero_point": {},
        "bit_width": {}
      }}
    }}"#,
                    info.quality_improvement,
                    info.adaptation_time.as_secs_f32() * 1000.0,
                    info.original_params.scale,
                    info.original_params.zero_point,
                    info.original_params.bit_width,
                    info.adapted_params.scale,
                    info.adapted_params.zero_point,
                    info.adapted_params.bit_width
                )
            }
            None => r#"{"performed": false}"#.to_string(),
        }
    }

    /// Generate CSV report line (for batch analysis)
    pub fn generate_csv_line(&self) -> String {
        format!(
            "{:.6},{},{},{},{:.2},{:.6},{:.2},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.2}",
            self.parameters.scale,
            self.parameters.zero_point,
            self.parameters.bit_width,
            self.parameters.scheme,
            self.quality_metrics.snr,
            self.quality_metrics.mse,
            self.quality_metrics.psnr,
            self.quality_metrics.perceptual_score,
            self.quality_metrics.ssim,
            self.pattern_info.as_ref().unwrap_or(&"unknown".to_string()),
            self.adaptation_info.is_some(),
            self.runtime_stats.total_operations,
            self.runtime_stats.avg_quality,
            self.runtime_stats.prediction_accuracy,
            self.calculate_adaptation_rate()
        )
    }

    /// Get CSV header for batch analysis
    pub fn csv_header() -> String {
        "scale,zero_point,bit_width,scheme,snr,mse,psnr,perceptual_score,ssim,pattern,adapted,operations,avg_quality,prediction_accuracy,adaptation_rate".to_string()
    }
}

impl RuntimeStatistics {
    /// Update average quality with new measurement
    pub fn update_avg_quality(&mut self, new_quality: f32) {
        if self.total_operations == 0 {
            self.avg_quality = new_quality;
        } else {
            self.avg_quality = (self.avg_quality * (self.total_operations - 1) as f32
                + new_quality)
                / self.total_operations as f32;
        }
    }

    /// Add performance improvement measurement
    pub fn add_performance_improvement(&mut self, improvement: f32) {
        self.performance_improvements.push(improvement);
        if self.performance_improvements.len() > 100 {
            self.performance_improvements.remove(0);
        }
    }

    /// Add energy savings measurement
    pub fn add_energy_savings(&mut self, savings: f32) {
        self.energy_savings.push(savings);
        if self.energy_savings.len() > 100 {
            self.energy_savings.remove(0);
        }
    }

    /// Calculate average performance improvement
    pub fn avg_performance_improvement(&self) -> f32 {
        if self.performance_improvements.is_empty() {
            0.0
        } else {
            self.performance_improvements.iter().sum::<f32>()
                / self.performance_improvements.len() as f32
        }
    }

    /// Calculate average energy savings
    pub fn avg_energy_savings(&self) -> f32 {
        if self.energy_savings.is_empty() {
            0.0
        } else {
            self.energy_savings.iter().sum::<f32>() / self.energy_savings.len() as f32
        }
    }
}
