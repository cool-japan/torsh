//! Real-time quality assessment system
//!
//! This module provides comprehensive quality assessment including SNR, MSE, PSNR,
//! perceptual quality metrics, and degradation detection for adaptive quantization.

use super::config::{AnomalyThresholds, QuantizationParameters};
use crate::TorshResult;
use std::collections::VecDeque;
use std::time::Instant;
use torsh_tensor::Tensor;

/// Real-time quality assessment system
#[derive(Debug, Clone)]
pub struct QualityAssessor {
    /// Quality metrics to track
    #[allow(dead_code)]
    metrics: QualityMetrics,
    /// Quality history for trend analysis
    quality_history: VecDeque<QualityMeasurement>,
    /// Anomaly detection thresholds
    #[allow(dead_code)]
    anomaly_thresholds: AnomalyThresholds,
    /// Degradation detector
    degradation_detector: DegradationDetector,
}

/// Quality metrics tracking
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Mean squared error
    pub mse: f32,
    /// Peak signal-to-noise ratio
    pub psnr: f32,
    /// Perceptual quality score
    pub perceptual_score: f32,
    /// Structural similarity index
    pub ssim: f32,
}

/// Quality measurement with timestamp
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Quality metrics at this measurement
    pub metrics: QualityMetrics,
    /// Quantization parameters used
    pub quant_params: QuantizationParameters,
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Operation count at measurement
    pub operation_count: usize,
}

/// Degradation detector for quality monitoring
#[derive(Debug, Clone)]
pub struct DegradationDetector {
    /// Window size for trend analysis
    window_size: usize,
    /// Minimum degradation slope to trigger alert
    #[allow(dead_code)]
    degradation_slope_threshold: f32,
    /// Recent quality measurements
    recent_measurements: VecDeque<f32>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            snr: 0.0,
            mse: 0.0,
            psnr: 0.0,
            perceptual_score: 1.0,
            ssim: 1.0,
        }
    }
}

impl QualityAssessor {
    /// Create new quality assessor
    pub fn new() -> Self {
        Self {
            metrics: QualityMetrics::default(),
            quality_history: VecDeque::new(),
            anomaly_thresholds: AnomalyThresholds {
                snr_threshold: 15.0,
                mse_threshold: 0.1,
                perceptual_threshold: 0.7,
            },
            degradation_detector: DegradationDetector {
                window_size: 10,
                degradation_slope_threshold: -0.05,
                recent_measurements: VecDeque::new(),
            },
        }
    }

    /// Assess quality of quantized tensor
    pub fn assess_quality(
        &mut self,
        original: &Tensor,
        quantized: &Tensor,
        params: &QuantizationParameters,
    ) -> TorshResult<QualityMetrics> {
        let orig_data = original.data()?;
        let quant_data = quantized.data()?;

        if orig_data.len() != quant_data.len() {
            return Err(torsh_core::TorshError::operation_error(
                "quality assessment: Tensor size mismatch",
            ));
        }

        // Calculate MSE
        let mse = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f32>()
            / orig_data.len() as f32;

        // Calculate SNR
        let signal_power =
            orig_data.iter().map(|x| x.powi(2)).sum::<f32>() / orig_data.len() as f32;
        let snr = if mse > 0.0 {
            10.0 * (signal_power / mse).log10()
        } else {
            f32::INFINITY
        };

        // Calculate PSNR
        let max_val = orig_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let psnr = if mse > 0.0 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else {
            f32::INFINITY
        };

        // Simplified perceptual quality score
        let perceptual_score = 1.0 / (1.0 + mse);

        // Simplified SSIM
        let ssim = self.calculate_ssim(&orig_data, &quant_data);

        let metrics = QualityMetrics {
            snr: snr.min(100.0), // Cap at reasonable value
            mse,
            psnr: psnr.min(100.0), // Cap at reasonable value
            perceptual_score,
            ssim,
        };

        // Record measurement
        let measurement = QualityMeasurement {
            metrics: metrics.clone(),
            quant_params: params.clone(),
            timestamp: Instant::now(),
            operation_count: self.quality_history.len(),
        };

        self.quality_history.push_back(measurement);
        if self.quality_history.len() > 1000 {
            self.quality_history.pop_front();
        }

        // Update degradation detector
        self.degradation_detector
            .recent_measurements
            .push_back(perceptual_score);
        if self.degradation_detector.recent_measurements.len()
            > self.degradation_detector.window_size
        {
            self.degradation_detector.recent_measurements.pop_front();
        }

        Ok(metrics)
    }

    /// Calculate simplified SSIM
    fn calculate_ssim(&self, orig: &[f32], quant: &[f32]) -> f32 {
        // Simplified SSIM calculation
        let mu1 = orig.iter().sum::<f32>() / orig.len() as f32;
        let mu2 = quant.iter().sum::<f32>() / quant.len() as f32;

        let sigma1_sq = orig.iter().map(|x| (x - mu1).powi(2)).sum::<f32>() / orig.len() as f32;
        let sigma2_sq = quant.iter().map(|x| (x - mu2).powi(2)).sum::<f32>() / quant.len() as f32;
        let sigma12 = orig
            .iter()
            .zip(quant.iter())
            .map(|(x, y)| (x - mu1) * (y - mu2))
            .sum::<f32>()
            / orig.len() as f32;

        let c1 = 0.01_f32.powi(2);
        let c2 = 0.03_f32.powi(2);

        let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
        let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            1.0
        }
    }

    /// Detect quality degradation trends
    pub fn detect_degradation(&self) -> bool {
        if self.degradation_detector.recent_measurements.len() < 3 {
            return false;
        }

        let measurements: Vec<f32> = self
            .degradation_detector
            .recent_measurements
            .iter()
            .cloned()
            .collect();

        // Simple trend analysis - check if quality is consistently decreasing
        let mut decreasing_count = 0;
        for i in 1..measurements.len() {
            if measurements[i] < measurements[i - 1] {
                decreasing_count += 1;
            }
        }

        // If more than half the recent measurements show degradation
        decreasing_count > measurements.len() / 2
    }

    /// Get quality statistics over recent history
    pub fn get_quality_statistics(&self) -> QualityStatistics {
        if self.quality_history.is_empty() {
            return QualityStatistics::default();
        }

        let recent_count = self.quality_history.len().min(100);
        let recent_measurements: Vec<&QualityMeasurement> = self
            .quality_history
            .iter()
            .rev()
            .take(recent_count)
            .collect();

        // Calculate averages
        let avg_snr = recent_measurements
            .iter()
            .map(|m| m.metrics.snr)
            .sum::<f32>()
            / recent_count as f32;
        let avg_mse = recent_measurements
            .iter()
            .map(|m| m.metrics.mse)
            .sum::<f32>()
            / recent_count as f32;
        let avg_psnr = recent_measurements
            .iter()
            .map(|m| m.metrics.psnr)
            .sum::<f32>()
            / recent_count as f32;
        let avg_perceptual = recent_measurements
            .iter()
            .map(|m| m.metrics.perceptual_score)
            .sum::<f32>()
            / recent_count as f32;
        let avg_ssim = recent_measurements
            .iter()
            .map(|m| m.metrics.ssim)
            .sum::<f32>()
            / recent_count as f32;

        // Find min/max
        let min_snr = recent_measurements
            .iter()
            .map(|m| m.metrics.snr)
            .fold(f32::INFINITY, f32::min);
        let max_snr = recent_measurements
            .iter()
            .map(|m| m.metrics.snr)
            .fold(f32::NEG_INFINITY, f32::max);

        QualityStatistics {
            sample_count: recent_count,
            avg_snr,
            avg_mse,
            avg_psnr,
            avg_perceptual_score: avg_perceptual,
            avg_ssim,
            min_snr,
            max_snr,
            degradation_detected: self.detect_degradation(),
        }
    }

    /// Get full quality history
    pub fn get_quality_history(&self) -> &VecDeque<QualityMeasurement> {
        &self.quality_history
    }

    /// Clear quality history
    pub fn clear_history(&mut self) {
        self.quality_history.clear();
        self.degradation_detector.recent_measurements.clear();
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality statistics over recent measurements
#[derive(Debug, Clone)]
pub struct QualityStatistics {
    pub sample_count: usize,
    pub avg_snr: f32,
    pub avg_mse: f32,
    pub avg_psnr: f32,
    pub avg_perceptual_score: f32,
    pub avg_ssim: f32,
    pub min_snr: f32,
    pub max_snr: f32,
    pub degradation_detected: bool,
}

impl Default for QualityStatistics {
    fn default() -> Self {
        Self {
            sample_count: 0,
            avg_snr: 0.0,
            avg_mse: 0.0,
            avg_psnr: 0.0,
            avg_perceptual_score: 1.0,
            avg_ssim: 1.0,
            min_snr: 0.0,
            max_snr: 0.0,
            degradation_detected: false,
        }
    }
}
