//! Quantization performance metrics

/// Quantization performance metrics
#[derive(Debug)]
pub struct QuantizationMetrics {
    pub mean_absolute_error: f32,
    pub max_absolute_error: f32,
    pub signal_to_noise_ratio: f32,
    pub sample_count: usize,
}
