//! Quantization benchmarking utilities

use super::metrics::QuantizationMetrics;

/// Quantization benchmarking utilities
pub struct QuantizationBenchmark;

impl QuantizationBenchmark {
    /// Measure quantization accuracy
    pub fn measure_accuracy(
        original_outputs: &[f32],
        quantized_outputs: &[f32],
    ) -> QuantizationMetrics {
        let mut total_error = 0.0f32;
        let mut max_error = 0.0f32;
        let mut snr_sum = 0.0f32;

        for (orig, quant) in original_outputs.iter().zip(quantized_outputs.iter()) {
            let error = (orig - quant).abs();
            total_error += error;
            max_error = max_error.max(error);

            if orig.abs() > 1e-8 {
                let snr = 20.0 * (orig.abs() / error).log10();
                snr_sum += snr;
            }
        }

        let mean_error = total_error / original_outputs.len() as f32;
        let mean_snr = snr_sum / original_outputs.len() as f32;

        QuantizationMetrics {
            mean_absolute_error: mean_error,
            max_absolute_error: max_error,
            signal_to_noise_ratio: mean_snr,
            sample_count: original_outputs.len(),
        }
    }
}
