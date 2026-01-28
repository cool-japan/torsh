//! Visualization tools for quantization analysis

use crate::analysis::config::SensitivityAnalysisResults;
use crate::analysis::size::SizeAnalyzer;
use crate::QScheme;
use crate::TorshResult;
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Visualization tools for quantization analysis
pub struct VisualizationTool;

impl VisualizationTool {
    /// Generate a text-based bar chart for sensitivity scores
    pub fn render_sensitivity_bar_chart(
        results: &SensitivityAnalysisResults,
        width: usize,
    ) -> String {
        let mut chart = String::new();
        chart.push_str("Layer Sensitivity Analysis\n");
        chart.push_str(&"=".repeat(width));
        chart.push('\n');

        // Sort layers by sensitivity for better visualization
        let mut sorted_results = results.layer_results.clone();
        sorted_results.sort_by(|a, b| {
            b.sensitivity_score
                .partial_cmp(&a.sensitivity_score)
                .expect("sensitivity scores should be comparable")
        });

        let max_sensitivity = sorted_results
            .first()
            .map(|r| r.sensitivity_score)
            .unwrap_or(1.0);

        for result in &sorted_results {
            let bar_length =
                ((result.sensitivity_score / max_sensitivity) * (width - 20) as f32) as usize;
            let bar = "█".repeat(bar_length);

            chart.push_str(&format!(
                "{:15} |{:<width$}| {:.3}\n",
                Self::truncate_string(&result.layer_name, 15),
                bar,
                result.sensitivity_score,
                width = width - 20
            ));
        }

        chart.push('\n');
        chart.push_str(&format!(
            "Overall Sensitivity: {:.3}\n",
            results.overall_sensitivity
        ));
        chart
    }

    /// Generate a text-based comparison table for quantization schemes
    pub fn render_quantization_comparison_table(
        num_parameters: usize,
        baseline_accuracy: f32,
        sensitivity_results: &SensitivityAnalysisResults,
    ) -> String {
        let mut table = String::new();
        table.push_str("Quantization Scheme Comparison\n");
        table.push_str(&"=".repeat(80));
        table.push('\n');

        table.push_str(&format!(
            "{:<20} | {:>10} | {:>10} | {:>15} | {:>15}\n",
            "Scheme", "Size (MB)", "Reduction", "Speed Factor", "Avg Accuracy"
        ));
        table.push_str(&"-".repeat(80));
        table.push('\n');

        let schemes = vec![
            ("FP32 (Baseline)", QScheme::MixedPrecision, 1.0),
            ("INT8 PerTensor", QScheme::PerTensorAffine, 1.0),
            ("INT8 PerChannel", QScheme::PerChannelAffine, 1.0),
            ("INT4", QScheme::Int4PerTensor, 1.0),
            ("Binary", QScheme::Binary, 1.0),
            ("Ternary", QScheme::Ternary, 1.0),
            ("Group-wise", QScheme::GroupWise, 1.0),
        ];

        for (name, scheme, accuracy_modifier) in schemes {
            let size_mb = SizeAnalyzer::calculate_model_size(num_parameters, scheme);
            let reduction_factor = SizeAnalyzer::size_reduction_factor(
                QScheme::MixedPrecision,
                scheme,
                num_parameters,
            );
            let speed_factor = Self::estimate_speed_improvement(scheme);
            let avg_accuracy =
                baseline_accuracy * accuracy_modifier - sensitivity_results.overall_sensitivity;

            table.push_str(&format!(
                "{name:<20} | {size_mb:>8.1} | {reduction_factor:>8.1}x | {speed_factor:>13.1}x | {avg_accuracy:>13.3}\n"
            ));
        }

        table
    }

    /// Generate histogram of quantization errors
    pub fn render_error_histogram(
        original: &Tensor,
        quantized: &Tensor,
        bins: usize,
        width: usize,
    ) -> TorshResult<String> {
        if original.shape() != quantized.shape() {
            return Err(TorshError::InvalidArgument(
                "Tensors must have the same shape".to_string(),
            ));
        }

        let original_data = original.data()?;
        let quantized_data = quantized.data()?;

        // Calculate errors
        let errors: Vec<f32> = original_data
            .iter()
            .zip(quantized_data.iter())
            .map(|(orig, quant)| orig - quant)
            .collect();

        let min_error = errors.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_error = errors.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if min_error == max_error {
            return Ok("All errors are identical (perfect quantization)\n".to_string());
        }

        // Create histogram bins
        let mut histogram = vec![0; bins];
        let bin_width = (max_error - min_error) / bins as f32;

        for &error in &errors {
            let bin_index = ((error - min_error) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        // Render histogram
        let mut chart = String::new();
        chart.push_str("Quantization Error Distribution\n");
        chart.push_str(&"=".repeat(width));
        chart.push('\n');

        let max_count = *histogram.iter().max().unwrap_or(&1);

        for (i, &count) in histogram.iter().enumerate() {
            let bin_start = min_error + i as f32 * bin_width;
            let bin_end = bin_start + bin_width;
            let bar_length = (count as f32 / max_count as f32 * (width - 25) as f32) as usize;
            let bar = "█".repeat(bar_length);

            chart.push_str(&format!(
                "[{:7.3}, {:7.3}) |{:<width$}| {:>5}\n",
                bin_start,
                bin_end,
                bar,
                count,
                width = width - 25
            ));
        }

        chart.push_str(&format!(
            "\nMean Error: {:.6}\n",
            errors.iter().sum::<f32>() / errors.len() as f32
        ));
        chart.push_str(&format!(
            "Std Error:  {:.6}\n",
            Self::calculate_std(&errors)
        ));

        Ok(chart)
    }

    /// Export data for external visualization tools
    pub fn export_sensitivity_data(
        results: &SensitivityAnalysisResults,
    ) -> HashMap<String, Vec<f32>> {
        let mut data = HashMap::new();

        let sensitivity_scores: Vec<f32> = results
            .layer_results
            .iter()
            .map(|r| r.sensitivity_score)
            .collect();
        let accuracy_drops: Vec<f32> = results
            .layer_results
            .iter()
            .map(|r| r.accuracy_drop_percentage())
            .collect();

        data.insert("sensitivity_scores".to_string(), sensitivity_scores);
        data.insert("accuracy_drops".to_string(), accuracy_drops);

        data
    }

    // Helper functions
    fn truncate_string(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    fn calculate_std(values: &[f32]) -> f32 {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }

    fn estimate_speed_improvement(scheme: QScheme) -> f32 {
        match scheme {
            QScheme::Binary => 8.0,
            QScheme::Ternary => 6.0,
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 4.0,
            QScheme::PerTensorAffine | QScheme::PerChannelAffine => 2.0,
            QScheme::PerTensorSymmetric | QScheme::PerChannelSymmetric => 2.0,
            QScheme::MixedPrecision => 1.5,
            QScheme::GroupWise => 2.5,
        }
    }
}
