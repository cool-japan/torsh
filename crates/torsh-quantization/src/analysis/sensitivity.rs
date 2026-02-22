//! Sensitivity analysis for quantization

use crate::analysis::config::{LayerSensitivityResult, SensitivityAnalysisResults};
use crate::{QScheme, QuantConfig, TorshResult};
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Quantization sensitivity analyzer
pub struct SensitivityAnalyzer {
    /// Test dataset for evaluation
    #[allow(dead_code)]
    test_data: Vec<(Tensor, Tensor)>, // (input, expected_output) pairs
    /// Tolerance for accuracy comparison
    tolerance: f32,
}

impl SensitivityAnalyzer {
    /// Create a new sensitivity analyzer
    pub fn new(test_data: Vec<(Tensor, Tensor)>) -> Self {
        Self {
            test_data,
            tolerance: 1e-6,
        }
    }

    /// Set tolerance for accuracy comparison
    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance;
    }

    /// Perform sensitivity analysis on a model's layers
    pub fn analyze_layer_sensitivity(
        &self,
        layer_names: &[String],
        evaluation_fn: impl Fn(&str, &QuantConfig) -> TorshResult<f32>,
    ) -> TorshResult<SensitivityAnalysisResults> {
        let mut layer_results = Vec::new();

        // Get baseline accuracy (no quantization)
        let baseline_accuracy = evaluation_fn("", &QuantConfig::default())?;

        for layer_name in layer_names {
            // Test different quantization schemes for this layer
            let mut best_accuracy = 0.0;
            let mut _best_scheme = QScheme::PerTensorAffine;

            let schemes_to_test = vec![
                QScheme::PerTensorAffine,
                QScheme::PerTensorSymmetric,
                QScheme::PerChannelAffine,
                QScheme::Int4PerTensor,
                QScheme::Binary,
            ];

            for &scheme in &schemes_to_test {
                let config = QuantConfig::new().with_scheme(scheme);

                match evaluation_fn(layer_name, &config) {
                    Ok(accuracy) => {
                        if accuracy > best_accuracy {
                            best_accuracy = accuracy;
                            _best_scheme = scheme;
                        }
                    }
                    Err(_) => {
                        // Skip this scheme if it fails
                        continue;
                    }
                }
            }

            let result =
                LayerSensitivityResult::new(layer_name.clone(), baseline_accuracy, best_accuracy);
            layer_results.push(result);
        }

        Ok(SensitivityAnalysisResults::new(layer_results))
    }

    /// Perform heuristic sensitivity analysis based on layer types
    pub fn heuristic_sensitivity_analysis(
        &self,
        layer_names: &[String],
    ) -> TorshResult<SensitivityAnalysisResults> {
        let mut layer_results = Vec::new();

        for layer_name in layer_names {
            let (sensitivity_score, _recommended_scheme) =
                self.estimate_layer_sensitivity(layer_name);

            let baseline_accuracy = 0.95; // Assumed baseline
            let quantized_accuracy = baseline_accuracy - sensitivity_score;

            let result = LayerSensitivityResult::new(
                layer_name.clone(),
                baseline_accuracy,
                quantized_accuracy,
            );
            layer_results.push(result);
        }

        Ok(SensitivityAnalysisResults::new(layer_results))
    }

    /// Estimate layer sensitivity based on layer type and name patterns
    fn estimate_layer_sensitivity(&self, layer_name: &str) -> (f32, QScheme) {
        let layer_name_lower = layer_name.to_lowercase();

        // Different layer types have different sensitivity levels
        if layer_name_lower.contains("embedding") {
            (0.08, QScheme::PerTensorAffine) // High sensitivity
        } else if layer_name_lower.contains("attention") || layer_name_lower.contains("self_attn") {
            (0.06, QScheme::PerChannelAffine) // Medium-high sensitivity
        } else if layer_name_lower.contains("output") || layer_name_lower.contains("classifier") {
            (0.05, QScheme::PerTensorAffine) // Medium sensitivity
        } else if layer_name_lower.contains("layer_norm") || layer_name_lower.contains("batch_norm")
        {
            (0.02, QScheme::Int4PerTensor) // Low sensitivity
        } else if layer_name_lower.contains("conv") && layer_name_lower.contains("1x1") {
            (0.01, QScheme::Int4PerTensor) // Very low sensitivity
        } else if layer_name_lower.contains("conv") {
            (0.03, QScheme::PerChannelAffine) // Low-medium sensitivity
        } else if layer_name_lower.contains("linear") || layer_name_lower.contains("dense") {
            (0.025, QScheme::PerTensorAffine) // Low sensitivity
        } else {
            (0.03, QScheme::PerTensorAffine) // Default medium sensitivity
        }
    }

    /// Compare accuracy between original and quantized tensors
    pub fn compare_tensor_accuracy(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> TorshResult<f32> {
        if original.shape() != quantized.shape() {
            return Err(TorshError::InvalidArgument(
                "Tensors must have the same shape for accuracy comparison".to_string(),
            ));
        }

        let original_data = original.data()?;
        let quantized_data = quantized.data()?;

        let mut correct_predictions = 0;
        let total_predictions = original_data.len();

        for (orig, quant) in original_data.iter().zip(quantized_data.iter()) {
            if (orig - quant).abs() <= self.tolerance {
                correct_predictions += 1;
            }
        }

        Ok(correct_predictions as f32 / total_predictions as f32)
    }

    /// Calculate Mean Squared Error between tensors
    pub fn calculate_mse(&self, original: &Tensor, quantized: &Tensor) -> TorshResult<f32> {
        if original.shape() != quantized.shape() {
            return Err(TorshError::InvalidArgument(
                "Tensors must have the same shape for MSE calculation".to_string(),
            ));
        }

        let original_data = original.data()?;
        let quantized_data = quantized.data()?;

        let mse = original_data
            .iter()
            .zip(quantized_data.iter())
            .map(|(orig, quant)| (orig - quant).powi(2))
            .sum::<f32>()
            / original_data.len() as f32;

        Ok(mse)
    }

    /// Calculate Signal-to-Noise Ratio
    pub fn calculate_snr(&self, original: &Tensor, quantized: &Tensor) -> TorshResult<f32> {
        let mse = self.calculate_mse(original, quantized)?;

        if mse == 0.0 {
            return Ok(f32::INFINITY); // Perfect reconstruction
        }

        let original_data = original.data()?;
        let signal_power =
            original_data.iter().map(|&x| x.powi(2)).sum::<f32>() / original_data.len() as f32;

        let snr_db = 10.0 * (signal_power / mse).log10();
        Ok(snr_db)
    }
}
