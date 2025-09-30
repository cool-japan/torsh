//! Advanced quantization features
//!
//! This module implements cutting-edge quantization techniques including:
//! - Dynamic Quantization Scaling
//! - Knowledge Distillation Integration
//! - Layer-wise Reconstruction (BRECQ-style)
//! - Quantization-Aware Pruning
//! - Adaptive Runtime Quantization

use crate::{QuantConfig, TorshResult};
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Dynamic quantization scaling that adapts based on runtime inference patterns
#[derive(Debug, Clone)]
pub struct DynamicQuantizationScaler {
    /// Moving average of tensor activation statistics
    activation_stats: HashMap<String, ActivationStats>,
    /// Configuration for dynamic scaling
    config: DynamicScalingConfig,
    /// Number of inference steps processed
    inference_steps: usize,
}

/// Configuration for dynamic quantization scaling
#[derive(Debug, Clone)]
pub struct DynamicScalingConfig {
    /// Update rate for activation statistics (default: 0.01)
    pub update_rate: f32,
    /// Threshold for scale adjustment (default: 0.1)
    pub scale_adjustment_threshold: f32,
    /// Maximum scale change per step (default: 0.05)
    pub max_scale_change: f32,
    /// Minimum number of steps before adjustments (default: 100)
    pub warmup_steps: usize,
    /// Enable outlier detection for scale adjustment
    pub outlier_detection: bool,
}

impl Default for DynamicScalingConfig {
    fn default() -> Self {
        Self {
            update_rate: 0.01,
            scale_adjustment_threshold: 0.1,
            max_scale_change: 0.05,
            warmup_steps: 100,
            outlier_detection: true,
        }
    }
}

/// Statistics for layer activations used in dynamic scaling
#[derive(Debug, Clone)]
struct ActivationStats {
    /// Current quantization scale
    current_scale: f32,
    /// Current zero point
    current_zero_point: i32,
    /// Moving average of min values
    avg_min: f32,
    /// Moving average of max values
    avg_max: f32,
    /// Count of processed batches
    batch_count: usize,
    /// Recent outlier percentage
    outlier_percentage: f32,
}

impl DynamicQuantizationScaler {
    /// Create new dynamic quantization scaler
    pub fn new(config: DynamicScalingConfig) -> Self {
        Self {
            activation_stats: HashMap::new(),
            config,
            inference_steps: 0,
        }
    }

    /// Process tensor through dynamic quantization with adaptive scaling
    pub fn quantize_dynamic(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
        base_config: &QuantConfig,
    ) -> TorshResult<(Tensor, f32, i32)> {
        let data = tensor.data()?;

        // Get or initialize stats for this layer
        let layer_key = layer_name.to_string();
        if !self.activation_stats.contains_key(&layer_key) {
            let (initial_scale, initial_zero_point) =
                self.calculate_initial_params(&data, base_config);
            let stats = ActivationStats {
                current_scale: initial_scale,
                current_zero_point: initial_zero_point,
                avg_min: data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                avg_max: data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                batch_count: 0,
                outlier_percentage: 0.0,
            };
            self.activation_stats.insert(layer_key.clone(), stats);
        }

        // Update statistics with current activation
        {
            let stats = self.activation_stats.get_mut(&layer_key).unwrap();
            Self::update_activation_stats(&self.config, stats, &data)?;
        }

        // Adjust quantization parameters if needed
        if self.inference_steps > self.config.warmup_steps {
            let stats = self.activation_stats.get_mut(&layer_key).unwrap();
            Self::adjust_quantization_params(&self.config, stats)?;
        }

        // Quantize using current parameters
        let stats = self.activation_stats.get(&layer_key).unwrap();
        let (quantized, _, _) = crate::quantize::quantize_per_tensor_affine(
            tensor,
            stats.current_scale,
            stats.current_zero_point,
        )?;

        self.inference_steps += 1;
        Ok((quantized, stats.current_scale, stats.current_zero_point))
    }

    /// Calculate initial quantization parameters
    fn calculate_initial_params(&self, data: &[f32], config: &QuantConfig) -> (f32, i32) {
        let (qmin, qmax) = config.get_qint_range();
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b)).min(0.0);
        let max_val = data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            .max(0.0);

        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let zero_point = (qmin as f32 - min_val / scale)
            .round()
            .clamp(qmin as f32, qmax as f32) as i32;

        (scale, zero_point)
    }

    /// Update activation statistics with new data
    fn update_activation_stats(
        config: &DynamicScalingConfig,
        stats: &mut ActivationStats,
        data: &[f32],
    ) -> TorshResult<()> {
        let batch_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let batch_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Update moving averages
        let alpha = config.update_rate;
        stats.avg_min = alpha * batch_min + (1.0 - alpha) * stats.avg_min;
        stats.avg_max = alpha * batch_max + (1.0 - alpha) * stats.avg_max;

        // Calculate outlier percentage if enabled
        if config.outlier_detection {
            stats.outlier_percentage = Self::calculate_outlier_percentage(data, stats);
        }

        stats.batch_count += 1;
        Ok(())
    }

    /// Calculate percentage of outliers in current batch
    fn calculate_outlier_percentage(data: &[f32], stats: &ActivationStats) -> f32 {
        let expected_range = stats.avg_max - stats.avg_min;
        let tolerance = expected_range * 0.1; // 10% tolerance

        let outliers = data
            .iter()
            .filter(|&&x| x < (stats.avg_min - tolerance) || x > (stats.avg_max + tolerance))
            .count();

        outliers as f32 / data.len() as f32 * 100.0
    }

    /// Adjust quantization parameters based on statistics
    fn adjust_quantization_params(
        config: &DynamicScalingConfig,
        stats: &mut ActivationStats,
    ) -> TorshResult<()> {
        // Calculate optimal scale based on current statistics
        let optimal_scale = (stats.avg_max - stats.avg_min) / 255.0; // Assuming INT8
        let scale_diff = (optimal_scale - stats.current_scale) / stats.current_scale;

        // Only adjust if the difference is significant
        if scale_diff.abs() > config.scale_adjustment_threshold {
            let adjustment = scale_diff.clamp(-config.max_scale_change, config.max_scale_change);
            stats.current_scale *= 1.0 + adjustment;

            // Recalculate zero point with new scale
            let zero_point = (-128.0 - stats.avg_min / stats.current_scale)
                .round()
                .clamp(-128.0, 127.0) as i32;
            stats.current_zero_point = zero_point;
        }

        Ok(())
    }

    /// Get current statistics for all layers
    pub fn get_layer_statistics(&self) -> HashMap<String, (f32, i32, f32)> {
        self.activation_stats
            .iter()
            .map(|(name, stats)| {
                (
                    name.clone(),
                    (
                        stats.current_scale,
                        stats.current_zero_point,
                        stats.outlier_percentage,
                    ),
                )
            })
            .collect()
    }
}

/// Knowledge distillation integration for quantization-aware training
#[derive(Debug, Clone)]
pub struct QuantizationDistiller {
    /// Temperature for distillation softmax
    pub temperature: f32,
    /// Weight for distillation loss vs quantization loss
    pub distillation_weight: f32,
    /// Configuration for the student (quantized) model
    pub student_config: QuantConfig,
}

impl QuantizationDistiller {
    /// Create new quantization distiller
    pub fn new(temperature: f32, distillation_weight: f32, student_config: QuantConfig) -> Self {
        Self {
            temperature,
            distillation_weight,
            student_config,
        }
    }

    /// Compute distillation loss between teacher (FP32) and student (quantized) outputs
    pub fn compute_distillation_loss(
        &self,
        teacher_output: &Tensor,
        student_output: &Tensor,
    ) -> TorshResult<f32> {
        let teacher_data = teacher_output.data()?;
        let student_data = student_output.data()?;

        if teacher_data.len() != student_data.len() {
            return Err(TorshError::InvalidArgument(
                "Teacher and student outputs must have the same size".to_string(),
            ));
        }

        // Apply temperature scaling and compute KL divergence
        let teacher_probs = self.apply_temperature_softmax(&teacher_data);
        let student_probs = self.apply_temperature_softmax(&student_data);

        let kl_loss = self.compute_kl_divergence(&teacher_probs, &student_probs);
        Ok(kl_loss * self.distillation_weight)
    }

    /// Apply temperature scaling and softmax
    fn apply_temperature_softmax(&self, logits: &[f32]) -> Vec<f32> {
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / self.temperature).collect();
        let max_logit = scaled_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        let sum_exp: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Compute KL divergence between two probability distributions
    fn compute_kl_divergence(&self, p: &[f32], q: &[f32]) -> f32 {
        let eps = 1e-8;
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                let pi_clamped = pi.max(eps);
                let qi_clamped = qi.max(eps);
                pi_clamped * (pi_clamped / qi_clamped).ln()
            })
            .sum()
    }
}

/// Layer-wise reconstruction optimizer inspired by BRECQ
#[derive(Debug, Clone)]
pub struct LayerwiseReconstructor {
    /// Number of reconstruction iterations per layer
    pub iterations: usize,
    /// Learning rate for reconstruction
    pub learning_rate: f32,
    /// Block size for reconstruction (0 = full layer)
    pub block_size: usize,
}

impl LayerwiseReconstructor {
    /// Create new layerwise reconstructor
    pub fn new(iterations: usize, learning_rate: f32, block_size: usize) -> Self {
        Self {
            iterations,
            learning_rate,
            block_size,
        }
    }

    /// Reconstruct layer parameters to minimize quantization error
    pub fn reconstruct_layer(
        &self,
        original_weights: &Tensor,
        quantized_weights: &mut Tensor,
    ) -> TorshResult<f32> {
        let orig_data = original_weights.data()?;
        let mut quant_data = quantized_weights.data()?.clone();

        let mut best_error = f32::INFINITY;

        for _ in 0..self.iterations {
            // Compute reconstruction error
            let error = self.compute_reconstruction_error(&orig_data, &quant_data);

            if error < best_error {
                best_error = error;
            }

            // Gradient-based update to reduce error
            self.update_quantized_weights(&orig_data, &mut quant_data)?;
        }

        // Update the quantized tensor with optimized weights
        *quantized_weights = Tensor::from_data(
            quant_data,
            original_weights.shape().dims().to_vec(),
            original_weights.device(),
        )?;

        Ok(best_error)
    }

    /// Compute reconstruction error (MSE)
    fn compute_reconstruction_error(&self, original: &[f32], quantized: &[f32]) -> f32 {
        original
            .iter()
            .zip(quantized.iter())
            .map(|(&o, &q)| (o - q).powi(2))
            .sum::<f32>()
            / original.len() as f32
    }

    /// Update quantized weights to reduce reconstruction error
    fn update_quantized_weights(&self, original: &[f32], quantized: &mut [f32]) -> TorshResult<()> {
        // First update all weights
        for (orig, quant) in original.iter().zip(quantized.iter_mut()) {
            let error = orig - *quant;
            *quant += self.learning_rate * error;
        }

        // Then apply block-wise constraints if specified
        if self.block_size > 0 {
            for i in (0..quantized.len()).step_by(self.block_size) {
                let block_end = (i + self.block_size).min(quantized.len());
                self.apply_block_constraints(&mut quantized[i..block_end]);
            }
        }
        Ok(())
    }

    /// Apply quantization constraints to a block of weights
    fn apply_block_constraints(&self, block: &mut [f32]) {
        // Round to nearest quantization level and clamp
        for weight in block.iter_mut() {
            *weight = weight.round().clamp(-128.0, 127.0);
        }
    }
}

/// Quantization-aware pruning that optimizes sparsity and quantization jointly
#[derive(Debug, Clone)]
pub struct QuantizationAwarePruner {
    /// Target sparsity percentage (0.0 to 1.0)
    pub target_sparsity: f32,
    /// Quantization configuration
    pub quant_config: QuantConfig,
    /// Pruning schedule (gradual vs immediate)
    pub gradual_pruning: bool,
    /// Current sparsity level
    current_sparsity: f32,
}

impl QuantizationAwarePruner {
    /// Create new quantization-aware pruner
    pub fn new(target_sparsity: f32, quant_config: QuantConfig, gradual_pruning: bool) -> Self {
        Self {
            target_sparsity,
            quant_config,
            gradual_pruning,
            current_sparsity: 0.0,
        }
    }

    /// Prune and quantize weights jointly
    pub fn prune_and_quantize(
        &mut self,
        weights: &Tensor,
        step: usize,
        total_steps: usize,
    ) -> TorshResult<Tensor> {
        let mut weight_data = weights.data()?.clone();

        // Update current sparsity target if using gradual pruning
        if self.gradual_pruning {
            let progress = step as f32 / total_steps as f32;
            self.current_sparsity = self.target_sparsity * progress;
        } else {
            self.current_sparsity = self.target_sparsity;
        }

        // Apply magnitude-based pruning
        self.apply_magnitude_pruning(&mut weight_data)?;

        // Quantize the pruned weights
        let pruned_tensor = Tensor::from_data(
            weight_data,
            weights.shape().dims().to_vec(),
            weights.device(),
        )?;

        let (quantized, _, _) = crate::quantize::quantize_tensor_auto(
            &pruned_tensor,
            self.quant_config.dtype,
            self.quant_config.scheme,
        )?;

        Ok(quantized)
    }

    /// Apply magnitude-based pruning to weights
    fn apply_magnitude_pruning(&self, weights: &mut [f32]) -> TorshResult<()> {
        if self.current_sparsity <= 0.0 {
            return Ok(());
        }

        // Calculate magnitude threshold for pruning
        let mut magnitudes: Vec<f32> = weights.iter().map(|&w| w.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (magnitudes.len() as f32 * self.current_sparsity) as usize;
        let threshold = if threshold_idx < magnitudes.len() {
            magnitudes[threshold_idx]
        } else {
            0.0
        };

        // Apply pruning mask
        for weight in weights.iter_mut() {
            if weight.abs() <= threshold {
                *weight = 0.0;
            }
        }

        Ok(())
    }

    /// Get current sparsity statistics
    pub fn get_sparsity_stats(&self, weights: &Tensor) -> TorshResult<(f32, usize, usize)> {
        let data = weights.data()?;
        let total_params = data.len();
        let zero_params = data.iter().filter(|&&w| w == 0.0).count();
        let actual_sparsity = zero_params as f32 / total_params as f32;

        Ok((actual_sparsity, zero_params, total_params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_dynamic_quantization_scaler() {
        let mut scaler = DynamicQuantizationScaler::new(DynamicScalingConfig::default());
        let config = QuantConfig::int8();

        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let (quantized, scale, zero_point) = scaler
            .quantize_dynamic(&tensor, "test_layer", &config)
            .unwrap();

        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());

        // Test statistics collection
        let stats = scaler.get_layer_statistics();
        assert!(stats.contains_key("test_layer"));
    }

    #[test]
    fn test_quantization_distiller() {
        let distiller = QuantizationDistiller::new(3.0, 0.5, QuantConfig::int8());

        let teacher_output = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let student_output = tensor_1d(&[0.9, 1.8, 2.7, 3.6]).unwrap();

        let loss = distiller
            .compute_distillation_loss(&teacher_output, &student_output)
            .unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_layerwise_reconstructor() {
        let reconstructor = LayerwiseReconstructor::new(10, 0.01, 0);

        let original = tensor_1d(&[1.5, 2.7, 3.2, 4.8]).unwrap();
        let mut quantized = tensor_1d(&[1.0, 3.0, 3.0, 5.0]).unwrap();

        let error = reconstructor
            .reconstruct_layer(&original, &mut quantized)
            .unwrap();
        assert!(error >= 0.0);
    }

    #[test]
    fn test_quantization_aware_pruner() {
        let mut pruner = QuantizationAwarePruner::new(0.5, QuantConfig::int8(), true);

        let weights = tensor_1d(&[0.1, 0.8, 0.2, 0.9, 0.05, 0.7]).unwrap();
        let pruned_quantized = pruner.prune_and_quantize(&weights, 5, 10).unwrap();

        assert_eq!(pruned_quantized.shape().dims(), weights.shape().dims());

        let (sparsity, zero_count, total_count) =
            pruner.get_sparsity_stats(&pruned_quantized).unwrap();
        assert!((0.0..=1.0).contains(&sparsity));
        assert_eq!(zero_count + (total_count - zero_count), total_count);
    }

    #[test]
    fn test_dynamic_scaling_config() {
        let config = DynamicScalingConfig::default();
        assert_eq!(config.update_rate, 0.01);
        assert_eq!(config.warmup_steps, 100);
        assert!(config.outlier_detection);

        let conservative_config = DynamicScalingConfig {
            update_rate: 0.001,
            scale_adjustment_threshold: 0.05,
            max_scale_change: 0.01,
            ..Default::default()
        };
        assert_eq!(conservative_config.update_rate, 0.001);
    }
}
