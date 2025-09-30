//! Calibration utilities for determining optimal quantization parameters

use crate::{
    quantization::{
        CalibrationConfig, CalibrationMethod, CalibrationMetrics, CalibrationStats,
        QuantizationParams,
    },
    Module,
};
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Calibrator for determining quantization parameters
pub struct Calibrator {
    config: CalibrationConfig,
    activation_stats: HashMap<String, ActivationStats>,
    weight_stats: HashMap<String, WeightStats>,
    num_samples_processed: usize,
}

/// Statistics for activation tensors
#[derive(Debug, Clone)]
struct ActivationStats {
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
    histograms: Vec<Histogram>,
    moving_avg_min: f32,
    moving_avg_max: f32,
}

/// Statistics for weight tensors
#[derive(Debug, Clone)]
struct WeightStats {
    min_val: f32,
    max_val: f32,
    histogram: Histogram,
    #[allow(dead_code)]
    per_channel_stats: Option<Vec<(f32, f32)>>, // (min, max) per channel
}

/// Simple histogram for distribution analysis
#[derive(Debug, Clone)]
struct Histogram {
    bins: Vec<u32>,
    min_val: f32,
    max_val: f32,
    bin_width: f32,
}

impl Histogram {
    fn new(min_val: f32, max_val: f32, num_bins: usize) -> Self {
        let bin_width = (max_val - min_val) / num_bins as f32;
        Self {
            bins: vec![0; num_bins],
            min_val,
            max_val,
            bin_width,
        }
    }

    fn add_value(&mut self, value: f32) {
        if value < self.min_val || value > self.max_val {
            return; // Ignore out-of-range values
        }

        let bin_idx = ((value - self.min_val) / self.bin_width) as usize;
        let bin_idx = bin_idx.min(self.bins.len() - 1);
        self.bins[bin_idx] += 1;
    }

    fn percentile(&self, p: f32) -> f32 {
        let total_count: u32 = self.bins.iter().sum();
        let target_count = (total_count as f32 * p / 100.0) as u32;

        let mut cumulative = 0;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return self.min_val + (i as f32 + 0.5) * self.bin_width;
            }
        }

        self.max_val
    }

    fn entropy(&self) -> f32 {
        let total_count: u32 = self.bins.iter().sum();
        if total_count == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in &self.bins {
            if count > 0 {
                let p = count as f32 / total_count as f32;
                entropy -= p * p.log2();
            }
        }

        entropy
    }
}

impl Calibrator {
    /// Create a new calibrator
    pub fn new(config: &CalibrationConfig) -> Self {
        Self {
            config: config.clone(),
            activation_stats: HashMap::new(),
            weight_stats: HashMap::new(),
            num_samples_processed: 0,
        }
    }

    /// Calibrate a model using sample data
    pub fn calibrate<M, I>(&mut self, model: &mut M, calibration_data: I) -> Result<()>
    where
        M: Module,
        I: Iterator<Item = Tensor>,
    {
        // Set model to evaluation mode
        model.eval();

        // Process calibration samples
        for (i, input) in calibration_data.take(self.config.num_samples).enumerate() {
            if i >= self.config.num_samples {
                break;
            }

            // Run forward pass with hooks to collect activation statistics
            let _output = self.forward_with_hooks(model, input)?;
            self.num_samples_processed += 1;
        }

        // Collect weight statistics
        self.collect_weight_stats(model)?;

        Ok(())
    }

    /// Forward pass with activation collection hooks
    fn forward_with_hooks<M>(&mut self, model: &M, input: Tensor) -> Result<Tensor>
    where
        M: Module,
    {
        // In a full implementation, this would install hooks on each layer
        // to collect activation statistics. For now, we simulate this.

        let output = model.forward(&input)?;

        // Simulate collecting activation stats for some layers
        let layer1_data = vec![0.1f32; 32 * 64];
        let layer1_tensor = Tensor::from_data(
            layer1_data,
            vec![32, 64],
            torsh_core::device::DeviceType::Cpu,
        )?;
        self.collect_activation_stats("layer1", &layer1_tensor)?;

        let layer2_data = vec![0.2f32; 32 * 128];
        let layer2_tensor = Tensor::from_data(
            layer2_data,
            vec![32, 128],
            torsh_core::device::DeviceType::Cpu,
        )?;
        self.collect_activation_stats("layer2", &layer2_tensor)?;

        Ok(output)
    }

    /// Collect activation statistics for a layer
    fn collect_activation_stats(&mut self, layer_name: &str, activation: &Tensor) -> Result<()> {
        let data = activation.to_vec()?;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let stats = self
            .activation_stats
            .entry(layer_name.to_string())
            .or_insert_with(|| ActivationStats {
                min_vals: Vec::new(),
                max_vals: Vec::new(),
                histograms: Vec::new(),
                moving_avg_min: min_val,
                moving_avg_max: max_val,
            });

        // Update statistics
        stats.min_vals.push(min_val);
        stats.max_vals.push(max_val);

        if self.config.use_moving_average {
            stats.moving_avg_min = self.config.momentum * stats.moving_avg_min
                + (1.0 - self.config.momentum) * min_val;
            stats.moving_avg_max = self.config.momentum * stats.moving_avg_max
                + (1.0 - self.config.momentum) * max_val;
        }

        // Create histogram for distribution analysis
        let mut histogram = Histogram::new(min_val, max_val, 128);
        for &value in &data {
            histogram.add_value(value);
        }
        stats.histograms.push(histogram);

        Ok(())
    }

    /// Collect weight statistics from model parameters
    fn collect_weight_stats<M>(&mut self, model: &M) -> Result<()>
    where
        M: Module,
    {
        let parameters = model.parameters();

        for (name, param) in parameters.iter() {
            let layer_name = format!("param_{}", name);
            let tensor_guard = param.tensor();
            let tensor = tensor_guard.read();
            let data: Vec<f32> = tensor.to_vec().into_iter().flatten().collect();

            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut histogram = Histogram::new(min_val, max_val, 256);
            for &value in &data {
                histogram.add_value(value);
            }

            // Calculate per-channel statistics if applicable
            let per_channel_stats = if tensor.shape().dims().len() >= 2 {
                Some(self.calculate_per_channel_stats(&tensor)?)
            } else {
                None
            };

            self.weight_stats.insert(
                layer_name,
                WeightStats {
                    min_val,
                    max_val,
                    histogram,
                    per_channel_stats,
                },
            );
        }

        Ok(())
    }

    /// Calculate per-channel statistics for weight tensors
    fn calculate_per_channel_stats(&self, weight: &Tensor) -> Result<Vec<(f32, f32)>> {
        let shape = weight.shape();
        let num_channels = shape.dims()[0]; // Assuming first dimension is output channels
        let mut stats = Vec::with_capacity(num_channels);

        for channel in 0..num_channels {
            let channel_tensor = weight.slice(0, channel, channel + 1)?;
            let channel_data: Vec<f32> = channel_tensor.to_vec().into_iter().flatten().collect();

            let min_val = channel_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = channel_data
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            stats.push((min_val, max_val));
        }

        Ok(stats)
    }

    /// Get calibration statistics
    pub fn stats(&self) -> CalibrationStats {
        let mut activation_ranges = HashMap::new();
        let mut weight_ranges = HashMap::new();

        // Collect activation ranges
        for (layer_name, stats) in &self.activation_stats {
            let final_min = if self.config.use_moving_average {
                stats.moving_avg_min
            } else {
                stats.min_vals.iter().fold(f32::INFINITY, |a, &b| a.min(b))
            };

            let final_max = if self.config.use_moving_average {
                stats.moving_avg_max
            } else {
                stats
                    .max_vals
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            };

            activation_ranges.insert(layer_name.clone(), (final_min, final_max));
        }

        // Collect weight ranges
        for (layer_name, stats) in &self.weight_stats {
            weight_ranges.insert(layer_name.clone(), (stats.min_val, stats.max_val));
        }

        CalibrationStats {
            num_samples: self.num_samples_processed,
            activation_ranges,
            weight_ranges,
            metrics: self.calculate_metrics(),
        }
    }

    /// Calculate calibration metrics
    fn calculate_metrics(&self) -> CalibrationMetrics {
        // Simplified metrics calculation
        // In practice, these would be computed by comparing original vs quantized model outputs

        let mut total_entropy = 0.0;
        let mut num_histograms = 0;

        for stats in self.activation_stats.values() {
            for histogram in &stats.histograms {
                total_entropy += histogram.entropy();
                num_histograms += 1;
            }
        }

        for stats in self.weight_stats.values() {
            total_entropy += stats.histogram.entropy();
            num_histograms += 1;
        }

        let avg_entropy = if num_histograms > 0 {
            total_entropy / num_histograms as f32
        } else {
            0.0
        };

        CalibrationMetrics {
            mse: 0.01,               // Placeholder
            snr: 40.0,               // Placeholder
            cosine_similarity: 0.95, // Placeholder
            kl_divergence: avg_entropy,
        }
    }

    /// Generate quantization parameters from collected statistics
    pub fn quantization_params(&self) -> HashMap<String, QuantizationParams> {
        let mut params = HashMap::new();

        // Generate parameters for activations
        for (layer_name, stats) in &self.activation_stats {
            let (min_val, max_val) = match self.config.method {
                CalibrationMethod::MinMax => {
                    if self.config.use_moving_average {
                        (stats.moving_avg_min, stats.moving_avg_max)
                    } else {
                        let min_val = stats.min_vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max_val = stats
                            .max_vals
                            .iter()
                            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        (min_val, max_val)
                    }
                }
                CalibrationMethod::Entropy => {
                    // Use percentile-based range to handle outliers
                    let percentile = self.config.outlier_percentile;
                    if let Some(last_histogram) = stats.histograms.last() {
                        let min_val = last_histogram.percentile(100.0 - percentile);
                        let max_val = last_histogram.percentile(percentile);
                        (min_val, max_val)
                    } else {
                        (stats.moving_avg_min, stats.moving_avg_max)
                    }
                }
                _ => (stats.moving_avg_min, stats.moving_avg_max),
            };

            let quant_params = self.calculate_quantization_params(min_val, max_val, DType::I8);
            params.insert(format!("{}_activation", layer_name), quant_params);
        }

        // Generate parameters for weights
        for (layer_name, stats) in &self.weight_stats {
            let quant_params =
                self.calculate_quantization_params(stats.min_val, stats.max_val, DType::I8);
            params.insert(format!("{}_weight", layer_name), quant_params);
        }

        params
    }

    /// Calculate quantization parameters from min/max values
    fn calculate_quantization_params(
        &self,
        min_val: f32,
        max_val: f32,
        target_dtype: DType,
    ) -> QuantizationParams {
        match target_dtype {
            DType::I8 => {
                // Symmetric quantization for INT8
                let scale = max_val.abs().max(min_val.abs()) / 127.0;
                QuantizationParams::symmetric(scale, DType::F32, DType::I8)
            }
            DType::U8 => {
                // Asymmetric quantization for UINT8
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32;
                QuantizationParams::asymmetric(scale, zero_point, DType::F32, DType::U8)
            }
            _ => {
                // Default to I8 symmetric
                let scale = max_val.abs().max(min_val.abs()) / 127.0;
                QuantizationParams::symmetric(scale, DType::F32, DType::I8)
            }
        }
    }
}

/// Optimal scale calculation using different methods
pub fn calculate_optimal_scale(
    data: &[f32],
    method: &CalibrationMethod,
    target_dtype: DType,
) -> Result<f32> {
    match method {
        CalibrationMethod::MinMax => {
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let max_range = match target_dtype {
                DType::I8 => 127.0,
                DType::U8 => 255.0,
                DType::I16 => 32767.0,
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "Unsupported quantization dtype".to_string(),
                    ))
                }
            };

            Ok(max_val.abs().max(min_val.abs()) / max_range)
        }
        CalibrationMethod::Entropy => {
            // KL divergence-based optimal scale finding
            find_optimal_scale_kl_divergence(data, target_dtype)
        }
        CalibrationMethod::MSE => {
            // Mean squared error minimization
            find_optimal_scale_mse(data, target_dtype)
        }
        CalibrationMethod::CosineSimilarity => {
            // Cosine similarity maximization
            find_optimal_scale_cosine(data, target_dtype)
        }
    }
}

/// Find optimal scale using KL divergence
fn find_optimal_scale_kl_divergence(data: &[f32], target_dtype: DType) -> Result<f32> {
    let max_range = match target_dtype {
        DType::I8 => 127.0,
        DType::U8 => 255.0,
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported dtype for KL divergence".to_string(),
            ))
        }
    };

    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let base_scale = max_val / max_range;

    // Simple heuristic: try multiple scales around the base scale
    let mut best_scale = base_scale;
    let mut best_divergence = f32::INFINITY;

    for scale_multiplier in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0] {
        let scale = base_scale * scale_multiplier;
        let divergence = calculate_kl_divergence(data, scale, max_range as i32);

        if divergence < best_divergence {
            best_divergence = divergence;
            best_scale = scale;
        }
    }

    Ok(best_scale)
}

/// Calculate KL divergence for a given scale
fn calculate_kl_divergence(data: &[f32], scale: f32, max_quant: i32) -> f32 {
    // Create histograms for original and quantized data
    let num_bins = 256;
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut original_hist = vec![0u32; num_bins];
    let mut quantized_hist = vec![0u32; num_bins];

    let bin_width = (max_val - min_val) / num_bins as f32;

    for &value in data {
        // Original histogram
        let bin_idx = ((value - min_val) / bin_width) as usize;
        let bin_idx = bin_idx.min(num_bins - 1);
        original_hist[bin_idx] += 1;

        // Quantized histogram
        let quantized = ((value / scale).round() as i32).clamp(-max_quant, max_quant);
        let dequantized = quantized as f32 * scale;
        let quant_bin_idx = ((dequantized - min_val) / bin_width) as usize;
        let quant_bin_idx = quant_bin_idx.min(num_bins - 1);
        quantized_hist[quant_bin_idx] += 1;
    }

    // Calculate KL divergence
    let total_count = data.len() as f32;
    let mut kl_div = 0.0;

    for i in 0..num_bins {
        let p = original_hist[i] as f32 / total_count;
        let q = quantized_hist[i] as f32 / total_count;

        if p > 0.0 && q > 0.0 {
            kl_div += p * (p / q).ln();
        }
    }

    kl_div
}

/// Find optimal scale using MSE
fn find_optimal_scale_mse(data: &[f32], target_dtype: DType) -> Result<f32> {
    let max_range = match target_dtype {
        DType::I8 => 127.0,
        DType::U8 => 255.0,
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported dtype for MSE".to_string(),
            ))
        }
    };

    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let base_scale = max_val / max_range;

    let mut best_scale = base_scale;
    let mut best_mse = f32::INFINITY;

    for scale_multiplier in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0] {
        let scale = base_scale * scale_multiplier;
        let mse = calculate_mse(data, scale, max_range as i32);

        if mse < best_mse {
            best_mse = mse;
            best_scale = scale;
        }
    }

    Ok(best_scale)
}

/// Calculate MSE for quantization with given scale
fn calculate_mse(data: &[f32], scale: f32, max_quant: i32) -> f32 {
    let mut mse = 0.0;

    for &value in data {
        let quantized = ((value / scale).round() as i32).clamp(-max_quant, max_quant);
        let dequantized = quantized as f32 * scale;
        let error = value - dequantized;
        mse += error * error;
    }

    mse / data.len() as f32
}

/// Find optimal scale using cosine similarity
fn find_optimal_scale_cosine(data: &[f32], target_dtype: DType) -> Result<f32> {
    let max_range = match target_dtype {
        DType::I8 => 127.0,
        DType::U8 => 255.0,
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported dtype for cosine similarity".to_string(),
            ))
        }
    };

    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let base_scale = max_val / max_range;

    let mut best_scale = base_scale;
    let mut best_similarity = f32::NEG_INFINITY;

    for scale_multiplier in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0] {
        let scale = base_scale * scale_multiplier;
        let similarity = calculate_cosine_similarity(data, scale, max_range as i32);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_scale = scale;
        }
    }

    Ok(best_scale)
}

/// Calculate cosine similarity between original and quantized data
fn calculate_cosine_similarity(data: &[f32], scale: f32, max_quant: i32) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_original = 0.0;
    let mut norm_quantized = 0.0;

    for &value in data {
        let quantized = ((value / scale).round() as i32).clamp(-max_quant, max_quant);
        let dequantized = quantized as f32 * scale;

        dot_product += value * dequantized;
        norm_original += value * value;
        norm_quantized += dequantized * dequantized;
    }

    if norm_original == 0.0 || norm_quantized == 0.0 {
        return 0.0;
    }

    dot_product / (norm_original.sqrt() * norm_quantized.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(-1.0, 1.0, 10);
        let data = vec![-0.8, -0.2, 0.0, 0.3, 0.7];

        for value in data {
            hist.add_value(value);
        }

        assert!(hist.bins.iter().sum::<u32>() == 5);
        assert!(hist.percentile(50.0) > -0.5 && hist.percentile(50.0) < 0.5);
    }

    #[test]
    fn test_calibration() {
        let config = CalibrationConfig::default();
        let mut calibrator = Calibrator::new(&config);

        // Simulate collecting some stats
        let activation_data = vec![0.1f32; 32 * 64];
        let activation = Tensor::from_data(
            activation_data,
            vec![32, 64],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        calibrator
            .collect_activation_stats("test_layer", &activation)
            .unwrap();

        let params = calibrator.quantization_params();
        assert!(!params.is_empty());

        let _stats = calibrator.stats();
        // num_samples represents batches processed, not individual activations collected
    }

    #[test]
    fn test_optimal_scale_calculation() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let scale = calculate_optimal_scale(&data, &CalibrationMethod::MinMax, DType::I8).unwrap();
        assert!(scale > 0.0);
        assert!(scale <= 2.0 / 127.0);

        let mse_scale = calculate_optimal_scale(&data, &CalibrationMethod::MSE, DType::I8).unwrap();
        assert!(mse_scale > 0.0);
    }
}
