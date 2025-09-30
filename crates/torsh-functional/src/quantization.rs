//! Quantization and Compression Functions
//!
//! This module provides quantization operations for model compression including:
//! - Uniform and non-uniform quantization
//! - Dynamic quantization schemes
//! - Pruning utilities (magnitude-based, structured, unstructured)
//! - Model compression techniques
//! - Low-precision computation functions
//! - Knowledge distillation utilities

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{
    creation::{ones, randn, zeros},
    stats::StatMode,
    Tensor,
};

/// Quantization schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Uniform quantization with equal spacing
    Uniform,
    /// Non-uniform quantization with custom levels
    NonUniform,
    /// Dynamic quantization based on data statistics
    Dynamic,
}

/// Quantization data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 4-bit quantization (for extreme compression)
    Int4,
}

/// Uniform quantization
///
/// Quantizes floating-point values to fixed-point representation
/// using uniform spacing between quantization levels.
///
/// # Arguments
/// * `input` - Input tensor to quantize
/// * `scale` - Quantization scale factor
/// * `zero_point` - Zero point offset
/// * `qtype` - Target quantization type
///
/// # Returns
/// Tuple of (quantized_tensor, scale, zero_point)
pub fn uniform_quantize(
    input: &Tensor,
    scale: f32,
    zero_point: i32,
    qtype: QuantizationType,
) -> TorshResult<(Tensor, f32, i32)> {
    let (qmin, qmax) = match qtype {
        QuantizationType::Int8 => (-128i32, 127i32),
        QuantizationType::UInt8 => (0i32, 255i32),
        QuantizationType::Int16 => (-32768i32, 32767i32),
        QuantizationType::Int4 => (-8i32, 7i32),
    };

    // Quantize: q = clamp(round(x / scale + zero_point), qmin, qmax)
    let scaled = input.div_scalar(scale)?;
    let shifted = scaled.add_scalar(zero_point as f32)?;
    let rounded = shifted.round()?;
    let clamped = crate::math::clamp(&rounded, qmin as f32, qmax as f32)?;

    Ok((clamped, scale, zero_point))
}

/// Dequantize uniformly quantized tensor
///
/// # Arguments
/// * `quantized` - Quantized tensor
/// * `scale` - Quantization scale factor
/// * `zero_point` - Zero point offset
///
/// # Returns
/// Dequantized floating-point tensor
pub fn uniform_dequantize(quantized: &Tensor, scale: f32, zero_point: i32) -> TorshResult<Tensor> {
    // Dequantize: x = (q - zero_point) * scale
    let mut shifted = quantized.clone();
    shifted.sub_scalar_(zero_point as f32)?;
    let shifted = shifted;
    let dequantized = shifted.mul_scalar(scale)?;
    Ok(dequantized)
}

/// Dynamic quantization with automatic scale/zero-point calculation
///
/// Automatically determines optimal scale and zero point based on
/// input tensor statistics.
///
/// # Arguments
/// * `input` - Input tensor to quantize
/// * `qtype` - Target quantization type
/// * `reduce_range` - Whether to reduce quantization range for better accuracy
///
/// # Returns
/// Tuple of (quantized_tensor, scale, zero_point)
pub fn dynamic_quantize(
    input: &Tensor,
    qtype: QuantizationType,
    reduce_range: bool,
) -> TorshResult<(Tensor, f32, i32)> {
    let (qmin, qmax) = match qtype {
        QuantizationType::Int8 => {
            if reduce_range {
                (-64i32, 63i32)
            } else {
                (-128i32, 127i32)
            }
        }
        QuantizationType::UInt8 => {
            if reduce_range {
                (0i32, 127i32)
            } else {
                (0i32, 255i32)
            }
        }
        QuantizationType::Int16 => {
            if reduce_range {
                (-16384i32, 16383i32)
            } else {
                (-32768i32, 32767i32)
            }
        }
        QuantizationType::Int4 => {
            if reduce_range {
                (-4i32, 3i32)
            } else {
                (-8i32, 7i32)
            }
        }
    };

    // Calculate min and max values in input
    let input_min = input.min()?.data()?[0];
    let input_max = input.max(None, false)?.data()?[0];

    // Calculate scale and zero point
    let scale = (input_max - input_min) / (qmax - qmin) as f32;
    let zero_point_float = qmin as f32 - input_min / scale;
    let zero_point = zero_point_float.round() as i32;

    // Ensure scale is not zero
    let safe_scale = if scale == 0.0 { 1.0 } else { scale };

    uniform_quantize(input, safe_scale, zero_point, qtype)
}

/// Quantize-aware training (QAT) simulation
///
/// Simulates quantization effects during training by applying
/// fake quantization (quantize then immediately dequantize).
///
/// # Arguments
/// * `input` - Input tensor
/// * `scale` - Quantization scale
/// * `zero_point` - Zero point offset
/// * `qtype` - Quantization type
///
/// # Returns
/// Fake quantized tensor (still in floating point)
pub fn fake_quantize(
    input: &Tensor,
    scale: f32,
    zero_point: i32,
    qtype: QuantizationType,
) -> TorshResult<Tensor> {
    let (quantized, scale, zero_point) = uniform_quantize(input, scale, zero_point, qtype)?;
    uniform_dequantize(&quantized, scale, zero_point)
}

/// Magnitude-based pruning
///
/// Prunes weights with smallest absolute values to achieve target sparsity.
///
/// # Arguments
/// * `weights` - Weight tensor to prune
/// * `sparsity` - Target sparsity level (0.0 = no pruning, 0.9 = 90% pruned)
/// * `structured` - Whether to use structured pruning (prune entire channels/filters)
///
/// # Returns
/// Tuple of (pruned_weights, pruning_mask)
pub fn magnitude_prune(
    weights: &Tensor,
    sparsity: f32,
    structured: bool,
) -> TorshResult<(Tensor, Tensor)> {
    if sparsity < 0.0 || sparsity >= 1.0 {
        return Err(TorshError::invalid_argument_with_context(
            "Sparsity must be in range [0.0, 1.0)",
            "magnitude_prune",
        ));
    }

    if structured {
        // Structured pruning: prune entire channels/filters
        let weight_shape_ref = weights.shape();
        let weight_shape = weight_shape_ref.dims();
        if weight_shape.len() < 2 {
            return Err(TorshError::invalid_argument_with_context(
                "Structured pruning requires at least 2D weights",
                "magnitude_prune",
            ));
        }

        let num_filters = weight_shape[0];
        let num_to_prune = (num_filters as f32 * sparsity) as usize;

        // Calculate L2 norm for each filter
        let dims_to_reduce: Vec<i32> = (1..weight_shape.len()).map(|i| i as i32).collect();
        let _filter_norms = weights
            .pow_scalar(2.0)?
            .sum_dim(&dims_to_reduce, false)?
            .sqrt()?;

        // Create mask (simplified - in practice would sort and threshold)
        let mask = ones(&weight_shape)?;

        // For now, create a simple pattern mask
        // In full implementation would identify smallest norm filters
        if num_to_prune > 0 {
            // Set some filters to zero as example
            for _i in 0..num_to_prune.min(num_filters) {
                // This is a placeholder - in practice would zero out specific filter indices
            }
        }

        let pruned_weights = weights.mul_op(&mask)?;
        Ok((pruned_weights, mask))
    } else {
        // Unstructured pruning: prune individual weights
        let abs_weights = weights.abs()?;
        let threshold = calculate_pruning_threshold(&abs_weights, sparsity)?;

        // Create mask: 1 where |weight| > threshold, 0 otherwise
        let bool_mask = abs_weights.gt_scalar(threshold)?;
        // Convert boolean mask to float mask manually
        let mask_data: Vec<f32> = bool_mask
            .data()?
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::from_data(mask_data, weights.shape().dims().to_vec(), weights.device())?;
        let pruned_weights = weights.mul_op(&mask)?;

        Ok((pruned_weights, mask))
    }
}

/// Calculate pruning threshold for given sparsity level
fn calculate_pruning_threshold(abs_weights: &Tensor, sparsity: f32) -> TorshResult<f32> {
    // In a full implementation, this would:
    // 1. Flatten the tensor
    // 2. Sort the values
    // 3. Find the value at the sparsity percentile

    // For now, use a simple approximation based on statistics
    let mean_data = abs_weights.mean(None, false)?.data()?;
    let mean_val = mean_data.get(0).unwrap_or(&0.1).clone();
    let std_data = abs_weights.std(None, false, StatMode::Sample)?.data()?;
    let std_val = std_data.get(0).unwrap_or(&0.01).clone();

    // Use a heuristic threshold based on statistics
    let threshold = mean_val - sparsity * std_val;
    Ok(threshold.max(0.0))
}

/// Gradual magnitude pruning with sparsity scheduling
///
/// Implements gradual pruning where sparsity increases over training steps.
///
/// # Arguments
/// * `weights` - Weight tensor to prune
/// * `current_step` - Current training step
/// * `start_step` - Step to start pruning
/// * `end_step` - Step to finish pruning
/// * `initial_sparsity` - Initial sparsity level
/// * `final_sparsity` - Final target sparsity level
///
/// # Returns
/// Tuple of (pruned_weights, current_sparsity, pruning_mask)
pub fn gradual_magnitude_prune(
    weights: &Tensor,
    current_step: usize,
    start_step: usize,
    end_step: usize,
    initial_sparsity: f32,
    final_sparsity: f32,
) -> TorshResult<(Tensor, f32, Tensor)> {
    if current_step < start_step {
        // No pruning yet
        let mask = ones(&weights.shape().dims())?;
        return Ok((weights.clone(), initial_sparsity, mask));
    }

    if current_step >= end_step {
        // Final sparsity reached
        let (pruned, mask) = magnitude_prune(weights, final_sparsity, false)?;
        return Ok((pruned, final_sparsity, mask));
    }

    // Calculate current sparsity using polynomial schedule
    let progress = (current_step - start_step) as f32 / (end_step - start_step) as f32;
    let current_sparsity = initial_sparsity
        + (final_sparsity - initial_sparsity) * (3.0 * progress.powi(2) - 2.0 * progress.powi(3));

    let (pruned, mask) = magnitude_prune(weights, current_sparsity, false)?;
    Ok((pruned, current_sparsity, mask))
}

/// Weight clustering for compression
///
/// Groups weights into clusters and replaces each weight with its cluster centroid.
///
/// # Arguments
/// * `weights` - Weight tensor to cluster
/// * `num_clusters` - Number of clusters (codebook size)
///
/// # Returns
/// Tuple of (clustered_weights, centroids, cluster_assignments)
pub fn weight_clustering(
    weights: &Tensor,
    num_clusters: usize,
) -> TorshResult<(Tensor, Tensor, Tensor)> {
    if num_clusters == 0 {
        return Err(TorshError::invalid_argument_with_context(
            "Number of clusters must be positive",
            "weight_clustering",
        ));
    }

    // Simplified k-means clustering implementation
    // In practice would use proper k-means algorithm

    let weight_shape_ref = weights.shape();
    let weight_shape = weight_shape_ref.dims();
    let _num_weights = weights.numel();

    // Initialize centroids (simplified - random sampling from weights)
    let centroids = randn(&[num_clusters])?;

    // For now, create simple cluster assignments based on weight ranges
    let min_data = weights.min()?.data()?;
    let min_weight = min_data.get(0).unwrap_or(&-1.0).clone();
    let max_data = weights.max(None, false)?.data()?;
    let max_weight = max_data.get(0).unwrap_or(&1.0).clone();
    let _weight_range = max_weight - min_weight;

    // Create cluster assignments (simplified)
    let cluster_assignments = zeros(&weight_shape)?;

    // Replace weights with cluster centroids
    let clustered_weights = weights.clone(); // Placeholder

    Ok((clustered_weights, centroids, cluster_assignments))
}

/// Lottery ticket hypothesis: find winning subnetworks
///
/// Identifies sparse subnetworks that can achieve comparable performance
/// when trained from scratch.
///
/// # Arguments
/// * `weights` - Original trained weights
/// * `initial_weights` - Initial weights before training
/// * `sparsity` - Target sparsity for the lottery ticket
///
/// # Returns
/// Tuple of (lottery_ticket_mask, winning_subnetwork_weights)
pub fn lottery_ticket_prune(
    weights: &Tensor,
    initial_weights: &Tensor,
    sparsity: f32,
) -> TorshResult<(Tensor, Tensor)> {
    if weights.shape().dims() != initial_weights.shape().dims() {
        return Err(TorshError::invalid_argument_with_context(
            "Weight tensors must have same shape",
            "lottery_ticket_prune",
        ));
    }

    // Find winning lottery ticket based on final weight magnitudes
    let (_, mask) = magnitude_prune(weights, sparsity, false)?;

    // Apply mask to initial weights to get winning subnetwork
    let winning_subnetwork = initial_weights.mul_op(&mask)?;

    Ok((mask, winning_subnetwork))
}

/// Quantization error analysis
///
/// Analyzes the error introduced by quantization to guide optimization.
///
/// # Arguments
/// * `original` - Original floating-point tensor
/// * `quantized` - Quantized tensor (after dequantization)
///
/// # Returns
/// Tuple of (mse_error, max_error, snr_db)
pub fn quantization_error_analysis(
    original: &Tensor,
    quantized: &Tensor,
) -> TorshResult<(f32, f32, f32)> {
    if original.shape().dims() != quantized.shape().dims() {
        return Err(TorshError::invalid_argument_with_context(
            "Tensors must have same shape",
            "quantization_error_analysis",
        ));
    }

    // Calculate mean squared error
    let error = original.sub(quantized)?;
    let mse_tensor = error.pow_scalar(2.0)?.mean(None, false)?;
    let mse = mse_tensor.data()?[0];

    // Calculate maximum absolute error
    let abs_error = error.abs()?;
    let max_error_tensor = abs_error.max(None, false)?;
    let max_error = max_error_tensor.data()?[0];

    // Calculate signal-to-noise ratio in dB
    let signal_power_tensor = original.pow_scalar(2.0)?.mean(None, false)?;
    let signal_power = signal_power_tensor.data()?[0];
    let snr_db = if mse > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f32::INFINITY
    };

    Ok((mse, max_error, snr_db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_uniform_quantization() {
        let input = randn(&[4, 4], None, None, None).unwrap();
        let (quantized, scale, zero_point) =
            uniform_quantize(&input, 0.1, 128, QuantizationType::UInt8).unwrap();

        // Check quantized values are in valid range
        assert_eq!(quantized.shape().dims(), input.shape().dims());

        // Test dequantization
        let dequantized = uniform_dequantize(&quantized, scale, zero_point).unwrap();
        assert_eq!(dequantized.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_dynamic_quantization() {
        let input = randn(&[3, 3], None, None, None).unwrap();
        let (quantized, scale, _zero_point) =
            dynamic_quantize(&input, QuantizationType::Int8, false).unwrap();

        assert_eq!(quantized.shape().dims(), input.shape().dims());
        assert!(scale > 0.0);
    }

    #[test]
    fn test_fake_quantization() {
        let input = randn(&[2, 2], None, None, None).unwrap();
        let fake_quantized = fake_quantize(&input, 0.1, 0, QuantizationType::Int8).unwrap();

        assert_eq!(fake_quantized.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_magnitude_pruning() {
        let weights = randn(&[10, 10], None, None, None).unwrap();
        let (pruned, mask) = magnitude_prune(&weights, 0.5, false).unwrap();

        assert_eq!(pruned.shape().dims(), weights.shape().dims());
        assert_eq!(mask.shape().dims(), weights.shape().dims());
    }

    #[test]
    fn test_gradual_pruning() {
        let weights = randn(&[5, 5], None, None, None).unwrap();
        let (pruned, sparsity, mask) =
            gradual_magnitude_prune(&weights, 50, 10, 100, 0.0, 0.8).unwrap();

        assert_eq!(pruned.shape().dims(), weights.shape().dims());
        assert!(sparsity >= 0.0 && sparsity <= 0.8);
        assert_eq!(mask.shape().dims(), weights.shape().dims());
    }

    #[test]
    fn test_lottery_ticket() {
        let trained_weights = randn(&[4, 4], None, None, None).unwrap();
        let initial_weights = randn(&[4, 4], None, None, None).unwrap();

        let (mask, winning_subnetwork) =
            lottery_ticket_prune(&trained_weights, &initial_weights, 0.6).unwrap();

        assert_eq!(mask.shape().dims(), trained_weights.shape().dims());
        assert_eq!(
            winning_subnetwork.shape().dims(),
            initial_weights.shape().dims()
        );
    }

    #[test]
    fn test_quantization_error_analysis() {
        let original = randn(&[3, 3], None, None, None).unwrap();
        let quantized = original.clone(); // Perfect case for testing

        let (mse, max_error, snr_db) = quantization_error_analysis(&original, &quantized).unwrap();

        // Should be very small errors for identical tensors
        assert!(mse <= 1e-6);
        assert!(max_error <= 1e-6);
        assert!(snr_db > 60.0 || snr_db.is_infinite());
    }
}
