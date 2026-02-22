//! Advanced model operations and utilities
//!
//! This module provides advanced operations for model management including:
//! - Model comparison and difference analysis
//! - Model merging and ensemble creation
//! - Model quantization helpers
//! - Model pruning utilities
//! - Model conversion helpers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Model difference analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiff {
    /// Parameters that exist in both models
    pub common_parameters: Vec<String>,
    /// Parameters only in first model
    pub only_in_first: Vec<String>,
    /// Parameters only in second model
    pub only_in_second: Vec<String>,
    /// Parameters with different shapes
    pub shape_differences: Vec<ShapeDifference>,
    /// Statistical differences for common parameters
    pub value_differences: Vec<ValueDifference>,
    /// Total parameter count for each model
    pub param_counts: (usize, usize),
    /// Memory footprint for each model (in bytes)
    pub memory_footprints: (u64, u64),
}

/// Shape difference between two parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeDifference {
    pub parameter_name: String,
    pub shape_first: Vec<usize>,
    pub shape_second: Vec<usize>,
}

/// Value difference statistics for a parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueDifference {
    pub parameter_name: String,
    pub mean_absolute_diff: f64,
    pub max_absolute_diff: f64,
    pub relative_diff_percent: f64,
    pub cosine_similarity: f64,
}

/// Model comparison options
#[derive(Debug, Clone)]
pub struct ComparisonOptions {
    /// Whether to compute detailed value differences (can be expensive)
    pub compute_value_diffs: bool,
    /// Threshold for considering values as different (relative)
    pub diff_threshold: f64,
    /// Maximum number of parameters to compare in detail
    pub max_params_to_compare: usize,
}

impl Default for ComparisonOptions {
    fn default() -> Self {
        Self {
            compute_value_diffs: true,
            diff_threshold: 1e-5,
            max_params_to_compare: 1000,
        }
    }
}

/// Compare two models and return their differences
pub fn compare_models(
    model1_state: &HashMap<String, Tensor<f32>>,
    model2_state: &HashMap<String, Tensor<f32>>,
    options: Option<ComparisonOptions>,
) -> Result<ModelDiff> {
    let options = options.unwrap_or_default();

    let keys1: std::collections::HashSet<_> = model1_state.keys().cloned().collect();
    let keys2: std::collections::HashSet<_> = model2_state.keys().cloned().collect();

    let common_parameters: Vec<String> = keys1.intersection(&keys2).cloned().collect();
    let only_in_first: Vec<String> = keys1.difference(&keys2).cloned().collect();
    let only_in_second: Vec<String> = keys2.difference(&keys1).cloned().collect();

    let mut shape_differences = Vec::new();
    let mut value_differences = Vec::new();

    // Check shape differences and compute value differences
    for param_name in &common_parameters {
        let tensor1 = &model1_state[param_name];
        let tensor2 = &model2_state[param_name];

        let shape1 = tensor1.shape().dims().to_vec();
        let shape2 = tensor2.shape().dims().to_vec();

        if shape1 != shape2 {
            shape_differences.push(ShapeDifference {
                parameter_name: param_name.clone(),
                shape_first: shape1,
                shape_second: shape2,
            });
        } else if options.compute_value_diffs
            && value_differences.len() < options.max_params_to_compare
        {
            // Compute value differences only if shapes match
            if let Ok(diff) =
                compute_value_difference(tensor1, tensor2, param_name, options.diff_threshold)
            {
                value_differences.push(diff);
            }
        }
    }

    let param_counts = (model1_state.len(), model2_state.len());
    let memory_footprints = (
        estimate_memory_footprint(model1_state),
        estimate_memory_footprint(model2_state),
    );

    Ok(ModelDiff {
        common_parameters,
        only_in_first,
        only_in_second,
        shape_differences,
        value_differences,
        param_counts,
        memory_footprints,
    })
}

/// Compute value difference statistics between two tensors
fn compute_value_difference(
    tensor1: &Tensor<f32>,
    tensor2: &Tensor<f32>,
    param_name: &str,
    _threshold: f64,
) -> Result<ValueDifference> {
    // Get tensor data
    let data1 = tensor1.to_vec()?;
    let data2 = tensor2.to_vec()?;

    if data1.len() != data2.len() {
        return Err(TorshError::InvalidArgument(
            "Tensors must have same number of elements".to_string(),
        ));
    }

    // Compute statistics
    let mut sum_abs_diff = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut dot_product = 0.0f64;
    let mut norm1_sq = 0.0f64;
    let mut norm2_sq = 0.0f64;

    for (&v1, &v2) in data1.iter().zip(data2.iter()) {
        let v1 = v1 as f64;
        let v2 = v2 as f64;
        let abs_diff = (v1 - v2).abs();

        sum_abs_diff += abs_diff;
        max_abs_diff = max_abs_diff.max(abs_diff);
        dot_product += v1 * v2;
        norm1_sq += v1 * v1;
        norm2_sq += v2 * v2;
    }

    let mean_absolute_diff = sum_abs_diff / data1.len() as f64;

    // Compute relative difference as percentage
    let mean1 = data1.iter().map(|&x| x as f64).sum::<f64>() / data1.len() as f64;
    let relative_diff_percent = if mean1.abs() > 1e-10 {
        (mean_absolute_diff / mean1.abs()) * 100.0
    } else {
        0.0
    };

    // Compute cosine similarity
    let cosine_similarity = if norm1_sq > 0.0 && norm2_sq > 0.0 {
        dot_product / (norm1_sq.sqrt() * norm2_sq.sqrt())
    } else {
        0.0
    };

    Ok(ValueDifference {
        parameter_name: param_name.to_string(),
        mean_absolute_diff,
        max_absolute_diff: max_abs_diff,
        relative_diff_percent,
        cosine_similarity,
    })
}

/// Estimate memory footprint of a model state dict
fn estimate_memory_footprint(state_dict: &HashMap<String, Tensor<f32>>) -> u64 {
    state_dict
        .values()
        .map(|tensor| {
            let num_elements = tensor.shape().numel();
            (num_elements * std::mem::size_of::<f32>()) as u64
        })
        .sum()
}

/// Model ensemble configuration
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Weights for each model in the ensemble
    pub weights: Vec<f32>,
    /// Whether to normalize weights
    pub normalize_weights: bool,
    /// Voting strategy for classification tasks
    pub voting_strategy: VotingStrategy,
}

/// Voting strategy for ensemble models
#[derive(Debug, Clone, Copy)]
pub enum VotingStrategy {
    /// Average predictions (for regression)
    Average,
    /// Weighted average
    WeightedAverage,
    /// Majority vote (for classification)
    MajorityVote,
    /// Soft voting (use probabilities)
    SoftVoting,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            weights: vec![1.0],
            normalize_weights: true,
            voting_strategy: VotingStrategy::WeightedAverage,
        }
    }
}

/// Create an ensemble of models by averaging their parameters
pub fn create_model_ensemble(
    models: &[HashMap<String, Tensor<f32>>],
    config: Option<EnsembleConfig>,
) -> Result<HashMap<String, Tensor<f32>>> {
    if models.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot create ensemble from empty model list".to_string(),
        ));
    }

    let config = config.unwrap_or_default();
    let mut weights = config.weights.clone();

    // Ensure we have the right number of weights
    if weights.len() != models.len() {
        weights = vec![1.0; models.len()];
    }

    // Normalize weights if requested
    if config.normalize_weights {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }
    }

    // Get common parameters
    let param_keys: std::collections::HashSet<_> = models[0].keys().cloned().collect();

    let mut ensemble_state = HashMap::new();

    for param_name in param_keys {
        // Collect all tensors for this parameter
        let tensors: Vec<&Tensor<f32>> = models.iter().filter_map(|m| m.get(&param_name)).collect();

        if tensors.len() != models.len() {
            continue; // Skip parameters that don't exist in all models
        }

        // Average the tensors with weights
        if let Ok(averaged) = weighted_average_tensors(&tensors, &weights) {
            ensemble_state.insert(param_name, averaged);
        }
    }

    Ok(ensemble_state)
}

/// Compute weighted average of tensors
fn weighted_average_tensors(tensors: &[&Tensor<f32>], weights: &[f32]) -> Result<Tensor<f32>> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument("Empty tensor list".to_string()));
    }

    // Verify all tensors have the same shape
    let shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != shape {
            return Err(TorshError::InvalidArgument(
                "All tensors must have the same shape".to_string(),
            ));
        }
    }

    // Convert to vectors and compute weighted average
    let data_vecs: Vec<Vec<f32>> = tensors
        .iter()
        .map(|t| t.to_vec())
        .collect::<Result<Vec<_>>>()?;
    let num_elements = data_vecs[0].len();

    let mut result = vec![0.0f32; num_elements];
    for (tensor_data, weight) in data_vecs.iter().zip(weights.iter()) {
        for (i, value) in tensor_data.iter().enumerate() {
            result[i] += value * weight;
        }
    }

    // Create result tensor
    Tensor::from_data(result, shape.dims().to_vec(), torsh_core::DeviceType::Cpu)
}

/// Model quantization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    pub original_size_bytes: u64,
    pub quantized_size_bytes: u64,
    pub compression_ratio: f32,
    pub parameters_quantized: usize,
    pub mean_quantization_error: f64,
    pub max_quantization_error: f64,
}

/// Model conversion metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMetadata {
    pub source_format: String,
    pub target_format: String,
    pub conversion_time_ms: u64,
    pub warnings: Vec<String>,
    pub unsupported_operations: Vec<String>,
}

/// Load model from file path with automatic format detection
pub fn load_model_auto(path: &Path) -> Result<HashMap<String, Tensor<f32>>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "torsh" | "pth" | "pt" => load_torsh_model(path),
        "onnx" => load_onnx_model_state(path),
        "h5" | "keras" => load_keras_model(path),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported model format: {}",
            extension
        ))),
    }
}

/// Load ToRSh/PyTorch model (placeholder - would need actual implementation)
fn load_torsh_model(_path: &Path) -> Result<HashMap<String, Tensor<f32>>> {
    // This would need actual implementation
    Ok(HashMap::new())
}

/// Load ONNX model state (placeholder - would need actual implementation)
fn load_onnx_model_state(_path: &Path) -> Result<HashMap<String, Tensor<f32>>> {
    // This would need actual implementation with ONNX integration
    Ok(HashMap::new())
}

/// Load Keras model (placeholder - would need actual implementation)
fn load_keras_model(_path: &Path) -> Result<HashMap<String, Tensor<f32>>> {
    // This would need actual implementation
    Ok(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;

    #[test]
    fn test_model_comparison() {
        let mut model1 = HashMap::new();
        let mut model2 = HashMap::new();

        // Add common parameter with same shape
        let tensor1 =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let tensor2 =
            Tensor::from_data(vec![1.1, 2.1, 3.1, 4.1], vec![2, 2], DeviceType::Cpu).unwrap();
        model1.insert("layer1.weight".to_string(), tensor1);
        model2.insert("layer1.weight".to_string(), tensor2);

        // Add parameter only in model1
        let tensor3 = Tensor::from_data(vec![5.0, 6.0], vec![2], DeviceType::Cpu).unwrap();
        model1.insert("layer1.bias".to_string(), tensor3);

        // Add parameter only in model2
        let tensor4 = Tensor::from_data(vec![7.0, 8.0], vec![2], DeviceType::Cpu).unwrap();
        model2.insert("layer2.weight".to_string(), tensor4);

        let diff = compare_models(&model1, &model2, None).unwrap();

        assert_eq!(diff.common_parameters.len(), 1);
        assert_eq!(diff.only_in_first.len(), 1);
        assert_eq!(diff.only_in_second.len(), 1);
        assert_eq!(diff.param_counts, (2, 2));
    }

    #[test]
    fn test_memory_footprint() {
        let mut model = HashMap::new();
        let tensor1 =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(vec![5.0, 6.0], vec![2], DeviceType::Cpu).unwrap();

        model.insert("weight".to_string(), tensor1);
        model.insert("bias".to_string(), tensor2);

        let footprint = estimate_memory_footprint(&model);
        // 4 elements + 2 elements = 6 elements * 4 bytes = 24 bytes
        assert_eq!(footprint, 24);
    }

    #[test]
    fn test_weighted_average() {
        let tensor1 = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let tensors = vec![&tensor1, &tensor2];
        let weights = vec![0.5, 0.5];

        let result = weighted_average_tensors(&tensors, &weights).unwrap();
        let result_data = result.to_vec().unwrap();

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - 2.0).abs() < 1e-5);
        assert!((result_data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_ensemble_creation() {
        let mut model1 = HashMap::new();
        let mut model2 = HashMap::new();

        let tensor1 = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu).unwrap();

        model1.insert("weight".to_string(), tensor1);
        model2.insert("weight".to_string(), tensor2);

        let models = vec![model1, model2];
        let config = EnsembleConfig {
            weights: vec![0.5, 0.5],
            normalize_weights: false,
            voting_strategy: VotingStrategy::WeightedAverage,
        };

        let ensemble = create_model_ensemble(&models, Some(config)).unwrap();

        assert_eq!(ensemble.len(), 1);
        let result = &ensemble["weight"];
        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 2.0).abs() < 1e-5);
        assert!((result_data[1] - 3.0).abs() < 1e-5);
    }
}
