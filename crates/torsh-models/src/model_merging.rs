//! Model merging and fusion utilities
//!
//! This module provides utilities for combining multiple models:
//! - Model averaging (simple, weighted, exponential moving average)
//! - LoRA merging and extraction
//! - Model soup (combining fine-tuned models)
//! - Task arithmetic (adding/subtracting task vectors)
//! - SLERP (Spherical Linear Interpolation)

use std::collections::HashMap;
use torsh_core::error::Result as TorshResult;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use crate::{ModelError, ModelResult};

/// Model merging strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Simple average of parameters
    Average,
    /// Weighted average with specified weights
    WeightedAverage,
    /// Exponential moving average
    ExponentialMovingAverage { alpha: f32 },
    /// Task arithmetic (subtract base, add task vectors)
    TaskArithmetic,
    /// SLERP - Spherical Linear Interpolation
    Slerp { t: f32 },
    /// Maximum magnitude (take parameter with largest magnitude)
    MaxMagnitude,
    /// Consensus (only merge if models agree within threshold)
    Consensus { threshold: f32 },
}

/// Model merger for combining multiple models
pub struct ModelMerger {
    /// Merging strategy
    strategy: MergeStrategy,
    /// Weights for weighted averaging (if applicable)
    weights: Option<Vec<f32>>,
    /// Base model for task arithmetic
    base_model: Option<HashMap<String, Parameter>>,
}

impl ModelMerger {
    /// Create a new model merger with simple averaging
    pub fn new() -> Self {
        Self {
            strategy: MergeStrategy::Average,
            weights: None,
            base_model: None,
        }
    }

    /// Create merger with weighted averaging
    pub fn with_weights(weights: Vec<f32>) -> ModelResult<Self> {
        // Validate weights
        if weights.is_empty() {
            return Err(ModelError::ValidationError {
                reason: "Weights vector cannot be empty".to_string(),
            });
        }

        let sum: f32 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-5 {
            return Err(ModelError::ValidationError {
                reason: format!("Weights must sum to 1.0, got {}", sum),
            });
        }

        Ok(Self {
            strategy: MergeStrategy::WeightedAverage,
            weights: Some(weights),
            base_model: None,
        })
    }

    /// Create merger with exponential moving average
    pub fn with_ema(alpha: f32) -> ModelResult<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(ModelError::ValidationError {
                reason: format!("Alpha must be between 0 and 1, got {}", alpha),
            });
        }

        Ok(Self {
            strategy: MergeStrategy::ExponentialMovingAverage { alpha },
            weights: None,
            base_model: None,
        })
    }

    /// Create merger with SLERP
    pub fn with_slerp(t: f32) -> ModelResult<Self> {
        if !(0.0..=1.0).contains(&t) {
            return Err(ModelError::ValidationError {
                reason: format!("t must be between 0 and 1, got {}", t),
            });
        }

        Ok(Self {
            strategy: MergeStrategy::Slerp { t },
            weights: None,
            base_model: None,
        })
    }

    /// Create merger with task arithmetic
    pub fn with_task_arithmetic(base_model: &dyn Module) -> Self {
        Self {
            strategy: MergeStrategy::TaskArithmetic,
            weights: None,
            base_model: Some(base_model.parameters()),
        }
    }

    /// Set merging strategy
    pub fn set_strategy(&mut self, strategy: MergeStrategy) {
        self.strategy = strategy;
    }

    /// Merge multiple models into one
    pub fn merge_models(&self, models: &[&dyn Module]) -> ModelResult<HashMap<String, Parameter>> {
        if models.is_empty() {
            return Err(ModelError::ValidationError {
                reason: "Cannot merge empty model list".to_string(),
            });
        }

        if models.len() == 1 {
            return Ok(models[0].parameters());
        }

        // Validate weights match number of models if using weighted averaging
        if let Some(ref weights) = self.weights {
            if weights.len() != models.len() {
                return Err(ModelError::ValidationError {
                    reason: format!(
                        "Number of weights ({}) must match number of models ({})",
                        weights.len(),
                        models.len()
                    ),
                });
            }
        }

        // Get all parameter names from first model
        let param_names: Vec<String> = models[0].parameters().keys().cloned().collect();

        // Validate all models have the same parameters
        for (i, model) in models.iter().enumerate().skip(1) {
            let model_params = model.parameters();
            for name in &param_names {
                if !model_params.contains_key(name) {
                    return Err(ModelError::ValidationError {
                        reason: format!(
                            "Model {} missing parameter '{}' present in model 0",
                            i, name
                        ),
                    });
                }
            }
        }

        // Merge parameters
        let mut merged_params = HashMap::new();

        for name in &param_names {
            // Collect tensors with proper Arc<RwLock> handling
            let tensor_arcs: Vec<_> = models
                .iter()
                .map(|m| {
                    m.parameters()
                        .get(name)
                        .expect("parameter should exist in all models")
                        .tensor()
                })
                .collect();

            let merged_tensor = match self.strategy {
                MergeStrategy::Average => self.average_tensors(&tensor_arcs)?,
                MergeStrategy::WeightedAverage => self.weighted_average_tensors(
                    &tensor_arcs,
                    self.weights
                        .as_ref()
                        .expect("weights should be set for weighted average strategy"),
                )?,
                MergeStrategy::ExponentialMovingAverage { alpha } => {
                    self.ema_tensors(&tensor_arcs, alpha)?
                }
                MergeStrategy::TaskArithmetic => {
                    self.task_arithmetic_tensors(&tensor_arcs, name)?
                }
                MergeStrategy::Slerp { t } => {
                    if tensor_arcs.len() != 2 {
                        return Err(ModelError::ValidationError {
                            reason: "SLERP requires exactly 2 models".to_string(),
                        });
                    }
                    self.slerp_tensors(&tensor_arcs[0], &tensor_arcs[1], t)?
                }
                MergeStrategy::MaxMagnitude => self.max_magnitude_tensors(&tensor_arcs)?,
                MergeStrategy::Consensus { threshold } => {
                    self.consensus_tensors(&tensor_arcs, threshold)?
                }
            };

            merged_params.insert(
                name.clone(),
                Parameter::from_tensor(std::sync::Arc::new(parking_lot::RwLock::new(
                    merged_tensor,
                ))),
            );
        }

        Ok(merged_params)
    }

    /// Simple averaging of tensors
    fn average_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
    ) -> TorshResult<Tensor> {
        if tensor_arcs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Cannot average empty tensor list".to_string(),
            ));
        }

        let first = tensor_arcs[0].read();
        let mut sum = first.clone();
        drop(first);

        for tensor_arc in &tensor_arcs[1..] {
            let tensor = tensor_arc.read();
            sum = sum.add(&*tensor)?;
        }

        sum.div_scalar(tensor_arcs.len() as f32)
    }

    /// Weighted averaging of tensors
    fn weighted_average_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
        weights: &[f32],
    ) -> TorshResult<Tensor> {
        if tensor_arcs.is_empty() || weights.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Cannot average empty tensor or weight list".to_string(),
            ));
        }

        let first = tensor_arcs[0].read();
        let mut result = first.mul_scalar(weights[0])?;
        drop(first);

        for (tensor_arc, &weight) in tensor_arcs.iter().zip(weights.iter()).skip(1) {
            let tensor = tensor_arc.read();
            let weighted = tensor.mul_scalar(weight)?;
            result = result.add(&weighted)?;
        }

        Ok(result)
    }

    /// Exponential moving average
    fn ema_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
        alpha: f32,
    ) -> TorshResult<Tensor> {
        if tensor_arcs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Cannot compute EMA of empty tensor list".to_string(),
            ));
        }

        let first = tensor_arcs[0].read();
        let mut result = first.clone();
        drop(first);

        for tensor_arc in &tensor_arcs[1..] {
            let tensor = tensor_arc.read();
            // result = alpha * tensor + (1 - alpha) * result
            let weighted_new = tensor.mul_scalar(alpha)?;
            let weighted_old = result.mul_scalar(1.0 - alpha)?;
            result = weighted_new.add(&weighted_old)?;
        }

        Ok(result)
    }

    /// Task arithmetic: (model - base) merging
    fn task_arithmetic_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
        param_name: &str,
    ) -> TorshResult<Tensor> {
        if let Some(ref base_params) = self.base_model {
            if let Some(base_param) = base_params.get(param_name) {
                let base_tensor_arc = base_param.tensor();
                let base_tensor = base_tensor_arc.read();

                // Compute task vectors: model - base
                let mut task_vectors = Vec::new();
                for tensor_arc in tensor_arcs {
                    let tensor = tensor_arc.read();
                    let task_vector = tensor.sub(&*base_tensor)?;
                    task_vectors.push(task_vector);
                }
                drop(base_tensor);

                // Average task vectors - need to create Arc<RwLock> wrappers
                let task_arcs: Vec<_> = task_vectors
                    .into_iter()
                    .map(|t| std::sync::Arc::new(parking_lot::RwLock::new(t)))
                    .collect();

                let avg_task_vector = self.average_tensors(&task_arcs)?;

                // Add back to base
                let base_tensor = base_tensor_arc.read();
                base_tensor.add(&avg_task_vector)
            } else {
                // No base parameter, just average
                self.average_tensors(tensor_arcs)
            }
        } else {
            // No base model, fall back to averaging
            self.average_tensors(tensor_arcs)
        }
    }

    /// SLERP - Spherical Linear Interpolation
    fn slerp_tensors(
        &self,
        tensor_arc1: &std::sync::Arc<parking_lot::RwLock<Tensor>>,
        tensor_arc2: &std::sync::Arc<parking_lot::RwLock<Tensor>>,
        t: f32,
    ) -> TorshResult<Tensor> {
        let tensor1 = tensor_arc1.read();
        let tensor2 = tensor_arc2.read();

        // Simplified SLERP - just linear interpolation for now
        // Full SLERP implementation would require more tensor operations
        let result = tensor1.mul_scalar(1.0 - t)?;
        let weighted2 = tensor2.mul_scalar(t)?;
        result.add(&weighted2)
    }

    /// Maximum magnitude merging
    fn max_magnitude_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
    ) -> TorshResult<Tensor> {
        if tensor_arcs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Cannot compute max magnitude of empty tensor list".to_string(),
            ));
        }

        let first = tensor_arcs[0].read();
        let mut result = first.clone();
        drop(first);

        for tensor_arc in &tensor_arcs[1..] {
            let tensor = tensor_arc.read();
            // Simplified: just take average for now
            // Full implementation would need element-wise comparison
            result = result.add(&*tensor)?.div_scalar(2.0)?;
        }

        Ok(result)
    }

    /// Consensus merging - only merge if models agree within threshold
    fn consensus_tensors(
        &self,
        tensor_arcs: &[std::sync::Arc<parking_lot::RwLock<Tensor>>],
        _threshold: f32,
    ) -> TorshResult<Tensor> {
        if tensor_arcs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Cannot compute consensus of empty tensor list".to_string(),
            ));
        }

        // Simplified: just average for now
        // Full implementation would check threshold
        self.average_tensors(tensor_arcs)
    }
}

impl Default for ModelMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// LoRA (Low-Rank Adaptation) merger
pub struct LoRAMerger {
    /// Scaling factor for LoRA weights
    alpha: f32,
    /// Rank of LoRA matrices
    rank: usize,
}

impl LoRAMerger {
    /// Create a new LoRA merger
    pub fn new(alpha: f32, rank: usize) -> Self {
        Self { alpha, rank }
    }

    /// Merge LoRA weights into base model
    pub fn merge_lora(
        &self,
        base_model: &dyn Module,
        lora_a: &HashMap<String, Parameter>,
        lora_b: &HashMap<String, Parameter>,
    ) -> ModelResult<HashMap<String, Parameter>> {
        let mut merged_params = base_model.parameters();

        for (name, base_param) in &merged_params.clone() {
            // Check if LoRA parameters exist for this layer
            let lora_a_name = format!("{}.lora_a", name);
            let lora_b_name = format!("{}.lora_b", name);

            if let (Some(a_param), Some(b_param)) =
                (lora_a.get(&lora_a_name), lora_b.get(&lora_b_name))
            {
                // Read tensors from Arc<RwLock>
                let a_tensor = a_param.tensor();
                let b_tensor = b_param.tensor();
                let base_tensor = base_param.tensor();

                let a = a_tensor.read();
                let b = b_tensor.read();
                let base = base_tensor.read();

                // Compute delta_W = alpha * B @ A
                let delta_w = b.matmul(&*a)?;
                let scaled_delta = delta_w.mul_scalar(self.alpha)?;

                // Add to base weight: W' = W + delta_W
                let merged = base.add(&scaled_delta)?;

                merged_params.insert(
                    name.clone(),
                    Parameter::from_tensor(std::sync::Arc::new(parking_lot::RwLock::new(merged))),
                );
            }
        }

        Ok(merged_params)
    }

    /// Extract LoRA parameters from fine-tuned model
    pub fn extract_lora(
        &self,
        base_model: &dyn Module,
        finetuned_model: &dyn Module,
    ) -> ModelResult<(HashMap<String, Parameter>, HashMap<String, Parameter>)> {
        let base_params = base_model.parameters();
        let finetuned_params = finetuned_model.parameters();

        let mut lora_a = HashMap::new();
        let mut lora_b = HashMap::new();

        for (name, base_param) in &base_params {
            if let Some(finetuned_param) = finetuned_params.get(name) {
                // Read tensors from Arc<RwLock>
                let base_tensor = base_param.tensor();
                let finetuned_tensor = finetuned_param.tensor();

                let base = base_tensor.read();
                let finetuned = finetuned_tensor.read();

                // Compute delta: W_finetuned - W_base
                let delta = finetuned.sub(&*base)?;

                // Perform low-rank decomposition (simplified SVD)
                // In practice, use proper SVD with rank truncation
                let (a, b) = self.low_rank_decomposition(&delta)?;

                lora_a.insert(
                    format!("{}.lora_a", name),
                    Parameter::from_tensor(std::sync::Arc::new(parking_lot::RwLock::new(a))),
                );
                lora_b.insert(
                    format!("{}.lora_b", name),
                    Parameter::from_tensor(std::sync::Arc::new(parking_lot::RwLock::new(b))),
                );
            }
        }

        Ok((lora_a, lora_b))
    }

    /// Low-rank decomposition using SVD
    ///
    /// Computes a rank-k approximation of the input tensor using Singular Value Decomposition.
    /// For a matrix W ∈ ℝ^(m×n), computes W ≈ A @ B where:
    /// - A ∈ ℝ^(m×k) contains the k largest left singular vectors scaled by singular values
    /// - B ∈ ℝ^(k×n) contains the k largest right singular vectors
    ///
    /// This is the optimal rank-k approximation in the Frobenius norm (Eckart-Young theorem).
    fn low_rank_decomposition(&self, tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
        let shape = tensor.shape();

        if shape.dims().len() != 2 {
            return Err(torsh_core::TorshError::InvalidArgument(
                "LoRA decomposition requires 2D tensor".to_string(),
            ));
        }

        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let rank = self.rank.min(rows).min(cols);

        // Perform SVD: tensor = U @ diag(S) @ V^T
        let (u, s, vt) = torsh_linalg::decomposition::svd(tensor, false)?;

        // Extract the first 'rank' components
        // A = U[:, :rank] @ diag(sqrt(S[:rank]))
        // B = diag(sqrt(S[:rank])) @ V^T[:rank, :]

        let mut a_data = Vec::with_capacity(rows * rank);
        let mut b_data = Vec::with_capacity(rank * cols);

        // Build A = U[:, :rank] @ diag(sqrt(S[:rank]))
        for i in 0..rows {
            for j in 0..rank {
                let s_val = s.get(&[j])?.sqrt();
                let u_val = u.get(&[i, j])?;
                a_data.push(u_val * s_val);
            }
        }

        // Build B = diag(sqrt(S[:rank])) @ V^T[:rank, :]
        for i in 0..rank {
            let s_val = s.get(&[i])?.sqrt();
            for j in 0..cols {
                let vt_val = vt.get(&[i, j])?;
                b_data.push(s_val * vt_val);
            }
        }

        let a = Tensor::from_data(a_data, vec![rows, rank], tensor.device())?;
        let b = Tensor::from_data(b_data, vec![rank, cols], tensor.device())?;

        Ok((a, b))
    }
}

/// Model soup - combining multiple fine-tuned models
pub struct ModelSoup {
    /// Models to combine
    models: Vec<Box<dyn Module>>,
    /// Greedy selection threshold
    greedy_threshold: Option<f32>,
}

impl ModelSoup {
    /// Create a new model soup
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            greedy_threshold: None,
        }
    }

    /// Add a model to the soup
    pub fn add_model(&mut self, model: Box<dyn Module>) {
        self.models.push(model);
    }

    /// Set greedy selection threshold
    pub fn with_greedy_threshold(mut self, threshold: f32) -> Self {
        self.greedy_threshold = Some(threshold);
        self
    }

    /// Create soup by averaging all models
    pub fn uniform_soup(&self) -> ModelResult<HashMap<String, Parameter>> {
        let merger = ModelMerger::new();
        let model_refs: Vec<&dyn Module> = self.models.iter().map(|m| m.as_ref()).collect();
        merger.merge_models(&model_refs)
    }

    /// Create soup using greedy selection
    /// Adds models one at a time if they improve validation performance
    pub fn greedy_soup<F>(&self, validate_fn: F) -> ModelResult<HashMap<String, Parameter>>
    where
        F: Fn(&HashMap<String, Parameter>) -> f32,
    {
        if self.models.is_empty() {
            return Err(ModelError::ValidationError {
                reason: "Cannot create soup from empty model list".to_string(),
            });
        }

        // Start with first model
        let mut best_params = self.models[0].parameters();
        let mut best_score = validate_fn(&best_params);

        // Try adding each model
        for model in &self.models[1..] {
            let merger = ModelMerger::new();

            // Create temporary soup with current best + this model
            let temp_soup = merger.merge_models(&[&*self.models[0], model.as_ref()])?;

            let temp_score = validate_fn(&temp_soup);

            // If score improves, keep it
            if temp_score > best_score {
                best_params = temp_soup;
                best_score = temp_score;
            }
        }

        Ok(best_params)
    }
}

impl Default for ModelSoup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_strategy_creation() {
        let merger = ModelMerger::new();
        assert_eq!(merger.strategy, MergeStrategy::Average);

        let weighted = ModelMerger::with_weights(vec![0.5, 0.5]).unwrap();
        assert_eq!(weighted.strategy, MergeStrategy::WeightedAverage);

        let ema = ModelMerger::with_ema(0.9).unwrap();
        assert!(matches!(
            ema.strategy,
            MergeStrategy::ExponentialMovingAverage { .. }
        ));
    }

    #[test]
    fn test_weight_validation() {
        // Invalid: doesn't sum to 1
        let result = ModelMerger::with_weights(vec![0.3, 0.3]);
        assert!(result.is_err());

        // Valid
        let result = ModelMerger::with_weights(vec![0.6, 0.4]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lora_merger_creation() {
        let lora = LoRAMerger::new(0.5, 8);
        assert_eq!(lora.alpha, 0.5);
        assert_eq!(lora.rank, 8);
    }
}
