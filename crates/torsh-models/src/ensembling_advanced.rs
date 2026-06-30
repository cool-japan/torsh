//! Advanced ensembling components: meta-learners, utility helpers.
//!
//! This module is included by `ensembling.rs` via the `#[path]` attribute.
//! All items in `ensembling.rs` are in scope through `use super::*`.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use super::*;

/// Meta-learner for stacking that applies learned per-model scalar weights to the
/// concatenated base-model output vector.
///
/// Given an input of shape `[1, n_models * out_dim]` it computes:
///   output[j] = Σ_m  weights[m] * input[m * out_dim + j]   for j in 0..out_dim
pub(super) struct StackingMetaLearner {
    /// Normalised per-model weights (sums to 1).
    weights: Vec<f64>,
    /// Output dimension expected per model slot.
    out_dim: usize,
    /// Whether the input also contains original features appended after the model predictions.
    include_original: bool,
}

impl StackingMetaLearner {
    pub(super) fn new(weights: Vec<f64>, out_dim: usize, include_original: bool) -> Self {
        Self {
            weights,
            out_dim: out_dim.max(1),
            include_original,
        }
    }
}

impl Module for StackingMetaLearner {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let data = input.to_vec()?;
        let n_models = self.weights.len();
        let out_dim = self.out_dim;
        let expected = n_models * out_dim;

        // If input is shorter than expected (e.g. scalar meta-features), fall back
        // to uniform weighted sum of whatever is there.
        let (model_data, _orig) = if data.len() >= expected {
            data.split_at(expected)
        } else {
            (data.as_slice(), &[] as &[f32])
        };

        let effective_out = if n_models > 0 && model_data.len() >= n_models * out_dim {
            out_dim
        } else if n_models > 0 {
            (model_data.len() / n_models).max(1)
        } else {
            1
        };

        let mut result = vec![0.0_f32; effective_out];
        for (m, &w) in self.weights.iter().enumerate() {
            let start = m * effective_out;
            let end = (start + effective_out).min(model_data.len());
            for (j, slot) in result.iter_mut().enumerate() {
                let idx = start + j;
                if idx < end {
                    *slot += (w as f32) * model_data[idx];
                }
            }
        }

        let device = input.device();
        let len = result.len();
        Tensor::from_data(result, vec![1, len], device)
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn named_parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
        Ok(())
    }
}

/// Uniform-average meta-learner.
///
/// Used as a fallback when a non-linear meta-learner type (SVM, Random Forest,
/// Neural Network) is requested but no training has taken place.  Rather than
/// passing the raw concatenated-prediction vector back unchanged (which would
/// silently return the wrong shape and magnitude), this module computes a true
/// element-wise mean over the base-model prediction slots.
///
/// Input  shape: `[1, n_models * out_dim]`
/// Output shape: `[1, out_dim]`
///
/// When `out_dim` cannot be inferred (e.g. `n_models == 0` or input length is
/// not divisible by any known `n_models`), the output is the scalar mean of
/// all input elements with shape `[1, 1]`.
pub(super) struct SimpleMeta {
    /// Number of base models whose predictions are concatenated in the input.
    /// Zero means "unknown" — derive heuristically.
    n_models: usize,
    /// Output dimensionality per model slot.  Zero means "unknown".
    out_dim: usize,
}

impl SimpleMeta {
    pub(super) fn new() -> Self {
        Self {
            n_models: 0,
            out_dim: 0,
        }
    }

    /// Construct with explicit shape hints for correct slicing.
    pub(super) fn with_shape(n_models: usize, out_dim: usize) -> Self {
        Self { n_models, out_dim }
    }
}

impl Module for SimpleMeta {
    /// Compute the element-wise uniform average across `n_models` prediction
    /// slots packed into the input tensor.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let data = input.to_vec()?;
        let total = data.len();

        if total == 0 {
            return Tensor::from_data(vec![0.0f32], vec![1, 1], input.device());
        }

        // Determine (n_models, out_dim) from stored hints or from the data length.
        let (n_models, out_dim) = if self.n_models > 0 && self.out_dim > 0 {
            (self.n_models, self.out_dim)
        } else if self.n_models > 0 {
            let od = (total / self.n_models).max(1);
            (self.n_models, od)
        } else {
            // No hints: fall back to scalar mean (most conservative assumption)
            let mean = data.iter().sum::<f32>() / total as f32;
            return Tensor::from_data(vec![mean], vec![1, 1], input.device());
        };

        // Guard against shapes that don't tile evenly
        if total < n_models * out_dim {
            let mean = data.iter().sum::<f32>() / total as f32;
            return Tensor::from_data(vec![mean], vec![1, 1], input.device());
        }

        // Average the n_models slots of length out_dim
        let mut result = vec![0.0f32; out_dim];
        for m in 0..n_models {
            for j in 0..out_dim {
                result[j] += data[m * out_dim + j];
            }
        }
        let scale = 1.0 / n_models as f32;
        for v in &mut result {
            *v *= scale;
        }

        Tensor::from_data(result, vec![1, out_dim], input.device())
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn named_parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
        Ok(())
    }
}

/// Utility functions for ensembling
pub mod ensembling_utils {
    use super::*;

    /// Create a simple averaging ensemble config
    pub fn simple_average_config() -> EnsembleConfig {
        EnsembleConfig {
            method: EnsembleMethod::SimpleAverage,
            validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.2 },
            diversity_regularization: None,
            performance_weighting: false,
            online_adaptation: None,
        }
    }

    /// Create a weighted averaging ensemble config
    pub fn weighted_average_config(weights: Vec<f64>, learnable: bool) -> EnsembleConfig {
        EnsembleConfig {
            method: EnsembleMethod::WeightedAverage { weights, learnable },
            validation_strategy: EnsembleValidationStrategy::KFold { k: 5 },
            diversity_regularization: None,
            performance_weighting: true,
            online_adaptation: None,
        }
    }

    /// Create a stacking ensemble config
    pub fn stacking_config(meta_learner_type: MetaLearnerType) -> EnsembleConfig {
        EnsembleConfig {
            method: EnsembleMethod::Stacking {
                meta_learner_config: MetaLearnerConfig {
                    learner_type: meta_learner_type,
                    config: HashMap::new(),
                    include_original_features: false,
                },
                use_cross_validation: true,
            },
            validation_strategy: EnsembleValidationStrategy::KFold { k: 5 },
            diversity_regularization: Some(DiversityRegularization {
                strength: 0.1,
                metric: DiversityMetric::Disagreement,
                encourage: true,
            }),
            performance_weighting: true,
            online_adaptation: None,
        }
    }

    /// Create an online adaptive ensemble config
    pub fn online_adaptive_config(learning_rate: f64) -> EnsembleConfig {
        EnsembleConfig {
            method: EnsembleMethod::WeightedAverage {
                weights: vec![],
                learnable: true,
            },
            validation_strategy: EnsembleValidationStrategy::HoldOut { split_ratio: 0.2 },
            diversity_regularization: None,
            performance_weighting: true,
            online_adaptation: Some(OnlineAdaptationConfig {
                learning_rate,
                window_size: 100,
                forgetting_factor: 0.99,
                min_weight: 0.01,
            }),
        }
    }

    /// Calculate optimal ensemble size based on diversity-accuracy tradeoff
    pub fn calculate_optimal_ensemble_size(
        individual_accuracies: &[f64],
        diversity_measures: &[f64],
        diversity_weight: f64,
    ) -> usize {
        let mut best_score = 0.0;
        let mut best_size = 1;

        for size in 1..=individual_accuracies.len() {
            let avg_accuracy = individual_accuracies[..size].iter().sum::<f64>() / size as f64;
            let avg_diversity = if size > 1 {
                diversity_measures[..size - 1].iter().sum::<f64>() / (size - 1) as f64
            } else {
                0.0
            };

            let score = avg_accuracy + diversity_weight * avg_diversity;

            if score > best_score {
                best_score = score;
                best_size = size;
            }
        }

        best_size
    }

    /// Estimate ensemble performance improvement
    pub fn estimate_ensemble_improvement(
        individual_accuracies: &[f64],
        ensemble_method: &EnsembleMethod,
    ) -> f64 {
        let avg_individual =
            individual_accuracies.iter().sum::<f64>() / individual_accuracies.len() as f64;
        let max_individual = individual_accuracies.iter().fold(0.0f64, |a, &b| a.max(b));

        let improvement_factor = match ensemble_method {
            EnsembleMethod::SimpleAverage => 1.1,
            EnsembleMethod::WeightedAverage { .. } => 1.15,
            EnsembleMethod::Stacking { .. } => 1.2,
            EnsembleMethod::MixtureOfExperts { .. } => 1.25,
            _ => 1.05,
        };

        let estimated_ensemble = avg_individual * improvement_factor;
        estimated_ensemble.min(max_individual * 1.1) // Cap at 110% of best individual
    }

    /// Calculate ensemble complexity score
    pub fn calculate_ensemble_complexity(num_models: usize, method: &EnsembleMethod) -> f64 {
        let base_complexity = num_models as f64;

        let method_multiplier = match method {
            EnsembleMethod::SimpleAverage => 1.0,
            EnsembleMethod::WeightedAverage { .. } => 1.1,
            EnsembleMethod::MajorityVoting => 1.05,
            EnsembleMethod::WeightedVoting { .. } => 1.1,
            EnsembleMethod::Stacking { .. } => 1.5,
            EnsembleMethod::BayesianAverage { .. } => 1.3,
            EnsembleMethod::DynamicSelection { .. } => 1.4,
            EnsembleMethod::MixtureOfExperts { .. } => 2.0,
        };

        base_complexity * method_multiplier
    }
}
