//! Model ensembling utilities for combining multiple models
//!
//! This module provides comprehensive ensembling techniques including:
//! - Voting ensembles (majority, weighted voting)
//! - Averaging ensembles (simple, weighted averaging)
//! - Stacking ensembles with meta-learners
//! - Bagging and boosting methods
//! - Dynamic ensemble selection

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Ensemble method configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Simple averaging of predictions
    SimpleAverage,
    /// Weighted averaging with learned or fixed weights
    WeightedAverage { weights: Vec<f64>, learnable: bool },
    /// Majority voting for classification
    MajorityVoting,
    /// Weighted voting for classification
    WeightedVoting { weights: Vec<f64> },
    /// Stacking with a meta-learner
    Stacking {
        meta_learner_config: MetaLearnerConfig,
        use_cross_validation: bool,
    },
    /// Bayesian model averaging
    BayesianAverage { prior_weights: Option<Vec<f64>> },
    /// Dynamic ensemble selection
    DynamicSelection {
        selection_method: SelectionMethod,
        validation_split: f64,
    },
    /// Mixture of experts
    MixtureOfExperts { gating_network: GatingNetworkConfig },
}

/// Meta-learner configuration for stacking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnerConfig {
    /// Type of meta-learner
    pub learner_type: MetaLearnerType,
    /// Configuration parameters
    pub config: HashMap<String, EnsembleConfigValue>,
    /// Whether to include original features
    pub include_original_features: bool,
}

/// Types of meta-learners
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearnerType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression  
    LogisticRegression,
    /// Ridge regression
    RidgeRegression { alpha: f64 },
    /// Random forest
    RandomForest { n_estimators: usize },
    /// Neural network
    NeuralNetwork {
        hidden_layers: Vec<usize>,
        activation: String,
    },
    /// Support Vector Machine
    SVM { kernel: String, c: f64 },
}

/// Configuration value types for ensemble operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleConfigValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Array(Vec<f64>),
}

/// Dynamic selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    /// Select based on local accuracy
    LocalAccuracy { k_neighbors: usize },
    /// Select based on confidence scores
    Confidence { threshold: f64 },
    /// Select based on diversity measures
    Diversity { diversity_metric: DiversityMetric },
    /// Dynamic weighted selection
    DynamicWeighted { adaptation_rate: f64 },
}

/// Diversity metrics for ensemble selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMetric {
    /// Disagreement measure
    Disagreement,
    /// Correlation-based diversity
    Correlation,
    /// Entropy-based diversity
    Entropy,
    /// Q-statistic
    QStatistic,
}

/// Gating network configuration for mixture of experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingNetworkConfig {
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation function
    pub activation: String,
    /// Dropout probability
    pub dropout: f64,
    /// Temperature for softmax
    pub temperature: f64,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Ensemble method to use
    pub method: EnsembleMethod,
    /// Model validation strategy
    pub validation_strategy: EnsembleValidationStrategy,
    /// Diversity regularization
    pub diversity_regularization: Option<DiversityRegularization>,
    /// Performance weighting
    pub performance_weighting: bool,
    /// Online adaptation settings
    pub online_adaptation: Option<OnlineAdaptationConfig>,
}

/// Validation strategies for ensemble training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleValidationStrategy {
    /// K-fold cross-validation
    KFold { k: usize },
    /// Hold-out validation
    HoldOut { split_ratio: f64 },
    /// Time series split
    TimeSeries { n_splits: usize },
    /// Bootstrap validation
    Bootstrap { n_iterations: usize },
}

/// Diversity regularization for ensemble training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityRegularization {
    /// Regularization strength
    pub strength: f64,
    /// Diversity metric to optimize
    pub metric: DiversityMetric,
    /// Whether to encourage or discourage diversity
    pub encourage: bool,
}

/// Online adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineAdaptationConfig {
    /// Learning rate for weight updates
    pub learning_rate: f64,
    /// Adaptation window size
    pub window_size: usize,
    /// Forgetting factor for exponential moving average
    pub forgetting_factor: f64,
    /// Minimum weight threshold
    pub min_weight: f64,
}

/// Ensemble performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleStats {
    /// Individual model performances
    pub individual_performances: Vec<ModelPerformance>,
    /// Ensemble performance
    pub ensemble_performance: ModelPerformance,
    /// Diversity measures
    pub diversity_measures: DiversityMeasures,
    /// Weight distributions
    pub weight_distribution: WeightDistribution,
    /// Prediction statistics
    pub prediction_stats: PredictionStats,
}

/// Individual model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Model identifier
    pub model_id: String,
    /// Accuracy (for classification)
    pub accuracy: Option<f64>,
    /// Mean squared error (for regression)
    pub mse: Option<f64>,
    /// Mean absolute error (for regression)
    pub mae: Option<f64>,
    /// F1 score (for classification)
    pub f1_score: Option<f64>,
    /// AUC score (for binary classification)
    pub auc: Option<f64>,
    /// Confidence scores
    pub confidence_scores: Vec<f64>,
}

/// Diversity measures for the ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMeasures {
    /// Pairwise disagreement
    pub disagreement: f64,
    /// Average correlation
    pub correlation: f64,
    /// Entropy-based diversity
    pub entropy: f64,
    /// Q-statistic
    pub q_statistic: f64,
    /// Kappa statistic
    pub kappa: f64,
}

/// Weight distribution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDistribution {
    /// Current weights
    pub current_weights: Vec<f64>,
    /// Weight history (for adaptive ensembles)
    pub weight_history: Vec<Vec<f64>>,
    /// Weight entropy (measure of concentration)
    pub weight_entropy: f64,
    /// Effective number of models
    pub effective_models: f64,
}

/// Prediction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStats {
    /// Prediction variance
    pub prediction_variance: f64,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Prediction disagreement
    pub disagreement_rate: f64,
    /// Uncertainty estimates
    pub uncertainty_estimates: Vec<f64>,
}

/// Main ensemble engine
pub struct ModelEnsemble<M: Module> {
    models: Vec<M>,
    config: EnsembleConfig,
    weights: Vec<f64>,
    meta_learner: Option<Box<dyn Module>>,
    stats: Option<EnsembleStats>,
    adaptation_state: Option<AdaptationState>,
    /// Gating weight matrix for MoE: shape [num_models x input_features].
    /// Row `m` is the weight vector for expert `m`.
    gating_matrix: Option<Vec<Vec<f64>>>,
}

/// State for online adaptation
#[derive(Debug, Clone)]
struct AdaptationState {
    /// Performance history
    performance_history: Vec<Vec<f64>>,
    /// Weight update momentum
    momentum: Vec<f64>,
    /// Adaptation step count
    step_count: usize,
}

impl<M: Module> ModelEnsemble<M> {
    /// Create a new model ensemble
    pub fn new(models: Vec<M>, config: EnsembleConfig) -> Self {
        let num_models = models.len();

        // Extract weights from config if specified, otherwise use uniform weights
        let weights = match &config.method {
            EnsembleMethod::WeightedAverage { weights, .. } => weights.clone(),
            _ => vec![1.0 / num_models as f64; num_models],
        };

        Self {
            models,
            config,
            weights,
            meta_learner: None,
            stats: None,
            adaptation_state: None,
            gating_matrix: None,
        }
    }

    /// Train the ensemble (for methods that require training)
    pub fn train(&mut self, inputs: &[Tensor], targets: &[Tensor]) -> Result<()> {
        let method = self.config.method.clone();
        match method {
            EnsembleMethod::Stacking {
                meta_learner_config,
                use_cross_validation,
            } => {
                self.train_stacking_ensemble(
                    inputs,
                    targets,
                    &meta_learner_config,
                    use_cross_validation,
                )?;
            }
            EnsembleMethod::WeightedAverage { weights, learnable } => {
                if learnable {
                    self.learn_ensemble_weights(inputs, targets)?;
                } else {
                    self.weights = weights;
                }
            }
            EnsembleMethod::DynamicSelection {
                selection_method,
                validation_split,
            } => {
                self.train_dynamic_selection(inputs, targets, &selection_method, validation_split)?;
            }
            EnsembleMethod::MixtureOfExperts { gating_network } => {
                self.train_mixture_of_experts(inputs, targets, &gating_network)?;
            }
            _ => {
                // For simple methods, just validate the models
                self.validate_models(inputs, targets)?;
            }
        }

        // Initialize online adaptation if configured
        if let Some(adaptation_config) = self.config.online_adaptation.clone() {
            self.initialize_online_adaptation(&adaptation_config);
        }

        Ok(())
    }

    /// Make predictions using the ensemble
    pub fn predict(&mut self, input: &Tensor) -> Result<Tensor> {
        match &self.config.method {
            EnsembleMethod::SimpleAverage => self.simple_average_prediction(input),
            EnsembleMethod::WeightedAverage { .. } => self.weighted_average_prediction(input),
            EnsembleMethod::MajorityVoting => self.majority_voting_prediction(input),
            EnsembleMethod::WeightedVoting { weights } => {
                self.weighted_voting_prediction(input, weights)
            }
            EnsembleMethod::Stacking { .. } => self.stacking_prediction(input),
            EnsembleMethod::BayesianAverage { prior_weights } => {
                self.bayesian_average_prediction(input, prior_weights.as_ref())
            }
            EnsembleMethod::DynamicSelection {
                selection_method, ..
            } => self.dynamic_selection_prediction(input, selection_method),
            EnsembleMethod::MixtureOfExperts { .. } => self.mixture_of_experts_prediction(input),
        }
    }

    /// Update ensemble weights based on recent performance (online adaptation)
    pub fn update_weights(
        &mut self,
        input: &Tensor,
        target: &Tensor,
        prediction: &Tensor,
    ) -> Result<()> {
        if let Some(adaptation_config) = self.config.online_adaptation.clone() {
            self.online_weight_update(input, target, prediction, &adaptation_config)?;
        }
        Ok(())
    }

    /// Get individual model predictions
    pub fn get_individual_predictions(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let mut predictions = Vec::new();
        for model in &self.models {
            predictions.push(model.forward(input)?);
        }
        Ok(predictions)
    }

    /// Calculate ensemble diversity measures
    pub fn calculate_diversity(&self, inputs: &[Tensor]) -> Result<DiversityMeasures> {
        let mut all_predictions = Vec::new();

        // Get predictions from all models
        for input in inputs {
            let mut predictions = Vec::new();
            for model in &self.models {
                predictions.push(model.forward(input)?);
            }
            all_predictions.push(predictions);
        }

        // Calculate diversity measures
        let disagreement = self.calculate_disagreement(&all_predictions)?;
        let correlation = self.calculate_correlation(&all_predictions)?;
        let entropy = self.calculate_entropy(&all_predictions)?;
        let q_statistic = self.calculate_q_statistic(&all_predictions)?;
        let kappa = self.calculate_kappa_statistic(&all_predictions)?;

        Ok(DiversityMeasures {
            disagreement,
            correlation,
            entropy,
            q_statistic,
            kappa,
        })
    }

    // Implementation methods for different ensemble strategies
    fn simple_average_prediction(&self, input: &Tensor) -> Result<Tensor> {
        let predictions = self.get_individual_predictions(input)?;
        let mut sum = predictions[0].clone();

        for prediction in &predictions[1..] {
            sum = sum.add(prediction)?;
        }

        sum.div_scalar(predictions.len() as f32)
    }

    fn weighted_average_prediction(&self, input: &Tensor) -> Result<Tensor> {
        let predictions = self.get_individual_predictions(input)?;
        let mut weighted_sum = predictions[0].mul_scalar(self.weights[0] as f32)?;

        for (prediction, &weight) in predictions[1..].iter().zip(&self.weights[1..]) {
            let weighted_pred = prediction.mul_scalar(weight as f32)?;
            weighted_sum = weighted_sum.add(&weighted_pred)?;
        }

        Ok(weighted_sum)
    }

    fn majority_voting_prediction(&self, input: &Tensor) -> Result<Tensor> {
        let predictions = self.get_individual_predictions(input)?;
        if predictions.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot perform majority voting on an empty ensemble".to_string(),
            ));
        }

        // Per-row argmax. The trailing dimension is treated as the class axis.
        let dims = predictions[0].shape().dims().to_vec();
        let num_classes = *dims.last().unwrap_or(&1);
        let num_rows = if num_classes == 0 {
            0
        } else {
            predictions[0].numel() / num_classes
        };

        // Collect argmax index per row, per model.
        let mut per_model_rows: Vec<Vec<usize>> = Vec::with_capacity(predictions.len());
        for prediction in &predictions {
            let data = prediction.to_vec()?;
            let mut row_argmaxes = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let start = row * num_classes;
                let end = start + num_classes;
                let (best_idx, _) = data[start..end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, &0.0));
                row_argmaxes.push(best_idx);
            }
            per_model_rows.push(row_argmaxes);
        }

        // For each row, take the mode across models.
        let mut output = vec![0.0f32; num_rows * num_classes];
        for row in 0..num_rows {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for model_rows in &per_model_rows {
                *counts.entry(model_rows[row]).or_insert(0) += 1;
            }
            let majority = counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class)
                .unwrap_or(0);
            if majority < num_classes {
                output[row * num_classes + majority] = 1.0;
            }
        }

        Tensor::from_data(output, dims, predictions[0].device())
    }

    fn weighted_voting_prediction(&self, input: &Tensor, weights: &[f64]) -> Result<Tensor> {
        let predictions = self.get_individual_predictions(input)?;
        if predictions.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot perform weighted voting on an empty ensemble".to_string(),
            ));
        }
        if weights.len() != predictions.len() {
            return Err(TorshError::ComputeError(format!(
                "Voting weights length {} does not match number of base models {}",
                weights.len(),
                predictions.len()
            )));
        }

        let dims = predictions[0].shape().dims().to_vec();
        let num_classes = *dims.last().unwrap_or(&1);
        let num_rows = if num_classes == 0 {
            0
        } else {
            predictions[0].numel() / num_classes
        };

        // Per-row argmax for each model.
        let mut per_model_rows: Vec<Vec<usize>> = Vec::with_capacity(predictions.len());
        for prediction in &predictions {
            let data = prediction.to_vec()?;
            let mut row_argmaxes = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let start = row * num_classes;
                let end = start + num_classes;
                let (best_idx, _) = data[start..end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, &0.0));
                row_argmaxes.push(best_idx);
            }
            per_model_rows.push(row_argmaxes);
        }

        // Weighted vote per row. The class with the largest accumulated weight wins.
        let mut output = vec![0.0f32; num_rows * num_classes];
        for row in 0..num_rows {
            let mut weighted_votes: HashMap<usize, f64> = HashMap::new();
            for (model_rows, &weight) in per_model_rows.iter().zip(weights.iter()) {
                *weighted_votes.entry(model_rows[row]).or_insert(0.0) += weight;
            }
            let majority = weighted_votes
                .into_iter()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(class, _)| class)
                .unwrap_or(0);
            if majority < num_classes {
                output[row * num_classes + majority] = 1.0;
            }
        }

        Tensor::from_data(output, dims, predictions[0].device())
    }

    fn stacking_prediction(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(meta_learner) = &self.meta_learner {
            // Get base model predictions
            let base_predictions = self.get_individual_predictions(input)?;

            // Concatenate predictions as features for meta-learner
            let base_refs: Vec<&Tensor> = base_predictions.iter().collect();
            let meta_features = Tensor::cat(&base_refs, -1)?;

            // Use meta-learner for final prediction
            meta_learner.forward(&meta_features)
        } else {
            // Fallback to weighted average
            self.weighted_average_prediction(input)
        }
    }

    fn bayesian_average_prediction(
        &self,
        input: &Tensor,
        prior_weights: Option<&Vec<f64>>,
    ) -> Result<Tensor> {
        // Simplified Bayesian model averaging
        // In practice, this would involve proper Bayesian inference
        let predictions = self.get_individual_predictions(input)?;

        let weights = if let Some(priors) = prior_weights {
            priors.clone()
        } else {
            self.weights.clone()
        };

        let mut weighted_sum = predictions[0].mul_scalar(weights[0] as f32)?;
        for (prediction, &weight) in predictions[1..].iter().zip(&weights[1..]) {
            let weighted_pred = prediction.mul_scalar(weight as f32)?;
            weighted_sum = weighted_sum.add(&weighted_pred)?;
        }

        Ok(weighted_sum)
    }

    fn dynamic_selection_prediction(
        &self,
        input: &Tensor,
        selection_method: &SelectionMethod,
    ) -> Result<Tensor> {
        match selection_method {
            SelectionMethod::Confidence { threshold } => {
                let predictions = self.get_individual_predictions(input)?;
                let mut confident_predictions = Vec::new();
                let mut confident_weights = Vec::new();

                for (i, prediction) in predictions.iter().enumerate() {
                    let confidence = self.calculate_confidence(prediction)?;
                    if confidence > *threshold {
                        confident_predictions.push(prediction.clone());
                        confident_weights.push(self.weights[i]);
                    }
                }

                if confident_predictions.is_empty() {
                    // Fallback to all models
                    self.weighted_average_prediction(input)
                } else {
                    // Average confident predictions
                    let weight_sum: f64 = confident_weights.iter().sum();
                    let normalized_weights: Vec<f32> = confident_weights
                        .iter()
                        .map(|&w| (w / weight_sum) as f32)
                        .collect();

                    let mut weighted_sum =
                        confident_predictions[0].mul_scalar(normalized_weights[0])?;
                    for (prediction, &weight) in confident_predictions[1..]
                        .iter()
                        .zip(&normalized_weights[1..])
                    {
                        let weighted_pred = prediction.mul_scalar(weight)?;
                        weighted_sum = weighted_sum.add(&weighted_pred)?;
                    }

                    Ok(weighted_sum)
                }
            }
            _ => {
                // Simplified implementation for other selection methods
                self.weighted_average_prediction(input)
            }
        }
    }

    fn mixture_of_experts_prediction(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified mixture of experts
        // In practice, this would use a trained gating network
        let predictions = self.get_individual_predictions(input)?;

        // Use current weights as gating network output (simplified)
        let mut weighted_sum = predictions[0].mul_scalar(self.weights[0] as f32)?;
        for (prediction, &weight) in predictions[1..].iter().zip(&self.weights[1..]) {
            let weighted_pred = prediction.mul_scalar(weight as f32)?;
            weighted_sum = weighted_sum.add(&weighted_pred)?;
        }

        Ok(weighted_sum)
    }

    // Training methods

    /// Stacking ensemble training: generate OOF meta-features via k-fold CV (k=5)
    /// then fit a ridge-regression meta-learner (gradient descent, 100 iterations,
    /// lr = 0.01, α from config) that learns per-base-model scalar weights
    /// minimising hold-out MSE.
    fn train_stacking_ensemble(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        meta_learner_config: &MetaLearnerConfig,
        use_cross_validation: bool,
    ) -> Result<()> {
        if inputs.is_empty() || targets.is_empty() {
            self.meta_learner = Some(self.create_meta_learner(meta_learner_config)?);
            return Ok(());
        }
        if inputs.len() != targets.len() {
            return Err(TorshError::ComputeError(format!(
                "stacking train: input/target length mismatch {} vs {}",
                inputs.len(),
                targets.len()
            )));
        }

        // Step 1: generate out-of-fold (OOF) predictions.
        let k = if use_cross_validation { 5 } else { 1 };
        let meta_features = if use_cross_validation {
            self.generate_cv_meta_features(inputs, targets, k)?
        } else {
            self.generate_holdout_meta_features(inputs, targets, 0.2)?
        };

        // Step 2: determine ridge alpha from config.
        let alpha = match &meta_learner_config.learner_type {
            MetaLearnerType::RidgeRegression { alpha } => *alpha,
            _ => 1.0,
        };

        // Step 3: extract meta-features and targets into flat f64 vecs for GD.
        // meta_features[n] shape = [1, num_models * output_dim]
        // We want per-model scalar weights w_m so that:
        //   ŷ_n = Σ_m  w_m * avg(pred_{n,m})   (averaged over output dims)
        // and we minimise MSE + alpha * ||w||^2.
        let n_models = self.models.len();
        if n_models == 0 {
            self.meta_learner = Some(self.create_meta_learner(meta_learner_config)?);
            return Ok(());
        }

        // Build per-sample per-model scalar predictions (avg over output).
        let n_samples = meta_features.len().min(targets.len());
        if n_samples == 0 {
            self.meta_learner = Some(self.create_meta_learner(meta_learner_config)?);
            return Ok(());
        }

        // Each meta-feature has width = n_models * out_dim.
        let meta_width = meta_features[0].numel();
        let out_dim = if n_models > 0 { meta_width / n_models } else { 1 };

        // per_model_avg[sample][model] = average output value of that model for that sample.
        let mut per_model_avg: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for mf in meta_features.iter().take(n_samples) {
            let data = mf.to_vec()?;
            let mut row = Vec::with_capacity(n_models);
            for m in 0..n_models {
                let start = m * out_dim;
                let end = (start + out_dim).min(data.len());
                let avg = if end > start {
                    data[start..end].iter().map(|&v| v as f64).sum::<f64>()
                        / (end - start) as f64
                } else {
                    0.0
                };
                row.push(avg);
            }
            per_model_avg.push(row);
        }

        // Target scalar per sample (average over output dims).
        let target_scalars: Vec<f64> = targets
            .iter()
            .take(n_samples)
            .map(|t| -> Result<f64> {
                let d = t.to_vec()?;
                let s = d.iter().map(|&v| v as f64).sum::<f64>()
                    / d.len().max(1) as f64;
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;

        // Step 4: gradient descent to learn per-model weights w (ridge regression).
        let lr = 0.01_f64;
        let n_iters = 100_usize;
        let mut w: Vec<f64> = vec![1.0 / n_models as f64; n_models];

        for _ in 0..n_iters {
            let mut grad = vec![0.0_f64; n_models];
            for (pma, &t_scalar) in per_model_avg.iter().zip(target_scalars.iter()) {
                let y_hat: f64 = w.iter().zip(pma.iter()).map(|(wi, xi)| wi * xi).sum();
                let err = y_hat - t_scalar;
                for m in 0..n_models {
                    grad[m] += 2.0 * err * pma[m];
                }
            }
            for m in 0..n_models {
                // Ridge penalty gradient: 2 * alpha * w_m
                grad[m] = grad[m] / n_samples as f64 + 2.0 * alpha * w[m];
                w[m] -= lr * grad[m];
                if !w[m].is_finite() {
                    w[m] = 1.0 / n_models as f64;
                }
            }
        }

        // Step 5: clamp negatives and normalise so weights live on the probability simplex.
        for wi in &mut w {
            *wi = wi.max(0.0);
        }
        let w_sum: f64 = w.iter().sum();
        if w_sum > 1e-10 {
            for wi in &mut w {
                *wi /= w_sum;
            }
        } else {
            w = vec![1.0 / n_models as f64; n_models];
        }

        // Store normalised weights on the ensemble.
        self.weights = w.clone();

        // Step 6: build the meta-learner box with the trained weights.
        self.meta_learner = Some(Box::new(StackingMetaLearner::new(
            w,
            out_dim,
            meta_learner_config.include_original_features,
        )));

        Ok(())
    }

    fn learn_ensemble_weights(&mut self, inputs: &[Tensor], targets: &[Tensor]) -> Result<()> {
        // Simplified weight learning using validation performance
        let mut performances = vec![0.0; self.models.len()];

        for (input, target) in inputs.iter().zip(targets.iter()) {
            for (i, model) in self.models.iter().enumerate() {
                let prediction = model.forward(input)?;
                let loss = self.calculate_loss(&prediction, target)?;
                performances[i] += loss;
            }
        }

        // Convert losses to weights (lower loss = higher weight)
        let max_loss = performances.iter().fold(0.0f64, |a, &b| a.max(b));
        let weight_sum: f64 = performances
            .iter()
            .map(|&loss| max_loss - loss + 1e-8)
            .sum();

        self.weights = performances
            .iter()
            .map(|&loss| (max_loss - loss + 1e-8) / weight_sum)
            .collect();

        Ok(())
    }

    fn train_dynamic_selection(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        _selection_method: &SelectionMethod,
        validation_split: f64,
    ) -> Result<()> {
        if inputs.is_empty() || targets.is_empty() {
            return Ok(());
        }
        if inputs.len() != targets.len() {
            return Err(TorshError::ComputeError(format!(
                "Input/target length mismatch: {} vs {}",
                inputs.len(),
                targets.len()
            )));
        }

        // Take the trailing fraction of the dataset as the validation split.
        let split_ratio = validation_split.clamp(0.0, 1.0);
        let val_count = ((inputs.len() as f64) * split_ratio).ceil() as usize;
        let val_count = val_count.max(1).min(inputs.len());
        let split_at = inputs.len() - val_count;

        // Per-model performance: accumulated negative MSE over the validation slice.
        let mut performances = vec![0.0f64; self.models.len()];
        for (input, target) in inputs[split_at..].iter().zip(targets[split_at..].iter()) {
            for (i, model) in self.models.iter().enumerate() {
                let prediction = model.forward(input)?;
                let loss = self.calculate_loss(&prediction, target)?;
                performances[i] += -loss;
            }
        }

        // Convert validation scores into normalized weights so the dynamic selector
        // has a meaningful preference signal.
        let min_score = performances
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let shifted: Vec<f64> = performances
            .iter()
            .map(|&p| p - min_score + 1e-8)
            .collect();
        let weight_sum: f64 = shifted.iter().sum();
        if weight_sum > 0.0 {
            self.weights = shifted.iter().map(|&w| w / weight_sum).collect();
        }
        Ok(())
    }

    /// Mixture-of-Experts gating: learn a linear gating matrix W ∈ ℝ^{M×D} such that
    /// gate_n = softmax(W · x_n) gives per-sample expert weights that minimise MSE on
    /// training data.  Training uses gradient descent (lr = 0.01, 100 iterations).
    fn train_mixture_of_experts(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        _gating_network: &GatingNetworkConfig,
    ) -> Result<()> {
        let n_models = self.models.len();
        if n_models == 0 || inputs.is_empty() || targets.is_empty() {
            return Ok(());
        }
        if inputs.len() != targets.len() {
            return Err(TorshError::ComputeError(format!(
                "MoE train: input/target length mismatch {} vs {}",
                inputs.len(),
                targets.len()
            )));
        }

        // Determine input feature dimension from the first sample.
        let in_dim = inputs[0].numel();
        if in_dim == 0 {
            return Ok(());
        }

        // Initialise gating weight matrix W[m][d] = 0 (produces uniform softmax).
        let mut gate_w: Vec<Vec<f64>> = vec![vec![0.0_f64; in_dim]; n_models];

        let lr = 0.01_f64;
        let n_iters = 100_usize;
        let n_samples = inputs.len();

        // Pre-fetch all input / target vecs to avoid repeated tensor extraction in loop.
        let input_vecs: Vec<Vec<f64>> = inputs
            .iter()
            .map(|t| -> Result<Vec<f64>> {
                Ok(t.to_vec()?.iter().map(|&v| v as f64).collect())
            })
            .collect::<Result<Vec<_>>>()?;

        // Collect per-expert, per-sample predictions as average scalar.
        let mut expert_preds: Vec<Vec<f64>> = vec![vec![0.0_f64; n_samples]; n_models];
        for m in 0..n_models {
            for (s, input) in inputs.iter().enumerate() {
                let pred = self.models[m].forward(input)?;
                let data = pred.to_vec()?;
                let avg = data.iter().map(|&v| v as f64).sum::<f64>()
                    / data.len().max(1) as f64;
                expert_preds[m][s] = avg;
            }
        }

        let target_scalars: Vec<f64> = targets
            .iter()
            .map(|t| -> Result<f64> {
                let d = t.to_vec()?;
                Ok(d.iter().map(|&v| v as f64).sum::<f64>() / d.len().max(1) as f64)
            })
            .collect::<Result<Vec<_>>>()?;

        // Gradient descent on the gating matrix.
        for _ in 0..n_iters {
            // Accumulate gradients across samples.
            let mut dw: Vec<Vec<f64>> = vec![vec![0.0_f64; in_dim]; n_models];

            for s in 0..n_samples {
                let x = &input_vecs[s];

                // Compute logits: logit[m] = dot(W[m], x).
                let logits: Vec<f64> = gate_w
                    .iter()
                    .map(|row| row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>())
                    .collect();

                // Softmax over logits.
                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
                let exp_sum: f64 = exps.iter().sum();
                let gates: Vec<f64> = if exp_sum > f64::EPSILON {
                    exps.iter().map(|e| e / exp_sum).collect()
                } else {
                    vec![1.0 / n_models as f64; n_models]
                };

                // Ensemble output: ŷ = Σ_m gate_m * expert_m(x).
                let y_hat: f64 = gates
                    .iter()
                    .zip(expert_preds.iter())
                    .map(|(g, ep)| g * ep[s])
                    .sum();
                let err = y_hat - target_scalars[s]; // residual

                // Gradient of MSE loss w.r.t. gate_m:
                //   ∂L/∂g_m = 2 * err * expert_m
                // Gradient of softmax gating w.r.t. logit_m (chain rule):
                //   ∂g_m/∂logit_k = g_m*(δ_{mk} - g_k)
                // Combined: ∂L/∂logit_k = 2*err * Σ_m expert_m * g_m*(δ_{mk}-g_k)
                //                        = 2*err * (expert_k*g_k - g_k * Σ_m expert_m*g_m)
                //                        = 2*err * g_k * (expert_k - ŷ)
                for k in 0..n_models {
                    let d_logit_k = 2.0 * err * gates[k] * (expert_preds[k][s] - y_hat);
                    // ∂logit_k/∂W[k][d] = x[d]
                    for d in 0..in_dim {
                        dw[k][d] += d_logit_k * x[d];
                    }
                }
            }

            // Apply gradient step.
            for m in 0..n_models {
                for d in 0..in_dim {
                    let g = dw[m][d] / n_samples as f64;
                    gate_w[m][d] -= lr * g;
                    if !gate_w[m][d].is_finite() {
                        gate_w[m][d] = 0.0;
                    }
                }
            }
        }

        // Store the learned gating matrix for use at inference time.
        self.gating_matrix = Some(gate_w.clone());

        // Compute final per-sample average gate weights and store as ensemble weights.
        // We evaluate the gating on all training inputs and average.
        let mut avg_gates = vec![0.0_f64; n_models];
        for x in &input_vecs {
            let logits: Vec<f64> = gate_w
                .iter()
                .map(|row| row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>())
                .collect();
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let exp_sum: f64 = exps.iter().sum();
            if exp_sum > f64::EPSILON {
                for m in 0..n_models {
                    avg_gates[m] += exps[m] / exp_sum;
                }
            } else {
                for m in 0..n_models {
                    avg_gates[m] += 1.0 / n_models as f64;
                }
            }
        }
        let gate_sum: f64 = avg_gates.iter().sum();
        if gate_sum > f64::EPSILON {
            for m in 0..n_models {
                avg_gates[m] /= gate_sum;
            }
        }
        self.weights = avg_gates;

        Ok(())
    }

    // Helper methods
    fn argmax(&self, tensor: &Tensor) -> Result<usize> {
        let float_data = tensor.to_vec()?;

        let (argmax, _) = float_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b)
                    .expect("tensor values should be comparable")
            })
            .unwrap_or((0, &0.0));

        Ok(argmax)
    }

    fn calculate_confidence(&self, prediction: &Tensor) -> Result<f64> {
        // Calculate prediction confidence (e.g., max probability for classification)
        let float_data = prediction.to_vec()?;

        let max_prob = float_data.iter().fold(0.0f32, |a, &b| a.max(b));
        Ok(max_prob as f64)
    }

    fn calculate_loss(&self, prediction: &Tensor, target: &Tensor) -> Result<f64> {
        // Simplified MSE loss calculation
        let pred_f32 = prediction.to_vec()?;
        let target_f32 = target.to_vec()?;

        let mse = pred_f32
            .iter()
            .zip(target_f32.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / pred_f32.len() as f32;

        Ok(mse as f64)
    }

    fn validate_models(&self, inputs: &[Tensor], targets: &[Tensor]) -> Result<()> {
        // Validate that all models can process the input format
        for (input, _target) in inputs.iter().zip(targets.iter()) {
            for model in &self.models {
                let _prediction = model.forward(input)?;
            }
        }
        Ok(())
    }

    fn initialize_online_adaptation(&mut self, _adaptation_config: &OnlineAdaptationConfig) {
        self.adaptation_state = Some(AdaptationState {
            performance_history: vec![vec![]; self.models.len()],
            momentum: vec![0.0; self.models.len()],
            step_count: 0,
        });
    }

    fn online_weight_update(
        &mut self,
        input: &Tensor,
        target: &Tensor,
        _prediction: &Tensor,
        adaptation_config: &OnlineAdaptationConfig,
    ) -> Result<()> {
        // Compute per-model losses up front so we can borrow `self` immutably here
        // before mutating `self.weights` / `self.adaptation_state` below.
        let mut per_model_losses = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let prediction = model.forward(input)?;
            let loss = self.calculate_loss(&prediction, target)?;
            per_model_losses.push(loss);
        }

        let lr = adaptation_config.learning_rate;
        let forgetting = adaptation_config.forgetting_factor;
        let min_weight = adaptation_config.min_weight;

        // Convert per-model losses into reward signals: lower loss => larger reward.
        let max_loss = per_model_losses
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        let rewards: Vec<f64> = per_model_losses
            .iter()
            .map(|&loss| (max_loss - loss).max(0.0))
            .collect();
        let reward_sum: f64 = rewards.iter().sum();
        let normalized_rewards: Vec<f64> = if reward_sum > 0.0 {
            rewards.iter().map(|r| r / reward_sum).collect()
        } else {
            // All models tied: use uniform reward to avoid division by zero.
            vec![1.0 / self.models.len() as f64; self.models.len()]
        };

        if let Some(state) = &mut self.adaptation_state {
            state.step_count += 1;

            // Track rolling per-model performance history (bounded window).
            for (i, &loss) in per_model_losses.iter().enumerate() {
                if i < state.performance_history.len() {
                    state.performance_history[i].push(loss);
                    let window = adaptation_config.window_size.max(1);
                    let history = &mut state.performance_history[i];
                    if history.len() > window {
                        let drop = history.len() - window;
                        history.drain(0..drop);
                    }
                }
            }

            // Update weights using a momentum-style EMA combining the previous weight,
            // the forgetting factor, and the gradient towards the per-step reward target.
            for (i, weight) in self.weights.iter_mut().enumerate() {
                let target_weight = normalized_rewards[i];
                let momentum = state.momentum.get_mut(i);
                let delta = lr * (target_weight - *weight);
                if let Some(m) = momentum {
                    *m = forgetting * *m + delta;
                    *weight += *m;
                } else {
                    *weight += delta;
                }
                if !weight.is_finite() {
                    *weight = min_weight.max(0.0);
                }
            }
        } else {
            // No adaptation state yet — fall back to a simple convex combination.
            for (weight, &target_weight) in
                self.weights.iter_mut().zip(normalized_rewards.iter())
            {
                *weight = forgetting * *weight + (1.0 - forgetting) * target_weight;
            }
        }

        // Clamp to the floor and renormalise so weights stay a probability simplex.
        for weight in &mut self.weights {
            *weight = weight.max(min_weight).max(0.0);
        }
        let weight_sum: f64 = self.weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut self.weights {
                *weight /= weight_sum;
            }
        }

        Ok(())
    }

    // ----- Diversity calculation helpers -----

    /// Helper: collapse a (sample, model) prediction grid into per-(sample, model)
    /// flat-argmax class labels. Returns `None` if the grid is empty or jagged.
    fn predictions_to_labels(
        &self,
        all_predictions: &[Vec<Tensor>],
    ) -> Result<Option<Vec<Vec<usize>>>> {
        if all_predictions.is_empty() {
            return Ok(None);
        }
        let num_models = all_predictions[0].len();
        if num_models == 0 {
            return Ok(None);
        }

        let mut labels: Vec<Vec<usize>> = Vec::with_capacity(all_predictions.len());
        for sample_preds in all_predictions {
            if sample_preds.len() != num_models {
                return Err(TorshError::ComputeError(
                    "Inconsistent number of model predictions across samples".to_string(),
                ));
            }
            let mut row = Vec::with_capacity(num_models);
            for prediction in sample_preds {
                row.push(self.argmax(prediction)?);
            }
            labels.push(row);
        }
        Ok(Some(labels))
    }

    /// Pairwise disagreement: average over all model pairs and samples of the indicator
    /// that the two models predict different classes. Bounded in `[0.0, 1.0]`.
    fn calculate_disagreement(&self, all_predictions: &[Vec<Tensor>]) -> Result<f64> {
        let labels = match self.predictions_to_labels(all_predictions)? {
            Some(l) => l,
            None => return Ok(0.0),
        };
        let num_models = labels[0].len();
        if num_models < 2 {
            return Ok(0.0);
        }

        let pair_count = num_models * (num_models - 1) / 2;
        let mut disagreement = 0.0;
        for row in &labels {
            for i in 0..num_models {
                for j in (i + 1)..num_models {
                    if row[i] != row[j] {
                        disagreement += 1.0;
                    }
                }
            }
        }
        let total = labels.len() as f64 * pair_count as f64;
        Ok(if total > 0.0 { disagreement / total } else { 0.0 })
    }

    /// Average Pearson correlation of raw output vectors across model pairs.
    fn calculate_correlation(&self, all_predictions: &[Vec<Tensor>]) -> Result<f64> {
        if all_predictions.is_empty() {
            return Ok(0.0);
        }
        let num_models = all_predictions[0].len();
        if num_models < 2 {
            return Ok(0.0);
        }

        // Stack each model's outputs across all samples into a single flat vector.
        let mut model_vectors: Vec<Vec<f32>> = vec![Vec::new(); num_models];
        for sample_preds in all_predictions {
            for (idx, prediction) in sample_preds.iter().enumerate() {
                let data = prediction.to_vec()?;
                model_vectors[idx].extend_from_slice(&data);
            }
        }

        let mut total = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_models {
            for j in (i + 1)..num_models {
                let r = pearson_correlation(&model_vectors[i], &model_vectors[j]);
                total += r;
                count += 1;
            }
        }
        Ok(if count > 0 { total / count as f64 } else { 0.0 })
    }

    /// Per-sample classification entropy averaged over the dataset (Cunningham &
    /// Carney definition): `-(p*log p + (1-p)*log(1-p))` averaged over samples,
    /// where `p` is the fraction of models that pick the modal class.
    fn calculate_entropy(&self, all_predictions: &[Vec<Tensor>]) -> Result<f64> {
        let labels = match self.predictions_to_labels(all_predictions)? {
            Some(l) => l,
            None => return Ok(0.0),
        };
        let num_models = labels[0].len();
        if num_models == 0 {
            return Ok(0.0);
        }
        let mut total = 0.0f64;
        for row in &labels {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &lbl in row {
                *counts.entry(lbl).or_insert(0) += 1;
            }
            let modal = counts.values().copied().max().unwrap_or(0) as f64;
            let p = modal / num_models as f64;
            let q = 1.0 - p;
            let term = if p > 0.0 && q > 0.0 {
                -(p * p.ln() + q * q.ln())
            } else {
                0.0
            };
            total += term;
        }
        Ok(total / labels.len() as f64)
    }

    /// Yule's Q-statistic averaged over model pairs. Requires binary correctness,
    /// which we approximate by comparing each model's argmax to the overall majority
    /// vote. `Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)`, range `[-1, 1]`.
    fn calculate_q_statistic(&self, all_predictions: &[Vec<Tensor>]) -> Result<f64> {
        let labels = match self.predictions_to_labels(all_predictions)? {
            Some(l) => l,
            None => return Ok(0.0),
        };
        let num_models = labels[0].len();
        if num_models < 2 {
            return Ok(0.0);
        }

        // Treat the per-sample majority vote as ground truth and convert each
        // model's prediction into a 0/1 correctness indicator.
        let mut correct: Vec<Vec<u8>> = Vec::with_capacity(labels.len());
        for row in &labels {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &lbl in row {
                *counts.entry(lbl).or_insert(0) += 1;
            }
            let majority = counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class)
                .unwrap_or(0);
            correct.push(row.iter().map(|&l| (l == majority) as u8).collect());
        }

        let mut total_q = 0.0f64;
        let mut pair_count = 0usize;
        for i in 0..num_models {
            for j in (i + 1)..num_models {
                let mut n11 = 0.0f64;
                let mut n10 = 0.0f64;
                let mut n01 = 0.0f64;
                let mut n00 = 0.0f64;
                for row in &correct {
                    match (row[i], row[j]) {
                        (1, 1) => n11 += 1.0,
                        (1, 0) => n10 += 1.0,
                        (0, 1) => n01 += 1.0,
                        _ => n00 += 1.0,
                    }
                }
                let numerator = n11 * n00 - n01 * n10;
                let denominator = n11 * n00 + n01 * n10;
                if denominator.abs() > f64::EPSILON {
                    total_q += numerator / denominator;
                    pair_count += 1;
                }
            }
        }
        Ok(if pair_count > 0 {
            total_q / pair_count as f64
        } else {
            0.0
        })
    }

    /// Average Cohen's kappa across model pairs (label agreement corrected for
    /// chance). Returns 0 when no agreement signal is available.
    fn calculate_kappa_statistic(&self, all_predictions: &[Vec<Tensor>]) -> Result<f64> {
        let labels = match self.predictions_to_labels(all_predictions)? {
            Some(l) => l,
            None => return Ok(0.0),
        };
        let num_models = labels[0].len();
        if num_models < 2 || labels.is_empty() {
            return Ok(0.0);
        }

        let n = labels.len() as f64;
        let mut total_kappa = 0.0f64;
        let mut pair_count = 0usize;
        for i in 0..num_models {
            for j in (i + 1)..num_models {
                let mut po = 0.0f64;
                let mut counts_i: HashMap<usize, f64> = HashMap::new();
                let mut counts_j: HashMap<usize, f64> = HashMap::new();
                for row in &labels {
                    if row[i] == row[j] {
                        po += 1.0;
                    }
                    *counts_i.entry(row[i]).or_insert(0.0) += 1.0;
                    *counts_j.entry(row[j]).or_insert(0.0) += 1.0;
                }
                po /= n;
                let mut pe = 0.0f64;
                for (cls, ci) in &counts_i {
                    if let Some(cj) = counts_j.get(cls) {
                        pe += (ci / n) * (cj / n);
                    }
                }
                let kappa = if (1.0 - pe).abs() > f64::EPSILON {
                    (po - pe) / (1.0 - pe)
                } else {
                    0.0
                };
                total_kappa += kappa;
                pair_count += 1;
            }
        }
        Ok(if pair_count > 0 {
            total_kappa / pair_count as f64
        } else {
            0.0
        })
    }

    /// Build cross-validation meta-features for stacking. With `k` folds, every
    /// example becomes a meta-feature vector formed by concatenating each base
    /// model's prediction on that example. The fold structure is preserved so the
    /// resulting features are aligned with the original input order.
    fn generate_cv_meta_features(
        &self,
        inputs: &[Tensor],
        targets: &[Tensor],
        k: usize,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        if inputs.len() != targets.len() {
            return Err(TorshError::ComputeError(format!(
                "CV meta-feature input/target length mismatch: {} vs {}",
                inputs.len(),
                targets.len()
            )));
        }
        let folds = k.max(1).min(inputs.len());
        let fold_size = inputs.len().div_ceil(folds);

        let mut meta_features = Vec::with_capacity(inputs.len());
        for fold_idx in 0..folds {
            let start = fold_idx * fold_size;
            let end = ((fold_idx + 1) * fold_size).min(inputs.len());
            for input in &inputs[start..end] {
                let prediction = self.concatenate_base_predictions(input)?;
                meta_features.push(prediction);
            }
        }
        Ok(meta_features)
    }

    /// Build hold-out meta-features for stacking. The trailing `split_ratio`
    /// fraction of the dataset becomes the hold-out, and we record concatenated
    /// base-model predictions for that slice.
    fn generate_holdout_meta_features(
        &self,
        inputs: &[Tensor],
        _targets: &[Tensor],
        split_ratio: f64,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let split_ratio = split_ratio.clamp(0.0, 1.0);
        let holdout_count = ((inputs.len() as f64) * split_ratio).ceil() as usize;
        let holdout_count = holdout_count.max(1).min(inputs.len());
        let split_at = inputs.len() - holdout_count;

        let mut meta_features = Vec::with_capacity(holdout_count);
        for input in &inputs[split_at..] {
            meta_features.push(self.concatenate_base_predictions(input)?);
        }
        Ok(meta_features)
    }

    /// Concatenate the per-base-model predictions for a single input into a flat
    /// meta-feature tensor (1 x sum(C_i)). Used by both CV and hold-out meta-feature
    /// generation.
    fn concatenate_base_predictions(&self, input: &Tensor) -> Result<Tensor> {
        let predictions = self.get_individual_predictions(input)?;
        if predictions.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot build meta-features from an empty ensemble".to_string(),
            ));
        }
        let device = predictions[0].device();
        let mut data: Vec<f32> = Vec::new();
        for pred in &predictions {
            data.extend(pred.to_vec()?);
        }
        let len = data.len();
        Tensor::from_data(data, vec![1, len], device)
    }

    /// Factory: construct a meta-learner module suitable for stacking.
    ///
    /// For `RidgeRegression` and `LinearRegression`, returns a
    /// [`StackingMetaLearner`] initialised with uniform per-model weights.
    /// For other learner types a [`SimpleMeta`] pass-through is returned as a
    /// safe fallback — the stacking training loop replaces it with a trained
    /// `StackingMetaLearner` anyway.
    pub fn create_meta_learner(&self, config: &MetaLearnerConfig) -> Result<Box<dyn Module>> {
        let n_models = self.models.len();
        match &config.learner_type {
            MetaLearnerType::LinearRegression
            | MetaLearnerType::LogisticRegression
            | MetaLearnerType::RidgeRegression { .. } => {
                // Default: uniform weights, output_dim=1 (scalar combination).
                let weights = vec![1.0 / n_models.max(1) as f64; n_models];
                Ok(Box::new(StackingMetaLearner::new(
                    weights,
                    1,
                    config.include_original_features,
                )))
            }
            _ => Ok(Box::new(SimpleMeta::new())),
        }
    }

    /// Get current ensemble statistics
    pub fn get_stats(&self) -> Option<&EnsembleStats> {
        self.stats.as_ref()
    }

    /// Get current ensemble weights
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    /// Set ensemble weights manually
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        if weights.len() != self.models.len() {
            return Err(TorshError::ComputeError(
                "Weight count must match model count".to_string(),
            ));
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum.abs() < 1e-8 {
            return Err(TorshError::ComputeError(
                "Weights must sum to a non-zero value".to_string(),
            ));
        }

        // Normalize weights
        self.weights = weights.iter().map(|w| w / weight_sum).collect();
        Ok(())
    }
}

/// Pearson correlation coefficient between two equal-length f32 series.
/// Returns `0.0` if the inputs are empty, mismatched, or have zero variance.
fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        num += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    if denom > f64::EPSILON {
        num / denom
    } else {
        0.0
    }
}

/// Meta-learner for stacking that applies learned per-model scalar weights to the
/// concatenated base-model output vector.
///
/// Given an input of shape `[1, n_models * out_dim]` it computes:
///   output[j] = Σ_m  weights[m] * input[m * out_dim + j]   for j in 0..out_dim
struct StackingMetaLearner {
    /// Normalised per-model weights (sums to 1).
    weights: Vec<f64>,
    /// Output dimension expected per model slot.
    out_dim: usize,
    /// Whether the input also contains original features appended after the model predictions.
    include_original: bool,
}

impl StackingMetaLearner {
    fn new(weights: Vec<f64>, out_dim: usize, include_original: bool) -> Self {
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

// Simplified pass-through meta-learner (used as fallback for non-linear learner types)
struct SimpleMeta;

impl SimpleMeta {
    fn new() -> Self {
        Self
    }
}

impl Module for SimpleMeta {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified meta-learner: just return input
        Ok(input.clone())
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

// Tests are in a separate file to keep this file under the 2000-line limit.
#[cfg(test)]
#[path = "ensembling_tests.rs"]
mod tests;

