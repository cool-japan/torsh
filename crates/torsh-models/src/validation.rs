//! Model validation and accuracy testing utilities
//!
//! This module provides comprehensive model validation tools including:
//! - Accuracy testing against ground truth datasets
//! - Model performance validation and regression testing
//! - Cross-validation utilities
//! - Statistical significance testing
//! - Model correctness verification

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Configuration for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation strategy to use
    pub strategy: ValidationStrategy,
    /// Metrics to compute
    pub metrics: Vec<ValidationMetric>,
    /// Cross-validation configuration
    pub cross_validation: Option<CrossValidationConfig>,
    /// Statistical testing configuration
    pub statistical_tests: Option<StatisticalTestConfig>,
    /// Tolerance settings for numerical comparisons
    pub tolerance: ToleranceConfig,
    /// Whether to save detailed results
    pub save_detailed_results: bool,
}

/// Validation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// Single holdout validation
    Holdout { test_ratio: f64, stratified: bool },
    /// K-fold cross-validation
    KFold {
        k: usize,
        shuffle: bool,
        stratified: bool,
    },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series split validation
    TimeSeriesSplit {
        n_splits: usize,
        test_size: Option<usize>,
    },
    /// Bootstrap validation
    Bootstrap {
        n_bootstrap: usize,
        sample_ratio: f64,
    },
    /// Custom validation splits
    Custom { splits: Vec<ValidationSplit> },
}

/// Individual validation split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub name: Option<String>,
}

/// Validation metrics to compute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMetric {
    /// Classification accuracy
    Accuracy,
    /// Top-K accuracy
    TopKAccuracy { k: usize },
    /// Precision (per-class and macro/micro)
    Precision { average: AverageType },
    /// Recall (per-class and macro/micro)
    Recall { average: AverageType },
    /// F1 score
    F1Score { average: AverageType },
    /// Area under ROC curve
    AucRoc,
    /// Area under Precision-Recall curve
    AucPr,
    /// Mean Squared Error (for regression)
    MeanSquaredError,
    /// Root Mean Squared Error
    RootMeanSquaredError,
    /// Mean Absolute Error
    MeanAbsoluteError,
    /// R-squared coefficient
    RSquared,
    /// Cross-entropy loss
    CrossEntropyLoss,
    /// Custom metric
    Custom { name: String },
}

/// Averaging types for classification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AverageType {
    /// Macro averaging (unweighted mean)
    Macro,
    /// Micro averaging (global)
    Micro,
    /// Weighted averaging
    Weighted,
    /// Per-class metrics
    None,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Whether to stratify splits
    pub stratified: bool,
}

/// Statistical testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestConfig {
    /// Significance level (e.g., 0.05)
    pub alpha: f64,
    /// Tests to perform
    pub tests: Vec<StatisticalTest>,
    /// Baseline model for comparison
    pub baseline_results: Option<ValidationResults>,
}

/// Statistical tests for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    /// Paired t-test
    PairedTTest,
    /// Wilcoxon signed-rank test
    WilcoxonSignedRank,
    /// McNemar's test (for classification)
    McNemarTest,
    /// Friedman test (for multiple models)
    FriedmanTest,
}

/// Tolerance configuration for numerical comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// Absolute tolerance
    pub absolute_tolerance: f64,
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Whether to use relative tolerance
    pub use_relative: bool,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-5,
            use_relative: true,
        }
    }
}

/// Validation results container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Metric scores for each split
    pub split_results: Vec<SplitResult>,
    /// Aggregated metrics across all splits
    pub aggregated_metrics: HashMap<String, MetricSummary>,
    /// Statistical test results (if performed)
    pub statistical_tests: Option<StatisticalTestResults>,
    /// Detailed predictions (if saved)
    pub detailed_results: Option<DetailedResults>,
    /// Validation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Results for a single validation split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitResult {
    /// Split identifier
    pub split_id: usize,
    /// Split name (if available)
    pub split_name: Option<String>,
    /// Metric scores for this split
    pub metrics: HashMap<String, f64>,
    /// Number of samples in test set
    pub test_size: usize,
    /// Number of samples in train set
    pub train_size: usize,
}

/// Summary statistics for a metric across splits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
    /// Individual values from each split
    pub values: Vec<f64>,
}

/// Statistical test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResults {
    /// Test results by metric
    pub test_results: HashMap<String, TestResult>,
    /// Overall conclusion
    pub conclusion: String,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Is significant at alpha level
    pub is_significant: bool,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
}

/// Detailed validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedResults {
    /// Predictions for each test sample
    pub predictions: Vec<PredictionResult>,
    /// Confusion matrix (for classification)
    pub confusion_matrix: Option<ConfusionMatrix>,
    /// Per-class metrics (for classification)
    pub per_class_metrics: Option<HashMap<usize, ClassMetrics>>,
}

/// Individual prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Sample index
    pub sample_id: usize,
    /// True label/value
    pub true_value: f64,
    /// Predicted label/value
    pub predicted_value: f64,
    /// Prediction confidence (if available)
    pub confidence: Option<f64>,
    /// Split this sample was tested in
    pub split_id: usize,
}

/// Confusion matrix for classification tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Matrix data (true_class, predicted_class) -> count
    pub matrix: HashMap<(usize, usize), usize>,
    /// Class labels
    pub class_labels: Vec<String>,
    /// Number of classes
    pub num_classes: usize,
}

/// Per-class metrics for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    /// Class label
    pub class_label: usize,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1-score
    pub f1_score: f64,
    /// Support (number of samples)
    pub support: usize,
}

/// Main model validator
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    /// Create a new model validator
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a model on the given dataset
    pub fn validate<M: Module>(
        &self,
        model: &M,
        dataset: &[(Tensor, Tensor)], // (input, target) pairs
        task_type: TaskType,
    ) -> Result<ValidationResults> {
        let splits = self.generate_splits(dataset)?;
        let mut split_results = Vec::new();
        let mut detailed_predictions = Vec::new();

        for (split_id, split) in splits.iter().enumerate() {
            let split_result = self.evaluate_split(
                model,
                dataset,
                split,
                split_id,
                &task_type,
                &mut detailed_predictions,
            )?;
            split_results.push(split_result);
        }

        let aggregated_metrics = self.aggregate_metrics(&split_results)?;
        let statistical_tests = self.perform_statistical_tests(&split_results)?;

        let detailed_results = if self.config.save_detailed_results {
            Some(self.create_detailed_results(detailed_predictions, &task_type)?)
        } else {
            None
        };

        Ok(ValidationResults {
            split_results,
            aggregated_metrics,
            statistical_tests,
            detailed_results,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Generate validation splits based on strategy
    fn generate_splits(&self, dataset: &[(Tensor, Tensor)]) -> Result<Vec<ValidationSplit>> {
        match &self.config.strategy {
            ValidationStrategy::Holdout {
                test_ratio,
                stratified,
            } => self.generate_holdout_split(dataset, *test_ratio, *stratified),
            ValidationStrategy::KFold {
                k,
                shuffle,
                stratified,
            } => self.generate_kfold_splits(dataset, *k, *shuffle, *stratified),
            ValidationStrategy::LeaveOneOut => self.generate_loo_splits(dataset),
            ValidationStrategy::TimeSeriesSplit {
                n_splits,
                test_size,
            } => self.generate_time_series_splits(dataset, *n_splits, *test_size),
            ValidationStrategy::Bootstrap {
                n_bootstrap,
                sample_ratio,
            } => self.generate_bootstrap_splits(dataset, *n_bootstrap, *sample_ratio),
            ValidationStrategy::Custom { splits } => Ok(splits.clone()),
        }
    }

    /// Generate holdout split
    fn generate_holdout_split(
        &self,
        dataset: &[(Tensor, Tensor)],
        test_ratio: f64,
        _stratified: bool,
    ) -> Result<Vec<ValidationSplit>> {
        use scirs2_core::random::Random;

        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        let mut rng = Random::seed(42);
        // Fisher-Yates shuffle algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let test_size = (dataset.len() as f64 * test_ratio) as usize;
        let train_indices = indices[test_size..].to_vec();
        let test_indices = indices[..test_size].to_vec();

        Ok(vec![ValidationSplit {
            train_indices,
            test_indices,
            name: Some("holdout".to_string()),
        }])
    }

    /// Generate K-fold splits
    fn generate_kfold_splits(
        &self,
        dataset: &[(Tensor, Tensor)],
        k: usize,
        shuffle: bool,
        _stratified: bool,
    ) -> Result<Vec<ValidationSplit>> {
        use scirs2_core::random::Random;

        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = Random::seed(42);
            // Fisher-Yates shuffle algorithm
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        let fold_size = dataset.len() / k;
        let mut splits = Vec::new();

        for fold_idx in 0..k {
            let start = fold_idx * fold_size;
            let end = if fold_idx == k - 1 {
                dataset.len() // Include remaining samples in last fold
            } else {
                start + fold_size
            };

            let test_indices = indices[start..end].to_vec();
            let train_indices = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            splits.push(ValidationSplit {
                train_indices,
                test_indices,
                name: Some(format!("fold_{}", fold_idx)),
            });
        }

        Ok(splits)
    }

    /// Generate leave-one-out splits
    fn generate_loo_splits(&self, dataset: &[(Tensor, Tensor)]) -> Result<Vec<ValidationSplit>> {
        let mut splits = Vec::new();

        for i in 0..dataset.len() {
            let test_indices = vec![i];
            let train_indices = (0..dataset.len()).filter(|&idx| idx != i).collect();

            splits.push(ValidationSplit {
                train_indices,
                test_indices,
                name: Some(format!("loo_{}", i)),
            });
        }

        Ok(splits)
    }

    /// Generate time series splits
    fn generate_time_series_splits(
        &self,
        dataset: &[(Tensor, Tensor)],
        n_splits: usize,
        test_size: Option<usize>,
    ) -> Result<Vec<ValidationSplit>> {
        let test_size = test_size.unwrap_or(dataset.len() / (n_splits + 1));
        let mut splits = Vec::new();

        for i in 0..n_splits {
            let test_start = (i + 1) * (dataset.len() - test_size) / n_splits;
            let test_end = test_start + test_size;

            if test_end > dataset.len() {
                break;
            }

            let train_indices = (0..test_start).collect();
            let test_indices = (test_start..test_end).collect();

            splits.push(ValidationSplit {
                train_indices,
                test_indices,
                name: Some(format!("time_split_{}", i)),
            });
        }

        Ok(splits)
    }

    /// Generate bootstrap splits
    fn generate_bootstrap_splits(
        &self,
        dataset: &[(Tensor, Tensor)],
        n_bootstrap: usize,
        sample_ratio: f64,
    ) -> Result<Vec<ValidationSplit>> {
        use scirs2_core::random::Random;

        let mut splits = Vec::new();
        let sample_size = (dataset.len() as f64 * sample_ratio) as usize;

        for i in 0..n_bootstrap {
            let mut rng = Random::seed(42);
            let all_indices: Vec<usize> = (0..dataset.len()).collect();

            // Bootstrap sample for training
            let train_indices: Vec<usize> = (0..sample_size)
                .map(|_| {
                    let idx = rng.gen_range(0..all_indices.len());
                    all_indices[idx]
                })
                .collect();

            // Out-of-bag samples for testing
            let test_indices: Vec<usize> = all_indices
                .iter()
                .filter(|&&idx| !train_indices.contains(&idx))
                .cloned()
                .collect();

            if !test_indices.is_empty() {
                splits.push(ValidationSplit {
                    train_indices,
                    test_indices,
                    name: Some(format!("bootstrap_{}", i)),
                });
            }
        }

        Ok(splits)
    }

    /// Evaluate a single split
    fn evaluate_split<M: Module>(
        &self,
        model: &M,
        dataset: &[(Tensor, Tensor)],
        split: &ValidationSplit,
        split_id: usize,
        task_type: &TaskType,
        detailed_predictions: &mut Vec<PredictionResult>,
    ) -> Result<SplitResult> {
        let mut predictions = Vec::new();
        let mut targets = Vec::new();

        // Run inference on test set
        for &test_idx in &split.test_indices {
            let (input, target) = &dataset[test_idx];
            let prediction = model.forward(input)?;

            predictions.push(prediction.clone());
            targets.push(target.clone());

            // Save detailed prediction if requested
            if self.config.save_detailed_results {
                let pred_result = self.create_prediction_result(
                    test_idx,
                    target,
                    &prediction,
                    split_id,
                    task_type,
                )?;
                detailed_predictions.push(pred_result);
            }
        }

        // Compute metrics
        let metrics = self.compute_metrics(&predictions, &targets, task_type)?;

        Ok(SplitResult {
            split_id,
            split_name: split.name.clone(),
            metrics,
            test_size: split.test_indices.len(),
            train_size: split.train_indices.len(),
        })
    }

    /// Create prediction result for detailed tracking
    fn create_prediction_result(
        &self,
        sample_id: usize,
        target: &Tensor,
        prediction: &Tensor,
        split_id: usize,
        task_type: &TaskType,
    ) -> Result<PredictionResult> {
        let (true_value, predicted_value, confidence) = match task_type {
            TaskType::Classification => {
                let true_label = target.argmax(Some(0))?.item()? as f64;
                let pred_probs = prediction.softmax(0)?;
                let predicted_label = pred_probs.argmax(Some(0))?.item()? as f64;
                let max_confidence = pred_probs.max(None, false)?.item()? as f64;

                (true_label, predicted_label, Some(max_confidence))
            }
            TaskType::Regression => {
                let true_val = target.item()? as f64;
                let pred_val = prediction.item()? as f64;

                (true_val, pred_val, None)
            }
        };

        Ok(PredictionResult {
            sample_id,
            true_value,
            predicted_value,
            confidence,
            split_id,
        })
    }

    /// Compute metrics for predictions
    fn compute_metrics(
        &self,
        predictions: &[Tensor],
        targets: &[Tensor],
        task_type: &TaskType,
    ) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        for metric in &self.config.metrics {
            let value = self.compute_single_metric(metric, predictions, targets, task_type)?;
            let metric_name = self.metric_name(metric);
            metrics.insert(metric_name, value);
        }

        Ok(metrics)
    }

    /// Compute a single metric
    fn compute_single_metric(
        &self,
        metric: &ValidationMetric,
        predictions: &[Tensor],
        targets: &[Tensor],
        task_type: &TaskType,
    ) -> Result<f64> {
        match (metric, task_type) {
            (ValidationMetric::Accuracy, TaskType::Classification) => {
                self.compute_accuracy(predictions, targets)
            }
            (ValidationMetric::TopKAccuracy { k }, TaskType::Classification) => {
                self.compute_top_k_accuracy(predictions, targets, *k)
            }
            (ValidationMetric::MeanSquaredError, TaskType::Regression) => {
                self.compute_mse(predictions, targets)
            }
            (ValidationMetric::RootMeanSquaredError, TaskType::Regression) => {
                self.compute_rmse(predictions, targets)
            }
            (ValidationMetric::MeanAbsoluteError, TaskType::Regression) => {
                self.compute_mae(predictions, targets)
            }
            (ValidationMetric::CrossEntropyLoss, TaskType::Classification) => {
                self.compute_cross_entropy_loss(predictions, targets)
            }
            _ => {
                // Return placeholder for unsupported metric/task combinations
                Ok(0.0)
            }
        }
    }

    /// Compute classification accuracy
    fn compute_accuracy(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mut correct = 0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = pred.argmax(Some(0))?.item()?;
            let true_class = target.argmax(Some(0))?.item()?;

            if pred_class == true_class {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    /// Compute top-k accuracy
    fn compute_top_k_accuracy(
        &self,
        predictions: &[Tensor],
        targets: &[Tensor],
        k: usize,
    ) -> Result<f64> {
        let mut correct = 0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let true_class = target.argmax(Some(0))?.item()?;
            let top_k_indices = self.get_top_k_indices(pred, k)?;

            if top_k_indices.contains(&(true_class as usize)) {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    /// Get top-k indices from tensor
    fn get_top_k_indices(&self, tensor: &Tensor, k: usize) -> Result<Vec<usize>> {
        let data = tensor.to_vec()?;
        let mut indexed_data: Vec<(usize, f32)> =
            data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(indexed_data.iter().take(k).map(|(i, _)| *i).collect())
    }

    /// Compute mean squared error
    fn compute_mse(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mut total_error = 0.0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_val = pred.item()? as f64;
            let true_val = target.item()? as f64;
            let error = pred_val - true_val;
            total_error += error * error;
        }

        Ok(total_error / total as f64)
    }

    /// Compute root mean squared error
    fn compute_rmse(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mse = self.compute_mse(predictions, targets)?;
        Ok(mse.sqrt())
    }

    /// Compute mean absolute error
    fn compute_mae(&self, predictions: &[Tensor], targets: &[Tensor]) -> Result<f64> {
        let mut total_error = 0.0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_val = pred.item()? as f64;
            let true_val = target.item()? as f64;
            total_error += (pred_val - true_val).abs();
        }

        Ok(total_error / total as f64)
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(
        &self,
        predictions: &[Tensor],
        targets: &[Tensor],
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let softmax_pred = pred.softmax(0)?;
            let true_class = target.argmax(Some(0))?.item()? as usize;
            let predicted_prob = softmax_pred.get(&[true_class])?;
            total_loss += -predicted_prob.ln() as f64;
        }

        Ok(total_loss / total as f64)
    }

    /// Get metric name as string
    fn metric_name(&self, metric: &ValidationMetric) -> String {
        match metric {
            ValidationMetric::Accuracy => "accuracy".to_string(),
            ValidationMetric::TopKAccuracy { k } => format!("top_{}_accuracy", k),
            ValidationMetric::Precision { average } => format!("precision_{:?}", average),
            ValidationMetric::Recall { average } => format!("recall_{:?}", average),
            ValidationMetric::F1Score { average } => format!("f1_score_{:?}", average),
            ValidationMetric::AucRoc => "auc_roc".to_string(),
            ValidationMetric::AucPr => "auc_pr".to_string(),
            ValidationMetric::MeanSquaredError => "mse".to_string(),
            ValidationMetric::RootMeanSquaredError => "rmse".to_string(),
            ValidationMetric::MeanAbsoluteError => "mae".to_string(),
            ValidationMetric::RSquared => "r_squared".to_string(),
            ValidationMetric::CrossEntropyLoss => "cross_entropy_loss".to_string(),
            ValidationMetric::Custom { name } => name.clone(),
        }
    }

    /// Aggregate metrics across splits
    fn aggregate_metrics(
        &self,
        split_results: &[SplitResult],
    ) -> Result<HashMap<String, MetricSummary>> {
        let mut aggregated = HashMap::new();

        // Collect all metric names
        let metric_names: std::collections::HashSet<String> = split_results
            .iter()
            .flat_map(|result| result.metrics.keys().cloned())
            .collect();

        for metric_name in metric_names {
            let values: Vec<f64> = split_results
                .iter()
                .filter_map(|result| result.metrics.get(&metric_name).copied())
                .collect();

            if !values.is_empty() {
                let summary = self.compute_metric_summary(&values)?;
                aggregated.insert(metric_name, summary);
            }
        }

        Ok(aggregated)
    }

    /// Compute summary statistics for a metric
    fn compute_metric_summary(&self, values: &[f64]) -> Result<MetricSummary> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // 95% confidence interval
        let t_value = 1.96; // Approximate for normal distribution
        let std_error = std / (values.len() as f64).sqrt();
        let margin = t_value * std_error;
        let confidence_interval = (mean - margin, mean + margin);

        Ok(MetricSummary {
            mean,
            std,
            min,
            max,
            confidence_interval,
            values: values.to_vec(),
        })
    }

    /// Perform statistical tests
    fn perform_statistical_tests(
        &self,
        _split_results: &[SplitResult],
    ) -> Result<Option<StatisticalTestResults>> {
        // Statistical tests would be implemented here
        // For now, return None
        Ok(None)
    }

    /// Create detailed results
    fn create_detailed_results(
        &self,
        predictions: Vec<PredictionResult>,
        task_type: &TaskType,
    ) -> Result<DetailedResults> {
        let confusion_matrix = match task_type {
            TaskType::Classification => Some(self.compute_confusion_matrix(&predictions)?),
            TaskType::Regression => None,
        };

        let per_class_metrics = match task_type {
            TaskType::Classification => Some(self.compute_per_class_metrics(&predictions)?),
            TaskType::Regression => None,
        };

        Ok(DetailedResults {
            predictions,
            confusion_matrix,
            per_class_metrics,
        })
    }

    /// Compute confusion matrix
    fn compute_confusion_matrix(
        &self,
        predictions: &[PredictionResult],
    ) -> Result<ConfusionMatrix> {
        let mut matrix = HashMap::new();
        let mut classes = std::collections::HashSet::new();

        for pred in predictions {
            let true_class = pred.true_value as usize;
            let pred_class = pred.predicted_value as usize;

            classes.insert(true_class);
            classes.insert(pred_class);

            *matrix.entry((true_class, pred_class)).or_insert(0) += 1;
        }

        let mut class_labels: Vec<_> = classes.into_iter().collect();
        class_labels.sort();
        let class_labels: Vec<String> = class_labels.iter().map(|&c| c.to_string()).collect();

        let num_classes = class_labels.len();
        Ok(ConfusionMatrix {
            matrix,
            class_labels,
            num_classes,
        })
    }

    /// Compute per-class metrics
    fn compute_per_class_metrics(
        &self,
        predictions: &[PredictionResult],
    ) -> Result<HashMap<usize, ClassMetrics>> {
        let mut class_metrics = HashMap::new();

        // Group predictions by class
        let mut class_predictions: HashMap<usize, Vec<&PredictionResult>> = HashMap::new();
        for pred in predictions {
            let true_class = pred.true_value as usize;
            class_predictions
                .entry(true_class)
                .or_insert_with(Vec::new)
                .push(pred);
        }

        // Compute metrics for each class
        for (&class_label, class_preds) in &class_predictions {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;

            for pred in class_preds {
                let predicted_class = pred.predicted_value as usize;
                if predicted_class == class_label {
                    tp += 1;
                } else {
                    fn_count += 1;
                }
            }

            // Count false positives (other classes predicted as this class)
            for pred in predictions {
                let predicted_class = pred.predicted_value as usize;
                let true_class = pred.true_value as usize;
                if predicted_class == class_label && true_class != class_label {
                    fp += 1;
                }
            }

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let f1_score = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            class_metrics.insert(
                class_label,
                ClassMetrics {
                    class_label,
                    precision,
                    recall,
                    f1_score,
                    support: class_preds.len(),
                },
            );
        }

        Ok(class_metrics)
    }
}

/// Task type for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
}

impl std::fmt::Display for ValidationResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model Validation Results")?;
        writeln!(f, "=======================")?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f, "Number of splits: {}", self.split_results.len())?;
        writeln!(f)?;

        writeln!(f, "Aggregated Metrics:")?;
        for (metric_name, summary) in &self.aggregated_metrics {
            writeln!(
                f,
                "  {}: {:.4} ± {:.4} (95% CI: {:.4}-{:.4})",
                metric_name,
                summary.mean,
                summary.std,
                summary.confidence_interval.0,
                summary.confidence_interval.1
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Per-split Results:")?;
        for result in &self.split_results {
            writeln!(
                f,
                "  Split {}: {} samples",
                result.split_id, result.test_size
            )?;
            for (metric, value) in &result.metrics {
                writeln!(f, "    {}: {:.4}", metric, value)?;
            }
        }

        Ok(())
    }
}

/// Utility functions for model validation
pub mod validation_utils {
    use super::*;

    /// Create a standard classification validation config
    pub fn create_classification_config(k_folds: usize) -> ValidationConfig {
        ValidationConfig {
            strategy: ValidationStrategy::KFold {
                k: k_folds,
                shuffle: true,
                stratified: true,
            },
            metrics: vec![
                ValidationMetric::Accuracy,
                ValidationMetric::TopKAccuracy { k: 5 },
                ValidationMetric::Precision {
                    average: AverageType::Macro,
                },
                ValidationMetric::Recall {
                    average: AverageType::Macro,
                },
                ValidationMetric::F1Score {
                    average: AverageType::Macro,
                },
                ValidationMetric::CrossEntropyLoss,
            ],
            cross_validation: Some(CrossValidationConfig {
                n_folds: k_folds,
                random_state: Some(42),
                shuffle: true,
                stratified: true,
            }),
            statistical_tests: None,
            tolerance: ToleranceConfig::default(),
            save_detailed_results: true,
        }
    }

    /// Create a standard regression validation config
    pub fn create_regression_config(k_folds: usize) -> ValidationConfig {
        ValidationConfig {
            strategy: ValidationStrategy::KFold {
                k: k_folds,
                shuffle: true,
                stratified: false,
            },
            metrics: vec![
                ValidationMetric::MeanSquaredError,
                ValidationMetric::RootMeanSquaredError,
                ValidationMetric::MeanAbsoluteError,
                ValidationMetric::RSquared,
            ],
            cross_validation: Some(CrossValidationConfig {
                n_folds: k_folds,
                random_state: Some(42),
                shuffle: true,
                stratified: false,
            }),
            statistical_tests: None,
            tolerance: ToleranceConfig::default(),
            save_detailed_results: true,
        }
    }

    /// Create a quick validation config for testing
    pub fn create_quick_config() -> ValidationConfig {
        ValidationConfig {
            strategy: ValidationStrategy::Holdout {
                test_ratio: 0.2,
                stratified: false,
            },
            metrics: vec![
                ValidationMetric::Accuracy,
                ValidationMetric::MeanSquaredError,
            ],
            cross_validation: None,
            statistical_tests: None,
            tolerance: ToleranceConfig::default(),
            save_detailed_results: false,
        }
    }

    /// Validate model correctness against known outputs
    pub fn validate_model_correctness<M: Module>(
        model: &M,
        test_cases: &[(Tensor, Tensor)], // (input, expected_output)
        tolerance: &ToleranceConfig,
    ) -> Result<bool> {
        for (input, expected) in test_cases {
            let actual = model.forward(input)?;

            if !tensors_close(&actual, expected, tolerance)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if two tensors are close within tolerance
    pub fn tensors_close(a: &Tensor, b: &Tensor, tolerance: &ToleranceConfig) -> Result<bool> {
        let diff = (a - b).abs()?;
        let max_diff = diff.max(None, false)?.item()?;

        if tolerance.use_relative {
            let max_val = a
                .abs()?
                .max(None, false)?
                .item()?
                .max(b.abs()?.max(None, false)?.item()?);
            let relative_error = max_diff / max_val.max(1e-8);
            Ok(relative_error <= tolerance.relative_tolerance as f32)
        } else {
            Ok(max_diff <= tolerance.absolute_tolerance as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;

    #[test]
    fn test_validation_config_creation() {
        let config = validation_utils::create_classification_config(5);
        assert_eq!(config.metrics.len(), 6);

        match config.strategy {
            ValidationStrategy::KFold { k, .. } => assert_eq!(k, 5),
            _ => panic!("Expected KFold strategy"),
        }
    }

    #[test]
    fn test_holdout_split_generation() {
        let validator = ModelValidator::new(validation_utils::create_quick_config());

        // Create mock dataset
        let mut dataset = Vec::new();
        for _i in 0..100 {
            let input = torsh_tensor::creation::zeros(&[3, 32, 32]).unwrap();
            let target = torsh_tensor::creation::zeros(&[10]).unwrap();
            dataset.push((input, target));
        }

        let splits = validator.generate_splits(&dataset).unwrap();
        assert_eq!(splits.len(), 1);

        let split = &splits[0];
        assert_eq!(split.train_indices.len() + split.test_indices.len(), 100);
        assert_eq!(split.test_indices.len(), 20); // 20% test ratio
    }

    #[test]
    fn test_kfold_split_generation() {
        let config = validation_utils::create_classification_config(5);
        let validator = ModelValidator::new(config);

        // Create mock dataset
        let mut dataset = Vec::new();
        for _i in 0..100 {
            let input = torsh_tensor::creation::zeros(&[3, 32, 32]).unwrap();
            let target = torsh_tensor::creation::zeros(&[10]).unwrap();
            dataset.push((input, target));
        }

        let splits = validator.generate_splits(&dataset).unwrap();
        assert_eq!(splits.len(), 5);

        // Check that each fold has approximately equal size
        for split in &splits {
            assert!(split.test_indices.len() >= 18 && split.test_indices.len() <= 22);
            assert_eq!(split.train_indices.len() + split.test_indices.len(), 100);
        }
    }

    #[test]
    fn test_metric_computation() {
        let config = validation_utils::create_quick_config();
        let validator = ModelValidator::new(config);

        // Create mock predictions and targets
        let predictions = vec![
            Tensor::from_data(vec![0.8, 0.2], vec![2], DeviceType::Cpu).unwrap(),
            Tensor::from_data(vec![0.3, 0.7], vec![2], DeviceType::Cpu).unwrap(),
        ];

        let targets = vec![
            Tensor::from_data(vec![1.0, 0.0], vec![2], DeviceType::Cpu).unwrap(),
            Tensor::from_data(vec![0.0, 1.0], vec![2], DeviceType::Cpu).unwrap(),
        ];

        let accuracy = validator.compute_accuracy(&predictions, &targets).unwrap();
        assert_eq!(accuracy, 1.0); // Both predictions are correct
    }

    #[test]
    fn test_metric_summary() {
        let config = validation_utils::create_quick_config();
        let validator = ModelValidator::new(config);

        let values = vec![0.8, 0.85, 0.9, 0.82, 0.88];
        let summary = validator.compute_metric_summary(&values).unwrap();

        assert!((summary.mean - 0.85).abs() < 1e-10);
        assert!(summary.std > 0.0);
        assert_eq!(summary.min, 0.8);
        assert_eq!(summary.max, 0.9);
        assert_eq!(summary.values.len(), 5);
    }

    #[test]
    fn test_tolerance_checking() {
        let tolerance = ToleranceConfig {
            absolute_tolerance: 1e-3,
            relative_tolerance: 1e-2,
            use_relative: false,
        };

        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0005, 2.0005, 3.0005], vec![3], DeviceType::Cpu).unwrap();

        let close = validation_utils::tensors_close(&a, &b, &tolerance).unwrap();
        assert!(close);

        let c = Tensor::from_data(vec![1.1, 2.1, 3.1], vec![3], DeviceType::Cpu).unwrap();
        let not_close = validation_utils::tensors_close(&a, &c, &tolerance).unwrap();
        assert!(!not_close);
    }
}
