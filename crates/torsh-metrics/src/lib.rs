//! Comprehensive evaluation metrics for ToRSh
//!
//! This module provides PyTorch-compatible metrics for model evaluation,
//! built on top of SciRS2's comprehensive metrics library.

pub mod advanced_ml;
pub mod classification;
pub mod clustering;
pub mod deep_learning;
pub mod explainability;
pub mod fairness;
pub mod gpu;
pub mod memory_efficient;
pub mod mlflow;
pub mod model_selection;
pub mod parallel;
pub mod ranking;
pub mod regression;
pub mod regression_diagnostics;
pub mod reporting;
pub mod robustness;
pub mod sklearn_compat;
pub mod statistical_tests;
pub mod statistics;
pub mod streaming;
pub mod tensorboard;
pub mod time_series;
pub mod uncertainty;
pub mod utils;
pub mod visualization;
pub mod wandb;

// Re-export high-performance vectorized metrics for convenience
pub use deep_learning::{
    BleuScore, DeepLearningMetrics, RougeMetrics, RougeScore, RougeType, SimilarityType,
    VectorizedFidScore, VectorizedInceptionScore, VectorizedPerplexity,
    VectorizedSemanticSimilarity,
};

// Re-export classification metrics
pub use classification::{ConfusionMatrix, MultiClassMetrics, ThresholdMetrics};

// Re-export ranking/IR metrics
pub use ranking::IRMetrics;

// Re-export uncertainty quantification metrics
pub use uncertainty::{
    BayesianUncertainty, CalibrationMetrics, EnsembleUncertainty, MCDropoutUncertainty,
    UncertaintyDecomposition,
};

// Re-export fairness metrics
pub use fairness::FairnessMetrics;

// Re-export statistical metrics
pub use statistics::{BootstrapResult, CrossValidationResult, HypothesisTestResult};

// Re-export GPU-accelerated metrics
pub use gpu::{GpuAccuracy, GpuBatchMetrics, GpuConfusionMatrix};

// Re-export parallel metrics
pub use parallel::{ParallelAccuracy, ParallelConfusionMatrix, ParallelMetricCollection};

// Re-export reporting utilities
pub use reporting::{ComparisonReport, MetricReport, ReportBuilder, ReportFormat};

// Re-export memory-efficient metrics
pub use memory_efficient::{
    ChunkedEvaluator, MemoryEfficientAccuracy, MemoryEfficientMAE, MemoryEfficientMSE,
    OnlineConfusionMatrix, StreamingMetric,
};

// Re-export TensorBoard integration
pub use tensorboard::{MetricLogger as TensorBoardLogger, TensorBoardWriter};

// Re-export MLflow integration
pub use mlflow::{ExperimentTracker, MLflowClient, MLflowRun};

// Re-export visualization utilities
pub use visualization::{
    CalibrationCurvePlot, ConfusionMatrixPlot, ExportFormat, FeatureImportancePlot,
    InteractiveDashboard, LatexReportBuilder, LearningCurvePlot, MetricComparisonPlot, PRCurvePlot,
    ROCCurvePlot, VisualizationAggregator,
};

// Re-export advanced ML metrics
pub use advanced_ml::{
    ContinualLearningMetrics, DomainAdaptationMetrics, FewShotMetrics, MetaLearningMetrics,
};

// Re-export scikit-learn compatibility layer
pub use sklearn_compat::{
    SklearnAccuracy, SklearnF1Score, SklearnMeanAbsoluteError, SklearnMeanSquaredError,
    SklearnMetric, SklearnPrecision, SklearnR2Score, SklearnRecall,
};

// Re-export Weights & Biases integration
pub use wandb::{LogEntry, WandbClient};

// Re-export model selection metrics
pub use model_selection::{
    AICc, CVModelComparison, CVModelSelection, CVScoreType, ModelComparisonReport,
    MultiModelComparison, AIC, BIC, HQIC,
};

// Re-export statistical tests
pub use statistical_tests::{
    FiveByTwoCVTest, FriedmanTest, KruskalWallisTest, MannWhitneyTest, McNemarTest, NemenyiTest,
    PairedTTest, WilcoxonTest,
};

// Re-export time series metrics
pub use time_series::{
    dtw_distance, error_autocorrelation, mape, mase, mean_directional_accuracy, msis, smape,
    theil_u, tracking_signal,
};

// Re-export regression diagnostics
pub use regression_diagnostics::{
    breusch_pagan_test, calculate_leverage, condition_number, cooks_distance, dffits,
    durbin_watson, variance_inflation_factor, RegressionDiagnosticReport, ResidualDiagnostics,
};

// Re-export explainability metrics
pub use explainability::{
    attribution_agreement, counterfactual_validity, explanation_completeness,
    explanation_faithfulness, feature_importance_stability, feature_monotonicity,
    interaction_strength, ExplainabilityMetrics,
};

// Re-export robustness metrics
pub use robustness::{
    adversarial_accuracy, attack_success_rate, certified_robustness_radius, confidence_stability,
    corruption_robustness, gradient_stability, noise_sensitivity, ood_detection_score,
    robustness_accuracy_tradeoff, RobustnessReport,
};

use torsh_tensor::Tensor;

/// Base trait for all metrics
pub trait Metric {
    /// Compute the metric
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64;

    /// Reset internal state (for stateful metrics)
    fn reset(&mut self) {}

    /// Update internal state with new batch
    fn update(&mut self, _predictions: &Tensor, _targets: &Tensor) {}

    /// Get the name of the metric
    fn name(&self) -> &str;
}

/// Metric collection for evaluating multiple metrics at once
pub struct MetricCollection {
    metrics: Vec<Box<dyn Metric>>,
    results: Vec<(String, f64)>,
}

impl MetricCollection {
    /// Create a new metric collection
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a metric to the collection
    pub fn add<M: Metric + 'static>(mut self, metric: M) -> Self {
        self.metrics.push(Box::new(metric));
        self
    }

    /// Compute all metrics
    pub fn compute(&mut self, predictions: &Tensor, targets: &Tensor) -> Vec<(String, f64)> {
        self.results.clear();

        for metric in &self.metrics {
            let name = metric.name().to_string();
            let value = metric.compute(predictions, targets);
            self.results.push((name, value));
        }

        self.results.clone()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        for metric in &mut self.metrics {
            metric.reset();
        }
        self.results.clear();
    }

    /// Get results as a formatted string
    pub fn format_results(&self) -> String {
        self.results
            .iter()
            .map(|(name, value)| format!("{}: {:.4}", name, value))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Default for MetricCollection {
    fn default() -> Self {
        Self::new()
    }
}
