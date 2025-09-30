//! Comprehensive evaluation metrics for ToRSh
//!
//! This module provides PyTorch-compatible metrics for model evaluation,
//! built on top of SciRS2's comprehensive metrics library.

pub mod classification;
pub mod clustering;
pub mod deep_learning;
pub mod fairness;
pub mod ranking;
pub mod regression;
pub mod statistics;
pub mod streaming;
pub mod uncertainty;
pub mod utils;

// Re-export high-performance vectorized metrics for convenience
pub use deep_learning::{
    SimilarityType, VectorizedFidScore, VectorizedInceptionScore, VectorizedPerplexity,
    VectorizedSemanticSimilarity,
};

use torsh_tensor::Tensor;

/// Base trait for all metrics
pub trait Metric {
    /// Compute the metric
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64;

    /// Reset internal state (for stateful metrics)
    fn reset(&mut self) {}

    /// Update internal state with new batch
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) {}

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
