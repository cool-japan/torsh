//! Clustering evaluation metrics and utilities

pub mod metrics;

pub use metrics::*;

use crate::error::ClusterResult;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use torsh_tensor::Tensor;

/// Trait for clustering evaluation metrics
pub trait ClusteringMetric {
    /// Compute the metric score
    fn score(&self, labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64>;

    /// Get metric name
    fn name(&self) -> &str;

    /// Whether higher scores are better
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Comprehensive evaluation result
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluationResult {
    /// Silhouette score
    pub silhouette_score: Option<f64>,
    /// Adjusted Rand Index
    pub adjusted_rand_score: Option<f64>,
    /// Normalized Mutual Information
    pub normalized_mutual_info: Option<f64>,
    /// Adjusted Mutual Information
    pub adjusted_mutual_info: Option<f64>,
    /// V-measure score
    pub v_measure: Option<f64>,
    /// Homogeneity score
    pub homogeneity: Option<f64>,
    /// Completeness score
    pub completeness: Option<f64>,
    /// Fowlkes-Mallows score
    pub fowlkes_mallows: Option<f64>,
    /// Calinski-Harabasz score
    pub calinski_harabasz: Option<f64>,
    /// Davies-Bouldin score
    pub davies_bouldin: Option<f64>,
}

impl EvaluationResult {
    /// Create a new evaluation result
    pub fn new() -> Self {
        Self::default()
    }

    /// Set silhouette score
    pub fn with_silhouette_score(mut self, score: f64) -> Self {
        self.silhouette_score = Some(score);
        self
    }

    /// Set adjusted rand score
    pub fn with_adjusted_rand_score(mut self, score: f64) -> Self {
        self.adjusted_rand_score = Some(score);
        self
    }

    /// Set normalized mutual information
    pub fn with_normalized_mutual_info(mut self, score: f64) -> Self {
        self.normalized_mutual_info = Some(score);
        self
    }

    /// Get summary of available scores
    pub fn summary(&self) -> Vec<(&str, f64)> {
        let mut scores = Vec::new();

        if let Some(score) = self.silhouette_score {
            scores.push(("Silhouette", score));
        }
        if let Some(score) = self.adjusted_rand_score {
            scores.push(("Adjusted Rand Index", score));
        }
        if let Some(score) = self.normalized_mutual_info {
            scores.push(("Normalized Mutual Info", score));
        }
        if let Some(score) = self.adjusted_mutual_info {
            scores.push(("Adjusted Mutual Info", score));
        }
        if let Some(score) = self.v_measure {
            scores.push(("V-measure", score));
        }
        if let Some(score) = self.homogeneity {
            scores.push(("Homogeneity", score));
        }
        if let Some(score) = self.completeness {
            scores.push(("Completeness", score));
        }
        if let Some(score) = self.fowlkes_mallows {
            scores.push(("Fowlkes-Mallows", score));
        }
        if let Some(score) = self.calinski_harabasz {
            scores.push(("Calinski-Harabasz", score));
        }
        if let Some(score) = self.davies_bouldin {
            scores.push(("Davies-Bouldin", score));
        }

        scores
    }
}
