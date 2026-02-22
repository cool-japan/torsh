//! Core traits for clustering algorithms

use crate::error::{ClusterError, ClusterResult};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use torsh_tensor::Tensor;

/// Trait for clustering algorithm results
pub trait ClusteringResult: Clone + std::fmt::Debug {
    /// Get cluster labels for each data point
    fn labels(&self) -> &Tensor;

    /// Get number of clusters found
    fn n_clusters(&self) -> usize;

    /// Get cluster centers/representatives (if applicable)
    fn centers(&self) -> Option<&Tensor> {
        None
    }

    /// Get inertia/within-cluster sum of squares (if applicable)
    fn inertia(&self) -> Option<f64> {
        None
    }

    /// Get number of iterations to convergence (if applicable)
    fn n_iter(&self) -> Option<usize> {
        None
    }

    /// Check if algorithm converged
    fn converged(&self) -> bool {
        true
    }

    /// Get additional metadata
    fn metadata(&self) -> Option<&std::collections::HashMap<String, String>> {
        None
    }
}

/// Trait for fitting clustering algorithms
pub trait Fit {
    type Result: ClusteringResult;

    /// Fit the clustering algorithm to data
    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result>;
}

/// Trait for fitting and predicting in one step
pub trait FitPredict {
    type Result: ClusteringResult;

    /// Fit the algorithm and return predictions
    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result>;
}

/// Trait for transforming data using fitted clustering model
pub trait Transform {
    /// Transform data to cluster distance space
    fn transform(&self, data: &Tensor) -> ClusterResult<Tensor>;

    /// Transform data to cluster probability space (for probabilistic methods)
    fn predict_proba(&self, _data: &Tensor) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "predict_proba not implemented for this algorithm".to_string(),
        ))
    }
}

/// Trait for predicting cluster assignments for new data
pub trait Predict {
    /// Predict cluster labels for new data
    fn predict(&self, data: &Tensor) -> ClusterResult<Tensor>;
}

/// Main trait for clustering algorithms
pub trait ClusteringAlgorithm: Fit + FitPredict {
    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm parameters as key-value pairs
    fn get_params(&self) -> std::collections::HashMap<String, String>;

    /// Set algorithm parameters
    fn set_params(
        &mut self,
        params: std::collections::HashMap<String, String>,
    ) -> ClusterResult<()>;

    /// Check if algorithm is fitted
    fn is_fitted(&self) -> bool {
        false
    }

    /// Validate input data
    fn validate_input(&self, data: &Tensor) -> ClusterResult<()> {
        if data.shape().dims().is_empty() {
            return Err(ClusterError::EmptyDataset);
        }

        if data.shape().dims()[0] == 0 {
            return Err(ClusterError::EmptyDataset);
        }

        Ok(())
    }

    /// Get supported distance metrics
    fn supported_distance_metrics(&self) -> Vec<&str> {
        vec!["euclidean"]
    }

    /// Get algorithm complexity information
    fn complexity_info(&self) -> AlgorithmComplexity {
        AlgorithmComplexity::default()
    }
}

/// Information about algorithm computational complexity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlgorithmComplexity {
    /// Time complexity description
    pub time_complexity: String,
    /// Space complexity description
    pub space_complexity: String,
    /// Whether algorithm is deterministic
    pub deterministic: bool,
    /// Whether algorithm supports online/incremental learning
    pub online_capable: bool,
    /// Typical memory usage pattern
    pub memory_pattern: MemoryPattern,
}

impl Default for AlgorithmComplexity {
    fn default() -> Self {
        Self {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(n)".to_string(),
            deterministic: false,
            online_capable: false,
            memory_pattern: MemoryPattern::Linear,
        }
    }
}

/// Memory usage patterns for clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryPattern {
    /// Linear in number of data points
    Linear,
    /// Quadratic in number of data points
    Quadratic,
    /// Constant memory usage
    Constant,
    /// Memory usage depends on cluster structure
    Adaptive,
}

/// Trait for incremental/online clustering algorithms
pub trait IncrementalClustering {
    type Result: ClusteringResult;

    /// Add a single data point
    fn partial_fit_one(&mut self, point: &Tensor) -> ClusterResult<()>;

    /// Add a batch of data points
    fn partial_fit(&mut self, data: &Tensor) -> ClusterResult<()>;

    /// Get current clustering result
    fn current_result(&self) -> ClusterResult<Self::Result>;

    /// Reset the incremental state
    fn reset(&mut self);
}

/// Trait for hierarchical clustering algorithms
pub trait HierarchicalClustering {
    type Tree;

    /// Get the full clustering tree/dendrogram
    fn get_tree(&self) -> Option<&Self::Tree> {
        None
    }

    /// Extract flat clustering at given number of clusters
    fn extract_flat_clustering(&self, n_clusters: usize) -> ClusterResult<Tensor>;

    /// Extract flat clustering at given distance threshold
    fn extract_clustering_by_distance(&self, threshold: f64) -> ClusterResult<Tensor>;
}

/// Trait for probabilistic clustering algorithms
pub trait ProbabilisticClustering {
    /// Get cluster membership probabilities
    fn membership_probabilities(&self, data: &Tensor) -> ClusterResult<Tensor>;

    /// Get cluster parameters (means, covariances, etc.)
    fn cluster_parameters(&self) -> ClusterResult<Vec<std::collections::HashMap<String, Tensor>>>;

    /// Compute log-likelihood of data
    fn log_likelihood(&self, data: &Tensor) -> ClusterResult<f64>;

    /// Sample from the cluster distribution
    fn sample(&self, n_samples: usize) -> ClusterResult<Tensor>;
}

/// Trait for density-based clustering algorithms
pub trait DensityBasedClustering {
    /// Get core points (points with sufficient local density)
    fn core_points(&self) -> Option<&Tensor> {
        None
    }

    /// Get noise points (outliers)
    fn noise_points(&self) -> Option<&Tensor> {
        None
    }

    /// Get local density estimates
    fn density_estimates(&self, data: &Tensor) -> ClusterResult<Tensor>;
}

/// Configuration trait for clustering algorithms
pub trait ClusteringConfig: Clone + std::fmt::Debug {
    /// Validate configuration parameters
    fn validate(&self) -> ClusterResult<()>;

    /// Get default configuration
    fn default() -> Self;

    /// Merge with another configuration
    fn merge(&mut self, other: &Self);
}
