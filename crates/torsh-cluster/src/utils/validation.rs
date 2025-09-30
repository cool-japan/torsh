//! Validation utilities for clustering operations

use crate::error::{ClusterError, ClusterResult};
use torsh_tensor::Tensor;

/// Validation trait for clustering operations
pub trait ClusterValidation {
    /// Validate input data
    fn validate(&self, data: &Tensor) -> ClusterResult<()>;
}

/// Validate cluster input data
pub fn validate_cluster_input(data: &Tensor) -> ClusterResult<()> {
    // Check if data is empty
    if data.shape().dims().is_empty() {
        return Err(ClusterError::EmptyDataset);
    }

    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims().get(1).copied().unwrap_or(1);

    // Check minimum samples
    if n_samples == 0 {
        return Err(ClusterError::EmptyDataset);
    }

    // Check minimum features
    if n_features == 0 {
        return Err(ClusterError::InvalidInput(
            "Data must have at least one feature".to_string(),
        ));
    }

    // Check for invalid values (NaN, infinite)
    if let Ok(data_vec) = data.to_vec() {
        for &val in &data_vec {
            if val.is_nan() || val.is_infinite() {
                return Err(ClusterError::InvalidInput(
                    "Data contains NaN or infinite values".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Validate number of clusters
pub fn validate_n_clusters(n_clusters: usize, n_samples: usize) -> ClusterResult<()> {
    if n_clusters == 0 {
        return Err(ClusterError::InvalidClusters(n_clusters));
    }

    if n_clusters > n_samples {
        return Err(ClusterError::InvalidClusters(n_clusters));
    }

    Ok(())
}

/// Validate clustering parameters
pub fn validate_clustering_params(
    data: &Tensor,
    n_clusters: usize,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
) -> ClusterResult<()> {
    // Validate input data
    validate_cluster_input(data)?;

    // Validate number of clusters
    let n_samples = data.shape().dims()[0];
    validate_n_clusters(n_clusters, n_samples)?;

    // Validate max iterations
    if let Some(max_iters) = max_iters {
        if max_iters == 0 {
            return Err(ClusterError::ConfigError(
                "max_iters must be positive".to_string(),
            ));
        }
    }

    // Validate tolerance
    if let Some(tolerance) = tolerance {
        if tolerance < 0.0 {
            return Err(ClusterError::ConfigError(
                "tolerance must be non-negative".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate DBSCAN parameters
pub fn validate_dbscan_params(eps: f64, min_samples: usize) -> ClusterResult<()> {
    if eps <= 0.0 {
        return Err(ClusterError::InvalidEpsilon(eps));
    }

    if min_samples == 0 {
        return Err(ClusterError::InvalidMinSamples(min_samples));
    }

    Ok(())
}

/// Validate distance metric
pub fn validate_distance_metric(metric: &str) -> ClusterResult<()> {
    match metric {
        "euclidean" | "manhattan" | "cosine" | "hamming" | "jaccard" => Ok(()),
        _ => Err(ClusterError::InvalidDistanceMetric(metric.to_string())),
    }
}

/// Validate linkage criterion for hierarchical clustering
pub fn validate_linkage(linkage: &str) -> ClusterResult<()> {
    match linkage {
        "ward" | "complete" | "average" | "single" => Ok(()),
        _ => Err(ClusterError::InvalidLinkage(linkage.to_string())),
    }
}

/// Validate affinity matrix for spectral clustering
pub fn validate_affinity_matrix(affinity: &Tensor) -> ClusterResult<()> {
    let shape = affinity.shape();
    let dims = shape.dims();

    // Must be square matrix
    if dims.len() != 2 || dims[0] != dims[1] {
        return Err(ClusterError::InvalidAffinityMatrix(
            "Affinity matrix must be square".to_string(),
        ));
    }

    // Check for valid values
    if let Ok(affinity_vec) = affinity.to_vec() {
        for &val in &affinity_vec {
            if val.is_nan() || val.is_infinite() || val < 0.0 {
                return Err(ClusterError::InvalidAffinityMatrix(
                    "Affinity matrix contains invalid values".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Validate feature dimensions match
pub fn validate_feature_dimensions(
    data1: &Tensor,
    data2: &Tensor,
    _context: &str,
) -> ClusterResult<()> {
    let shape1 = data1.shape();
    let dims1 = shape1.dims();
    let shape2 = data2.shape();
    let dims2 = shape2.dims();

    let n_features1 = dims1.get(1).copied().unwrap_or(1);
    let n_features2 = dims2.get(1).copied().unwrap_or(1);

    if n_features1 != n_features2 {
        return Err(ClusterError::InvalidFeatureDimension {
            expected: n_features1,
            actual: n_features2,
        });
    }

    Ok(())
}

/// Validate cluster labels
pub fn validate_cluster_labels(labels: &Tensor, n_samples: usize) -> ClusterResult<()> {
    // Check dimensions
    let label_shape = labels.shape();
    let label_dims = label_shape.dims();
    if label_dims.len() != 1 || label_dims[0] != n_samples {
        return Err(ClusterError::InvalidAssignment(format!(
            "Labels must be 1D array with {} elements",
            n_samples
        )));
    }

    // Check for valid label values
    if let Ok(labels_vec) = labels.to_vec() {
        for &label in &labels_vec {
            if label.is_nan() || label.is_infinite() || label < -1.0 {
                return Err(ClusterError::InvalidAssignment(
                    "Labels contain invalid values".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Check for numerical stability issues
pub fn check_numerical_stability(values: &[f64], context: &str) -> ClusterResult<()> {
    for &val in values {
        if val.is_nan() {
            return Err(ClusterError::NumericalInstability(format!(
                "NaN detected in {}",
                context
            )));
        }

        if val.is_infinite() {
            return Err(ClusterError::NumericalInstability(format!(
                "Infinite value detected in {}",
                context
            )));
        }

        // Check for very small or very large values that might cause issues
        if val.abs() < 1e-15 && val != 0.0 {
            return Err(ClusterError::NumericalInstability(format!(
                "Very small value detected in {}: {}",
                context, val
            )));
        }

        if val.abs() > 1e15 {
            return Err(ClusterError::NumericalInstability(format!(
                "Very large value detected in {}: {}",
                context, val
            )));
        }
    }

    Ok(())
}
