//! Adaptive parameter selection utilities for clustering algorithms
//!
//! This module provides automatic parameter estimation methods for clustering algorithms,
//! particularly DBSCAN and HDBSCAN, to reduce the need for manual parameter tuning.

use crate::error::{ClusterError, ClusterResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};
use std::cmp::Ordering;
use torsh_tensor::Tensor;

/// Adaptive epsilon selection for DBSCAN using the k-distance graph method
///
/// This function automatically estimates a good epsilon value by analyzing
/// the distribution of k-nearest neighbor distances.
///
/// # Algorithm
///
/// 1. For each point, compute distance to k-th nearest neighbor
/// 2. Sort these k-distances in ascending order
/// 3. Find the "elbow" point where the sorted curve shows maximum curvature
/// 4. Return the k-distance at the elbow as recommended epsilon
///
/// # Mathematical Foundation
///
/// The k-distance graph plots sorted k-distances. Dense clusters create a
/// plateau (small k-distances), while noise creates a steep incline (large k-distances).
/// The elbow point indicates the transition between dense regions and noise.
///
/// **Curvature Calculation:**
/// ```text
/// For point i in sorted k-distances:
/// curvature(i) ≈ (d[i+1] - d[i]) - (d[i] - d[i-1])
/// ```
///
/// # Parameters
///
/// - `data`: Input data tensor (n_samples × n_features)
/// - `k`: Number of neighbors to consider (typically min_samples - 1)
/// - `method`: Selection method ("elbow" or "knee" or "percentile")
/// - `percentile`: If method="percentile", use this percentile value (0-100)
///
/// # Returns
///
/// Recommended epsilon value
///
/// # Example
///
/// ```rust
/// use torsh_cluster::utils::adaptive::suggest_epsilon;
/// use torsh_tensor::creation::randn;
///
/// let data = randn::<f32>(&[1000, 10])?;
/// let eps = suggest_epsilon(&data, 4, "elbow", None)?;
/// println!("Recommended epsilon: {}", eps);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn suggest_epsilon(
    data: &Tensor,
    k: usize,
    method: &str,
    percentile: Option<f64>,
) -> ClusterResult<f64> {
    if k == 0 {
        return Err(ClusterError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    // Convert tensor to array
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let shape = data.shape();
    let data_shape = shape.dims();

    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if k >= n_samples {
        return Err(ClusterError::InvalidInput(format!(
            "k ({}) must be less than number of samples ({})",
            k, n_samples
        )));
    }

    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Compute k-distances for all points
    let k_distances = compute_k_distances(&data_array, k)?;

    // Select epsilon based on method
    match method {
        "elbow" => find_elbow_point(&k_distances),
        "knee" => find_knee_point(&k_distances),
        "percentile" => {
            let p = percentile.ok_or_else(|| {
                ClusterError::ConfigError(
                    "percentile parameter required for percentile method".to_string(),
                )
            })?;
            find_percentile(&k_distances, p)
        }
        _ => Err(ClusterError::ConfigError(format!(
            "Unknown epsilon selection method: {}. Use 'elbow', 'knee', or 'percentile'",
            method
        ))),
    }
}

/// Compute k-th nearest neighbor distances for all points
///
/// # Returns
///
/// Sorted array of k-distances (ascending order)
fn compute_k_distances(data: &Array2<f32>, k: usize) -> ClusterResult<Vec<f64>> {
    let n_samples = data.nrows();
    let mut k_distances = Vec::with_capacity(n_samples);

    // Use parallel computation for large datasets
    if n_samples >= 500 {
        // Parallel version using scirs2_core::parallel_ops
        let distances: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut dists = Vec::with_capacity(n_samples - 1);

                for j in 0..n_samples {
                    if i != j {
                        let dist = euclidean_distance(&data.row(i), &data.row(j));
                        dists.push(dist);
                    }
                }

                // Sort to find k-th nearest neighbor
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

                // Return k-th distance (k-1 index because 0-indexed)
                if dists.len() > k - 1 {
                    dists[k - 1]
                } else {
                    *dists.last().unwrap_or(&f64::MAX)
                }
            })
            .collect();

        k_distances = distances;
    } else {
        // Sequential version for small datasets
        for i in 0..n_samples {
            let mut dists = Vec::with_capacity(n_samples - 1);

            for j in 0..n_samples {
                if i != j {
                    let dist = euclidean_distance(&data.row(i), &data.row(j));
                    dists.push(dist);
                }
            }

            // Sort to find k-th nearest neighbor
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Get k-th distance
            if dists.len() > k - 1 {
                k_distances.push(dists[k - 1]);
            } else {
                k_distances.push(*dists.last().unwrap_or(&f64::MAX));
            }
        }
    }

    // Sort all k-distances for elbow detection
    k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    Ok(k_distances)
}

/// Find elbow point using maximum curvature method
///
/// The elbow is the point of maximum curvature in the sorted k-distance graph
fn find_elbow_point(sorted_distances: &[f64]) -> ClusterResult<f64> {
    if sorted_distances.len() < 3 {
        return Err(ClusterError::InvalidInput(
            "Need at least 3 points to find elbow".to_string(),
        ));
    }

    let n = sorted_distances.len();

    // Normalize distances and indices to [0, 1] range for curvature calculation
    let max_dist = sorted_distances[n - 1];
    let min_dist = sorted_distances[0];
    let dist_range = max_dist - min_dist;

    if dist_range < 1e-10 {
        // All distances are the same, return median
        return Ok(sorted_distances[n / 2]);
    }

    // Compute curvature at each point (except first and last)
    let mut max_curvature = 0.0;
    let mut elbow_idx = n / 2; // Default to middle

    for i in 1..(n - 1) {
        // Normalize points
        let x_prev = (i - 1) as f64 / (n - 1) as f64;
        let x_curr = i as f64 / (n - 1) as f64;
        let x_next = (i + 1) as f64 / (n - 1) as f64;

        let y_prev = (sorted_distances[i - 1] - min_dist) / dist_range;
        let y_curr = (sorted_distances[i] - min_dist) / dist_range;
        let y_next = (sorted_distances[i + 1] - min_dist) / dist_range;

        // Approximate curvature using second derivative
        let dx1 = x_curr - x_prev;
        let dx2 = x_next - x_curr;
        let dy1 = y_curr - y_prev;
        let dy2 = y_next - y_curr;

        let slope1 = dy1 / dx1;
        let slope2 = dy2 / dx2;

        let curvature = ((slope2 - slope1) / ((dx1 + dx2) / 2.0)).abs();

        if curvature > max_curvature {
            max_curvature = curvature;
            elbow_idx = i;
        }
    }

    Ok(sorted_distances[elbow_idx])
}

/// Find knee point using perpendicular distance method
///
/// The knee is the point with maximum perpendicular distance to the line
/// connecting the first and last points
fn find_knee_point(sorted_distances: &[f64]) -> ClusterResult<f64> {
    if sorted_distances.len() < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 points to find knee".to_string(),
        ));
    }

    let n = sorted_distances.len();

    // Line from first to last point: y = mx + b
    let x1 = 0.0;
    let y1 = sorted_distances[0];
    let x2 = (n - 1) as f64;
    let y2 = sorted_distances[n - 1];

    let m = (y2 - y1) / (x2 - x1);
    let b = y1 - m * x1;

    // Find point with maximum perpendicular distance to this line
    let mut max_distance = 0.0;
    let mut knee_idx = n / 2; // Default to middle

    for (i, &y) in sorted_distances.iter().enumerate() {
        let x = i as f64;

        // Perpendicular distance from point (x, y) to line mx - y + b = 0
        // distance = |mx - y + b| / sqrt(m^2 + 1)
        let distance = (m * x - y + b).abs() / (m * m + 1.0).sqrt();

        if distance > max_distance {
            max_distance = distance;
            knee_idx = i;
        }
    }

    Ok(sorted_distances[knee_idx])
}

/// Find epsilon at specified percentile
fn find_percentile(sorted_distances: &[f64], percentile: f64) -> ClusterResult<f64> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(ClusterError::InvalidInput(format!(
            "Percentile must be between 0 and 100, got {}",
            percentile
        )));
    }

    if sorted_distances.is_empty() {
        return Err(ClusterError::InvalidInput(
            "Cannot compute percentile of empty array".to_string(),
        ));
    }

    let n = sorted_distances.len();
    let idx = ((percentile / 100.0) * (n - 1) as f64).round() as usize;
    let idx = idx.min(n - 1); // Ensure within bounds

    Ok(sorted_distances[idx])
}

/// Compute Euclidean distance between two points
#[inline]
fn euclidean_distance(point1: &ArrayView1<f32>, point2: &ArrayView1<f32>) -> f64 {
    let mut sum_sq = 0.0_f64;
    for (a, b) in point1.iter().zip(point2.iter()) {
        let diff = (*a as f64) - (*b as f64);
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Suggest DBSCAN parameters (epsilon and min_samples) based on data characteristics
///
/// This function provides a complete automatic parameter suggestion for DBSCAN.
///
/// # Heuristics
///
/// - **min_samples**: Typically set to `2 * n_features` for dimensional data,
///   or `n_features + 1` as a minimum.
/// - **epsilon**: Estimated using k-distance graph with k = min_samples - 1
///
/// # Parameters
///
/// - `data`: Input data tensor
/// - `method`: Epsilon selection method ("auto", "elbow", "knee", "percentile")
/// - `percentile`: Optional percentile for percentile method (default: 90.0)
///
/// # Returns
///
/// Tuple of (recommended_epsilon, recommended_min_samples)
///
/// # Example
///
/// ```rust
/// use torsh_cluster::utils::adaptive::suggest_dbscan_params;
/// use torsh_tensor::creation::randn;
///
/// let data = randn::<f32>(&[1000, 10])?;
/// let (eps, min_samples) = suggest_dbscan_params(&data, "auto", None)?;
/// println!("Recommended parameters: eps={}, min_samples={}", eps, min_samples);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn suggest_dbscan_params(
    data: &Tensor,
    method: &str,
    percentile: Option<f64>,
) -> ClusterResult<(f64, usize)> {
    let shape = data.shape();
    let data_shape = shape.dims();

    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_features = data_shape[1];

    // Heuristic for min_samples: 2 * n_features (rule of thumb from literature)
    // Minimum of 4 to ensure meaningful density
    let min_samples = (2 * n_features).max(4);

    // Use k = min_samples - 1 for k-distance graph
    let k = min_samples - 1;

    // Select epsilon using specified method (or elbow for "auto")
    let actual_method = if method == "auto" { "elbow" } else { method };

    let epsilon = suggest_epsilon(data, k, actual_method, percentile)?;

    Ok((epsilon, min_samples))
}

/// Estimate optimal number of clusters for DBSCAN by trying different epsilon values
///
/// This function performs a grid search over epsilon values and evaluates
/// clustering quality using silhouette score.
///
/// **Note:** This is computationally expensive (O(n_eps × n²)) and should
/// only be used for small to medium datasets.
///
/// # Parameters
///
/// - `data`: Input data tensor
/// - `min_eps`: Minimum epsilon to try
/// - `max_eps`: Maximum epsilon to try
/// - `n_values`: Number of epsilon values to try
/// - `min_samples`: Fixed min_samples parameter
///
/// # Returns
///
/// Tuple of (best_epsilon, estimated_n_clusters, best_silhouette_score)
#[allow(dead_code)]
pub fn optimize_epsilon(
    _data: &Tensor,
    min_eps: f64,
    max_eps: f64,
    n_values: usize,
    _min_samples: usize,
) -> ClusterResult<(f64, usize, f64)> {
    if n_values < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 epsilon values to optimize".to_string(),
        ));
    }

    if min_eps >= max_eps {
        return Err(ClusterError::InvalidInput(
            "min_eps must be less than max_eps".to_string(),
        ));
    }

    // Generate epsilon values (log scale for better coverage)
    let _eps_values: Vec<f64> = (0..n_values)
        .map(|i| {
            let log_min = min_eps.ln();
            let log_max = max_eps.ln();
            let log_eps = log_min + (log_max - log_min) * (i as f64 / (n_values - 1) as f64);
            log_eps.exp()
        })
        .collect();

    // Run DBSCAN (this would require importing DBSCAN, which creates circular dependency)
    // For now, return error - this function should be used externally
    // with manual DBSCAN calls
    Err(ClusterError::NotImplemented(
        "optimize_epsilon requires external DBSCAN evaluation - use suggest_epsilon instead or implement manually".to_string(),
    ))

    // TODO: When possible, implement as:
    // let mut best_eps = eps_values[0];
    // let mut best_n_clusters = 0;
    // let mut best_score = f64::NEG_INFINITY;
    //
    // for &eps in &eps_values {
    //     let dbscan = DBSCAN::new(eps, min_samples);
    //     let result = dbscan.fit(data)?;
    //
    //     if result.n_clusters > 0 {
    //         let score = silhouette_score(data, &result.labels)?;
    //         if score > best_score {
    //             best_score = score;
    //             best_eps = eps;
    //             best_n_clusters = result.n_clusters;
    //         }
    //     }
    // }
    //
    // Ok((best_eps, best_n_clusters, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_suggest_epsilon_basic() -> ClusterResult<()> {
        // Create simple clustered data
        let data = Tensor::from_vec(
            vec![
                // Tight cluster (should have small k-distances)
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0,
                0.2, // Sparse points (should have large k-distances)
                5.0, 5.0, 10.0, 10.0,
            ],
            &[6, 2],
        )?;

        let eps = suggest_epsilon(&data, 3, "elbow", None)?;

        // Epsilon should be positive and reasonable
        assert!(eps > 0.0);
        assert!(eps < 20.0); // Should be less than max distance

        Ok(())
    }

    #[test]
    fn test_suggest_epsilon_methods() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 10.0, 10.0], &[4, 2])?;

        let eps_elbow = suggest_epsilon(&data, 2, "elbow", None)?;
        let eps_knee = suggest_epsilon(&data, 2, "knee", None)?;
        let eps_percentile = suggest_epsilon(&data, 2, "percentile", Some(75.0))?;

        // All methods should return positive values
        assert!(eps_elbow > 0.0);
        assert!(eps_knee > 0.0);
        assert!(eps_percentile > 0.0);

        // Methods may differ but should be in reasonable range
        assert!(eps_elbow < 20.0);
        assert!(eps_knee < 20.0);
        assert!(eps_percentile < 20.0);

        Ok(())
    }

    #[test]
    fn test_suggest_epsilon_percentile() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[3, 2])?;

        let eps_50 = suggest_epsilon(&data, 2, "percentile", Some(50.0))?;
        let eps_90 = suggest_epsilon(&data, 2, "percentile", Some(90.0))?;

        // Higher percentile should give higher epsilon
        assert!(eps_90 >= eps_50);

        Ok(())
    }

    #[test]
    fn test_suggest_dbscan_params() -> ClusterResult<()> {
        let data = Tensor::from_vec(
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, // cluster
                5.0, 5.0, 5.1, 5.1, 5.2, 5.0, // cluster
            ],
            &[6, 2],
        )?;

        let (eps, min_samples) = suggest_dbscan_params(&data, "auto", None)?;

        // min_samples should be >= 4 (heuristic: 2 * n_features)
        assert!(min_samples >= 4);

        // epsilon should be positive and reasonable
        assert!(eps > 0.0);
        assert!(eps < 10.0);

        Ok(())
    }

    #[test]
    fn test_compute_k_distances() -> ClusterResult<()> {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let k_dists = compute_k_distances(&data, 2)?;

        // Should have 4 k-distances (one per point)
        assert_eq!(k_dists.len(), 4);

        // Should be sorted
        for i in 1..k_dists.len() {
            assert!(k_dists[i] >= k_dists[i - 1]);
        }

        // All distances should be positive
        for &dist in &k_dists {
            assert!(dist > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_find_elbow_point() -> ClusterResult<()> {
        // Create artificial k-distance distribution with clear elbow
        let distances = vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 5.0, 10.0, 20.0];

        let elbow = find_elbow_point(&distances)?;

        // Elbow should be somewhere in the middle where curvature is highest
        assert!(elbow >= 0.1);
        assert!(elbow <= 20.0);

        Ok(())
    }

    #[test]
    fn test_find_knee_point() -> ClusterResult<()> {
        // Create artificial k-distance distribution
        let distances = vec![0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 7.0, 15.0];

        let knee = find_knee_point(&distances)?;

        // Knee should be at point with max perpendicular distance
        assert!(knee >= 0.1);
        assert!(knee <= 15.0);

        Ok(())
    }

    #[test]
    fn test_find_percentile() -> ClusterResult<()> {
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let p50 = find_percentile(&distances, 50.0)?;
        let p90 = find_percentile(&distances, 90.0)?;

        // 50th percentile should be around median
        assert_relative_eq!(p50, 5.5, epsilon = 1.0);

        // 90th percentile should be near the end
        assert!(p90 >= 9.0);
        assert!(p90 <= 10.0);

        Ok(())
    }

    #[test]
    fn test_invalid_inputs() {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2]).unwrap();

        // k too large
        assert!(suggest_epsilon(&data, 10, "elbow", None).is_err());

        // k = 0
        assert!(suggest_epsilon(&data, 0, "elbow", None).is_err());

        // Invalid method
        assert!(suggest_epsilon(&data, 1, "invalid_method", None).is_err());

        // Percentile without value
        assert!(suggest_epsilon(&data, 1, "percentile", None).is_err());

        // Invalid percentile
        assert!(suggest_epsilon(&data, 1, "percentile", Some(150.0)).is_err());
    }

    #[test]
    fn test_euclidean_distance() {
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![3.0, 4.0]);

        let dist = euclidean_distance(&p1.view(), &p2.view());

        // 3-4-5 triangle
        assert_relative_eq!(dist, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_k_distances() -> ClusterResult<()> {
        // Test with dataset large enough to trigger parallel computation
        let mut data_vec = Vec::new();
        for i in 0..1000 {
            data_vec.push((i % 10) as f32);
            data_vec.push((i / 10) as f32);
        }

        let data = Tensor::from_vec(data_vec, &[1000, 2])?;

        let eps = suggest_epsilon(&data, 4, "elbow", None)?;

        // Should complete without error and return reasonable value
        assert!(eps > 0.0);
        assert!(eps.is_finite());

        Ok(())
    }
}
