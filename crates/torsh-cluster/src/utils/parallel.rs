//! Parallel processing utilities for clustering algorithms
//!
//! This module provides parallel implementations of common clustering operations
//! using SciRS2's parallel_ops for optimal performance on multi-core systems.
//!
//! # SciRS2 POLICY Compliance
//!
//! All parallel operations use `scirs2_core::parallel_ops` instead of direct rayon,
//! ensuring consistent API and optimal integration with the SciRS2 ecosystem.

use crate::error::{ClusterError, ClusterResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::parallel_ops::{is_parallel_enabled, parallel_map};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Compute pairwise Euclidean distances in parallel using SIMD
///
/// # Arguments
/// * `data` - Data matrix (n_samples x n_features)
///
/// # Returns
/// Distance matrix (n_samples x n_samples)
pub fn parallel_pairwise_distances_f32(data: &Array2<f32>) -> ClusterResult<Array2<f32>> {
    let n_samples = data.nrows();
    let rows: Vec<ArrayView1<f32>> = data.outer_iter().collect();

    // Parallel computation of distances
    let distance_vec: Vec<f32> = parallel_map(&(0..n_samples).collect::<Vec<_>>(), |&i| {
        let mut row_distances = Vec::with_capacity(n_samples);
        for j in 0..n_samples {
            if i == j {
                row_distances.push(0.0);
            } else {
                // Use SIMD-optimized distance computation
                let diff = f32::simd_sub(&rows[i], &rows[j]);
                let dist_sq = f32::simd_sum_squares(&diff.view());
                row_distances.push(dist_sq.sqrt());
            }
        }
        row_distances
    })
    .into_iter()
    .flatten()
    .collect();

    Array2::from_shape_vec((n_samples, n_samples), distance_vec)
        .map_err(|_| ClusterError::InvalidInput("Failed to create distance matrix".to_string()))
}

/// Compute distances from data points to centroids in parallel
///
/// # Arguments
/// * `data` - Data points (n_samples x n_features)
/// * `centroids` - Cluster centroids (n_clusters x n_features)
///
/// # Returns
/// Distance matrix (n_samples x n_clusters)
pub fn parallel_distances_to_centroids_f32(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
) -> ClusterResult<Array2<f32>> {
    let n_samples = data.nrows();
    let n_clusters = centroids.nrows();
    let data_rows: Vec<ArrayView1<f32>> = data.outer_iter().collect();
    let centroid_rows: Vec<ArrayView1<f32>> = centroids.outer_iter().collect();

    // Parallel computation of distances from each point to all centroids
    let distance_vec: Vec<f32> = parallel_map(&(0..n_samples).collect::<Vec<_>>(), |&i| {
        let mut point_distances = Vec::with_capacity(n_clusters);
        for k in 0..n_clusters {
            // Use SIMD-optimized distance computation
            let diff = f32::simd_sub(&data_rows[i], &centroid_rows[k]);
            let dist_sq = f32::simd_sum_squares(&diff.view());
            point_distances.push(dist_sq.sqrt());
        }
        point_distances
    })
    .into_iter()
    .flatten()
    .collect();

    Array2::from_shape_vec((n_samples, n_clusters), distance_vec)
        .map_err(|_| ClusterError::InvalidInput("Failed to create distance matrix".to_string()))
}

/// Compute squared distances to centroids in parallel (faster, avoids sqrt)
///
/// # Arguments
/// * `data` - Data points (n_samples x n_features)
/// * `centroids` - Cluster centroids (n_clusters x n_features)
///
/// # Returns
/// Squared distance matrix (n_samples x n_clusters)
pub fn parallel_squared_distances_to_centroids_f32(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
) -> ClusterResult<Array2<f32>> {
    let n_samples = data.nrows();
    let n_clusters = centroids.nrows();
    let data_rows: Vec<ArrayView1<f32>> = data.outer_iter().collect();
    let centroid_rows: Vec<ArrayView1<f32>> = centroids.outer_iter().collect();

    // Parallel computation of squared distances (faster, no sqrt)
    let distance_vec: Vec<f32> = parallel_map(&(0..n_samples).collect::<Vec<_>>(), |&i| {
        let mut point_distances = Vec::with_capacity(n_clusters);
        for k in 0..n_clusters {
            // Use SIMD-optimized squared distance computation (no sqrt)
            let diff = f32::simd_sub(&data_rows[i], &centroid_rows[k]);
            let dist_sq = f32::simd_sum_squares(&diff.view());
            point_distances.push(dist_sq);
        }
        point_distances
    })
    .into_iter()
    .flatten()
    .collect();

    Array2::from_shape_vec((n_samples, n_clusters), distance_vec)
        .map_err(|_| ClusterError::InvalidInput("Failed to create distance matrix".to_string()))
}

/// Find nearest centroids for all data points in parallel
///
/// # Arguments
/// * `data` - Data points (n_samples x n_features)
/// * `centroids` - Cluster centroids (n_clusters x n_features)
///
/// # Returns
/// Tuple of (labels, distances) where `labels[i]` is the nearest centroid for point i
pub fn parallel_nearest_centroids_f32(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
) -> ClusterResult<(Array1<usize>, Array1<f32>)> {
    let n_samples = data.nrows();
    let data_rows: Vec<ArrayView1<f32>> = data.outer_iter().collect();
    let centroid_rows: Vec<ArrayView1<f32>> = centroids.outer_iter().collect();

    // Parallel computation of nearest centroid for each point
    let results: Vec<(usize, f32)> = parallel_map(&(0..n_samples).collect::<Vec<_>>(), |&i| {
        let mut min_dist = f32::INFINITY;
        let mut nearest_centroid = 0;

        for (k, centroid) in centroid_rows.iter().enumerate() {
            // Use SIMD-optimized squared distance (faster)
            let diff = f32::simd_sub(&data_rows[i], centroid);
            let dist_sq = f32::simd_sum_squares(&diff.view());

            if dist_sq < min_dist {
                min_dist = dist_sq;
                nearest_centroid = k;
            }
        }

        (nearest_centroid, min_dist.sqrt())
    });

    let labels: Vec<usize> = results.iter().map(|&(label, _)| label).collect();
    let distances: Vec<f32> = results.iter().map(|&(_, dist)| dist).collect();

    Ok((Array1::from_vec(labels), Array1::from_vec(distances)))
}

/// Compute cluster assignments and update centroids in parallel
///
/// # Arguments
/// * `data` - Data points (n_samples x n_features)
/// * `centroids` - Current cluster centroids (n_clusters x n_features)
///
/// # Returns
/// Tuple of (new_centroids, labels, inertia)
pub fn parallel_kmeans_iteration_f32(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
) -> ClusterResult<(Array2<f32>, Array1<usize>, f32)> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let n_clusters = centroids.nrows();

    // Step 1: Assign points to nearest centroids in parallel
    let (labels, _) = parallel_nearest_centroids_f32(data, centroids)?;

    // Step 2: Compute new centroids in parallel (one per cluster)
    let cluster_indices: Vec<usize> = (0..n_clusters).collect();
    let labels_vec = labels.to_vec();
    let data_clone = data.clone();

    let new_centroids_vec: Vec<Vec<f32>> = parallel_map(&cluster_indices, |&k| {
        let mut centroid = vec![0.0f32; n_features];
        let mut count = 0;

        for (i, &label) in labels_vec.iter().enumerate() {
            if label == k {
                for (j, val) in centroid.iter_mut().enumerate() {
                    *val += data_clone[[i, j]];
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in centroid.iter_mut() {
                *val /= count as f32;
            }
        }

        centroid
    });

    // Convert to Array2
    let new_centroids_flat: Vec<f32> = new_centroids_vec.into_iter().flatten().collect();
    let new_centroids = Array2::from_shape_vec((n_clusters, n_features), new_centroids_flat)
        .map_err(|_| ClusterError::InvalidInput("Failed to create centroids".to_string()))?;

    // Step 3: Compute inertia (within-cluster sum of squares) in parallel
    let data_rows_inertia: Vec<ArrayView1<f32>> = data.outer_iter().collect();
    let centroid_rows_inertia: Vec<ArrayView1<f32>> = new_centroids.outer_iter().collect();
    let sample_indices_inertia: Vec<usize> = (0..n_samples).collect();

    let per_sample_sq: Vec<f32> = parallel_map(&sample_indices_inertia, |&i| {
        let label = labels_vec[i];
        let diff = f32::simd_sub(&data_rows_inertia[i], &centroid_rows_inertia[label]);
        f32::simd_sum_squares(&diff.view())
    });
    let inertia: f32 = per_sample_sq.into_iter().sum();

    Ok((new_centroids, labels, inertia))
}

/// Compute silhouette scores for all samples in parallel
///
/// # Arguments
/// * `data` - Data points (n_samples x n_features)
/// * `labels` - Cluster labels for each point
/// * `n_clusters` - Number of clusters
///
/// # Returns
/// Silhouette scores for each sample
pub fn parallel_silhouette_scores_f32(
    data: &Array2<f32>,
    labels: &Array1<usize>,
    n_clusters: usize,
) -> ClusterResult<Array1<f32>> {
    let n_samples = data.nrows();
    let data_rows: Vec<ArrayView1<f32>> = data.outer_iter().collect();
    let labels_vec = labels.to_vec();

    // Parallel computation of silhouette scores
    let scores: Vec<f32> = parallel_map(&(0..n_samples).collect::<Vec<_>>(), |&i| {
        let label_i = labels_vec[i];

        // Compute a(i): mean distance to points in same cluster
        let mut a_sum = 0.0f32;
        let mut a_count = 0;
        for j in 0..n_samples {
            if i != j && labels_vec[j] == label_i {
                let diff = f32::simd_sub(&data_rows[i], &data_rows[j]);
                let dist = f32::simd_sum_squares(&diff.view()).sqrt();
                a_sum += dist;
                a_count += 1;
            }
        }
        let a_i = if a_count > 0 {
            a_sum / a_count as f32
        } else {
            0.0
        };

        // Compute b(i): mean distance to nearest cluster
        let mut b_i = f32::INFINITY;
        for k in 0..n_clusters {
            if k != label_i {
                let mut b_sum = 0.0f32;
                let mut b_count = 0;
                for j in 0..n_samples {
                    if labels_vec[j] == k {
                        let diff = f32::simd_sub(&data_rows[i], &data_rows[j]);
                        let dist = f32::simd_sum_squares(&diff.view()).sqrt();
                        b_sum += dist;
                        b_count += 1;
                    }
                }
                if b_count > 0 {
                    let mean_dist = b_sum / b_count as f32;
                    if mean_dist < b_i {
                        b_i = mean_dist;
                    }
                }
            }
        }

        // Compute silhouette score
        if a_i < b_i {
            1.0 - a_i / b_i
        } else if a_i > b_i {
            b_i / a_i - 1.0
        } else {
            0.0
        }
    });

    Ok(Array1::from_vec(scores))
}

/// Check if parallel processing is enabled
#[inline]
pub fn parallel_enabled() -> bool {
    is_parallel_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_parallel_enabled() {
        let enabled = parallel_enabled();
        println!("Parallel processing enabled: {}", enabled);
    }

    #[test]
    fn test_parallel_pairwise_distances() -> ClusterResult<()> {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .map_err(|_| ClusterError::InvalidInput("Failed to create test data".to_string()))?;

        let distances = parallel_pairwise_distances_f32(&data)?;

        assert_eq!(distances.shape(), &[3, 3]);
        assert_relative_eq!(distances[[0, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(distances[[0, 1]], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[[0, 2]], 1.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_parallel_distances_to_centroids() -> ClusterResult<()> {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0])
            .map_err(|_| ClusterError::InvalidInput("Failed to create test data".to_string()))?;
        let centroids = Array2::from_shape_vec((1, 2), vec![0.0, 0.0])
            .map_err(|_| ClusterError::InvalidInput("Failed to create centroids".to_string()))?;

        let distances = parallel_distances_to_centroids_f32(&data, &centroids)?;

        assert_eq!(distances.shape(), &[2, 1]);
        assert_relative_eq!(distances[[0, 0]], 2.0f32.sqrt(), epsilon = 1e-4);
        assert_relative_eq!(distances[[1, 0]], 8.0f32.sqrt(), epsilon = 1e-4);

        Ok(())
    }

    #[test]
    fn test_parallel_nearest_centroids() -> ClusterResult<()> {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
            .map_err(|_| ClusterError::InvalidInput("Failed to create test data".to_string()))?;
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 5.0])
            .map_err(|_| ClusterError::InvalidInput("Failed to create centroids".to_string()))?;

        let (labels, distances) = parallel_nearest_centroids_f32(&data, &centroids)?;

        assert_eq!(labels[0], 0); // Point [0,0] nearest to centroid 0
        assert_eq!(labels[1], 0); // Point [0.1,0.1] nearest to centroid 0
        assert_eq!(labels[2], 1); // Point [5,5] nearest to centroid 1
        assert_eq!(labels[3], 1); // Point [5.1,5.1] nearest to centroid 1

        // Check distances are reasonable
        assert!(distances[0] < 0.01); // Point [0,0] is at centroid 0
        assert!(distances[2] < 0.01); // Point [5,5] is at centroid 1

        Ok(())
    }

    #[test]
    fn test_parallel_kmeans_iteration() -> ClusterResult<()> {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
            .map_err(|_| ClusterError::InvalidInput("Failed to create test data".to_string()))?;
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 5.0])
            .map_err(|_| ClusterError::InvalidInput("Failed to create centroids".to_string()))?;

        let (new_centroids, labels, inertia) = parallel_kmeans_iteration_f32(&data, &centroids)?;

        assert_eq!(new_centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 4);
        assert!(inertia >= 0.0);

        // New centroids should be means of assigned points
        assert_relative_eq!(new_centroids[[0, 0]], 0.05, epsilon = 1e-4);
        assert_relative_eq!(new_centroids[[0, 1]], 0.05, epsilon = 1e-4);
        assert_relative_eq!(new_centroids[[1, 0]], 5.05, epsilon = 1e-4);
        assert_relative_eq!(new_centroids[[1, 1]], 5.05, epsilon = 1e-4);

        Ok(())
    }

    #[test]
    fn test_parallel_kmeans_iteration_inertia_parallel() -> ClusterResult<()> {
        // 3 well-separated clusters (4 points each) in 2D.
        // Cluster 0 centred at (0, 0)  — points: (1,0), (-1,0), (0,1), (0,-1)
        // Cluster 1 centred at (10, 0) — points: (11,0), (9,0), (10,1), (10,-1)
        // Cluster 2 centred at (0, 10) — points: (1,10), (-1,10), (0,11), (0,9)
        // Because the clusters are symmetric around their true means, the new
        // centroids computed by the k-means step equal the initial centroids, and
        // every point sits exactly 1 unit away from its centroid (dist² = 1).
        // => expected inertia = 12 × 1.0 = 12.0
        let data_flat: Vec<f32> = vec![
            1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, // cluster 0
            11.0, 0.0, 9.0, 0.0, 10.0, 1.0, 10.0, -1.0, // cluster 1
            1.0, 10.0, -1.0, 10.0, 0.0, 11.0, 0.0, 9.0, // cluster 2
        ];
        let data = Array2::from_shape_vec((12, 2), data_flat)
            .map_err(|_| ClusterError::InvalidInput("Failed to create test data".to_string()))?;

        let centroids_flat: Vec<f32> = vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
        let centroids = Array2::from_shape_vec((3, 2), centroids_flat)
            .map_err(|_| ClusterError::InvalidInput("Failed to create centroids".to_string()))?;

        let (new_centroids, labels, inertia) = parallel_kmeans_iteration_f32(&data, &centroids)?;

        // Serial reference loop to validate the parallel result
        let mut expected_inertia = 0.0f32;
        for i in 0..12_usize {
            let label = labels[i];
            for j in 0..2_usize {
                let diff = data[[i, j]] - new_centroids[[label, j]];
                expected_inertia += diff * diff;
            }
        }

        assert!(
            (inertia - expected_inertia).abs() < 1e-4,
            "Parallel inertia {inertia} differs from serial reference {expected_inertia}"
        );

        // Verify against the analytically known value
        assert!(
            (inertia - 12.0_f32).abs() < 1e-4,
            "Expected inertia ~12.0, got {inertia}"
        );

        Ok(())
    }
}
