//! Distance metrics for clustering
//!
//! This module provides optimized distance metrics using SciRS2's SIMD operations
//! for high-performance distance computations in clustering algorithms.

use crate::error::{ClusterError, ClusterResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use torsh_tensor::Tensor;

/// Distance metric trait
pub trait DistanceMetric {
    /// Compute distance between two points
    fn distance(&self, x: &[f32], y: &[f32]) -> f64;

    /// Compute pairwise distances between all points
    fn pairwise_distances(&self, data: &Tensor) -> ClusterResult<Tensor>;

    /// Get metric name
    fn name(&self) -> &str;
}

/// Euclidean distance metric
#[derive(Debug, Default)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() {
            return f64::INFINITY;
        }

        let mut sum = 0.0;
        for i in 0..x.len() {
            let diff = (x[i] - y[i]) as f64;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn pairwise_distances(&self, data: &Tensor) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        let mut distances = Vec::with_capacity(n_samples * n_samples);

        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut dist = 0.0;
                for k in 0..n_features {
                    let diff = data_vec[i * n_features + k] - data_vec[j * n_features + k];
                    dist += (diff * diff) as f64;
                }
                distances.push(dist.sqrt() as f32);
            }
        }

        Tensor::from_vec(distances, &[n_samples, n_samples]).map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "euclidean"
    }
}

/// Manhattan distance metric
#[derive(Debug, Default)]
pub struct ManhattanDistance;

impl DistanceMetric for ManhattanDistance {
    fn distance(&self, x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() {
            return f64::INFINITY;
        }

        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += (x[i] - y[i]).abs() as f64;
        }
        sum
    }

    fn pairwise_distances(&self, data: &Tensor) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        let mut distances = Vec::with_capacity(n_samples * n_samples);

        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut dist = 0.0_f32;
                for k in 0..n_features {
                    dist += (data_vec[i * n_features + k] - data_vec[j * n_features + k]).abs();
                }
                distances.push(dist);
            }
        }

        Tensor::from_vec(distances, &[n_samples, n_samples]).map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "manhattan"
    }
}

/// Cosine distance metric
#[derive(Debug, Default)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() {
            return f64::INFINITY;
        }

        let mut dot_product = 0.0;
        let mut norm_x = 0.0;
        let mut norm_y = 0.0;

        for i in 0..x.len() {
            dot_product += (x[i] * y[i]) as f64;
            norm_x += (x[i] * x[i]) as f64;
            norm_y += (y[i] * y[i]) as f64;
        }

        let cosine_similarity = dot_product / (norm_x.sqrt() * norm_y.sqrt());
        1.0 - cosine_similarity
    }

    fn pairwise_distances(&self, data: &Tensor) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        let mut distances = Vec::with_capacity(n_samples * n_samples);

        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut dot_product = 0.0;
                let mut norm_i = 0.0;
                let mut norm_j = 0.0;

                for k in 0..n_features {
                    let val_i = data_vec[i * n_features + k] as f64;
                    let val_j = data_vec[j * n_features + k] as f64;

                    dot_product += val_i * val_j;
                    norm_i += val_i * val_i;
                    norm_j += val_j * val_j;
                }

                let cosine_similarity = dot_product / (norm_i.sqrt() * norm_j.sqrt());
                let cosine_distance = 1.0 - cosine_similarity;
                distances.push(cosine_distance as f32);
            }
        }

        Tensor::from_vec(distances, &[n_samples, n_samples]).map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "cosine"
    }
}

/// Convenience functions for distance computation
pub fn euclidean_distance(x: &[f32], y: &[f32]) -> f64 {
    EuclideanDistance.distance(x, y)
}

pub fn manhattan_distance(x: &[f32], y: &[f32]) -> f64 {
    ManhattanDistance.distance(x, y)
}

pub fn cosine_distance(x: &[f32], y: &[f32]) -> f64 {
    CosineDistance.distance(x, y)
}

// ================================================================================================
// SIMD-Optimized Distance Computations (SciRS2 POLICY Compliant)
// ================================================================================================

/// Compute Euclidean distance using SIMD operations from scirs2_core
///
/// This function leverages SciRS2's SIMD operations for high-performance
/// distance computation on modern CPUs with vector instructions.
///
/// # Arguments
/// * `x` - First point as ArrayView1
/// * `y` - Second point as ArrayView1
///
/// # Returns
/// Euclidean distance between x and y
#[inline]
pub fn euclidean_distance_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    if x.len() != y.len() {
        return f32::INFINITY;
    }

    // Use SciRS2's SIMD operations for optimal performance
    let diff = f32::simd_sub(x, y);
    let sum_squares = f32::simd_sum_squares(&diff.view());
    sum_squares.sqrt()
}

/// Compute Euclidean distance using SIMD operations (f64 version)
#[inline]
pub fn euclidean_distance_simd_f64(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    if x.len() != y.len() {
        return f64::INFINITY;
    }

    // Use SciRS2's SIMD operations for optimal performance
    let diff = f64::simd_sub(x, y);
    let sum_squares = f64::simd_sum_squares(&diff.view());
    sum_squares.sqrt()
}

/// Compute squared Euclidean distance using SIMD (avoids sqrt for better performance)
#[inline]
pub fn euclidean_distance_squared_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    if x.len() != y.len() {
        return f32::INFINITY;
    }

    // Use SciRS2's SIMD operations - squared distance is faster (no sqrt)
    let diff = f32::simd_sub(x, y);
    f32::simd_sum_squares(&diff.view())
}

/// Compute Manhattan distance using SIMD operations
#[inline]
pub fn manhattan_distance_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    if x.len() != y.len() {
        return f32::INFINITY;
    }

    // Use SciRS2's SIMD operations for optimal performance
    let diff = f32::simd_sub(x, y);
    let abs_diff = f32::simd_abs(&diff.view());
    f32::simd_sum(&abs_diff.view())
}

/// Compute dot product using SIMD operations
#[inline]
pub fn dot_product_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    if x.len() != y.len() {
        return 0.0;
    }

    // Use SciRS2's highly optimized SIMD dot product
    f32::simd_dot(x, y)
}

/// Compute cosine similarity using SIMD operations
#[inline]
pub fn cosine_similarity_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    if x.len() != y.len() {
        return 0.0;
    }

    // Use SciRS2's SIMD operations for optimal performance
    let dot = f32::simd_dot(x, y);
    let norm_x = f32::simd_norm(x);
    let norm_y = f32::simd_norm(y);

    if norm_x == 0.0 || norm_y == 0.0 {
        return 0.0;
    }

    dot / (norm_x * norm_y)
}

/// Compute cosine distance using SIMD operations
#[inline]
pub fn cosine_distance_simd_f32(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    1.0 - cosine_similarity_simd_f32(x, y)
}

/// Batch computation of Euclidean distances between multiple points and centroids using SIMD
///
/// # Arguments
/// * `points` - Data points (n_samples x n_features)
/// * `centroids` - Cluster centroids (n_clusters x n_features)
///
/// # Returns
/// Distance matrix (n_samples x n_clusters)
pub fn batch_euclidean_distances_simd_f32(
    points: &[f32],
    centroids: &[f32],
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
) -> Vec<f32> {
    let mut distances = vec![0.0f32; n_samples * n_clusters];

    for i in 0..n_samples {
        let point_start = i * n_features;
        let point_end = point_start + n_features;
        let point_slice = &points[point_start..point_end];
        let point_array = Array1::from_vec(point_slice.to_vec());
        let point_view = point_array.view();

        for k in 0..n_clusters {
            let centroid_start = k * n_features;
            let centroid_end = centroid_start + n_features;
            let centroid_slice = &centroids[centroid_start..centroid_end];
            let centroid_array = Array1::from_vec(centroid_slice.to_vec());
            let centroid_view = centroid_array.view();

            distances[i * n_clusters + k] =
                euclidean_distance_simd_f32(&point_view, &centroid_view);
        }
    }

    distances
}

/// Batch computation of squared Euclidean distances (faster, avoids sqrt)
pub fn batch_euclidean_distances_squared_simd_f32(
    points: &[f32],
    centroids: &[f32],
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
) -> Vec<f32> {
    let mut distances = vec![0.0f32; n_samples * n_clusters];

    for i in 0..n_samples {
        let point_start = i * n_features;
        let point_end = point_start + n_features;
        let point_slice = &points[point_start..point_end];
        let point_array = Array1::from_vec(point_slice.to_vec());
        let point_view = point_array.view();

        for k in 0..n_clusters {
            let centroid_start = k * n_features;
            let centroid_end = centroid_start + n_features;
            let centroid_slice = &centroids[centroid_start..centroid_end];
            let centroid_array = Array1::from_vec(centroid_slice.to_vec());
            let centroid_view = centroid_array.view();

            distances[i * n_clusters + k] =
                euclidean_distance_squared_simd_f32(&point_view, &centroid_view);
        }
    }

    distances
}

/// Check if SIMD operations are available on the current platform
#[inline]
pub fn simd_available() -> bool {
    f32::simd_available()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_simd_euclidean_distance() {
        let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0f32, 5.0, 6.0]);

        let dist_simd = euclidean_distance_simd_f32(&x.view(), &y.view());
        let expected = ((3.0f32 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0) as f32).sqrt();

        assert_relative_eq!(dist_simd, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0f32, 5.0, 6.0]);

        let dist_simd = manhattan_distance_simd_f32(&x.view(), &y.view());
        let expected = 9.0f32; // |3| + |3| + |3| = 9

        assert_relative_eq!(dist_simd, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_simd_dot_product() {
        let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0f32, 5.0, 6.0]);

        let dot_simd = dot_product_simd_f32(&x.view(), &y.view());
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // = 32.0

        assert_relative_eq!(dot_simd, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_simd_cosine_similarity() {
        let x = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
        let y = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);

        let sim = cosine_similarity_simd_f32(&x.view(), &y.view());
        assert_relative_eq!(sim, 1.0, epsilon = 1e-5);

        let z = Array1::from_vec(vec![0.0f32, 1.0, 0.0]);
        let sim_orthogonal = cosine_similarity_simd_f32(&x.view(), &z.view());
        assert_relative_eq!(sim_orthogonal, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_simd_availability() {
        // Just check that the function runs without crashing
        let available = simd_available();
        println!("SIMD available: {}", available);
    }

    #[test]
    fn test_batch_euclidean_distances_simd() {
        let points = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 points, 3 features each
        let centroids = vec![0.0f32, 0.0, 0.0]; // 1 centroid, 3 features

        let distances = batch_euclidean_distances_simd_f32(&points, &centroids, 2, 3, 1);

        // Distance from [1,2,3] to [0,0,0] = sqrt(1 + 4 + 9) = sqrt(14)
        // Distance from [4,5,6] to [0,0,0] = sqrt(16 + 25 + 36) = sqrt(77)
        assert_relative_eq!(distances[0], 14.0f32.sqrt(), epsilon = 1e-4);
        assert_relative_eq!(distances[1], 77.0f32.sqrt(), epsilon = 1e-4);
    }
}
