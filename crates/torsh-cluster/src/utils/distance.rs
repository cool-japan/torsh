//! Distance metrics for clustering

use crate::error::{ClusterError, ClusterResult};
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
