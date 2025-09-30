//! K-means++ initialization strategy

use super::InitializationStrategy;
use crate::error::{ClusterError, ClusterResult};
use scirs2_core::random::{Random, Rng};
use torsh_tensor::Tensor;

/// K-means++ initialization for better cluster initialization
#[derive(Debug, Default)]
pub struct KMeansPlusPlus;

impl InitializationStrategy for KMeansPlusPlus {
    fn initialize(
        &self,
        data: &Tensor,
        n_clusters: usize,
        seed: Option<u64>,
    ) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];

        if n_clusters > n_samples {
            return Err(ClusterError::InvalidClusters(n_clusters));
        }

        let mut rng = Random::seed(seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        }));

        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let mut centroids_data = Vec::with_capacity(n_clusters * n_features);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            centroids_data.push(data_vec[first_idx * n_features + j]);
        }

        // Choose remaining centroids using K-means++ strategy
        for k in 1..n_clusters {
            let mut distances = vec![f32::INFINITY; n_samples];

            // Compute minimum distance to existing centroids for each point
            for i in 0..n_samples {
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff =
                            data_vec[i * n_features + j] - centroids_data[c * n_features + j];
                        dist += diff * diff;
                    }
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to squared distance
            let total_dist: f32 = distances.iter().sum();
            if total_dist <= 0.0 {
                // Fallback to random selection
                let idx = rng.gen_range(0..n_samples);
                for j in 0..n_features {
                    centroids_data.push(data_vec[idx * n_features + j]);
                }
            } else {
                let threshold = rng.random::<f32>() * total_dist;
                let mut cumsum = 0.0;
                let mut selected_idx = 0;

                for (i, &distance) in distances.iter().enumerate() {
                    cumsum += distance;
                    if cumsum >= threshold {
                        selected_idx = i;
                        break;
                    }
                }

                for j in 0..n_features {
                    centroids_data.push(data_vec[selected_idx * n_features + j]);
                }
            }
        }

        Tensor::from_vec(centroids_data, &[n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "K-means++"
    }
}
