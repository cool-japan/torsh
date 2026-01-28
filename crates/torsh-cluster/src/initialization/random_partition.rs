//! Random partition initialization strategy

use super::InitializationStrategy;
use crate::error::{ClusterError, ClusterResult};
use scirs2_core::random::Random;
use torsh_tensor::Tensor;

/// Random partition initialization: randomly assign points to clusters, then compute centroids
#[derive(Debug, Default)]
pub struct RandomPartition;

impl InitializationStrategy for RandomPartition {
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
                .expect("system time should be after UNIX_EPOCH")
                .as_secs()
        }));

        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        // Randomly assign points to clusters
        let mut cluster_assignments = Vec::new();
        for _ in 0..n_samples {
            cluster_assignments.push(rng.gen_range(0..n_clusters));
        }

        // Compute centroid for each cluster
        let mut centroids_data = vec![0.0; n_clusters * n_features];
        let mut cluster_counts = vec![0; n_clusters];

        for i in 0..n_samples {
            let cluster = cluster_assignments[i];
            cluster_counts[cluster] += 1;
            for j in 0..n_features {
                centroids_data[cluster * n_features + j] += data_vec[i * n_features + j];
            }
        }

        // Average to get centroids
        for k in 0..n_clusters {
            if cluster_counts[k] > 0 {
                for j in 0..n_features {
                    centroids_data[k * n_features + j] /= cluster_counts[k] as f32;
                }
            }
        }

        Tensor::from_vec(centroids_data, &[n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "Random Partition"
    }
}
