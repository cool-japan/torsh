//! Forgy initialization strategy

use super::InitializationStrategy;
use crate::error::{ClusterError, ClusterResult};
use scirs2_core::random::Random;
use torsh_tensor::Tensor;

/// Forgy initialization: randomly select k data points as centroids
#[derive(Debug, Default)]
pub struct Forgy;

impl InitializationStrategy for Forgy {
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
        let mut selected = std::collections::HashSet::new();
        let mut centroids_data = Vec::with_capacity(n_clusters * n_features);

        for _ in 0..n_clusters {
            let mut idx = rng.gen_range(0..n_samples);
            while selected.contains(&idx) {
                idx = rng.gen_range(0..n_samples);
            }
            selected.insert(idx);

            for j in 0..n_features {
                centroids_data.push(data_vec[idx * n_features + j]);
            }
        }

        Tensor::from_vec(centroids_data, &[n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    fn name(&self) -> &str {
        "Forgy"
    }
}
