//! Initialization strategies for clustering algorithms

pub mod forgy;
pub mod kmeans_plus_plus;
pub mod random_partition;

pub use forgy::Forgy;
pub use kmeans_plus_plus::KMeansPlusPlus;
pub use random_partition::RandomPartition;

use crate::error::ClusterResult;
use torsh_tensor::Tensor;

/// Trait for initialization strategies
pub trait InitializationStrategy {
    /// Initialize cluster centers
    fn initialize(
        &self,
        data: &Tensor,
        n_clusters: usize,
        seed: Option<u64>,
    ) -> ClusterResult<Tensor>;

    /// Get strategy name
    fn name(&self) -> &str;
}
