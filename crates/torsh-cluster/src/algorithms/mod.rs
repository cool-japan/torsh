//! Clustering algorithm implementations
//!
//! This module contains implementations of various clustering algorithms,
//! all built on top of the SciRS2 ecosystem for high performance and
//! numerical stability.

pub mod dbscan;
pub mod gaussian_mixture;
pub mod hierarchical;
pub mod incremental;
pub mod kmeans;
pub mod optics;
pub mod spectral;

// Re-export main algorithm types
pub use dbscan::{DBSCANConfig, DBSCANResult, HDBSCANConfig, HDBSCANResult, DBSCAN, HDBSCAN};
pub use gaussian_mixture::{GMConfig, GMResult, GaussianMixture};
pub use hierarchical::{AgglomerativeClustering, HierarchicalResult, Linkage};
pub use incremental::{
    IncrementalClustering, OnlineKMeans, OnlineKMeansConfig, OnlineKMeansResult,
};
pub use kmeans::{InitMethod, KMeans, KMeansConfig, KMeansResult};
pub use optics::{OPTICSConfig, OPTICSResult, OPTICS};
pub use spectral::{SpectralClustering, SpectralConfig, SpectralResult};
