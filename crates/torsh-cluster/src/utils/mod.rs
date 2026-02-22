//! Utility functions for clustering operations

pub mod adaptive;
pub mod distance;
pub mod drift_detection;
pub mod memory_efficient;
pub mod parallel;
pub mod preprocessing;
pub mod profiling;
pub mod validation;

pub use adaptive::{suggest_dbscan_params, suggest_epsilon};
pub use distance::*;
pub use drift_detection::{CompositeDriftDetector, DriftStatus, PageHinkleyTest, ADWIN, DDM};
pub use memory_efficient::{
    ChunkedDataProcessor, IncrementalCentroidUpdater, MemoryEfficientConfig,
};
pub use parallel::*;
pub use preprocessing::*;
pub use profiling::*;
pub use validation::*;

// Note: Both memory_efficient and profiling export estimate_memory_usage
// and suggest_clustering_strategy. Access them via the full path if needed:
// - utils::memory_efficient::estimate_memory_usage
// - utils::profiling::estimate_memory_usage
// - utils::memory_efficient::suggest_clustering_strategy
// - utils::profiling::suggest_clustering_algorithm
