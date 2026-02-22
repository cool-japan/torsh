//! Common components for sparse neural networks

pub mod traits;
pub mod types;
pub mod utils;

// Re-export commonly used items
pub use traits::{
    ModelSparsityAnalysis, SavingsEstimate, SparseActivation, SparseAnalyzer, SparseConverter,
    SparseInitializer, SparseLayer, SparseNormalization, SparseOptimizer, SparsePooling,
    SparsePruner,
};
pub use types::{
    SparseFormat, SparseInitConfig, SparseInitStrategy, SparseLayerConfig, SparseOps, SparseStats,
};
pub use utils::{
    SparseAnalyzer as SparsePatternAnalyzer, SparseConverter as SparseFormatConverter,
    SparsePatternAnalysis, SparseWeightGenerator,
};
