//! Sparse neural network layers
//!
//! This module provides a comprehensive collection of neural network layers
//! optimized for sparse tensors, organized by functionality.

pub mod activation;
pub mod attention;
pub mod conv;
pub mod linear;
pub mod norm;
pub mod pooling;

// Re-export all layer types for convenience
pub use activation::{SparseGELU, SparseLeakyReLU, SparseReLU, SparseSigmoid, SparseTanh};
pub use attention::SparseAttention;
pub use conv::{GraphConvolution, SparseConv1d, SparseConv2d};
pub use linear::{SparseEmbedding, SparseLinear};
pub use norm::{SparseBatchNorm, SparseLayerNorm};
pub use pooling::{
    SparseAdaptiveAvgPool2d, SparseAdaptiveMaxPool2d, SparseAvgPool1d, SparseAvgPool2d,
    SparseMaxPool1d, SparseMaxPool2d,
};
