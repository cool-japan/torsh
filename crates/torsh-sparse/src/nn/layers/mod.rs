//! Sparse neural network layers

pub mod embedding;
pub mod linear;

// Re-export commonly used layer types
pub use embedding::{SparseEmbedding, SparseEmbeddingStats};
pub use linear::SparseMemoryStats;
// Import SparseLinear from the main layers module that has pruning methods
pub use crate::layers::linear::SparseLinear;
