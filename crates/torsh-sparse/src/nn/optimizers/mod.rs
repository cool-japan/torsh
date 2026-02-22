//! Sparse neural network optimizers

pub mod sgd;

// Re-export commonly used optimizer types
pub use sgd::{SparseSGD, SparseSGDBuilder};
