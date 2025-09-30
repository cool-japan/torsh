//! Sparse neural network components
//!
//! This module provides neural network layers, optimizers, and utilities
//! that are optimized for sparse tensors, enabling efficient training
//! and inference with sparse models.

// Core components
pub mod common;

// Layer implementations
pub mod attention;
pub mod convolution;
pub mod graph;
pub mod layers;

// Optimization algorithms
pub mod optimizers;

// Re-export commonly used types from submodules
pub use common::{
    traits::{
        ModelSparsityAnalysis, SavingsEstimate, SparseActivation, SparseAnalyzer, SparseConverter,
        SparseInitializer, SparseLayer, SparseNormalization, SparseOptimizer, SparsePooling,
        SparsePruner,
    },
    types::{
        SparseFormat, SparseInitConfig, SparseInitStrategy, SparseLayerConfig, SparseOps,
        SparseStats,
    },
    utils::{
        SparseAnalyzer as SparsePatternAnalyzer, SparseConverter as SparseFormatConverter,
        SparsePatternAnalysis, SparseWeightGenerator,
    },
};

// Re-export layer types
pub use layers::{SparseEmbedding, SparseEmbeddingStats, SparseLinear, SparseMemoryStats};

// Re-export advanced layer types
pub use attention::SparseAttention;
pub use convolution::SparseConv2d;
pub use graph::GraphConvolution;

// Re-export optimizer types
pub use crate::optimizers::{SparseAdam, SparseSGD};

// Convenience type aliases
/// Type alias for sparse tensor format
pub type Format = SparseFormat;

/// Type alias for sparse layer configuration
pub type LayerConfig = SparseLayerConfig;

/// Type alias for sparse initialization configuration
pub type InitConfig = SparseInitConfig;
