//! Sparse neural network layers and operations - Clean Modular Interface
//!
//! This module provides neural network layers that are optimized for sparse tensors,
//! now organized into specialized modules for improved maintainability and functionality.
//!
//! # Architecture
//!
//! The sparse neural network system is organized into specialized modules:
//!
//! - **layers**: Neural network layer implementations organized by type
//!   - **linear**: Linear and embedding layers (SparseLinear, SparseEmbedding)
//!   - **conv**: Convolution layers (SparseConv1d, SparseConv2d, GraphConvolution)
//!   - **attention**: Attention mechanisms (SparseAttention)
//!   - **pooling**: Pooling operations (Max, Average, Adaptive variants)
//!   - **norm**: Normalization layers (SparseBatchNorm, SparseLayerNorm)
//!   - **activation**: Activation functions (ReLU, Sigmoid, Tanh, GELU, LeakyReLU)
//! - **optimizers**: Sparse optimization algorithms (SGD, Adam, AdamW, RMSprop)
//!
//! All layers and optimizers are fully backward compatible and accessible through this interface.

// Modular architecture imports
pub mod layers;
pub mod optimizers;

// Re-export all layer categories
pub use layers::{
    linear, conv, attention, pooling, norm, activation
};

// Re-export optimizers
pub use optimizers::*;

// Complete backward compatibility re-exports for all components
// ============================================================================

// Linear/Dense Layers
pub use layers::linear::{
    SparseLinear,
    SparseEmbedding,
};

// Convolution Layers
pub use layers::conv::{
    GraphConvolution,
    SparseConv1d,
    SparseConv2d,
};

// Attention Mechanisms
pub use layers::attention::{
    SparseAttention,
};

// Pooling Layers
pub use layers::pooling::{
    SparseMaxPool1d,
    SparseMaxPool2d,
    SparseAvgPool1d,
    SparseAvgPool2d,
    SparseAdaptiveMaxPool2d,
    SparseAdaptiveAvgPool2d,
};

// Normalization Layers
pub use layers::norm::{
    SparseBatchNorm,
    SparseLayerNorm,
};

// Activation Functions
pub use layers::activation::{
    SparseReLU,
    SparseSigmoid,
    SparseTanh,
    SparseGELU,
    SparseLeakyReLU,
};

// Optimizers
pub use optimizers::{
    SparseOptimizer,
    SparseSGD,
    SparseAdam,
    SparseAdamW,
    SparseRMSprop,
};

/// Performance benchmarking utilities for sparse operations
pub mod benchmarks {
    use super::*;

    /// Benchmark sparse layer performance
    pub fn benchmark_layer_performance<L>(layer: &L, input_sizes: &[usize]) -> Vec<f64>
    where
        L: std::fmt::Debug,
    {
        input_sizes.iter().map(|&size| {
            // Placeholder benchmarking logic
            let start = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_nanos(size as u64));
            start.elapsed().as_secs_f64()
        }).collect()
    }

    /// Get sparse operation speedup factor
    pub fn get_sparse_speedup_factor(dense_time: f64, sparse_time: f64) -> f64 {
        dense_time / sparse_time
    }
}

/// Sparse operation selection utilities
pub mod selection {
    use super::*;

    /// Automatically select the best sparse layer configuration based on sparsity
    pub fn auto_select_sparsity_level(data_sparsity: f32) -> f32 {
        match data_sparsity {
            s if s > 0.9 => 0.95,  // Very sparse data
            s if s > 0.7 => 0.8,   // Moderately sparse data
            s if s > 0.5 => 0.6,   // Somewhat sparse data
            _ => 0.3,              // Dense data
        }
    }

    /// Select optimal pooling strategy for sparse tensors
    pub fn auto_select_pooling_strategy(sparsity: f32, size: (usize, usize)) -> String {
        if sparsity > 0.8 {
            if size.0 * size.1 > 1000000 {
                "adaptive_max".to_string()
            } else {
                "max".to_string()
            }
        } else {
            "average".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::{CooTensor, CsrTensor, TorshResult};
    use torsh_core::Shape;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_modular_sparse_system() {
        // Test that all modules are accessible and working

        // Test linear layers
        let linear = SparseLinear::new(10, 5, 0.5, true).unwrap();
        assert_eq!(linear.sparsity(), 0.5);

        // Test embedding layers
        let embedding = SparseEmbedding::new(100, 50, 0.3).unwrap();
        assert_eq!(embedding.sparsity(), 0.3);

        // Test convolution layers
        let conv1d = SparseConv1d::new(3, 16, 3, None, None, None, 0.4, true).unwrap();
        assert_eq!(conv1d.kernel_sparsity(), 0.4);

        let conv2d = SparseConv2d::new(3, 16, (3, 3), None, None, None, 0.6, true).unwrap();
        assert_eq!(conv2d.kernel_sparsity(), 0.6);

        // Test graph convolution
        let graph_conv = GraphConvolution::new(10, 5, true, true, true).unwrap();
        assert_eq!(graph_conv.in_features(), 10);
        assert_eq!(graph_conv.out_features(), 5);

        // Test attention mechanisms
        let attention = SparseAttention::new(64, 8, 0.1, 0.1).unwrap();
        assert_eq!(attention.model_dim(), 64);
        assert_eq!(attention.num_heads(), 8);

        // Test pooling layers
        let maxpool2d = SparseMaxPool2d::new((2, 2), None, None, None);
        let avgpool2d = SparseAvgPool2d::new((2, 2), None, None, false);
        let maxpool1d = SparseMaxPool1d::new(2, None, None, None);
        let avgpool1d = SparseAvgPool1d::new(2, None, None, false);

        // Test adaptive pooling
        let adaptive_max = SparseAdaptiveMaxPool2d::new((1, 1));
        let adaptive_avg = SparseAdaptiveAvgPool2d::new((1, 1));

        // Test normalization layers
        let batch_norm = SparseBatchNorm::new(10, 1e-5, 0.1, true).unwrap();
        let layer_norm = SparseLayerNorm::new(vec![10], 1e-5, true).unwrap();

        // Test activation functions
        let relu = SparseReLU::new(false);
        let sigmoid = SparseSigmoid::new();
        let tanh = SparseTanh::new();
        let gelu = SparseGELU::new();
        let leaky_relu = SparseLeakyReLU::new(0.01);

        // Test optimizers
        let sgd = SparseSGD::default(0.01);
        assert_eq!(sgd.lr(), 0.01);

        let adam = SparseAdam::default(0.001);
        assert_eq!(adam.lr(), 0.001);

        let adamw = SparseAdamW::default(0.001);
        assert_eq!(adamw.lr(), 0.001);

        let rmsprop = SparseRMSprop::default(0.01);
        assert_eq!(rmsprop.lr(), 0.01);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure that the modular system maintains full backward compatibility
        // All original function names and APIs should work exactly as before

        // Test layer creation with original API
        let linear = SparseLinear::new(128, 64, 0.7, true).unwrap();
        assert_eq!(linear.num_parameters() > 0, true);

        let embedding = SparseEmbedding::new(1000, 128, 0.5).unwrap();
        assert_eq!(embedding.num_parameters() > 0, true);

        // Test optimizer creation with original API
        let sgd = SparseSGD::new(0.01, 0.9, 1e-4, false);
        assert_eq!(sgd.lr(), 0.01);

        let adam = SparseAdam::new(0.001, 0.9, 0.999, 1e-8, 0.0, false);
        assert_eq!(adam.lr(), 0.001);
    }

    #[test]
    fn test_selection_utilities() {
        // Test auto-selection functions
        let sparsity = selection::auto_select_sparsity_level(0.95);
        assert_eq!(sparsity, 0.95);

        let sparsity = selection::auto_select_sparsity_level(0.3);
        assert_eq!(sparsity, 0.3);

        let pooling_strategy = selection::auto_select_pooling_strategy(0.9, (1000, 1000));
        assert_eq!(pooling_strategy, "adaptive_max");

        let pooling_strategy = selection::auto_select_pooling_strategy(0.3, (100, 100));
        assert_eq!(pooling_strategy, "average");
    }

    #[test]
    fn test_benchmarking_utilities() {
        // Test benchmarking functions
        let dummy_layer = "test_layer";
        let input_sizes = vec![100, 1000, 10000];
        let times = benchmarks::benchmark_layer_performance(&dummy_layer, &input_sizes);
        assert_eq!(times.len(), 3);

        let speedup = benchmarks::get_sparse_speedup_factor(2.0, 1.0);
        assert_eq!(speedup, 2.0);
    }

    #[test]
    fn test_modular_structure_integrity() {
        // Test that all modules are properly accessible

        // Linear layers module
        let _: SparseLinear = SparseLinear::new(10, 5, 0.5, false).unwrap();
        let _: SparseEmbedding = SparseEmbedding::new(100, 50, 0.3).unwrap();

        // Convolution layers module
        let _: SparseConv1d = SparseConv1d::new(3, 16, 3, None, None, None, 0.4, false).unwrap();
        let _: SparseConv2d = SparseConv2d::new(3, 16, (3, 3), None, None, None, 0.4, false).unwrap();
        let _: GraphConvolution = GraphConvolution::new(10, 5, false, false, false).unwrap();

        // Attention module
        let _: SparseAttention = SparseAttention::new(64, 8, 0.1, 0.1).unwrap();

        // Pooling module
        let _: SparseMaxPool1d = SparseMaxPool1d::new(2, None, None, None);
        let _: SparseMaxPool2d = SparseMaxPool2d::new((2, 2), None, None, None);
        let _: SparseAvgPool1d = SparseAvgPool1d::new(2, None, None, false);
        let _: SparseAvgPool2d = SparseAvgPool2d::new((2, 2), None, None, false);
        let _: SparseAdaptiveMaxPool2d = SparseAdaptiveMaxPool2d::new((1, 1));
        let _: SparseAdaptiveAvgPool2d = SparseAdaptiveAvgPool2d::new((1, 1));

        // Normalization module
        let _: SparseBatchNorm = SparseBatchNorm::new(10, 1e-5, 0.1, true).unwrap();
        let _: SparseLayerNorm = SparseLayerNorm::new(vec![10], 1e-5, true).unwrap();

        // Activation module
        let _: SparseReLU = SparseReLU::new(false);
        let _: SparseSigmoid = SparseSigmoid::new();
        let _: SparseTanh = SparseTanh::new();
        let _: SparseGELU = SparseGELU::new();
        let _: SparseLeakyReLU = SparseLeakyReLU::new(0.01);

        // Optimizers module
        let _: SparseSGD = SparseSGD::default(0.01);
        let _: SparseAdam = SparseAdam::default(0.001);
        let _: SparseAdamW = SparseAdamW::default(0.001);
        let _: SparseRMSprop = SparseRMSprop::default(0.01);
    }
}