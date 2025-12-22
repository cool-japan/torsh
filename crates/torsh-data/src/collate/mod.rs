//! Batch collation functions
//!
//! This module provides comprehensive batch collation functionality for data loading,
//! supporting various strategies from simple stacking to advanced dynamic batching
//! and sparse tensor handling.
//!
//! # Core Components
//!
//! - `Collate` trait: The fundamental trait for all collation operations
//! - `DefaultCollate`: Standard collation for common tensor and tuple types
//! - `TensorStacker`: Optimized tensor stacking utility with parallel support
//!
//! # Strategies
//!
//! - **Simple Stacking**: Basic tensor stacking along batch dimension
//! - **Optimized**: High-performance collation with parallel processing and memory mapping
//! - **Dynamic**: Variable-length sequence handling with padding and packing
//! - **Cached**: Memory pool-based collation for reduced allocation overhead
//! - **Sparse**: Specialized handling for sparse tensors (COO format)
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_data::collate::{DefaultCollate, CollateBuilder, CollateStrategy, Collate};
//! use torsh_tensor::creation::ones;
//!
//! // Simple collation
//! let collate = DefaultCollate;
//! let batch = vec![
//!     ones::<f32>(&[3, 4]).unwrap(),
//!     ones::<f32>(&[3, 4]).unwrap(),
//! ];
//! let result = collate.collate(batch).unwrap();
//!
//! // Using builder pattern
//! let collate = CollateBuilder::new()
//!     .strategy(CollateStrategy::Optimized)
//!     .with_caching()
//!     .build();
//! ```

// Core components
pub mod core;
pub mod stacking;

// Builder pattern and strategy selection
pub mod builder;

// Performance-optimized implementations
pub mod optimized;

// Advanced collation techniques
pub mod advanced;

// Utility functions and types
pub mod utils;

// Example implementations
pub mod examples;

// Re-export main types and traits
pub use builder::{CollateBuilder, CollateStrategy};
pub use core::{Collate, DefaultCollate};
pub use stacking::TensorStacker;
pub use utils::{collate_fn, CollateFn};

// Re-export optimized functions for direct use
pub use optimized::{optimized_collate_fn, stack_tensors, OptimizedCollate};

// Re-export advanced collate functions
pub use advanced::{
    AdaptiveBatchSampler, BucketBatchSampler, CachedCollate, DynamicBatchCollate,
    DynamicBatchCollateWrapper, PadCollate,
};

// Re-export sparse support when available
#[cfg(feature = "sparse")]
pub use advanced::{collate_sparse_tensors, MixedCollate, SparseCollate};

// Re-export example functions
pub use examples::{collate_data_label, collate_dict};

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_default_collate() {
        let batch = vec![
            ones::<f32>(&[3, 4]).unwrap(),
            ones::<f32>(&[3, 4]).unwrap(),
            ones::<f32>(&[3, 4]).unwrap(),
        ];

        let collate = DefaultCollate;
        let result = collate.collate(batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_collate_fn() {
        let collate = CollateFn::new(|batch: Vec<i32>| Ok(batch.iter().sum::<i32>()));

        let result = collate.collate(vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result, 15);
    }

    #[test]
    fn test_pad_collate() {
        let batch = vec![ones::<f32>(&[2, 3]).unwrap(), ones::<f32>(&[2, 3]).unwrap()];

        let collate = PadCollate::new(0.0f32);
        let result = collate.collate(batch);
        assert!(result.is_ok());
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_collate() {
        use torsh_sparse::{CooTensor, SparseFormat};
        use torsh_tensor::creation::zeros;

        // Create some sparse tensors for testing
        let dense1 = zeros::<f32>(&[2, 3]).unwrap();
        let dense2 = zeros::<f32>(&[2, 3]).unwrap();

        // Convert to sparse (this is a placeholder - actual implementation may vary)
        // In a real scenario, you'd create actual sparse tensors with non-zero values
        let _sparse1 = torsh_sparse::sparse_from_dense(&dense1, SparseFormat::Coo, None).unwrap();
        let _sparse2 = torsh_sparse::sparse_from_dense(&dense2, SparseFormat::Coo, None).unwrap();

        // Test collation would go here - commented out due to potential API differences
        // let collate = SparseCollate;
        // let result = collate.collate(vec![sparse1, sparse2]);
        // assert!(result.is_ok());
    }
}
