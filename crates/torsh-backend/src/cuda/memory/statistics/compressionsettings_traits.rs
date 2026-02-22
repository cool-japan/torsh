//! # CompressionSettings - Trait Implementations
//!
//! This module contains trait implementations for `CompressionSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CompressionAlgorithm, CompressionSettings};

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_ratio: 0.3,
            algorithm: CompressionAlgorithm::Zstd,
        }
    }
}
