//! # QuantizationStrategy - Trait Implementations
//!
//! This module contains trait implementations for `QuantizationStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QuantizationStrategy;

impl Default for QuantizationStrategy {
    fn default() -> Self {
        Self::StaticInt8
    }
}
