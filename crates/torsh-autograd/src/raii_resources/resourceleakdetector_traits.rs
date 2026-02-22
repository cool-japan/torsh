//! # ResourceLeakDetector - Trait Implementations
//!
//! This module contains trait implementations for `ResourceLeakDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceLeakDetector;

impl Default for ResourceLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}
