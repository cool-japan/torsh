//! # AutogradScope - Trait Implementations
//!
//! This module contains trait implementations for `AutogradScope`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AutogradScope;

impl Default for AutogradScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutogradScope {
    fn drop(&mut self) {
        if !self.resources.is_empty() {
            println!(
                "AutogradScope dropping {} resources after {:?}",
                self.resources.len(),
                self.duration()
            );
        }
    }
}
