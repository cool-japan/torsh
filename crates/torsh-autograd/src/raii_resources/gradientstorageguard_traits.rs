//! # GradientStorageGuard - Trait Implementations
//!
//! This module contains trait implementations for `GradientStorageGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{GradientStorageGuard, ResourceStats};

impl AutogradResource for GradientStorageGuard {
    fn resource_type(&self) -> &'static str {
        "GradientStorage"
    }
    fn resource_size(&self) -> usize {
        self.storage_size
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.gradient_manager.lock() {
            manager.release_gradient(self.tensor_id)?;
            self.stats.is_active = false;
        }
        Ok(())
    }
    fn is_valid(&self) -> bool {
        self.stats.is_active
    }
    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for GradientStorageGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup gradient storage for tensor {}: {}",
                self.tensor_id, e
            );
        }
    }
}
