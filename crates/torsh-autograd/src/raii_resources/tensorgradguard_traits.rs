//! # TensorGradGuard - Trait Implementations
//!
//! This module contains trait implementations for `TensorGradGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{ResourceStats, TensorGradGuard};

impl AutogradResource for TensorGradGuard {
    fn resource_type(&self) -> &'static str {
        "TensorGradient"
    }
    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        self.gradient_enabled = self.requires_grad_original;
        self.stats.is_active = false;
        Ok(())
    }
    fn is_valid(&self) -> bool {
        self.stats.is_active
    }
    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for TensorGradGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup tensor gradient guard for tensor {}: {}",
                self.tensor_id, e
            );
        }
    }
}
