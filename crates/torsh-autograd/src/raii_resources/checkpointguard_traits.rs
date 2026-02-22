//! # CheckpointGuard - Trait Implementations
//!
//! This module contains trait implementations for `CheckpointGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{CheckpointGuard, ResourceStats};

impl AutogradResource for CheckpointGuard {
    fn resource_type(&self) -> &'static str {
        "GradientCheckpoint"
    }
    fn resource_size(&self) -> usize {
        self.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        self.checkpoint_data = None;
        self.memory_usage = 0;
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

impl Drop for CheckpointGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup checkpoint guard {}: {}",
                self.checkpoint_id, e
            );
        }
    }
}
