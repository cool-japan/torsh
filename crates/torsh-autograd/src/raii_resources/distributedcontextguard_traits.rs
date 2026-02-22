//! # DistributedContextGuard - Trait Implementations
//!
//! This module contains trait implementations for `DistributedContextGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{DistributedContextGuard, ResourceStats};

impl AutogradResource for DistributedContextGuard {
    fn resource_type(&self) -> &'static str {
        "DistributedContext"
    }
    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        self.communication_buffers.clear();
        self.stats.memory_usage = 0;
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

impl Drop for DistributedContextGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup distributed context guard {}: {}",
                self.context_id, e
            );
        }
    }
}
