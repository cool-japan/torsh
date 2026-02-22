//! # AutogradContextGuard - Trait Implementations
//!
//! This module contains trait implementations for `AutogradContextGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{AutogradContextGuard, ResourceStats};

impl AutogradResource for AutogradContextGuard {
    fn resource_type(&self) -> &'static str {
        "AutogradContext"
    }
    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.resource_manager.lock() {
            manager.cleanup_context(self.context_id)?;
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

impl Drop for AutogradContextGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup autograd context {}: {}",
                self.context_id, e
            );
        }
    }
}
