//! # MemoryBufferGuard - Trait Implementations
//!
//! This module contains trait implementations for `MemoryBufferGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{MemoryBufferGuard, ResourceStats};

impl AutogradResource for MemoryBufferGuard {
    fn resource_type(&self) -> &'static str {
        "MemoryBuffer"
    }
    fn resource_size(&self) -> usize {
        self.size
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.buffer_manager.lock() {
            manager.release_buffer(self.buffer_id)?;
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

impl Drop for MemoryBufferGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!("Failed to cleanup memory buffer {}: {}", self.buffer_id, e);
        }
    }
}
