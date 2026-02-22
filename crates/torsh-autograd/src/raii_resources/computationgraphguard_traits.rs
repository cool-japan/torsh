//! # ComputationGraphGuard - Trait Implementations
//!
//! This module contains trait implementations for `ComputationGraphGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{ComputationGraphGuard, ResourceStats};

impl AutogradResource for ComputationGraphGuard {
    fn resource_type(&self) -> &'static str {
        "ComputationGraphNode"
    }
    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.graph_manager.lock() {
            manager.remove_node(self.node_id)?;
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

impl Drop for ComputationGraphGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup computation graph node {}: {}",
                self.node_id, e
            );
        }
    }
}
