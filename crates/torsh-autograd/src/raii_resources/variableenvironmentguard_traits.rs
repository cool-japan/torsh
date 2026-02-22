//! # VariableEnvironmentGuard - Trait Implementations
//!
//! This module contains trait implementations for `VariableEnvironmentGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;

use super::functions::AutogradResource;
use super::types::{ResourceStats, VariableEnvironmentGuard};

impl AutogradResource for VariableEnvironmentGuard {
    fn resource_type(&self) -> &'static str {
        "VariableEnvironment"
    }
    fn resource_size(&self) -> usize {
        self.memory_usage
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        self.variables_count = 0;
        self.memory_usage = 0;
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

impl Drop for VariableEnvironmentGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup variable environment guard {}: {}",
                self.environment_id, e
            );
        }
    }
}
