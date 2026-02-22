//! # ProfileSessionGuard - Trait Implementations
//!
//! This module contains trait implementations for `ProfileSessionGuard`.
//!
//! ## Implemented Traits
//!
//! - `AutogradResource`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;
use std::sync::atomic::Ordering;

use super::functions::AutogradResource;
use super::types::{ProfileSessionGuard, ResourceStats};

impl AutogradResource for ProfileSessionGuard {
    fn resource_type(&self) -> &'static str {
        "ProfileSession"
    }
    fn resource_size(&self) -> usize {
        self.stats.memory_usage + self.collected_samples.load(Ordering::Relaxed) * 64
    }
    fn cleanup(&mut self) -> AutogradResult<()> {
        self.is_active.store(false, Ordering::Relaxed);
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

impl Drop for ProfileSessionGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup profile session guard '{}': {}",
                self.session_name, e
            );
        }
    }
}
