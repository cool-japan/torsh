//! # TrainingScheduler - Trait Implementations
//!
//! This module contains trait implementations for `TrainingScheduler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ResourceLimits, ScheduleType, TrainingScheduler};

impl Default for TrainingScheduler {
    fn default() -> Self {
        Self {
            schedule_type: ScheduleType::Prioritized,
            priority_queue: Vec::new(),
            resource_limits: ResourceLimits {
                max_memory: 8 * 1024 * 1024 * 1024,
                max_cpu_cores: 8,
                max_gpu_memory: 4 * 1024 * 1024 * 1024,
                max_concurrent_tasks: 4,
            },
            concurrent_training: true,
        }
    }
}

