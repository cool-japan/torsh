//! # UserPreferences - Trait Implementations
//!
//! This module contains trait implementations for `UserPreferences`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HardwareRequirements, UserPreferences};

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            preferred_architecture: None,
            preferred_tasks: vec![],
            performance_weight: 0.7,
            efficiency_weight: 0.3,
            hardware_constraints: HardwareRequirements::default(),
            preferred_license: Some("MIT".to_string()),
            max_model_size_mb: None,
            min_accuracy: None,
        }
    }
}
