//! # ConfigMetadata - Trait Implementations
//!
//! This module contains trait implementations for `ConfigMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{ConfigEnvironment, ConfigMetadata, ConfigPriority, ConfigSource, ConfigStatus, ConfigVersion};

impl Default for ConfigMetadata {
    fn default() -> Self {
        Self {
            version: ConfigVersion::new(1, 0, 0),
            name: "default_optimization_config".to_string(),
            description: "Default optimization configuration".to_string(),
            environment: ConfigEnvironment::Development,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            author: "system".to_string(),
            tags: vec!["default".to_string()],
            schema_version: "1.0".to_string(),
            checksum: "".to_string(),
            source: ConfigSource::Default,
            status: ConfigStatus::Active,
            priority: ConfigPriority::Normal,
            dependencies: Vec::new(),
        }
    }
}

