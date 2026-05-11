//! # ConfigManagerConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConfigManagerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConfigManagerConfig;

impl Default for ConfigManagerConfig {
    fn default() -> Self {
        Self {
            registry_config: Default::default(),
            validation_config: Default::default(),
            versioning_config: Default::default(),
            dynamic_config: Default::default(),
            template_config: Default::default(),
            environment_config: Default::default(),
            persistence_config: Default::default(),
            audit_config: Default::default(),
            backup_config: Default::default(),
            sync_config: Default::default(),
            schema_config: Default::default(),
            migration_config: Default::default(),
        }
    }
}

