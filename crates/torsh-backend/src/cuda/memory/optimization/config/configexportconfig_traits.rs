//! # ConfigExportConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConfigExportConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConfigExportConfig;

impl Default for ConfigExportConfig {
    fn default() -> Self {
        Self {
            format: Default::default(),
            include_metadata: true,
            include_history: false,
        }
    }
}
