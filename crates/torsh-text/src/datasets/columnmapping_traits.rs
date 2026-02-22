//! # ColumnMapping - Trait Implementations
//!
//! This module contains trait implementations for `ColumnMapping`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::ColumnMapping;

impl Default for ColumnMapping {
    fn default() -> Self {
        Self {
            text_column: Some(0),
            label_column: Some(1),
            source_column: None,
            target_column: None,
            custom_mapping: HashMap::new(),
        }
    }
}
