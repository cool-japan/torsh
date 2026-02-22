//! # PreprocessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `PreprocessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PreprocessingConfig;

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: false,
            remove_punctuation: false,
            remove_stopwords: false,
            max_length: None,
            min_length: None,
        }
    }
}
