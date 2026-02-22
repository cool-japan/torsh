//! # DatasetDownloader - Trait Implementations
//!
//! This module contains trait implementations for `DatasetDownloader`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DatasetDownloader;

impl Default for DatasetDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create dataset downloader")
    }
}
