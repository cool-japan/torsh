//! # DatasetConfig - Trait Implementations
//!
//! This module contains trait implementations for `DatasetConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ColumnMapping, DatasetConfig, FileFormat, PreprocessingConfig, SplitRatios, TaskType,
};

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: "default_dataset".to_string(),
            task_type: TaskType::TextClassification,
            file_format: FileFormat::Csv {
                delimiter: ',',
                has_header: true,
            },
            columns: ColumnMapping::default(),
            preprocessing: PreprocessingConfig::default(),
            split_ratios: Some(SplitRatios {
                train: 0.8,
                validation: 0.1,
                test: 0.1,
            }),
        }
    }
}
