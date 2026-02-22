//! # QualityWeights - Trait Implementations
//!
//! This module contains trait implementations for `QualityWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityWeights;

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            language_model_weight: 0.20,
            syntactic_weight: 0.15,
            lexical_weight: 0.15,
            semantic_weight: 0.20,
            prosodic_weight: 0.10,
            pragmatic_weight: 0.15,
            statistical_weight: 0.05,
        }
    }
}

