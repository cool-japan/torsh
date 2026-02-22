//! # ModelExplainability - Trait Implementations
//!
//! This module contains trait implementations for `ModelExplainability`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{ExplanationMethod, ModelExplainability};

impl Default for ModelExplainability {
    fn default() -> Self {
        Self {
            enabled: true,
            explanation_methods: vec![
                ExplanationMethod::SHAP, ExplanationMethod::PermutationImportance,
                ExplanationMethod::PartialDependence,
            ],
            global_explanations: HashMap::new(),
            local_explanations: Vec::new(),
        }
    }
}

