//! # QualityStandards - Trait Implementations
//!
//! This module contains trait implementations for `QualityStandards`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap};

use super::types::QualityStandards;

impl Default for QualityStandards {
    fn default() -> Self {
        let mut academic_standards = HashMap::new();
        academic_standards.insert("clarity".to_string(), 0.85);
        academic_standards.insert("coherence".to_string(), 0.90);
        academic_standards.insert("precision".to_string(), 0.88);
        let mut professional_standards = HashMap::new();
        professional_standards.insert("effectiveness".to_string(), 0.80);
        professional_standards.insert("appropriateness".to_string(), 0.85);
        professional_standards.insert("impact".to_string(), 0.75);
        let mut creative_standards = HashMap::new();
        creative_standards.insert("originality".to_string(), 0.70);
        creative_standards.insert("engagement".to_string(), 0.80);
        creative_standards.insert("expression".to_string(), 0.75);
        let mut technical_standards = HashMap::new();
        technical_standards.insert("accuracy".to_string(), 0.95);
        technical_standards.insert("completeness".to_string(), 0.90);
        technical_standards.insert("precision".to_string(), 0.92);
        let mut conversational_standards = HashMap::new();
        conversational_standards.insert("naturalness".to_string(), 0.85);
        conversational_standards.insert("appropriateness".to_string(), 0.80);
        conversational_standards.insert("engagement".to_string(), 0.75);
        Self {
            academic_standards,
            professional_standards,
            creative_standards,
            technical_standards,
            conversational_standards,
        }
    }
}

