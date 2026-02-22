//! # ModelSelectionStrategy - Trait Implementations
//!
//! This module contains trait implementations for `ModelSelectionStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CriterionType, ModelSelectionStrategy, ModelSelectionType, SelectionCriterion, ValidationMethod};

impl Default for ModelSelectionStrategy {
    fn default() -> Self {
        Self {
            strategy_type: ModelSelectionType::BestSingle,
            criteria: vec![
                SelectionCriterion { criterion_name : "accuracy".to_string(),
                criterion_type : CriterionType::Accuracy, weight : 0.4, threshold : 0.8,
                }, SelectionCriterion { criterion_name : "f1_score".to_string(),
                criterion_type : CriterionType::F1Score, weight : 0.3, threshold : 0.75,
                }, SelectionCriterion { criterion_name : "training_time".to_string(),
                criterion_type : CriterionType::TrainingTime, weight : 0.2, threshold :
                300.0, }, SelectionCriterion { criterion_name : "interpretability"
                .to_string(), criterion_type : CriterionType::Interpretability, weight :
                0.1, threshold : 0.6, },
            ],
            validation_method: ValidationMethod::CrossValidation,
            ensemble_threshold: 0.85,
        }
    }
}

