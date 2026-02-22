//! # EnsembleConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnsembleConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{EnsembleConfig, EnsembleType, VotingStrategy};

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            ensemble_type: EnsembleType::Voting,
            voting_strategy: VotingStrategy::Weighted,
            model_weights: HashMap::new(),
            diversity_threshold: 0.1,
            max_models: 5,
        }
    }
}

