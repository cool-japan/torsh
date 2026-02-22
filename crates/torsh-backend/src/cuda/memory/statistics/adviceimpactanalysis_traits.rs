//! # AdviceImpactAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `AdviceImpactAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::AdviceImpactAnalysis;

impl Default for AdviceImpactAnalysis {
    fn default() -> Self {
        Self {
            advice_effectiveness: HashMap::new(),
            overall_impact: 0.0,
            improvements: HashMap::new(),
            optimization_opportunities: Vec::new(),
        }
    }
}
