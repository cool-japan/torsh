//! # TransferEfficiencyAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `TransferEfficiencyAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::TransferEfficiencyAnalysis;

impl Default for TransferEfficiencyAnalysis {
    fn default() -> Self {
        Self {
            overall_efficiency: 1.0,
            efficiency_by_size: HashMap::new(),
            efficiency_by_direction: HashMap::new(),
            recommendations: Vec::new(),
        }
    }
}
