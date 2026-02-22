//! # PrefetchEffectiveness - Trait Implementations
//!
//! This module contains trait implementations for `PrefetchEffectiveness`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{PrefetchCostEffectiveness, PrefetchEffectiveness};

impl Default for PrefetchEffectiveness {
    fn default() -> Self {
        Self {
            total_prefetches: 0,
            successful_prefetches: 0,
            success_rate: 0.0,
            performance_improvement: 0.0,
            pattern_accuracy: HashMap::new(),
            cost_effectiveness: PrefetchCostEffectiveness::default(),
        }
    }
}
