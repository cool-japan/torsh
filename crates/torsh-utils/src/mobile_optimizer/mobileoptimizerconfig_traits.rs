//! # MobileOptimizerConfig - Trait Implementations
//!
//! This module contains trait implementations for `MobileOptimizerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    MobileBackend, MobileOptimizerConfig, PlatformOptimization, QuantizationStrategy,
    SizeOptimizationConfig,
};

impl Default for MobileOptimizerConfig {
    fn default() -> Self {
        Self {
            quantize: true,
            quantization_strategy: QuantizationStrategy::default(),
            fuse_ops: true,
            remove_dropout: true,
            fold_bn: true,
            optimize_for_inference: true,
            backend: MobileBackend::Cpu,
            platform_optimization: PlatformOptimization::default(),
            size_optimization: SizeOptimizationConfig::default(),
            preserve_layers: vec![],
            custom_passes: vec![],
        }
    }
}
