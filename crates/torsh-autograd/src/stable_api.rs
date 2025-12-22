// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stable API Surface for torsh-autograd
//!
//! This module defines the stable, public API that follows semantic versioning.
//! APIs marked as stable will only have breaking changes in major version updates.
//!
//! # Stability Guarantees
//!
//! ## Stable (v1.0+)
//! - Core autograd functionality (grad, backward)
//! - Gradient mode management (no_grad, enable_grad)
//! - Basic tensor operations with gradients
//! - Gradient computation and accumulation
//!
//! ## Beta (v0.1+)
//! - Advanced gradient features (checkpointing, clipping)
//! - Hardware acceleration APIs
//! - Distributed training support
//! - Custom function framework
//!
//! ## Experimental
//! - Quantum autograd
//! - Neural architecture search
//! - Advanced optimization differentiation
//!
//! # Version History
//!
//! - v0.1.0-alpha.2: Current version with comprehensive autograd features
//! - v0.1.0-alpha.1: Initial implementation
//!
//! # Breaking Change Policy
//!
//! - **Major versions** (x.0.0): Breaking changes allowed
//! - **Minor versions** (0.x.0): New features, no breaking changes to stable APIs
//! - **Patch versions** (0.0.x): Bug fixes only

use crate::{AutogradError, AutogradResult};

/// Stability level of an API
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityLevel {
    /// Stable API with semantic versioning guarantees
    /// Breaking changes only in major versions
    Stable,
    /// Beta API that may change in minor versions
    /// Generally safe to use but may evolve
    Beta,
    /// Experimental API that may change at any time
    /// Use with caution in production
    Experimental,
    /// Deprecated API that will be removed in a future version
    /// Migration path provided
    Deprecated {
        since: &'static str,
        note: &'static str,
    },
}

impl StabilityLevel {
    pub fn is_stable(&self) -> bool {
        matches!(self, StabilityLevel::Stable)
    }

    pub fn is_beta(&self) -> bool {
        matches!(self, StabilityLevel::Beta)
    }

    pub fn is_experimental(&self) -> bool {
        matches!(self, StabilityLevel::Experimental)
    }

    pub fn is_deprecated(&self) -> bool {
        matches!(self, StabilityLevel::Deprecated { .. })
    }
}

/// Marks an API feature with its stability level
#[derive(Debug, Clone)]
pub struct ApiFeature {
    pub name: &'static str,
    pub stability: StabilityLevel,
    pub since_version: &'static str,
    pub description: &'static str,
}

macro_rules! stable_api {
    ($name:expr, $since:expr, $desc:expr) => {
        ApiFeature {
            name: $name,
            stability: StabilityLevel::Stable,
            since_version: $since,
            description: $desc,
        }
    };
}

macro_rules! beta_api {
    ($name:expr, $since:expr, $desc:expr) => {
        ApiFeature {
            name: $name,
            stability: StabilityLevel::Beta,
            since_version: $since,
            description: $desc,
        }
    };
}

macro_rules! experimental_api {
    ($name:expr, $since:expr, $desc:expr) => {
        ApiFeature {
            name: $name,
            stability: StabilityLevel::Experimental,
            since_version: $since,
            description: $desc,
        }
    };
}

/// Get all API features with their stability levels
pub fn get_api_features() -> Vec<ApiFeature> {
    vec![
        // Stable APIs (v1.0 target)
        stable_api!(
            "grad_mode",
            "0.1.0",
            "Gradient computation mode management (enable_grad, no_grad)"
        ),
        stable_api!("backward", "0.1.0", "Backward pass computation for tensors"),
        stable_api!("grad", "0.1.0", "Gradient computation function"),
        stable_api!(
            "AutogradTensor",
            "0.1.0",
            "Core tensor trait with automatic differentiation"
        ),
        stable_api!(
            "GradientAccumulation",
            "0.1.0",
            "Gradient accumulation functionality"
        ),
        // Beta APIs
        beta_api!(
            "gradient_clipping",
            "0.1.0",
            "Gradient clipping strategies for optimization"
        ),
        beta_api!(
            "checkpoint",
            "0.1.0",
            "Gradient checkpointing for memory efficiency"
        ),
        beta_api!(
            "anomaly_detection",
            "0.1.0",
            "Numerical anomaly detection in gradients"
        ),
        beta_api!(
            "hardware_acceleration",
            "0.1.0-alpha.2",
            "Hardware-specific autograd acceleration (CUDA, Metal, WebGPU)"
        ),
        beta_api!(
            "custom_functions",
            "0.1.0",
            "Custom differentiable function framework"
        ),
        beta_api!(
            "higher_order_gradients",
            "0.1.0",
            "Higher-order derivative computation"
        ),
        beta_api!(
            "complex_ops",
            "0.1.0",
            "Complex number operations with Wirtinger derivatives"
        ),
        beta_api!(
            "distributed",
            "0.1.0",
            "Distributed gradient computation and synchronization"
        ),
        beta_api!(
            "profiler",
            "0.1.0",
            "Autograd operation profiling and analysis"
        ),
        beta_api!(
            "gradient_validation",
            "0.1.0",
            "Gradient correctness validation utilities"
        ),
        // Experimental APIs
        experimental_api!(
            "quantum_autograd",
            "0.1.0",
            "Quantum computing automatic differentiation"
        ),
        experimental_api!(
            "neural_architecture_search",
            "0.1.0",
            "Differentiable neural architecture search (DARTS)"
        ),
        experimental_api!(
            "optimization_diff",
            "0.1.0",
            "Differentiation through optimization problems"
        ),
        experimental_api!(
            "symbolic",
            "0.1.0",
            "Symbolic differentiation for simple expressions"
        ),
        experimental_api!(
            "stochastic_graphs",
            "0.1.0",
            "Stochastic computation graphs for probabilistic programming"
        ),
    ]
}

/// Check if an API feature is stable
pub fn is_stable_api(feature_name: &str) -> bool {
    get_api_features()
        .iter()
        .find(|f| f.name == feature_name)
        .map(|f| f.stability.is_stable())
        .unwrap_or(false)
}

/// Get the stability level of an API feature
pub fn get_stability_level(feature_name: &str) -> Option<StabilityLevel> {
    get_api_features()
        .iter()
        .find(|f| f.name == feature_name)
        .map(|f| f.stability)
}

/// Emit a warning if using an experimental API
pub fn check_experimental_api(feature_name: &str) {
    if let Some(feature) = get_api_features().iter().find(|f| f.name == feature_name) {
        match feature.stability {
            StabilityLevel::Experimental => {
                tracing::warn!(
                    "Using experimental API '{}': {}. This API may change without notice.",
                    feature_name,
                    feature.description
                );
            }
            StabilityLevel::Deprecated { since, note } => {
                tracing::warn!(
                    "Using deprecated API '{}' (since {}): {}",
                    feature_name,
                    since,
                    note
                );
            }
            _ => {}
        }
    }
}

/// API compatibility checker
pub struct ApiCompatibilityChecker {
    _current_version: semver::Version,
}

impl ApiCompatibilityChecker {
    pub fn new() -> Self {
        Self {
            _current_version: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("Invalid package version"),
        }
    }

    /// Get the current package version
    pub fn current_version(&self) -> &semver::Version {
        &self._current_version
    }

    /// Check if a feature is compatible with a specific version
    pub fn is_compatible(
        &self,
        feature_name: &str,
        required_version: &str,
    ) -> AutogradResult<bool> {
        let required = semver::Version::parse(required_version).map_err(|e| {
            AutogradError::gradient_computation("version_parse", format!("Invalid version: {}", e))
        })?;

        if let Some(feature) = get_api_features().iter().find(|f| f.name == feature_name) {
            let since = semver::Version::parse(feature.since_version).map_err(|e| {
                AutogradError::gradient_computation(
                    "version_parse",
                    format!("Invalid version: {}", e),
                )
            })?;

            Ok(required >= since && !feature.stability.is_deprecated())
        } else {
            Ok(false)
        }
    }

    /// Get minimum version required for a feature
    pub fn get_minimum_version(&self, feature_name: &str) -> Option<&'static str> {
        get_api_features()
            .iter()
            .find(|f| f.name == feature_name)
            .map(|f| f.since_version)
    }
}

impl Default for ApiCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable API re-exports
///
/// These are the core APIs that follow semantic versioning.
/// Breaking changes will only occur in major version updates.
pub mod stable {
    //! Stable APIs with semantic versioning guarantees

    // Core gradient mode management
    pub use crate::grad_mode::{
        is_grad_enabled, no_grad, pop_grad_enabled, push_grad_enabled, set_grad_enabled,
        with_grad_mode, with_no_grad,
    };

    // Core autograd traits
    pub use crate::autograd_traits::{
        AutogradTensor, AutogradTensorFactory, BackwardTensor, GradientAccumulation,
    };

    // Core error types
    pub use crate::error_handling::{AutogradError, AutogradResult};

    // Guards for RAII gradient mode management
    pub use crate::guards::{enable_grad, EnableGradGuard, GradModeGuard, NoGradGuard};

    // Global adapter for convenient access
    pub use crate::global_adapter::{
        backward_global, create_gradient_tensor, get_global_adapter, get_gradient_global,
    };
}

/// Beta API re-exports
///
/// These APIs are generally stable but may evolve in minor versions.
/// Use with confidence but be prepared for occasional changes.
pub mod beta {
    //! Beta APIs that may evolve in minor versions

    // Gradient clipping
    pub use crate::gradient_clipping::*;

    // Anomaly detection
    pub use crate::anomaly_detection::*;

    // Custom functions
    pub use crate::function::*;

    // Higher-order gradients
    pub use crate::higher_order_gradients::*;

    // Complex operations
    pub use crate::complex_ops::*;

    // Hardware acceleration
    pub use crate::hardware_acceleration::{
        AcceleratorType, HardwareAccelerationManager, HardwareAccelerator,
    };

    // Profiling
    pub use crate::profiler::*;

    // Gradient validation
    pub use crate::gradient_validation::*;
}

/// Experimental API re-exports
///
/// These APIs are experimental and may change at any time.
/// Use with caution in production environments.
pub mod experimental {
    //! Experimental APIs that may change without notice

    // Quantum autograd
    pub use crate::quantum_autograd::*;

    // Neural architecture search
    pub use crate::neural_architecture_search::*;

    // Optimization differentiation
    pub use crate::optimization_diff::*;

    // Symbolic differentiation
    pub use crate::symbolic::*;

    // Stochastic graphs
    pub use crate::stochastic_graphs::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_levels() {
        assert!(StabilityLevel::Stable.is_stable());
        assert!(!StabilityLevel::Beta.is_stable());
        assert!(!StabilityLevel::Experimental.is_stable());

        let deprecated = StabilityLevel::Deprecated {
            since: "0.1.0",
            note: "Use new API instead",
        };
        assert!(deprecated.is_deprecated());
    }

    #[test]
    fn test_api_features() {
        let features = get_api_features();
        assert!(!features.is_empty());

        // Check that core APIs are marked as stable
        assert!(is_stable_api("grad_mode"));
        assert!(is_stable_api("backward"));
        assert!(is_stable_api("grad"));
    }

    #[test]
    fn test_compatibility_checker() {
        let checker = ApiCompatibilityChecker::new();

        // Check compatibility with current version
        assert!(checker.is_compatible("grad_mode", "0.1.0").unwrap());

        // Check minimum version
        assert_eq!(checker.get_minimum_version("grad_mode"), Some("0.1.0"));
    }

    #[test]
    fn test_get_stability_level() {
        assert_eq!(
            get_stability_level("grad_mode"),
            Some(StabilityLevel::Stable)
        );
        assert_eq!(
            get_stability_level("gradient_clipping"),
            Some(StabilityLevel::Beta)
        );
        assert_eq!(
            get_stability_level("quantum_autograd"),
            Some(StabilityLevel::Experimental)
        );
    }
}
