//! Neural network modules for ToRSh
//!
//! This crate provides PyTorch-compatible neural network layers and modules,
//! built on top of scirs2-neural for optimized implementations.
//!
//! # Modular Architecture
//!
//! The neural network core system is organized into specialized modules for improved maintainability:
//!
//! - **core**: Core Module trait system and essential interfaces
//! - **parameter**: Comprehensive parameter management and initialization
//! - **hooks**: Hook system infrastructure for module callbacks
//! - **base**: ModuleBase helper for module implementations
//! - **composition**: Module composition patterns (sequential, parallel, etc.)
//! - **construction**: Module construction and configuration patterns
//! - **diagnostics**: Module and parameter diagnostics and health checking
//! - **utils**: Module utilities and helper functions
//!
//! All components maintain full backward compatibility through comprehensive re-exports.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(ambiguous_glob_reexports)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// pub mod checkpoint; // Temporarily disabled for testing
pub mod compile_time;
pub mod container;
#[cfg(feature = "std")]
pub mod conversion;
pub mod cuda_kernels;
#[cfg(feature = "std")]
pub mod export;
pub mod functional;
pub mod gradcheck;
pub mod hardware_opts;
pub mod init;
pub mod layers;
pub mod lazy;
pub mod mixed_precision;
pub mod model_zoo;
pub mod modules;
pub mod numerical_stability;
pub mod optimization;
pub mod parameter_updates;
pub mod pruning;
pub mod quantization;
pub mod research;
pub mod scirs2_neural_integration;
#[cfg(feature = "serialize")]
pub mod serialization;
pub mod sparse;
pub mod summary;
pub mod visualization;

// =============================================================================
// MODULAR ARCHITECTURE IMPORTS
// =============================================================================

// Core module trait system
pub mod core;
pub use core::Module;

// Parameter management system
pub mod parameter;
pub use parameter::{
    LayerType, Parameter, ParameterCollection, ParameterDiagnostics, ParameterStats,
};

// Hook system infrastructure
pub mod hooks;
pub use hooks::{HookCallback, HookHandle, HookRegistry, HookType};

// Module base infrastructure
pub mod base;
pub use base::ModuleBase;

// Module composition system
pub mod composition;
pub use composition::{
    ComposedModule, ConditionalModule, ModuleBuilder, ModuleComposition, ParallelModule,
    ResidualModule,
};

// Module construction and configuration
pub mod construction;
pub use construction::{ModuleConfig, ModuleConstruct};

// Module diagnostics and analysis
pub mod diagnostics;
pub use diagnostics::{ModuleDiagnostics, ModuleInfo};

// Module utilities
pub mod utils;
pub use utils::{ModuleApply, ModuleParameterStats};

// =============================================================================
// BACKWARD COMPATIBILITY IMPORTS AND RE-EXPORTS
// =============================================================================

use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

// Note: impl_module_constructor macro is already available via #[macro_export]

/// Sparse Matrix placeholder for compatibility
pub struct SparseMatrix;

impl SparseMatrix {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SparseMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::container::*;
    #[cfg(feature = "std")]
    pub use crate::conversion::{
        pytorch_compat, tensorflow_compat, ConversionConfig, FrameworkSource, MigrationHelper,
        ModelConverter,
    };
    pub use crate::cuda_kernels::{
        global_kernel_registry, CudaKernelRegistry, CudaNeuralOps, CudaOptimizations,
        CustomActivations,
    };
    #[cfg(feature = "std")]
    pub use crate::export::{
        DeploymentOptimizer, ExportConfig, ExportFormat, ModelExporter, OptimizationLevel,
        TargetDevice,
    };
    pub use crate::gradcheck::{
        fast_gradcheck, gradcheck, precise_gradcheck, GradCheckConfig, GradCheckResult, GradChecker,
    };
    pub use crate::init::{
        self,
        // Automatic initialization
        auto_init,
        coordinate_mlp_init,
        delta_orthogonal_init,
        // Modern initialization methods
        fixup_init,
        gan_balanced_init,
        lsuv_init,
        metainit,
        recommend_init_method,
        rezero_alpha_init,
        rezero_init,
        zero_centered_variance_init,
        ActivationHint,
        ArchitectureHint,
        // Core types
        FanMode,
        InitMethod,
        Initializer,
        Nonlinearity,
    };
    pub use crate::layers::*;
    pub use crate::lazy::{lazy_linear, lazy_linear_no_bias, LazyLinear, LazyModule, LazyWrapper};
    pub use crate::mixed_precision::prelude::*;
    #[allow(unused_imports)]
    pub use crate::modules::*;
    pub use crate::numerical_stability::utils::{
        comprehensive_stability_analysis, quick_stability_check,
    };
    pub use crate::numerical_stability::{
        StabilityConfig, StabilityIssue, StabilityResults, StabilityTester,
    };
    pub use crate::optimization::{
        optimize_for_inference, optimize_module, MemoryProfiler, NetworkOptimizer,
        OptimizationReport,
    };
    // pub use crate::parameter::sharing::{ParameterSharingRegistry, SharingStats}; // Module not yet implemented
    pub use crate::parameter_updates::{
        LayerSpecificOptimizers, ParameterUpdater, UpdateConfig, UpdateStatistics,
    };
    pub use crate::pruning::{Pruner, PruningConfig, PruningMask, PruningScope, PruningStrategy};
    pub use crate::quantization::prelude::*;
    pub use crate::scirs2_neural_integration::{
        LayerNorm, MemoryEfficientSequential, Mish, MultiHeadAttention, NeuralConfig,
        SciRS2NeuralProcessor, Swish, TransformerEncoderLayer, GELU,
    };
    pub use crate::summary::profiling::{
        AnalysisConfig, AnalysisReport, BatchProfiler, BatchProfilingConfig, BatchProfilingResult,
        FLOPSAnalysis, FLOPSCounter, MemoryAnalysis, ModelAnalyzer,
    };
    pub use crate::summary::utils::*;
    pub use crate::summary::{summarize, LayerInfo, ModelProfiler, ModelSummary, SummaryConfig};
    pub use crate::visualization::utils::*;
    pub use crate::visualization::{GraphEdge, GraphNode, NetworkGraph, VisualizationConfig};
    pub use crate::{ComposedModule, ConditionalModule, ParallelModule, ResidualModule};
    pub use crate::{
        HookCallback, HookHandle, HookRegistry, HookType, LayerType, Module, ModuleBase,
        ModuleConfig, ModuleConstruct, Parameter, ParameterCollection, ParameterDiagnostics,
        ParameterStats,
    };
    pub use crate::{ModuleBuilder, ModuleComposition, ModuleDiagnostics, ModuleInfo};
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::error::Result;

    // Conditional imports for std/no_std compatibility
    #[cfg(feature = "std")]
    use std::{boxed::Box, sync::Arc, vec::Vec};

    #[cfg(not(feature = "std"))]
    use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

    // Use parking_lot::Mutex for both std and no_std
    use parking_lot::Mutex;

    #[test]
    fn test_parameter() {
        let tensor = torsh_tensor::creation::ones(&[3, 4]).unwrap();
        let param = Parameter::new(tensor);
        assert!(param.requires_grad());
    }

    #[test]
    fn test_hook_registry() {
        let mut registry = HookRegistry::new();

        // Test registering hooks
        let call_count = Arc::new(Mutex::new(0));
        let call_count_clone = call_count.clone();

        let hook = Box::new(
            move |_module: &dyn Module, _input: &Tensor, _output: Option<&Tensor>| {
                *call_count_clone.lock() += 1;
                Ok(())
            },
        );

        let handle = registry.register_hook(HookType::PreForward, hook);

        assert!(registry.has_hooks(HookType::PreForward));
        assert_eq!(registry.hook_count(HookType::PreForward), 1);
        assert!(!registry.has_hooks(HookType::PostForward));

        // Test removing hooks
        assert!(registry.remove_hook(HookType::PreForward, handle));
        assert!(!registry.has_hooks(HookType::PreForward));
        assert_eq!(registry.hook_count(HookType::PreForward), 0);

        // Test removing non-existent hook
        assert!(!registry.remove_hook(HookType::PreForward, handle));
    }

    #[test]
    fn test_hook_execution() -> Result<()> {
        let mut registry = HookRegistry::new();

        // Track hook execution
        let execution_log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = execution_log.clone();

        let pre_hook = Box::new(
            move |_module: &dyn Module, _input: &Tensor, _output: Option<&Tensor>| {
                log_clone.lock().push("pre_forward".to_string());
                Ok(())
            },
        );

        let log_clone2 = execution_log.clone();
        let post_hook = Box::new(
            move |_module: &dyn Module, _input: &Tensor, output: Option<&Tensor>| {
                assert!(output.is_some()); // Post-forward should have output
                log_clone2.lock().push("post_forward".to_string());
                Ok(())
            },
        );

        registry.register_hook(HookType::PreForward, pre_hook);
        registry.register_hook(HookType::PostForward, post_hook);

        // Create a dummy module and tensor for testing
        struct DummyModule;
        impl Module for DummyModule {
            fn forward(&self, input: &Tensor) -> Result<Tensor> {
                Ok(input.clone())
            }
        }

        let dummy_module = DummyModule;
        let input = torsh_tensor::creation::zeros(&[2, 3])?;
        let output = torsh_tensor::creation::ones(&[2, 3])?;

        // Execute hooks
        registry.execute_hooks(HookType::PreForward, &dummy_module, &input, None)?;
        registry.execute_hooks(HookType::PostForward, &dummy_module, &input, Some(&output))?;

        // Check execution log
        let log = execution_log.lock();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], "pre_forward");
        assert_eq!(log[1], "post_forward");

        Ok(())
    }

    #[test]
    fn test_module_base_hooks() -> Result<()> {
        let mut base = ModuleBase::new();

        // Test hook registration
        let call_count = Arc::new(Mutex::new(0));
        let call_count_clone = call_count.clone();

        let hook = Box::new(
            move |_module: &dyn Module, _input: &Tensor, _output: Option<&Tensor>| {
                *call_count_clone.lock() += 1;
                Ok(())
            },
        );

        let handle = base.register_hook(HookType::PreForward, hook);
        assert!(base.has_hooks(HookType::PreForward));
        assert_eq!(base.hook_count(HookType::PreForward), 1);

        // Test hook removal
        assert!(base.remove_hook(HookType::PreForward, handle));
        assert!(!base.has_hooks(HookType::PreForward));

        Ok(())
    }

    #[test]
    fn test_hook_error_propagation() -> Result<()> {
        let mut registry = HookRegistry::new();

        // Hook that returns an error
        let error_hook = Box::new(
            |_module: &dyn Module, _input: &Tensor, _output: Option<&Tensor>| {
                Err(torsh_core::error::TorshError::Other(
                    "Hook error".to_string(),
                ))
            },
        );

        registry.register_hook(HookType::PreForward, error_hook);

        struct DummyModule;
        impl Module for DummyModule {
            fn forward(&self, input: &Tensor) -> Result<Tensor> {
                Ok(input.clone())
            }
        }

        let dummy_module = DummyModule;
        let input = torsh_tensor::creation::zeros(&[2, 3])?;

        // Hook execution should propagate the error
        let result = registry.execute_hooks(HookType::PreForward, &dummy_module, &input, None);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_multiple_hooks_execution_order() -> Result<()> {
        let mut registry = HookRegistry::new();

        let execution_order = Arc::new(Mutex::new(Vec::new()));

        // Register multiple hooks
        for i in 0..3 {
            let order_clone = execution_order.clone();
            let hook = Box::new(
                move |_module: &dyn Module, _input: &Tensor, _output: Option<&Tensor>| {
                    order_clone.lock().push(i);
                    Ok(())
                },
            );
            registry.register_hook(HookType::PreForward, hook);
        }

        assert_eq!(registry.hook_count(HookType::PreForward), 3);

        struct DummyModule;
        impl Module for DummyModule {
            fn forward(&self, input: &Tensor) -> Result<Tensor> {
                Ok(input.clone())
            }
        }

        let dummy_module = DummyModule;
        let input = torsh_tensor::creation::zeros(&[2, 3])?;

        registry.execute_hooks(HookType::PreForward, &dummy_module, &input, None)?;

        // Hooks should execute in registration order
        let order = execution_order.lock();
        assert_eq!(*order, vec![0, 1, 2]);

        Ok(())
    }

    #[test]
    fn test_hook_clear_operations() {
        let mut registry = HookRegistry::new();

        // Register hooks for different types
        let dummy_hook = Box::new(|_: &dyn Module, _: &Tensor, _: Option<&Tensor>| Ok(()));
        registry.register_hook(HookType::PreForward, dummy_hook);

        let dummy_hook2 = Box::new(|_: &dyn Module, _: &Tensor, _: Option<&Tensor>| Ok(()));
        registry.register_hook(HookType::PostForward, dummy_hook2);

        assert!(registry.has_hooks(HookType::PreForward));
        assert!(registry.has_hooks(HookType::PostForward));

        // Clear specific hook type
        registry.clear_hooks(HookType::PreForward);
        assert!(!registry.has_hooks(HookType::PreForward));
        assert!(registry.has_hooks(HookType::PostForward));

        // Register another hook
        let dummy_hook3 = Box::new(|_: &dyn Module, _: &Tensor, _: Option<&Tensor>| Ok(()));
        registry.register_hook(HookType::PreBackward, dummy_hook3);
        assert!(registry.has_hooks(HookType::PreBackward));

        // Clear all hooks
        registry.clear_all_hooks();
        assert!(!registry.has_hooks(HookType::PreForward));
        assert!(!registry.has_hooks(HookType::PostForward));
        assert!(!registry.has_hooks(HookType::PreBackward));
        assert!(!registry.has_hooks(HookType::PostBackward));
    }

    #[test]
    fn test_modular_system_integrity() {
        // Test that all modules are properly accessible and modular architecture works

        // Test parameter creation
        let tensor = torsh_tensor::creation::randn(&[3, 4]).unwrap();
        let param = Parameter::new(tensor);
        assert!(param.requires_grad());

        // Test parameter statistics
        let stats = param.stats().unwrap();
        assert_eq!(stats.numel, 12);

        // Test parameter collection
        let mut collection = ParameterCollection::new();
        collection.add("test_param".to_string(), param);
        assert_eq!(collection.len(), 1);
        assert!(!collection.is_empty());

        // Test module base
        let base = ModuleBase::new();
        assert!(base.training());

        // Test hook registry
        let registry = HookRegistry::new();
        assert!(!registry.has_hooks(HookType::PreForward));

        // Test module config
        let config = ModuleConfig::new();
        assert!(config.training);
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the modular system maintains full backward compatibility
        // All original APIs should work exactly as before

        // Test parameter creation (original API)
        let tensor = torsh_tensor::creation::ones(&[2, 3]).unwrap();
        let param = Parameter::new(tensor);
        assert!(param.requires_grad());

        // Test parameter access methods (original API)
        let shape = param.shape().unwrap();
        assert_eq!(shape, vec![2, 3]);

        let numel = param.numel().unwrap();
        assert_eq!(numel, 6);

        // Test module base functionality (original API)
        let mut base = ModuleBase::new();
        base.register_parameter("test".to_string(), param);
        assert_eq!(base.named_parameters().len(), 1);

        // Test hook system (original API)
        let mut registry = HookRegistry::new();
        let hook = Box::new(|_: &dyn Module, _: &Tensor, _: Option<&Tensor>| Ok(()));
        let handle = registry.register_hook(HookType::PreForward, hook);
        assert!(registry.has_hooks(HookType::PreForward));
        assert!(registry.remove_hook(HookType::PreForward, handle));
    }
}
