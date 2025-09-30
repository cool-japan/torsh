//! Autograd context system with computation graph management
//!
//! This module provides a comprehensive autograd context system for automatic differentiation
//! and computation graph management. It has been modularized for better maintainability and
//! organization while preserving complete backward compatibility.
//!
//! ## Architecture Overview
//!
//! The autograd context system is organized into specialized modules:
//!
//! ### Core Module ([`core`])
//! Contains the fundamental autograd context and computation graph management:
//! - [`AutogradContext`] - Main context for managing computation graphs
//! - [`GraphNode`] - Represents operations in the computation graph
//! - [`GradientFunction`] - Trait for implementing gradient computations
//! - Thread-local context management functions
//!
//! ### Gradient Functions Module ([`gradient_functions`])
//! Standard implementations of gradient functions for common operations:
//! - Basic arithmetic: [`AddGradient`], [`MulGradient`], [`SubGradient`], [`DivGradient`]
//! - Mathematical functions: [`SquareGradient`], [`PowGradient`], [`ExpGradient`], [`LogGradient`]
//! - Activation functions: [`ReLUGradient`], [`SigmoidGradient`], [`TanhGradient`]
//! - Matrix operations: [`MatMulGradient`]
//! - Reduction operations: [`SumGradient`], [`MeanGradient`]
//!
//! ### Optimization Module ([`optimization`])
//! Graph optimization, pruning, and compression functionality:
//! - [`GraphStats`] - Statistics about computation graph structure
//! - [`GraphOptimizer`] - Advanced graph optimization strategies
//! - [`OptimizationResult`] - Results of optimization operations
//! - Graph pruning and memory management utilities
//!
//! ### Debug Module ([`debug`])
//! Graph debugging, analysis, and visualization utilities:
//! - [`GraphDebugger`] - Comprehensive graph analysis and debugging
//! - [`GraphAnalysis`] - Detailed analysis results with performance metrics
//! - [`CriticalPath`] - Critical path analysis for performance optimization
//! - [`GraphProfiler`] - Performance profiling for graph execution
//!
//! ### Memory Module ([`memory`])
//! Advanced memory management and leak prevention:
//! - [`MemoryManagementConfig`] - Configuration for automatic memory management
//! - [`CleanupStatistics`] - Statistics about cleanup operations
//! - [`MemoryPressureMonitor`] - Memory pressure monitoring and alerts
//! - Orphaned gradient cleanup and circular reference breaking
//!
//! ## Key Features
//!
//! ### Automatic Differentiation
//! ```rust
//! use torsh_autograd::context::{AutogradContext, with_context};
//! use torsh_autograd::context::gradient_functions::AddGradient;
//! use std::sync::Arc;
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! with_context(|ctx| {
//!     // Add operation to computation graph
//!     ctx.add_operation(
//!         "add".to_string(),
//!         vec![1, 2], // input tensor IDs
//!         3,          // output tensor ID
//!         true,       // requires gradient
//!         Some(Arc::new(AddGradient)),
//!     )?;
//!
//!     // Perform backward pass
//!     ctx.backward_from_tensor(3, vec![1.0])?;
//!     Ok(())
//! })
//! # }
//! ```
//!
//! ### Graph Analysis and Debugging
//! ```rust
//! use torsh_autograd::context::{AutogradContext, GraphDebugger};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! let ctx = AutogradContext::new();
//! let debugger = GraphDebugger::new().verbose();
//!
//! // Analyze computation graph
//! let analysis = debugger.analyze_graph(&ctx)?;
//! println!("Graph has {} nodes, max depth: {}",
//!          analysis.node_count, analysis.max_depth);
//!
//! // Generate detailed report
//! let report = debugger.generate_graph_report(&ctx)?;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```
//!
//! ### Memory Management
//! ```rust
//! use torsh_autograd::context::{AutogradContext, MemoryManagementConfig};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! let mut ctx = AutogradContext::new();
//! let config = MemoryManagementConfig::default();
//!
//! // Perform automatic cleanup
//! let stats = ctx.advanced_memory_cleanup(&config)?;
//! println!("Cleaned up {} orphaned gradients", stats.orphaned_gradients_cleaned);
//!
//! // Monitor memory pressure
//! let monitor = ctx.monitor_memory_pressure(&config)?;
//! println!("Current usage: {} bytes", monitor.current_usage_bytes);
//! # Ok(())
//! # }
//! ```
//!
//! ### Graph Optimization
//! ```rust
//! use torsh_autograd::context::{AutogradContext, GraphOptimizer};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! let mut ctx = AutogradContext::new();
//! let optimizer = GraphOptimizer::new();
//!
//! // Apply optimizations
//! let result = optimizer.optimize(&mut ctx)?;
//! println!("Removed {} nodes through optimization", result.nodes_removed);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - **Zero-cost inference**: Graph building is skipped in inference mode for optimal performance
//! - **Memory efficiency**: Automatic cleanup of orphaned gradients and circular references
//! - **SIMD optimization**: Vectorized operations through scirs2 backend integration
//! - **Parallel execution**: Multi-threaded gradient computation where beneficial
//! - **Memory pressure handling**: Automatic cleanup when memory thresholds are exceeded

// Module declarations
pub mod core;
pub mod debug;
pub mod gradient_functions;
pub mod memory;
pub mod optimization;

// Re-export core types for backward compatibility
pub use core::{get_or_create_context, with_context, AutogradContext, GradientFunction, GraphNode};

// Re-export all gradient function implementations
pub use gradient_functions::{
    AbsGradient,

    // Basic arithmetic operations
    AddGradient,
    DivGradient,

    ExpGradient,
    LogGradient,
    // Matrix operations
    MatMulGradient,

    MeanGradient,
    MulGradient,
    PowGradient,
    // Activation functions
    ReLUGradient,
    SigmoidGradient,
    // Mathematical functions
    SquareGradient,
    SubGradient,
    // Reduction operations
    SumGradient,
    TanhGradient,
};

// Re-export optimization types and utilities
pub use optimization::{
    GraphCheckpoint, GraphDiff, GraphOptimizer, GraphStats, OptimizationResult,
};

// Re-export debugging and analysis utilities
pub use debug::{CriticalPath, GraphAnalysis, GraphDebugger, GraphProfiler, ProfileResult};

// Re-export memory management utilities
pub use memory::{
    CleanupStatistics, MemoryManagementConfig, MemoryPressureMonitor, OptimizationStatistics,
};

// Re-export utility modules
pub use memory::utils as memory_utils;

/// Prelude module for convenient imports
pub mod prelude {
    //! Common types and traits for autograd context usage

    pub use super::core::{with_context, AutogradContext, GradientFunction, GraphNode};

    pub use super::gradient_functions::{AddGradient, MulGradient, ReLUGradient, SigmoidGradient};

    pub use super::debug::GraphDebugger;
    pub use super::memory::MemoryManagementConfig;
    pub use super::optimization::GraphOptimizer;
}

/// Advanced features for power users
pub mod advanced {
    //! Advanced autograd context features for specialized use cases

    pub use super::debug::{GraphProfiler, ProfileResult};

    pub use super::optimization::{GraphCheckpoint, GraphDiff, GraphStats};

    pub use super::memory::{CleanupStatistics, MemoryPressureMonitor, OptimizationStatistics};

    /// Re-export memory utility functions
    pub use super::memory::utils::*;
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::Arc;
    use torsh_core::error::Result;

    #[test]
    fn test_context_integration() -> Result<()> {
        // Test basic context creation and operation
        with_context(|ctx| {
            assert_eq!(ctx.graph_size(), 0);

            // Add a simple operation
            ctx.add_operation(
                "test_op".to_string(),
                vec![],
                1,
                true,
                Some(Arc::new(gradient_functions::AddGradient)),
            )?;

            assert_eq!(ctx.graph_size(), 1);
            Ok(())
        })
    }

    #[test]
    fn test_gradient_functions_integration() -> Result<()> {
        // Test that all gradient functions can be created
        let _add = AddGradient;
        let _mul = MulGradient {
            x_values: vec![1.0],
            y_values: vec![2.0],
        };
        let _relu = ReLUGradient {
            input_values: vec![0.5, -0.5],
        };
        let _sigmoid = SigmoidGradient {
            output_values: vec![0.5],
        };

        // Test backward pass
        let grad_output = vec![1.0, 2.0];
        let result = _add.backward(&grad_output)?;
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], grad_output);
        assert_eq!(result[1], grad_output);

        Ok(())
    }

    #[test]
    fn test_optimization_integration() -> Result<()> {
        let mut ctx = AutogradContext::new();
        let optimizer = GraphOptimizer::new();

        // Test optimization on empty graph
        let result = optimizer.optimize(&mut ctx)?;
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.total_nodes_reduced, 0);

        Ok(())
    }

    #[test]
    fn test_debug_integration() -> Result<()> {
        let ctx = AutogradContext::new();
        let debugger = GraphDebugger::new();

        // Test analysis on empty graph
        let analysis = debugger.analyze_graph(&ctx)?;
        assert_eq!(analysis.node_count, 0);
        assert_eq!(analysis.leaf_nodes, 0);
        assert_eq!(analysis.root_nodes, 0);

        // Test report generation
        let report = debugger.generate_graph_report(&ctx)?;
        assert!(report.contains("=== Computation Graph Analysis Report ==="));

        Ok(())
    }

    #[test]
    fn test_memory_integration() -> Result<()> {
        let mut ctx = AutogradContext::new();
        let config = MemoryManagementConfig::default();

        // Test memory cleanup
        let stats = ctx.advanced_memory_cleanup(&config)?;
        assert_eq!(stats.orphaned_gradients_cleaned, 0);

        // Test memory breakdown
        let breakdown = ctx.get_memory_breakdown();
        assert!(breakdown.contains_key("computation_graph"));
        assert!(breakdown.contains_key("gradient_cache"));

        // Test leak detection
        let issues = ctx.check_memory_leaks();
        assert!(issues.is_empty()); // Empty context should have no issues

        Ok(())
    }

    #[test]
    fn test_full_workflow_integration() -> Result<()> {
        // Test a complete workflow using all modules
        with_context(|ctx| {
            // 1. Add operations to graph
            ctx.add_operation("input".to_string(), vec![], 1, true, None)?;

            ctx.add_operation(
                "square".to_string(),
                vec![1],
                2,
                true,
                Some(Arc::new(gradient_functions::SquareGradient {
                    input_values: vec![2.0],
                })),
            )?;

            // 2. Analyze the graph
            let debugger = GraphDebugger::new();
            let analysis = debugger.analyze_graph(ctx)?;
            assert_eq!(analysis.node_count, 2);
            assert_eq!(analysis.leaf_nodes, 1);
            assert_eq!(analysis.root_nodes, 1);

            // 3. Perform backward pass
            ctx.backward_from_tensor(2, vec![1.0])?;

            // 4. Check for gradients - both input and output tensors should have gradients
            assert!(
                ctx.has_gradient(1),
                "Input tensor should have gradient after backward pass"
            );
            assert!(
                ctx.has_gradient(2),
                "Output tensor should have gradient after backward pass"
            );

            // 5. Memory management
            let config = MemoryManagementConfig::default();
            let stats = ctx.advanced_memory_cleanup(&config)?;

            // 6. Optimization
            let optimizer = GraphOptimizer::new();
            let _opt_result = optimizer.optimize(ctx)?;

            Ok(())
        })
    }

    #[test]
    fn test_memory_utilities() {
        use memory::utils::*;

        // Test memory formatting
        assert_eq!(format_memory_size(1024), "1.00 KB");
        assert_eq!(format_memory_size(1024 * 1024), "1.00 MB");

        // Test configuration creation
        let conservative = conservative_memory_config();
        let aggressive = aggressive_memory_config();

        assert!(conservative.max_gradient_age > aggressive.max_gradient_age);
        assert!(conservative.memory_pressure_threshold > aggressive.memory_pressure_threshold);
    }

    #[test]
    fn test_prelude_imports() -> Result<()> {
        use crate::context::prelude::*;

        // Test that prelude imports work
        let _ctx = AutogradContext::new();
        let _debugger = GraphDebugger::new();
        let _optimizer = GraphOptimizer::new();
        let _config = MemoryManagementConfig::default();

        // Test gradient functions
        let _add = AddGradient;
        let _mul = MulGradient {
            x_values: vec![1.0],
            y_values: vec![2.0],
        };

        Ok(())
    }

    #[test]
    fn test_advanced_features() -> Result<()> {
        use crate::context::advanced::*;

        // Test advanced features
        let ctx = AutogradContext::new();
        let profiler = GraphProfiler::new();

        let profile_result = profiler.profile_execution(&ctx)?;
        assert_eq!(profile_result.total_operations, 0); // Empty graph

        // Test checkpoint functionality
        let checkpoint = ctx.checkpoint();
        let diff = ctx.diff_from_checkpoint(&checkpoint);
        assert_eq!(diff.nodes_added, 0);
        assert_eq!(diff.edges_added, 0);

        Ok(())
    }
}
