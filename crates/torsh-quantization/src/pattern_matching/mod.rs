//! Pattern matching for graph optimization
//!
//! This module provides pattern matching capabilities for computational graphs,
//! enabling optimization passes to identify specific operation patterns that
//! can be fused, eliminated, or optimized.
//!
//! # Overview
//!
//! The pattern matching system consists of several key components:
//!
//! - **Graph**: Core graph data structures and operations
//! - **Patterns**: Pattern definition system for matching operation sequences
//! - **Matcher**: Pattern matching engine with caching and confidence scoring
//! - **Passes**: Optimization passes for graph transformation
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_quantization::pattern_matching::{
//!     ComputationGraph, GraphNode, PatternMatcher, GraphPattern,
//!     passes::PassManager
//! };
//!
//! // Create a computation graph
//! let mut graph = ComputationGraph::new();
//! let node = GraphNode::new("conv1".to_string(), "conv2d".to_string());
//! graph.add_node(node);
//!
//! // Run optimization passes
//! let mut pass_manager = PassManager::new();
//! let result = pass_manager.run_all(&mut graph)?;
//! ```
//!
//! # Modules
//!
//! - [`graph`]: Core graph data structures and operations
//! - [`patterns`]: Pattern definition system and common patterns
//! - [`matcher`]: Pattern matching engine with advanced features
//! - [`passes`]: Optimization passes and pass management

use crate::error::TorshResult;
use crate::QuantConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Re-export sub-modules
pub mod graph;
pub mod matcher;
pub mod passes;
pub mod patterns;

// Re-export core types for backward compatibility
pub use graph::{ComputationGraph, GraphEdge, GraphMetrics, GraphNode, GraphValidation, NodeType};
pub use matcher::{
    MatchConfidence, MatchingConfig, MatchingStatistics, PatternMatch, PatternMatcher,
};
pub use passes::{
    ConstantFoldingPass, ConstantValue, DeadCodeEliminationPass, EliminationConfig,
    EliminationStatistics, FoldingConfig, FoldingStatistics, OptimizationConfig,
    OptimizationStatistics, PassConfig, PassManager, PassResult, PassType, PatternOptimizationPass,
};
pub use patterns::{
    CommonPatterns, GraphPattern, PatternConfig, PatternConstraint, PatternEdge, PatternLibrary,
    PatternNode, PatternType,
};

/// Unified optimization pass that coordinates multiple optimization strategies
#[derive(Debug)]
pub struct OptimizationPass {
    pass_manager: PassManager,
    config: OptimizationPassConfig,
}

/// Configuration for the unified optimization pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPassConfig {
    /// Enable pattern-based optimizations
    pub enable_pattern_optimization: bool,
    /// Enable dead code elimination
    pub enable_dead_code_elimination: bool,
    /// Enable constant folding
    pub enable_constant_folding: bool,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Enable detailed logging
    pub verbose: bool,
    /// Quantization configuration for pattern optimization
    pub qconfig: Option<QuantConfig>,
}

impl Default for OptimizationPassConfig {
    fn default() -> Self {
        Self {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: true,
            max_iterations: 10,
            convergence_threshold: 1e-6,
            verbose: false,
            qconfig: None,
        }
    }
}

/// Result of running the unified optimization pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Total execution time
    pub execution_time: Duration,
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
    /// Number of optimization iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of patterns matched and applied
    pub patterns_applied: usize,
    /// Number of dead nodes eliminated
    pub dead_nodes_eliminated: usize,
    /// Number of constants folded
    pub constants_folded: usize,
    /// Overall optimization score (0.0 to 1.0)
    pub optimization_score: f64,
    /// Detailed pass results
    pub pass_result: PassResult,
}

impl OptimizationPass {
    /// Create a new optimization pass with default configuration
    pub fn new() -> Self {
        Self::with_config(OptimizationPassConfig::default())
    }

    /// Create a new optimization pass with custom configuration
    pub fn with_config(config: OptimizationPassConfig) -> Self {
        let pass_config = PassConfig {
            enable_pattern_optimization: config.enable_pattern_optimization,
            enable_dead_code_elimination: config.enable_dead_code_elimination,
            enable_constant_folding: config.enable_constant_folding,
            max_iterations: config.max_iterations,
            convergence_threshold: config.convergence_threshold,
            verbose: config.verbose,
            custom_order: None,
        };

        let mut pass_manager = PassManager::with_config(pass_config);

        // Configure individual passes based on config
        if let Some(ref qconfig) = config.qconfig {
            pass_manager.configure_pattern_optimization(OptimizationConfig {
                enable_fusion: true,
                enable_elimination: true,
                max_pattern_size: 10,
                confidence_threshold: 0.8,
                enable_statistics: true,
                quantization_config: Some(qconfig.clone()),
                ..Default::default()
            });
        }

        Self {
            pass_manager,
            config,
        }
    }

    /// Run optimization on the computation graph
    pub fn optimize(&mut self, graph: &mut ComputationGraph) -> TorshResult<OptimizationResult> {
        let pass_result = self.pass_manager.run_all(graph)?;

        // Extract statistics from pass results
        let patterns_applied = pass_result
            .pattern_stats
            .as_ref()
            .map(|s| s.total_optimizations)
            .unwrap_or(0);

        let dead_nodes_eliminated = pass_result
            .elimination_stats
            .as_ref()
            .map(|s| s.nodes_removed)
            .unwrap_or(0);

        let constants_folded = pass_result
            .folding_stats
            .as_ref()
            .map(|s| s.nodes_folded)
            .unwrap_or(0);

        Ok(OptimizationResult {
            execution_time: pass_result.total_time,
            nodes_before: pass_result.nodes_before,
            nodes_after: pass_result.nodes_after,
            iterations: pass_result.iterations,
            converged: pass_result.converged,
            patterns_applied,
            dead_nodes_eliminated,
            constants_folded,
            optimization_score: pass_result.improvement_score,
            pass_result,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &OptimizationPassConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: OptimizationPassConfig) {
        self.config = config.clone();

        let pass_config = PassConfig {
            enable_pattern_optimization: config.enable_pattern_optimization,
            enable_dead_code_elimination: config.enable_dead_code_elimination,
            enable_constant_folding: config.enable_constant_folding,
            max_iterations: config.max_iterations,
            convergence_threshold: config.convergence_threshold,
            verbose: config.verbose,
            custom_order: None,
        };

        self.pass_manager.set_config(pass_config);
    }

    /// Reset all statistics
    pub fn reset_statistics(&mut self) {
        self.pass_manager.reset_statistics();
    }

    /// Get the underlying pass manager
    pub fn pass_manager(&mut self) -> &mut PassManager {
        &mut self.pass_manager
    }
}

impl Default for OptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for pattern matching operations
pub mod utils {
    use super::*;

    /// Create a simple computation graph for testing
    pub fn create_test_graph() -> ComputationGraph {
        let mut graph = ComputationGraph::new();

        // Add some test nodes
        let input = GraphNode::new("input".to_string(), "input".to_string());
        let conv1 = GraphNode::new("conv1".to_string(), "conv2d".to_string());
        let relu1 = GraphNode::new("relu1".to_string(), "relu".to_string());
        let output = GraphNode::new("output".to_string(), "output".to_string());

        graph.add_node(input);
        graph.add_node(conv1);
        graph.add_node(relu1);
        graph.add_node(output);

        // Connect nodes
        graph.connect_nodes("input", "conv1").unwrap();
        graph.connect_nodes("conv1", "relu1").unwrap();
        graph.connect_nodes("relu1", "output").unwrap();

        graph
    }

    /// Create an optimization pass configured for aggressive optimization
    pub fn aggressive_optimization() -> OptimizationPass {
        OptimizationPass::with_config(OptimizationPassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: true,
            max_iterations: 20,
            convergence_threshold: 1e-8,
            verbose: false,
            qconfig: None,
        })
    }

    /// Create an optimization pass configured for fast compilation
    pub fn fast_optimization() -> OptimizationPass {
        OptimizationPass::with_config(OptimizationPassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: false, // Skip for speed
            max_iterations: 3,
            convergence_threshold: 1e-3,
            verbose: false,
            qconfig: None,
        })
    }

    /// Create an optimization pass configured for quantization-aware optimization
    pub fn quantization_aware_optimization(qconfig: QuantConfig) -> OptimizationPass {
        OptimizationPass::with_config(OptimizationPassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: true,
            max_iterations: 15,
            convergence_threshold: 1e-6,
            verbose: false,
            qconfig: Some(qconfig),
        })
    }

    /// Validate a computation graph for common issues
    pub fn validate_graph(graph: &ComputationGraph) -> Result<(), String> {
        // Check for cycles
        if let Err(e) = graph.validate_acyclic() {
            return Err(format!("Graph contains cycles: {}", e));
        }

        // Check for orphaned nodes
        let orphaned = graph.find_orphaned_nodes();
        if !orphaned.is_empty() {
            return Err(format!("Graph contains orphaned nodes: {:?}", orphaned));
        }

        // Check for invalid connections
        for (node_id, node) in &graph.nodes {
            for input_id in &node.inputs {
                if !graph.nodes.contains_key(input_id) {
                    return Err(format!(
                        "Node {} references non-existent input {}",
                        node_id, input_id
                    ));
                }
            }
            for output_id in &node.outputs {
                if !graph.nodes.contains_key(output_id) {
                    return Err(format!(
                        "Node {} references non-existent output {}",
                        node_id, output_id
                    ));
                }
            }
        }

        Ok(())
    }

    /// Compute basic statistics about a computation graph
    pub fn compute_graph_statistics(graph: &ComputationGraph) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        stats.insert("total_nodes".to_string(), graph.nodes.len());
        stats.insert(
            "execution_order_length".to_string(),
            graph.execution_order.len(),
        );

        // Count nodes by type
        let mut op_counts = HashMap::new();
        for node in graph.nodes.values() {
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        for (op_type, count) in op_counts {
            stats.insert(format!("{}_nodes", op_type), count);
        }

        // Count edges
        let total_edges: usize = graph
            .nodes
            .values()
            .map(|node| node.inputs.len() + node.outputs.len())
            .sum();
        stats.insert("total_edges".to_string(), total_edges);

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_pass_creation() {
        let pass = OptimizationPass::new();
        assert!(pass.config().enable_pattern_optimization);
        assert!(pass.config().enable_dead_code_elimination);
        assert!(pass.config().enable_constant_folding);
    }

    #[test]
    fn test_optimization_pass_config() {
        let config = OptimizationPassConfig {
            enable_pattern_optimization: false,
            max_iterations: 5,
            ..Default::default()
        };

        let pass = OptimizationPass::with_config(config);
        assert!(!pass.config().enable_pattern_optimization);
        assert_eq!(pass.config().max_iterations, 5);
    }

    #[test]
    fn test_utils_test_graph() {
        let graph = utils::create_test_graph();
        assert_eq!(graph.nodes.len(), 4);
        assert!(graph.nodes.contains_key("input"));
        assert!(graph.nodes.contains_key("conv1"));
        assert!(graph.nodes.contains_key("relu1"));
        assert!(graph.nodes.contains_key("output"));
    }

    #[test]
    fn test_utils_graph_validation() {
        let graph = utils::create_test_graph();
        assert!(utils::validate_graph(&graph).is_ok());

        // Test with empty graph
        let empty_graph = ComputationGraph::new();
        assert!(utils::validate_graph(&empty_graph).is_ok());
    }

    #[test]
    fn test_utils_graph_statistics() {
        let graph = utils::create_test_graph();
        let stats = utils::compute_graph_statistics(&graph);

        assert_eq!(stats.get("total_nodes").unwrap(), &4);
        assert_eq!(stats.get("input_nodes").unwrap(), &1);
        assert_eq!(stats.get("conv2d_nodes").unwrap(), &1);
        assert_eq!(stats.get("relu_nodes").unwrap(), &1);
        assert_eq!(stats.get("output_nodes").unwrap(), &1);
    }

    #[test]
    fn test_optimization_result_serialization() {
        let result = OptimizationResult {
            execution_time: Duration::from_millis(100),
            nodes_before: 10,
            nodes_after: 8,
            iterations: 3,
            converged: true,
            patterns_applied: 2,
            dead_nodes_eliminated: 1,
            constants_folded: 1,
            optimization_score: 0.2,
            pass_result: PassResult {
                total_time: Duration::from_millis(100),
                pass_times: HashMap::new(),
                iterations: 3,
                converged: true,
                pattern_stats: None,
                elimination_stats: None,
                folding_stats: None,
                nodes_before: 10,
                nodes_after: 8,
                improvement_score: 0.2,
            },
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: OptimizationResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.nodes_before, deserialized.nodes_before);
        assert_eq!(result.nodes_after, deserialized.nodes_after);
        assert_eq!(result.converged, deserialized.converged);
    }
}
