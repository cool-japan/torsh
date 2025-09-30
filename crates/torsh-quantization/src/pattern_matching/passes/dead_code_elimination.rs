//! Dead Code Elimination Pass
//!
//! This module provides dead code elimination capabilities for computational graphs.
//! It identifies and removes nodes that don't contribute to the final output,
//! helping to reduce graph complexity and improve performance.

use crate::pattern_matching::graph::{ComputationGraph, GraphNode};
use crate::TorshResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use torsh_core::TorshError;

// =============================================================================
// Dead Code Elimination Configuration
// =============================================================================

/// Configuration for dead code elimination behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminationConfig {
    /// Whether to apply aggressive elimination
    pub aggressive: bool,
    /// Maximum number of elimination iterations
    pub max_iterations: usize,
    /// Preserve nodes with specific operation types
    pub preserve_op_types: HashSet<String>,
    /// Preserve nodes with specific name patterns
    pub preserve_name_patterns: Vec<String>,
    /// Whether to preserve debug information nodes
    pub preserve_debug_nodes: bool,
    /// Whether to preserve side-effect nodes
    pub preserve_side_effects: bool,
}

impl Default for EliminationConfig {
    fn default() -> Self {
        let mut preserve_op_types = HashSet::new();
        preserve_op_types.insert("input".to_string());
        preserve_op_types.insert("output".to_string());
        preserve_op_types.insert("placeholder".to_string());
        preserve_op_types.insert("variable".to_string());
        preserve_op_types.insert("constant".to_string());
        preserve_op_types.insert("parameter".to_string());

        Self {
            aggressive: false,
            max_iterations: 10,
            preserve_op_types,
            preserve_name_patterns: vec![
                "output".to_string(),
                "param".to_string(),
                "weight".to_string(),
                "bias".to_string(),
            ],
            preserve_debug_nodes: true,
            preserve_side_effects: true,
        }
    }
}

// =============================================================================
// Dead Code Elimination Pass
// =============================================================================

/// Advanced dead code elimination pass with configurable behavior
#[derive(Debug)]
pub struct DeadCodeEliminationPass {
    /// Configuration for elimination behavior
    config: EliminationConfig,
    /// Statistics tracking
    stats: EliminationStatistics,
}

impl DeadCodeEliminationPass {
    /// Create a new dead code elimination pass
    pub fn new() -> Self {
        Self {
            config: EliminationConfig::default(),
            stats: EliminationStatistics::default(),
        }
    }

    /// Create a dead code elimination pass with custom configuration
    pub fn with_config(config: EliminationConfig) -> Self {
        Self {
            config,
            stats: EliminationStatistics::default(),
        }
    }

    /// Enable or disable aggressive elimination
    pub fn set_aggressive(&mut self, aggressive: bool) {
        self.config.aggressive = aggressive;
    }

    /// Add an operation type to preserve
    pub fn preserve_op_type(&mut self, op_type: String) {
        self.config.preserve_op_types.insert(op_type);
    }

    /// Add a name pattern to preserve
    pub fn preserve_name_pattern(&mut self, pattern: String) {
        self.config.preserve_name_patterns.push(pattern);
    }

    /// Get elimination statistics
    pub fn get_statistics(&self) -> &EliminationStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = EliminationStatistics::default();
    }

    /// Apply dead code elimination to a graph
    pub fn eliminate(&mut self, graph: &mut ComputationGraph) -> TorshResult<EliminationResult> {
        let initial_node_count = graph.nodes.len();
        let mut total_eliminated = 0;
        let mut iteration = 0;

        self.stats.elimination_runs += 1;

        // Repeat until no more dead code is found or max iterations reached
        while iteration < self.config.max_iterations {
            let dead_nodes = self.find_dead_nodes(graph)?;

            if dead_nodes.is_empty() {
                break; // No more dead nodes found
            }

            let eliminated_this_iteration = dead_nodes.len();

            // Remove dead nodes
            for node_id in &dead_nodes {
                graph.remove_node(node_id);
                self.stats.nodes_eliminated += 1;

                // Track elimination by operation type
                if let Some(node) = graph.get_node(node_id) {
                    *self
                        .stats
                        .eliminated_by_op_type
                        .entry(node.op_type.clone())
                        .or_insert(0) += 1;
                }
            }

            total_eliminated += eliminated_this_iteration;
            iteration += 1;

            // If not aggressive and only one iteration, break
            if !self.config.aggressive && iteration >= 1 {
                break;
            }
        }

        let final_node_count = graph.nodes.len();

        Ok(EliminationResult {
            nodes_eliminated: total_eliminated,
            iterations: iteration,
            initial_node_count,
            final_node_count,
            elimination_rate: total_eliminated as f64 / initial_node_count as f64,
            success: true,
        })
    }

    /// Find dead nodes in the graph
    fn find_dead_nodes(&self, graph: &ComputationGraph) -> TorshResult<Vec<String>> {
        let mut dead_nodes = Vec::new();

        // Build a comprehensive analysis of node relationships
        let node_analysis = self.analyze_node_relationships(graph);

        for (node_id, node) in &graph.nodes {
            if self.is_dead_node(node, graph, &node_analysis) {
                dead_nodes.push(node_id.clone());
            }
        }

        Ok(dead_nodes)
    }

    /// Analyze relationships between nodes in the graph
    fn analyze_node_relationships(&self, graph: &ComputationGraph) -> NodeAnalysis {
        let mut reachable_from_outputs = HashSet::new();
        let mut has_side_effects = HashSet::new();

        // Find all nodes reachable from output nodes
        let output_nodes: Vec<&GraphNode> = graph
            .nodes
            .values()
            .filter(|node| self.is_output_node(node))
            .collect();

        for output_node in output_nodes {
            self.mark_reachable_nodes(graph, &output_node.id, &mut reachable_from_outputs);
        }

        // Find nodes with side effects
        for (node_id, node) in &graph.nodes {
            if self.has_side_effects(node) {
                has_side_effects.insert(node_id.clone());
                // Side effect nodes are also considered reachable
                self.mark_reachable_nodes(graph, node_id, &mut reachable_from_outputs);
            }
        }

        NodeAnalysis {
            reachable_from_outputs,
            has_side_effects,
        }
    }

    /// Mark all nodes reachable from a given node (backward traversal)
    fn mark_reachable_nodes(
        &self,
        graph: &ComputationGraph,
        node_id: &str,
        reachable: &mut HashSet<String>,
    ) {
        if reachable.contains(node_id) {
            return; // Already processed
        }

        reachable.insert(node_id.to_string());

        if let Some(node) = graph.get_node(node_id) {
            for input_id in &node.inputs {
                self.mark_reachable_nodes(graph, input_id, reachable);
            }
        }
    }

    /// Check if a node is dead (can be eliminated)
    fn is_dead_node(
        &self,
        node: &GraphNode,
        graph: &ComputationGraph,
        analysis: &NodeAnalysis,
    ) -> bool {
        // Never eliminate nodes that are reachable from outputs
        if analysis.reachable_from_outputs.contains(&node.id) {
            return false;
        }

        // Never eliminate nodes with side effects
        if analysis.has_side_effects.contains(&node.id) {
            return false;
        }

        // Never eliminate preserved nodes
        if self.should_preserve_node(node) {
            return false;
        }

        // Basic dead node detection: no outputs and not referenced
        if node.outputs.is_empty() {
            let is_referenced = graph
                .nodes
                .values()
                .any(|other_node| other_node.inputs.contains(&node.id));

            if !is_referenced {
                return true;
            }
        }

        // Aggressive mode: eliminate nodes that only connect to other dead nodes
        if self.config.aggressive {
            let all_outputs_dead = node.outputs.iter().all(|output_id| {
                if let Some(output_node) = graph.get_node(output_id) {
                    self.is_dead_node(output_node, graph, analysis)
                } else {
                    true // Node doesn't exist, consider it dead
                }
            });

            if all_outputs_dead && !node.outputs.is_empty() {
                return true;
            }
        }

        false
    }

    /// Check if a node should be preserved even if it appears dead
    fn should_preserve_node(&self, node: &GraphNode) -> bool {
        let node_name = node.id.to_lowercase();
        let op_type = node.op_type.to_lowercase();

        // Check preserved operation types
        if self.config.preserve_op_types.contains(&node.op_type)
            || self.config.preserve_op_types.contains(&op_type)
        {
            return true;
        }

        // Check preserved name patterns
        for pattern in &self.config.preserve_name_patterns {
            if node_name.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        // Preserve debug nodes if configured
        if self.config.preserve_debug_nodes && self.is_debug_node(node) {
            return true;
        }

        // Preserve side effect nodes if configured
        if self.config.preserve_side_effects && self.has_side_effects(node) {
            return true;
        }

        false
    }

    /// Check if a node is an output node
    fn is_output_node(&self, node: &GraphNode) -> bool {
        let node_name = node.id.to_lowercase();
        let op_type = node.op_type.to_lowercase();

        // Heuristics for output detection
        node_name.contains("output")
            || op_type.contains("output")
            || node.outputs.is_empty() && !self.is_intermediate_node(node)
    }

    /// Check if a node is an intermediate computation node
    fn is_intermediate_node(&self, node: &GraphNode) -> bool {
        matches!(
            node.op_type.to_lowercase().as_str(),
            "relu"
                | "sigmoid"
                | "tanh"
                | "add"
                | "mul"
                | "sub"
                | "div"
                | "conv2d"
                | "linear"
                | "matmul"
                | "batch_norm"
                | "dropout"
                | "reshape"
                | "transpose"
                | "squeeze"
                | "unsqueeze"
        )
    }

    /// Check if a node is a debug node
    fn is_debug_node(&self, node: &GraphNode) -> bool {
        let node_name = node.id.to_lowercase();
        let op_type = node.op_type.to_lowercase();

        node_name.contains("debug")
            || op_type.contains("debug")
            || op_type.contains("print")
            || op_type.contains("log")
    }

    /// Check if a node has side effects
    fn has_side_effects(&self, node: &GraphNode) -> bool {
        let op_type = node.op_type.to_lowercase();

        matches!(
            op_type.as_str(),
            "print"
                | "debug"
                | "save"
                | "checkpoint"
                | "write"
                | "update"
                | "assign"
                | "modify"
                | "send"
                | "receive"
        )
    }
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Analysis Structures
// =============================================================================

/// Analysis of node relationships in the graph
#[derive(Debug, Clone)]
struct NodeAnalysis {
    /// Nodes reachable from output nodes
    reachable_from_outputs: HashSet<String>,
    /// Nodes with side effects
    has_side_effects: HashSet<String>,
}

// =============================================================================
// Statistics and Results
// =============================================================================

/// Statistics for dead code elimination
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EliminationStatistics {
    /// Number of elimination runs
    pub elimination_runs: usize,
    /// Total nodes eliminated
    pub nodes_eliminated: usize,
    /// Nodes eliminated by operation type
    pub eliminated_by_op_type: HashMap<String, usize>,
}

/// Result of dead code elimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminationResult {
    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Initial node count
    pub initial_node_count: usize,
    /// Final node count
    pub final_node_count: usize,
    /// Elimination rate (0.0 to 1.0)
    pub elimination_rate: f64,
    /// Whether elimination was successful
    pub success: bool,
}

// =============================================================================
// Specialized Elimination Passes
// =============================================================================

/// Create a conservative elimination pass (preserves more nodes)
pub fn create_conservative_pass() -> DeadCodeEliminationPass {
    let config = EliminationConfig {
        aggressive: false,
        max_iterations: 3,
        preserve_debug_nodes: true,
        preserve_side_effects: true,
        ..Default::default()
    };

    DeadCodeEliminationPass::with_config(config)
}

/// Create an aggressive elimination pass
pub fn create_aggressive_pass() -> DeadCodeEliminationPass {
    let config = EliminationConfig {
        aggressive: true,
        max_iterations: 15,
        preserve_debug_nodes: false,
        preserve_side_effects: true,
        ..Default::default()
    };

    DeadCodeEliminationPass::with_config(config)
}

/// Create a production elimination pass (balanced approach)
pub fn create_production_pass() -> DeadCodeEliminationPass {
    let config = EliminationConfig {
        aggressive: true,
        max_iterations: 8,
        preserve_debug_nodes: false,
        preserve_side_effects: true,
        ..Default::default()
    };

    DeadCodeEliminationPass::with_config(config)
}

/// Analyze dead code in a graph without eliminating it
pub fn analyze_dead_code(graph: &ComputationGraph) -> TorshResult<DeadCodeAnalysis> {
    let pass = DeadCodeEliminationPass::new();
    let dead_nodes = pass.find_dead_nodes(graph)?;
    let analysis = pass.analyze_node_relationships(graph);

    let mut dead_by_type = HashMap::new();
    for node_id in &dead_nodes {
        if let Some(node) = graph.get_node(node_id) {
            *dead_by_type.entry(node.op_type.clone()).or_insert(0) += 1;
        }
    }

    Ok(DeadCodeAnalysis {
        total_nodes: graph.nodes.len(),
        dead_nodes: dead_nodes.len(),
        dead_node_ids: dead_nodes,
        dead_by_type,
        reachable_nodes: analysis.reachable_from_outputs.len(),
        side_effect_nodes: analysis.has_side_effects.len(),
    })
}

/// Analysis of dead code in a graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeAnalysis {
    /// Total number of nodes in the graph
    pub total_nodes: usize,
    /// Number of dead nodes
    pub dead_nodes: usize,
    /// IDs of dead nodes
    pub dead_node_ids: Vec<String>,
    /// Dead nodes by operation type
    pub dead_by_type: HashMap<String, usize>,
    /// Number of nodes reachable from outputs
    pub reachable_nodes: usize,
    /// Number of nodes with side effects
    pub side_effect_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_matching::graph::{create_linear_graph, ComputationGraph, GraphNode};

    #[test]
    fn test_elimination_pass_creation() {
        let pass = DeadCodeEliminationPass::new();
        assert!(!pass.config.aggressive);
        assert_eq!(pass.config.max_iterations, 10);

        let aggressive_pass = create_aggressive_pass();
        assert!(aggressive_pass.config.aggressive);
        assert_eq!(aggressive_pass.config.max_iterations, 15);
    }

    #[test]
    fn test_dead_node_detection() {
        let mut graph = ComputationGraph::new();

        // Add a connected chain
        let node1 = GraphNode::new("input".to_string(), "input".to_string());
        let node2 = GraphNode::new("compute".to_string(), "relu".to_string());
        let node3 = GraphNode::new("output".to_string(), "output".to_string());

        // Add an isolated node (should be dead)
        let dead_node = GraphNode::new("isolated".to_string(), "relu".to_string());

        graph.add_node(node1);
        graph.add_node(node2);
        graph.add_node(node3);
        graph.add_node(dead_node);

        graph.connect_nodes("input", "compute").unwrap();
        graph.connect_nodes("compute", "output").unwrap();

        let mut pass = DeadCodeEliminationPass::new();
        let result = pass.eliminate(&mut graph).unwrap();

        assert!(result.success);
        assert!(result.nodes_eliminated > 0);
        // The isolated node should be eliminated
        assert!(graph.get_node("isolated").is_none());
    }

    #[test]
    fn test_preserved_nodes() {
        let pass = DeadCodeEliminationPass::new();

        // Test output node preservation
        let output_node = GraphNode::new("model_output".to_string(), "output".to_string());
        assert!(pass.should_preserve_node(&output_node));

        // Test parameter node preservation
        let param_node = GraphNode::new("weight_param".to_string(), "parameter".to_string());
        assert!(pass.should_preserve_node(&param_node));

        // Test debug node preservation
        let debug_node = GraphNode::new("debug_print".to_string(), "print".to_string());
        assert!(pass.should_preserve_node(&debug_node));

        // Test regular computation node
        let compute_node = GraphNode::new("hidden_relu".to_string(), "relu".to_string());
        assert!(!pass.should_preserve_node(&compute_node));
    }

    #[test]
    fn test_side_effect_detection() {
        let pass = DeadCodeEliminationPass::new();

        let print_node = GraphNode::new("debug".to_string(), "print".to_string());
        assert!(pass.has_side_effects(&print_node));

        let save_node = GraphNode::new("checkpoint".to_string(), "save".to_string());
        assert!(pass.has_side_effects(&save_node));

        let relu_node = GraphNode::new("activation".to_string(), "relu".to_string());
        assert!(!pass.has_side_effects(&relu_node));
    }

    #[test]
    fn test_dead_code_analysis() {
        let mut graph = ComputationGraph::new();

        // Add connected nodes
        let input_node = GraphNode::new("input".to_string(), "input".to_string());
        let compute_node = GraphNode::new("compute".to_string(), "relu".to_string());
        let output_node = GraphNode::new("output".to_string(), "output".to_string());

        // Add isolated dead node
        let dead_node = GraphNode::new("dead".to_string(), "relu".to_string());

        graph.add_node(input_node);
        graph.add_node(compute_node);
        graph.add_node(output_node);
        graph.add_node(dead_node);

        graph.connect_nodes("input", "compute").unwrap();
        graph.connect_nodes("compute", "output").unwrap();

        let analysis = analyze_dead_code(&graph).unwrap();
        assert_eq!(analysis.total_nodes, 4);
        assert!(analysis.dead_nodes > 0);
        assert!(analysis.dead_node_ids.contains(&"dead".to_string()));
    }

    #[test]
    fn test_aggressive_vs_conservative() {
        let conservative = create_conservative_pass();
        let aggressive = create_aggressive_pass();

        assert!(!conservative.config.aggressive);
        assert!(aggressive.config.aggressive);
        assert!(conservative.config.max_iterations < aggressive.config.max_iterations);
    }

    #[test]
    fn test_elimination_statistics() {
        let mut pass = DeadCodeEliminationPass::new();
        let stats = pass.get_statistics();

        assert_eq!(stats.elimination_runs, 0);
        assert_eq!(stats.nodes_eliminated, 0);

        // Statistics should be updated after elimination
        let mut graph = create_linear_graph(&["input", "relu", "output"]);
        let _result = pass.eliminate(&mut graph).unwrap();

        let updated_stats = pass.get_statistics();
        assert!(updated_stats.elimination_runs > 0);
    }

    #[test]
    fn test_custom_configuration() {
        let mut config = EliminationConfig::default();
        config.preserve_op_types.insert("custom_op".to_string());
        config.preserve_name_patterns.push("special".to_string());

        let pass = DeadCodeEliminationPass::with_config(config);

        let custom_node = GraphNode::new("test".to_string(), "custom_op".to_string());
        assert!(pass.should_preserve_node(&custom_node));

        let special_node = GraphNode::new("special_node".to_string(), "relu".to_string());
        assert!(pass.should_preserve_node(&special_node));
    }
}
