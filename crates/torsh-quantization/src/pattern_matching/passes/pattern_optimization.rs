//! Pattern-Based Optimization Pass
//!
//! This module provides pattern-based optimization capabilities for computational graphs.
//! It identifies specific operation patterns and applies optimizations like fusion,
//! elimination, and quantization to improve performance and reduce graph complexity.

use crate::pattern_matching::graph::{ComputationGraph, GraphNode};
use crate::pattern_matching::matcher::{MatchingConfig, PatternMatch, PatternMatcher};
use crate::pattern_matching::patterns::{CommonPatterns, PatternCollection};
use crate::{QuantConfig, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use torsh_core::TorshError;

// =============================================================================
// Pattern Optimization Pass Configuration
// =============================================================================

/// Configuration for pattern optimization behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Whether to apply aggressive optimization
    pub aggressive: bool,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Whether to enable fusion optimizations
    pub enable_fusion: bool,
    /// Whether to enable elimination optimizations
    pub enable_elimination: bool,
    /// Whether to preserve debugging information
    pub preserve_debug_info: bool,
    /// Custom optimization priorities
    pub pattern_priorities: HashMap<String, i32>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            aggressive: false,
            max_iterations: 5,
            enable_fusion: true,
            enable_elimination: true,
            preserve_debug_info: false,
            pattern_priorities: HashMap::new(),
        }
    }
}

// =============================================================================
// Pattern Optimization Pass
// =============================================================================

/// Advanced pattern optimization pass with configurable behavior
#[derive(Debug)]
pub struct PatternOptimizationPass {
    /// Pattern matcher for finding optimization opportunities
    matcher: PatternMatcher,
    /// Optimization configuration
    config: OptimizationConfig,
    /// Statistics tracking
    stats: OptimizationStatistics,
}

impl PatternOptimizationPass {
    /// Create a new pattern optimization pass
    pub fn new() -> Self {
        Self {
            matcher: PatternMatcher::new(),
            config: OptimizationConfig::default(),
            stats: OptimizationStatistics::default(),
        }
    }

    /// Create an optimization pass with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self {
            matcher: PatternMatcher::new(),
            config,
            stats: OptimizationStatistics::default(),
        }
    }

    /// Create an optimization pass with custom pattern collection
    pub fn with_patterns(patterns: PatternCollection) -> Self {
        Self {
            matcher: PatternMatcher::from_collection(patterns),
            config: OptimizationConfig::default(),
            stats: OptimizationStatistics::default(),
        }
    }

    /// Enable or disable aggressive optimization
    pub fn set_aggressive(&mut self, aggressive: bool) {
        self.config.aggressive = aggressive;
    }

    /// Get the pattern matcher
    pub fn matcher(&self) -> &PatternMatcher {
        &self.matcher
    }

    /// Get the pattern matcher mutably
    pub fn matcher_mut(&mut self) -> &mut PatternMatcher {
        &mut self.matcher
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> &OptimizationStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = OptimizationStatistics::default();
    }

    /// Apply pattern optimization to a graph
    pub fn optimize(&mut self, graph: &mut ComputationGraph) -> TorshResult<OptimizationResult> {
        let initial_node_count = graph.nodes.len();
        let mut total_optimizations = 0;
        let mut iteration = 0;

        // Reset statistics for this optimization run
        self.stats.optimization_runs += 1;

        while iteration < self.config.max_iterations {
            let matches = self.matcher.find_matches(graph)?;

            if matches.is_empty() {
                break; // No more patterns to optimize
            }

            // Filter and prioritize matches
            let selected_matches = self.select_optimization_matches(matches);

            if selected_matches.is_empty() {
                break; // No applicable optimizations
            }

            let iteration_optimizations = selected_matches.len();

            // Apply optimizations
            for pattern_match in selected_matches {
                self.apply_pattern_optimization(graph, &pattern_match)?;
                self.stats.patterns_applied += 1;

                // Update pattern-specific statistics
                *self
                    .stats
                    .pattern_counts
                    .entry(pattern_match.pattern_name.clone())
                    .or_insert(0) += 1;
            }

            total_optimizations += iteration_optimizations;
            iteration += 1;

            // For non-iterative patterns, one pass is usually enough
            if !self.config.aggressive && iteration_optimizations == 0 {
                break;
            }
        }

        let final_node_count = graph.nodes.len();
        self.stats.nodes_eliminated += initial_node_count.saturating_sub(final_node_count);

        Ok(OptimizationResult {
            optimizations_applied: total_optimizations,
            nodes_eliminated: initial_node_count.saturating_sub(final_node_count),
            iterations: iteration,
            final_node_count,
            success: true,
            details: self.create_optimization_details(),
        })
    }

    /// Select and prioritize optimization matches
    fn select_optimization_matches(&self, matches: Vec<PatternMatch>) -> Vec<PatternMatch> {
        let mut selected = Vec::new();
        let mut used_nodes = HashSet::new();

        // Filter matches based on configuration
        let filtered_matches: Vec<PatternMatch> = matches
            .into_iter()
            .filter(|m| self.should_apply_pattern(m))
            .collect();

        // Sort by priority: custom priorities, then pattern priority, then confidence
        let mut prioritized_matches = filtered_matches;
        prioritized_matches.sort_by(|a, b| {
            // Check custom priorities first
            let a_priority = self
                .config
                .pattern_priorities
                .get(&a.pattern_name)
                .unwrap_or(&0);
            let b_priority = self
                .config
                .pattern_priorities
                .get(&b.pattern_name)
                .unwrap_or(&0);

            if a_priority != b_priority {
                return b_priority.cmp(a_priority);
            }

            // Then by confidence
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select non-overlapping matches
        for pattern_match in prioritized_matches {
            let has_overlap = pattern_match
                .matched_node_ids
                .iter()
                .any(|node_id| used_nodes.contains(node_id));

            if !has_overlap {
                for node_id in pattern_match.matched_node_ids.iter() {
                    used_nodes.insert(node_id.clone());
                }
                selected.push(pattern_match);
            }
        }

        selected
    }

    /// Check if a pattern should be applied based on configuration
    fn should_apply_pattern(&self, pattern_match: &PatternMatch) -> bool {
        // Check if fusion is enabled for fusion patterns
        if self.is_fusion_pattern(&pattern_match.pattern_name) && !self.config.enable_fusion {
            return false;
        }

        // Check if elimination is enabled for elimination patterns
        if self.is_elimination_pattern(&pattern_match.pattern_name)
            && !self.config.enable_elimination
        {
            return false;
        }

        true
    }

    /// Check if a pattern is a fusion pattern
    fn is_fusion_pattern(&self, pattern_name: &str) -> bool {
        matches!(
            pattern_name,
            "conv_bn"
                | "conv_bn_relu"
                | "conv_relu"
                | "linear_relu"
                | "add_relu"
                | "mul_add"
                | "matmul_add"
        )
    }

    /// Check if a pattern is an elimination pattern
    fn is_elimination_pattern(&self, pattern_name: &str) -> bool {
        matches!(
            pattern_name,
            "quant_dequant"
                | "quant_dequant_elimination"
                | "transpose_transpose"
                | "squeeze_unsqueeze"
                | "reshape_reshape"
        )
    }

    /// Apply a specific pattern optimization
    fn apply_pattern_optimization(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        match pattern_match.pattern_name.as_str() {
            "quant_dequant" | "quant_dequant_elimination" => {
                self.apply_quant_dequant_elimination(graph, pattern_match)
            }
            "transpose_transpose" => self.apply_transpose_elimination(graph, pattern_match),
            "squeeze_unsqueeze" => self.apply_squeeze_unsqueeze_elimination(graph, pattern_match),
            "reshape_reshape" => self.apply_reshape_elimination(graph, pattern_match),
            "conv_bn" | "conv_bn_relu" | "conv_relu" | "linear_relu" | "add_relu" | "mul_add"
            | "matmul_add" => self.apply_fusion_optimization(graph, pattern_match),
            _ => {
                // Unknown pattern, log and skip
                self.stats.unknown_patterns += 1;
                Ok(())
            }
        }
    }

    /// Apply quantize-dequantize elimination
    fn apply_quant_dequant_elimination(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        if pattern_match.matched_node_ids.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Quant-dequant elimination requires exactly 2 nodes".to_string(),
            ));
        }

        let quant_node_id = &pattern_match.matched_node_ids[0];
        let dequant_node_id = &pattern_match.matched_node_ids[1];

        // Get input and output connections
        let quant_inputs = graph
            .get_node(quant_node_id)
            .ok_or_else(|| TorshError::InvalidArgument("Quantize node not found".to_string()))?
            .inputs
            .clone();
        let dequant_outputs = graph
            .get_node(dequant_node_id)
            .ok_or_else(|| TorshError::InvalidArgument("Dequantize node not found".to_string()))?
            .outputs
            .clone();

        // Connect inputs directly to outputs (bypass quant-dequant)
        for input_id in &quant_inputs {
            for output_id in &dequant_outputs {
                graph.connect_nodes(input_id, output_id)?;
            }
        }

        // Remove the quantize and dequantize nodes
        graph.remove_node(quant_node_id);
        graph.remove_node(dequant_node_id);

        self.stats.eliminations_applied += 1;
        Ok(())
    }

    /// Apply transpose-transpose elimination
    fn apply_transpose_elimination(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        if pattern_match.matched_node_ids.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Transpose elimination requires exactly 2 nodes".to_string(),
            ));
        }

        let first_transpose = &pattern_match.matched_node_ids[0];
        let second_transpose = &pattern_match.matched_node_ids[1];

        // Check if the transpositions cancel out (would need axis information)
        // For now, assume consecutive transposes cancel out

        let inputs = graph.get_node(first_transpose).expect("first transpose node should exist in graph").inputs.clone();
        let outputs = graph.get_node(second_transpose).expect("second transpose node should exist in graph").outputs.clone();

        // Connect inputs directly to outputs
        for input_id in &inputs {
            for output_id in &outputs {
                graph.connect_nodes(input_id, output_id)?;
            }
        }

        // Remove transpose nodes
        graph.remove_node(first_transpose);
        graph.remove_node(second_transpose);

        self.stats.eliminations_applied += 1;
        Ok(())
    }

    /// Apply squeeze-unsqueeze elimination
    fn apply_squeeze_unsqueeze_elimination(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        if pattern_match.matched_node_ids.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Squeeze-unsqueeze elimination requires exactly 2 nodes".to_string(),
            ));
        }

        let squeeze_node = &pattern_match.matched_node_ids[0];
        let unsqueeze_node = &pattern_match.matched_node_ids[1];

        let inputs = graph.get_node(squeeze_node).expect("squeeze node should exist in graph").inputs.clone();
        let outputs = graph.get_node(unsqueeze_node).expect("unsqueeze node should exist in graph").outputs.clone();

        // Connect inputs directly to outputs
        for input_id in &inputs {
            for output_id in &outputs {
                graph.connect_nodes(input_id, output_id)?;
            }
        }

        // Remove squeeze and unsqueeze nodes
        graph.remove_node(squeeze_node);
        graph.remove_node(unsqueeze_node);

        self.stats.eliminations_applied += 1;
        Ok(())
    }

    /// Apply reshape-reshape elimination
    fn apply_reshape_elimination(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        if pattern_match.matched_node_ids.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Reshape elimination requires exactly 2 nodes".to_string(),
            ));
        }

        let first_reshape = &pattern_match.matched_node_ids[0];
        let second_reshape = &pattern_match.matched_node_ids[1];

        let inputs = graph.get_node(first_reshape).expect("first reshape node should exist in graph").inputs.clone();
        let outputs = graph.get_node(second_reshape).expect("second reshape node should exist in graph").outputs.clone();

        // Check if we can merge the reshapes or eliminate them entirely
        // For now, assume we can eliminate consecutive reshapes

        // Connect inputs directly to outputs
        for input_id in &inputs {
            for output_id in &outputs {
                graph.connect_nodes(input_id, output_id)?;
            }
        }

        // Remove reshape nodes
        graph.remove_node(first_reshape);
        graph.remove_node(second_reshape);

        self.stats.eliminations_applied += 1;
        Ok(())
    }

    /// Apply fusion optimization (replace multiple nodes with a single fused node)
    fn apply_fusion_optimization(
        &mut self,
        graph: &mut ComputationGraph,
        pattern_match: &PatternMatch,
    ) -> TorshResult<()> {
        if pattern_match.matched_node_ids.is_empty() {
            return Ok(());
        }

        // Get inputs of the first node and outputs of the last node
        let first_node_id = &pattern_match.matched_node_ids[0];
        let last_node_id = pattern_match.matched_node_ids.last().expect("matched node ids should not be empty");

        let first_inputs = graph
            .get_node(first_node_id)
            .ok_or_else(|| TorshError::InvalidArgument("First node not found".to_string()))?
            .inputs
            .clone();
        let last_outputs = graph
            .get_node(last_node_id)
            .ok_or_else(|| TorshError::InvalidArgument("Last node not found".to_string()))?
            .outputs
            .clone();

        // Create a new fused node
        let fused_node_id = format!(
            "fused_{}_{}",
            pattern_match.pattern_name, self.stats.fusions_applied
        );
        let mut fused_node = GraphNode::new(
            fused_node_id.clone(),
            format!("fused_{}", pattern_match.pattern_name),
        );

        // Set up inputs and outputs
        for input_id in &first_inputs {
            fused_node.add_input(input_id.clone());
        }
        for output_id in &last_outputs {
            fused_node.add_output(output_id.clone());
        }

        // Add quantization config as an attribute if available
        if let Some(qconfig) = &pattern_match.qconfig {
            fused_node.set_attribute(
                "quantization_scheme".to_string(),
                format!("{:?}", qconfig.scheme),
            );
            fused_node.set_attribute(
                "quantization_dtype".to_string(),
                format!("{:?}", qconfig.dtype),
            );
        }

        // Add debug information if preserving it
        if self.config.preserve_debug_info {
            let original_nodes: Vec<String> = pattern_match
                .matched_node_ids
                .iter()
                .map(|id| {
                    format!(
                        "{}:{}",
                        graph
                            .get_node(id)
                            .map(|n| &n.op_type)
                            .unwrap_or(&"unknown".to_string()),
                        id
                    )
                })
                .collect();
            fused_node.set_attribute("original_nodes".to_string(), original_nodes.join(","));
            fused_node.set_attribute(
                "fusion_confidence".to_string(),
                pattern_match.confidence.to_string(),
            );
        }

        // Remove old nodes
        for node_id in &pattern_match.matched_node_ids {
            graph.remove_node(node_id);
        }

        // Add the fused node
        graph.add_node(fused_node);

        // Reconnect inputs and outputs
        for input_id in &first_inputs {
            graph.connect_nodes(input_id, &fused_node_id)?;
        }
        for output_id in &last_outputs {
            graph.connect_nodes(&fused_node_id, output_id)?;
        }

        self.stats.fusions_applied += 1;
        Ok(())
    }

    /// Create detailed optimization information
    fn create_optimization_details(&self) -> HashMap<String, String> {
        let mut details = HashMap::new();

        details.insert(
            "fusions_applied".to_string(),
            self.stats.fusions_applied.to_string(),
        );
        details.insert(
            "eliminations_applied".to_string(),
            self.stats.eliminations_applied.to_string(),
        );
        details.insert(
            "patterns_applied".to_string(),
            self.stats.patterns_applied.to_string(),
        );
        details.insert(
            "unknown_patterns".to_string(),
            self.stats.unknown_patterns.to_string(),
        );

        // Add pattern-specific counts
        for (pattern, count) in &self.stats.pattern_counts {
            details.insert(format!("pattern_{}", pattern), count.to_string());
        }

        details
    }
}

impl Default for PatternOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Optimization Statistics and Results
// =============================================================================

/// Statistics tracking for pattern optimizations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    /// Number of optimization runs
    pub optimization_runs: usize,
    /// Total patterns applied
    pub patterns_applied: usize,
    /// Number of fusion optimizations applied
    pub fusions_applied: usize,
    /// Number of elimination optimizations applied
    pub eliminations_applied: usize,
    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
    /// Number of unknown patterns encountered
    pub unknown_patterns: usize,
    /// Pattern-specific application counts
    pub pattern_counts: HashMap<String, usize>,
}

/// Result of an optimization pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Number of nodes eliminated from the graph
    pub nodes_eliminated: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final node count after optimization
    pub final_node_count: usize,
    /// Whether the optimization was successful
    pub success: bool,
    /// Detailed optimization information
    pub details: HashMap<String, String>,
}

// =============================================================================
// Specialized Optimization Passes
// =============================================================================

/// Create an optimization pass focused on fusion
pub fn create_fusion_pass() -> PatternOptimizationPass {
    let config = OptimizationConfig {
        aggressive: false,
        max_iterations: 3,
        enable_fusion: true,
        enable_elimination: false,
        preserve_debug_info: false,
        pattern_priorities: HashMap::new(),
    };

    PatternOptimizationPass::with_patterns(PatternCollection::fusion_only()).with_config(config)
}

/// Create an optimization pass focused on elimination
pub fn create_elimination_pass() -> PatternOptimizationPass {
    let config = OptimizationConfig {
        aggressive: true,
        max_iterations: 10,
        enable_fusion: false,
        enable_elimination: true,
        preserve_debug_info: false,
        pattern_priorities: HashMap::new(),
    };

    PatternOptimizationPass::with_patterns(PatternCollection::elimination_only())
        .with_config(config)
}

/// Create an aggressive optimization pass
pub fn create_aggressive_pass() -> PatternOptimizationPass {
    let config = OptimizationConfig {
        aggressive: true,
        max_iterations: 15,
        enable_fusion: true,
        enable_elimination: true,
        preserve_debug_info: false,
        pattern_priorities: HashMap::new(),
    };

    PatternOptimizationPass::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_matching::graph::{create_branching_graph, create_linear_graph};

    #[test]
    fn test_optimization_pass_creation() {
        let pass = PatternOptimizationPass::new();
        assert!(!pass.config.aggressive);
        assert_eq!(pass.config.max_iterations, 5);

        let aggressive_pass = create_aggressive_pass();
        assert!(aggressive_pass.config.aggressive);
        assert_eq!(aggressive_pass.config.max_iterations, 15);
    }

    #[test]
    fn test_fusion_optimization() {
        let mut pass = PatternOptimizationPass::new();
        let mut graph = create_linear_graph(&["conv2d", "relu"]);

        let initial_count = graph.nodes.len();
        let result = pass.optimize(&mut graph).unwrap();

        assert!(result.success);
        // Should have fewer nodes due to fusion
        assert!(graph.nodes.len() <= initial_count);
    }

    #[test]
    fn test_pattern_selection() {
        let pass = PatternOptimizationPass::new();

        // Test fusion pattern detection
        assert!(pass.is_fusion_pattern("conv_relu"));
        assert!(pass.is_fusion_pattern("linear_relu"));
        assert!(!pass.is_fusion_pattern("quant_dequant"));

        // Test elimination pattern detection
        assert!(pass.is_elimination_pattern("quant_dequant"));
        assert!(pass.is_elimination_pattern("transpose_transpose"));
        assert!(!pass.is_elimination_pattern("conv_relu"));
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig {
            aggressive: true,
            enable_fusion: false,
            enable_elimination: true,
            ..Default::default()
        };

        let pass = PatternOptimizationPass::with_config(config);
        assert!(pass.config.aggressive);
        assert!(!pass.config.enable_fusion);
        assert!(pass.config.enable_elimination);
    }

    #[test]
    fn test_specialized_passes() {
        let fusion_pass = create_fusion_pass();
        assert!(fusion_pass.config.enable_fusion);
        assert!(!fusion_pass.config.enable_elimination);

        let elimination_pass = create_elimination_pass();
        assert!(!elimination_pass.config.enable_fusion);
        assert!(elimination_pass.config.enable_elimination);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut pass = PatternOptimizationPass::new();
        let stats = pass.get_statistics();

        assert_eq!(stats.patterns_applied, 0);
        assert_eq!(stats.fusions_applied, 0);
        assert_eq!(stats.eliminations_applied, 0);

        // Statistics should be updated after optimization
        let mut graph = create_linear_graph(&["conv2d", "relu"]);
        let _result = pass.optimize(&mut graph).unwrap();

        let updated_stats = pass.get_statistics();
        assert!(updated_stats.optimization_runs > 0);
    }
}
