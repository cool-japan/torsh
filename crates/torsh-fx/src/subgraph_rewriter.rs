//! Subgraph pattern matching and rewriting

use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use torsh_core::error::TorshError;

/// Pattern matcher for subgraphs
pub struct PatternMatcher {
    /// Pattern to match
    pattern: SubgraphPattern,
}

/// Represents a subgraph pattern to match
#[derive(Debug, Clone)]
pub struct SubgraphPattern {
    /// Pattern name
    pub name: String,
    /// Sequence of operations in the pattern
    pub operations: Vec<String>,
    /// Replacement operation
    pub replacement: String,
}

impl SubgraphPattern {
    /// Create a new pattern
    pub fn new(name: String, operations: Vec<String>, replacement: String) -> Self {
        Self {
            name,
            operations,
            replacement,
        }
    }

    /// Create a linear activation fusion pattern
    pub fn linear_relu_fusion() -> Self {
        Self::new(
            "linear_relu_fusion".to_string(),
            vec!["linear".to_string(), "relu".to_string()],
            "linear_relu".to_string(),
        )
    }

    /// Create a conv activation fusion pattern
    pub fn conv_relu_fusion() -> Self {
        Self::new(
            "conv_relu_fusion".to_string(),
            vec!["conv2d".to_string(), "relu".to_string()],
            "conv2d_relu".to_string(),
        )
    }

    /// Create a batch norm fusion pattern
    pub fn conv_bn_fusion() -> Self {
        Self::new(
            "conv_bn_fusion".to_string(),
            vec!["conv2d".to_string(), "batch_norm".to_string()],
            "conv2d_bn".to_string(),
        )
    }

    /// Create a three-operation fusion pattern
    pub fn conv_bn_relu_fusion() -> Self {
        Self::new(
            "conv_bn_relu_fusion".to_string(),
            vec![
                "conv2d".to_string(),
                "batch_norm".to_string(),
                "relu".to_string(),
            ],
            "conv2d_bn_relu".to_string(),
        )
    }
}

/// Match result for a pattern
#[derive(Debug)]
pub struct PatternMatch {
    /// Matched node indices in order
    pub nodes: Vec<NodeIndex>,
    /// Pattern that was matched
    pub pattern: SubgraphPattern,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(pattern: SubgraphPattern) -> Self {
        Self { pattern }
    }

    /// Find all matches of the pattern in the graph
    pub fn find_matches(&self, graph: &FxGraph) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        // Iterate through all nodes to find potential starting points
        for (start_idx, start_node) in graph.nodes() {
            if let Some(pattern_match) = self.match_pattern_at(graph, start_idx, start_node) {
                matches.push(pattern_match);
            }
        }

        matches
    }

    /// Try to match pattern starting at given node
    fn match_pattern_at(
        &self,
        graph: &FxGraph,
        start_idx: NodeIndex,
        start_node: &Node,
    ) -> Option<PatternMatch> {
        // Check if the first operation matches
        if let Node::Call(op_name, _) = start_node {
            if self.pattern.operations.is_empty() || &self.pattern.operations[0] != op_name {
                return None;
            }
        } else {
            return None;
        }

        // Try to match the complete pattern
        if let Some(matched_nodes) = self.match_sequence(graph, start_idx, &self.pattern.operations)
        {
            return Some(PatternMatch {
                nodes: matched_nodes,
                pattern: self.pattern.clone(),
            });
        }

        None
    }

    /// Match a sequence of operations starting from a node
    fn match_sequence(
        &self,
        graph: &FxGraph,
        start_idx: NodeIndex,
        operations: &[String],
    ) -> Option<Vec<NodeIndex>> {
        if operations.is_empty() {
            return Some(vec![]);
        }

        let mut current_nodes = vec![start_idx];
        let mut matched_nodes = vec![start_idx];

        // Match subsequent operations
        for expected_op in &operations[1..] {
            let mut next_nodes = Vec::new();

            for &current_idx in &current_nodes {
                // Find successors of current node
                let successors: Vec<_> = graph
                    .graph
                    .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
                    .collect();

                for successor_idx in successors {
                    if let Some(Node::Call(op_name, _)) = graph.get_node(successor_idx) {
                        if op_name == expected_op {
                            next_nodes.push(successor_idx);
                            matched_nodes.push(successor_idx);
                        }
                    }
                }
            }

            if next_nodes.is_empty() {
                return None; // Pattern doesn't match
            }

            current_nodes = next_nodes;
        }

        Some(matched_nodes)
    }
}

/// Subgraph rewriter for applying pattern transformations
pub struct SubgraphRewriter {
    patterns: Vec<SubgraphPattern>,
}

impl SubgraphRewriter {
    /// Create a new rewriter
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a pattern to the rewriter
    pub fn add_pattern(&mut self, pattern: SubgraphPattern) {
        self.patterns.push(pattern);
    }

    /// Create a rewriter with common fusion patterns
    pub fn with_common_fusions() -> Self {
        let mut rewriter = Self::new();
        rewriter.add_pattern(SubgraphPattern::linear_relu_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_relu_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_bn_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_bn_relu_fusion());
        rewriter
    }

    /// Apply all patterns to the graph
    pub fn apply(&self, graph: &mut FxGraph) -> TorshResult<usize> {
        let mut total_replacements = 0;

        for pattern in &self.patterns {
            let replacements = self.apply_pattern(graph, pattern)?;
            total_replacements += replacements;
        }

        Ok(total_replacements)
    }

    /// Apply a specific pattern to the graph
    fn apply_pattern(&self, graph: &mut FxGraph, pattern: &SubgraphPattern) -> TorshResult<usize> {
        let matcher = PatternMatcher::new(pattern.clone());
        let matches = matcher.find_matches(graph);
        let match_count = matches.len();

        for pattern_match in matches {
            self.replace_match(graph, &pattern_match)?;
        }

        Ok(match_count)
    }

    /// Replace a matched pattern with the replacement operation
    fn replace_match(&self, graph: &mut FxGraph, pattern_match: &PatternMatch) -> TorshResult<()> {
        if pattern_match.nodes.is_empty() {
            return Ok(());
        }

        let first_node_idx = pattern_match.nodes[0];
        let _last_node_idx = *pattern_match
            .nodes
            .last()
            .expect("pattern_match.nodes should not be empty");

        // Get the arguments from the first node
        let args = if let Some(Node::Call(_, args)) = graph.get_node(first_node_idx) {
            args.clone()
        } else {
            vec![]
        };

        // Replace the first node with the fused operation
        graph.graph[first_node_idx] = Node::Call(pattern_match.pattern.replacement.clone(), args);

        // Remove intermediate nodes (keeping first, removing rest)
        for &node_idx in &pattern_match.nodes[1..] {
            // Redirect edges from the removed node to the fused node
            let successors: Vec<_> = graph
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .collect();

            for successor_idx in successors {
                // Find the edge between node_idx and successor_idx
                if let Some(edge_idx) = graph.graph.find_edge(node_idx, successor_idx) {
                    let edge = graph.graph[edge_idx].clone();
                    // Remove old edge and add new edge from fused node
                    graph.graph.remove_edge(edge_idx);
                    graph.graph.add_edge(first_node_idx, successor_idx, edge);
                }
            }

            // Remove the node
            graph.graph.remove_node(node_idx);
        }

        Ok(())
    }
}

impl Default for SubgraphRewriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for replacing patterns
pub fn replace_pattern(graph: &mut FxGraph, pattern: &str, _replacement: &str) -> TorshResult<()> {
    let pattern_obj = match pattern {
        "linear->relu" => SubgraphPattern::linear_relu_fusion(),
        "conv2d->relu" => SubgraphPattern::conv_relu_fusion(),
        "conv2d->batch_norm" => SubgraphPattern::conv_bn_fusion(),
        "conv2d->batch_norm->relu" => SubgraphPattern::conv_bn_relu_fusion(),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unknown pattern: {}",
                pattern
            )));
        }
    };

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(pattern_obj);
    rewriter.apply(graph)?;

    Ok(())
}

/// Apply common fusion optimizations
pub fn apply_fusion_optimizations(graph: &mut FxGraph) -> TorshResult<usize> {
    let rewriter = SubgraphRewriter::with_common_fusions();
    rewriter.apply(graph)
}

/// Replace specific operation sequences
pub fn replace_operation_sequence(
    graph: &mut FxGraph,
    sequence: &[&str],
    replacement: &str,
) -> TorshResult<()> {
    let operations: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
    let pattern = SubgraphPattern::new(
        "custom_pattern".to_string(),
        operations,
        replacement.to_string(),
    );

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(pattern);
    rewriter.apply(graph)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;

    #[test]
    fn test_pattern_creation() {
        let pattern = SubgraphPattern::linear_relu_fusion();
        assert_eq!(pattern.name, "linear_relu_fusion");
        assert_eq!(pattern.operations, vec!["linear", "relu"]);
        assert_eq!(pattern.replacement, "linear_relu");
    }

    #[test]
    fn test_pattern_matching() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let graph = tracer.finalize();

        let pattern = SubgraphPattern::linear_relu_fusion();
        let matcher = PatternMatcher::new(pattern);
        let matches = matcher.find_matches(&graph);

        assert!(!matches.is_empty());
    }

    #[test]
    fn test_subgraph_rewriting() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let original_node_count = graph.node_count();

        let mut rewriter = SubgraphRewriter::new();
        rewriter.add_pattern(SubgraphPattern::linear_relu_fusion());
        let replacements = rewriter.apply(&mut graph).unwrap();

        assert!(replacements > 0);
        // Node count should decrease due to fusion
        assert!(graph.node_count() < original_node_count);
    }

    #[test]
    fn test_convenience_functions() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        // Test string-based pattern replacement
        assert!(replace_pattern(&mut graph, "linear->relu", "linear_relu").is_ok());

        // Test operation sequence replacement
        let mut tracer2 = ModuleTracer::new();
        tracer2.add_input("x");
        tracer2.add_call("conv2d", vec!["x".to_string()]);
        tracer2.add_call("batch_norm", vec!["node_0".to_string()]);
        tracer2.add_call("relu", vec!["node_1".to_string()]);
        tracer2.add_output("node_2");
        let mut graph2 = tracer2.finalize();

        assert!(replace_operation_sequence(
            &mut graph2,
            &["conv2d", "batch_norm", "relu"],
            "conv2d_bn_relu"
        )
        .is_ok());
    }

    #[test]
    fn test_fusion_optimizations() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("conv2d", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let _replacements = apply_fusion_optimizations(&mut graph).unwrap();
        // Should run without error - replacements is a valid count
    }
}
