//! Graph analysis and linting utilities for FX graphs
//!
//! This module provides comprehensive graph analysis capabilities including:
//! - Graph linting with best practice suggestions
//! - Graph diff and merge functionality for version control
//! - Advanced graph metrics and health checking
//! - Pattern detection and architectural analysis

use crate::{Edge, FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Graph linting utilities for best practice validation
#[derive(Debug)]
pub struct GraphLinter {
    rules: Vec<LintRule>,
    severity_threshold: LintSeverity,
}

/// Lint rule for graph validation
#[derive(Debug, Clone)]
pub struct LintRule {
    pub name: String,
    pub description: String,
    pub severity: LintSeverity,
    pub checker: fn(&FxGraph) -> Vec<LintIssue>,
}

/// Severity levels for lint issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LintSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Individual lint issue found in graph
#[derive(Debug, Clone)]
pub struct LintIssue {
    pub rule_name: String,
    pub severity: LintSeverity,
    pub message: String,
    pub node_index: Option<NodeIndex>,
    pub suggestions: Vec<String>,
}

/// Complete lint report for a graph
#[derive(Debug, Serialize, Deserialize)]
pub struct LintReport {
    pub total_issues: u32,
    pub issues_by_severity: HashMap<LintSeverity, u32>,
    #[serde(skip)]
    pub issues: Vec<LintIssue>,
    pub overall_score: f64, // 0.0 (bad) to 1.0 (perfect)
    pub recommendations: Vec<String>,
}

impl GraphLinter {
    /// Create a new graph linter with default rules
    pub fn new() -> Self {
        let mut linter = Self {
            rules: Vec::new(),
            severity_threshold: LintSeverity::Info,
        };
        linter.add_default_rules();
        linter
    }

    /// Set minimum severity threshold for reporting
    pub fn with_severity_threshold(mut self, threshold: LintSeverity) -> Self {
        self.severity_threshold = threshold;
        self
    }

    /// Add a custom lint rule
    pub fn add_rule(&mut self, rule: LintRule) {
        self.rules.push(rule);
    }

    /// Lint a graph and return comprehensive report
    pub fn lint_graph(&self, graph: &FxGraph) -> LintReport {
        let mut all_issues = Vec::new();

        // Run all lint rules
        for rule in &self.rules {
            let mut issues = (rule.checker)(graph);
            // Add rule name to each issue
            for issue in &mut issues {
                issue.rule_name = rule.name.clone();
            }
            all_issues.extend(issues);
        }

        // Filter by severity threshold
        all_issues.retain(|issue| issue.severity >= self.severity_threshold);

        // Generate statistics
        let total_issues = all_issues.len() as u32;
        let mut issues_by_severity = HashMap::new();
        for issue in &all_issues {
            *issues_by_severity
                .entry(issue.severity.clone())
                .or_insert(0) += 1;
        }

        // Calculate overall score (0.0 to 1.0)
        let overall_score = self.calculate_overall_score(&all_issues, graph);

        // Generate global recommendations
        let recommendations = self.generate_global_recommendations(&all_issues, graph);

        LintReport {
            total_issues,
            issues_by_severity,
            issues: all_issues,
            overall_score,
            recommendations,
        }
    }

    /// Add default lint rules
    fn add_default_rules(&mut self) {
        // Rule: Check for disconnected nodes
        self.add_rule(LintRule {
            name: "disconnected_nodes".to_string(),
            description: "Detect nodes with no incoming or outgoing connections".to_string(),
            severity: LintSeverity::Warning,
            checker: |graph| {
                let mut issues = Vec::new();
                for (idx, node) in graph.nodes() {
                    let has_incoming = graph
                        .graph
                        .edges_directed(idx, petgraph::Incoming)
                        .next()
                        .is_some();
                    let has_outgoing = graph
                        .graph
                        .edges_directed(idx, petgraph::Outgoing)
                        .next()
                        .is_some();

                    if !has_incoming
                        && !has_outgoing
                        && !matches!(node, Node::Input(_) | Node::Output)
                    {
                        issues.push(LintIssue {
                            rule_name: "".to_string(), // Will be filled by caller
                            severity: LintSeverity::Warning,
                            message: format!("Node {idx:?} is disconnected from the graph"),
                            node_index: Some(idx),
                            suggestions: vec![
                                "Remove unused node or connect it to the graph".to_string()
                            ],
                        });
                    }
                }
                issues
            },
        });

        // Rule: Check for cycles in the graph
        self.add_rule(LintRule {
            name: "cycles".to_string(),
            description: "Detect cycles that may cause infinite loops".to_string(),
            severity: LintSeverity::Error,
            checker: |graph| {
                let mut issues = Vec::new();
                if petgraph::algo::is_cyclic_directed(&graph.graph) {
                    issues.push(LintIssue {
                        rule_name: "".to_string(),
                        severity: LintSeverity::Error,
                        message: "Graph contains cycles which may cause infinite loops".to_string(),
                        node_index: None,
                        suggestions: vec![
                            "Review loop constructs and ensure proper termination conditions"
                                .to_string(),
                            "Consider breaking cycles with merge nodes".to_string(),
                        ],
                    });
                }
                issues
            },
        });

        // Rule: Check for missing inputs/outputs
        self.add_rule(LintRule {
            name: "missing_io".to_string(),
            description: "Ensure graph has proper input and output nodes".to_string(),
            severity: LintSeverity::Error,
            checker: |graph| {
                let mut issues = Vec::new();

                if graph.inputs().is_empty() {
                    issues.push(LintIssue {
                        rule_name: "".to_string(),
                        severity: LintSeverity::Error,
                        message: "Graph has no input nodes".to_string(),
                        node_index: None,
                        suggestions: vec![
                            "Add input nodes to define graph entry points".to_string()
                        ],
                    });
                }

                if graph.outputs().is_empty() {
                    issues.push(LintIssue {
                        rule_name: "".to_string(),
                        severity: LintSeverity::Error,
                        message: "Graph has no output nodes".to_string(),
                        node_index: None,
                        suggestions: vec!["Add output nodes to define graph results".to_string()],
                    });
                }
                issues
            },
        });

        // Rule: Check for inefficient patterns
        self.add_rule(LintRule {
            name: "inefficient_patterns".to_string(),
            description: "Detect known inefficient operation patterns".to_string(),
            severity: LintSeverity::Info,
            checker: |graph| {
                let mut issues = Vec::new();

                // Check for consecutive transpose operations
                for (idx, node) in graph.nodes() {
                    if let Node::Call(op, _) = node {
                        if op == "transpose" {
                            // Check if followed by another transpose
                            for neighbor in graph.graph.neighbors(idx) {
                                if let Some(Node::Call(neighbor_op, _)) = graph.get_node(neighbor) {
                                    if neighbor_op == "transpose" {
                                        issues.push(LintIssue {
                                            rule_name: "".to_string(),
                                            severity: LintSeverity::Info,
                                            message: "Consecutive transpose operations detected".to_string(),
                                            node_index: Some(idx),
                                            suggestions: vec!["Consider fusing consecutive transposes or eliminating them if they cancel out".to_string()],
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                issues
            },
        });

        // Rule: Check for large fan-out
        self.add_rule(LintRule {
            name: "large_fanout".to_string(),
            description: "Detect nodes with excessive fan-out".to_string(),
            severity: LintSeverity::Warning,
            checker: |graph| {
                let mut issues = Vec::new();
                const MAX_FANOUT: usize = 10;

                for (idx, _node) in graph.nodes() {
                    let fanout = graph.graph.edges_directed(idx, petgraph::Outgoing).count();
                    if fanout > MAX_FANOUT {
                        issues.push(LintIssue {
                            rule_name: "".to_string(),
                            severity: LintSeverity::Warning,
                            message: format!("Node {idx:?} has high fan-out of {fanout}"),
                            node_index: Some(idx),
                            suggestions: vec![
                                "Consider adding intermediate nodes to reduce fan-out".to_string(),
                                "Verify if all outputs are necessary".to_string(),
                            ],
                        });
                    }
                }
                issues
            },
        });
    }

    /// Calculate overall graph health score
    fn calculate_overall_score(&self, issues: &[LintIssue], graph: &FxGraph) -> f64 {
        let total_nodes = graph.node_count() as f64;
        if total_nodes == 0.0 {
            return 0.0;
        }

        let mut penalty = 0.0;
        for issue in issues {
            penalty += match issue.severity {
                LintSeverity::Info => 0.1,
                LintSeverity::Warning => 0.3,
                LintSeverity::Error => 0.7,
                LintSeverity::Critical => 1.0,
            };
        }

        // Normalize penalty by graph size
        let normalized_penalty = penalty / total_nodes;
        (1.0 - normalized_penalty).max(0.0)
    }

    /// Generate global recommendations based on all issues
    fn generate_global_recommendations(
        &self,
        issues: &[LintIssue],
        graph: &FxGraph,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Count issue types
        let error_count = issues
            .iter()
            .filter(|i| i.severity >= LintSeverity::Error)
            .count();
        let warning_count = issues
            .iter()
            .filter(|i| i.severity == LintSeverity::Warning)
            .count();

        if error_count > 0 {
            recommendations.push("Fix critical errors before deploying the graph".to_string());
        }

        if warning_count > 3 {
            recommendations.push("Consider refactoring to address multiple warnings".to_string());
        }

        if graph.node_count() > 100 {
            recommendations
                .push("Consider breaking large graphs into smaller subgraphs".to_string());
        }

        if graph.edge_count() > graph.node_count() * 2 {
            recommendations.push(
                "Graph appears to have high connectivity - verify if all connections are necessary"
                    .to_string(),
            );
        }

        recommendations.push("Run graph optimization passes to improve performance".to_string());
        recommendations.push("Add comprehensive documentation for complex operations".to_string());

        recommendations
    }
}

/// Graph diff and merge functionality for version control
pub struct GraphDiff;

#[derive(Debug, Clone)]
pub struct GraphDifference {
    pub added_nodes: Vec<(NodeIndex, Node)>,
    pub removed_nodes: Vec<(NodeIndex, Node)>,
    pub modified_nodes: Vec<(NodeIndex, Node, Node)>, // (index, old, new)
    pub added_edges: Vec<(NodeIndex, NodeIndex, Edge)>,
    pub removed_edges: Vec<(NodeIndex, NodeIndex, Edge)>,
}

impl GraphDiff {
    /// Calculate differences between two graphs
    pub fn diff(old_graph: &FxGraph, new_graph: &FxGraph) -> GraphDifference {
        let mut diff = GraphDifference {
            added_nodes: Vec::new(),
            removed_nodes: Vec::new(),
            modified_nodes: Vec::new(),
            added_edges: Vec::new(),
            removed_edges: Vec::new(),
        };

        // Create node maps for comparison
        let old_nodes: HashMap<String, (NodeIndex, &Node)> = old_graph
            .nodes()
            .map(|(idx, node)| (Self::node_key(node), (idx, node)))
            .collect();

        let new_nodes: HashMap<String, (NodeIndex, &Node)> = new_graph
            .nodes()
            .map(|(idx, node)| (Self::node_key(node), (idx, node)))
            .collect();

        // Find added and modified nodes
        for (key, (new_idx, new_node)) in &new_nodes {
            if let Some((_old_idx, old_node)) = old_nodes.get(key) {
                if !Self::nodes_equal(old_node, new_node) {
                    diff.modified_nodes
                        .push((*new_idx, (*old_node).clone(), (*new_node).clone()));
                }
            } else {
                diff.added_nodes.push((*new_idx, (*new_node).clone()));
            }
        }

        // Find removed nodes
        for (key, (old_idx, old_node)) in &old_nodes {
            if !new_nodes.contains_key(key) {
                diff.removed_nodes.push((*old_idx, (*old_node).clone()));
            }
        }

        // Compare edges (simplified comparison)
        let _old_edges: HashSet<String> = old_graph
            .graph
            .edge_references()
            .map(|edge| {
                use petgraph::visit::EdgeRef;
                format!(
                    "{}->{}:{}",
                    edge.source().index(),
                    edge.target().index(),
                    edge.weight().name
                )
            })
            .collect();

        let _new_edges: HashSet<String> = new_graph
            .graph
            .edge_references()
            .map(|edge| {
                use petgraph::visit::EdgeRef;
                format!(
                    "{}->{}:{}",
                    edge.source().index(),
                    edge.target().index(),
                    edge.weight().name
                )
            })
            .collect();

        // For now, just track edge count differences
        // A more sophisticated implementation would track actual edge changes

        diff
    }

    /// Merge changes from one graph into another
    pub fn merge(base_graph: &FxGraph, diff: &GraphDifference) -> TorshResult<FxGraph> {
        let mut merged_graph = base_graph.clone();

        // Apply node additions
        for (_idx, node) in &diff.added_nodes {
            merged_graph.graph.add_node(node.clone());
        }

        // Apply node modifications (simplified)
        for (idx, _old_node, new_node) in &diff.modified_nodes {
            if let Some(node_weight) = merged_graph.graph.node_weight_mut(*idx) {
                *node_weight = new_node.clone();
            }
        }

        // Apply edge changes would require more complex tracking
        // This is a simplified implementation

        Ok(merged_graph)
    }

    /// Generate a unique key for a node for comparison
    fn node_key(node: &Node) -> String {
        match node {
            Node::Input(name) => format!("input:{name}"),
            Node::Call(op, args) => {
                let args_str = args.join(",");
                format!("call:{op}:{args_str}")
            }
            Node::Output => "output".to_string(),
            Node::Conditional { condition, .. } => format!("conditional:{condition}"),
            Node::Loop { condition, .. } => format!("loop:{}", condition),
            Node::Merge { inputs } => format!("merge:{}", inputs.join(",")),
            Node::GetAttr { target, attr } => format!("getattr:{}:{}", target, attr),
        }
    }

    /// Check if two nodes are functionally equal
    fn nodes_equal(node1: &Node, node2: &Node) -> bool {
        std::mem::discriminant(node1) == std::mem::discriminant(node2)
            && Self::node_key(node1) == Self::node_key(node2)
    }
}

/// Advanced graph metrics and health checking
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub max_depth: usize,
    pub average_fanout: f64,
    pub connectivity_ratio: f64,
    pub complexity_score: f64,
    pub operation_distribution: HashMap<String, u32>,
    pub critical_path_length: usize,
}

/// Graph pattern detection
pub struct PatternDetector;

#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern_type: String,
    pub description: String,
    pub nodes: Vec<NodeIndex>,
    pub confidence: f64,
    pub optimization_potential: String,
}

impl PatternDetector {
    /// Detect common patterns in the graph
    pub fn detect_patterns(graph: &FxGraph) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        // Detect linear chains
        patterns.extend(Self::detect_linear_chains(graph));

        // Detect fan-out patterns
        patterns.extend(Self::detect_fanout_patterns(graph));

        // Detect bottlenecks
        patterns.extend(Self::detect_bottlenecks(graph));

        patterns
    }

    /// Detect linear chains of operations
    fn detect_linear_chains(graph: &FxGraph) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();
        let mut visited = HashSet::new();

        for (start_idx, _) in graph.nodes() {
            if visited.contains(&start_idx) {
                continue;
            }

            let chain = Self::trace_linear_chain(graph, start_idx, &mut visited);
            if chain.len() > 3 {
                // Consider chains of 4+ nodes as patterns
                patterns.push(DetectedPattern {
                    pattern_type: "linear_chain".to_string(),
                    description: format!("Linear chain of {} operations", chain.len()),
                    nodes: chain,
                    confidence: 0.9,
                    optimization_potential: "Consider operation fusion for better performance"
                        .to_string(),
                });
            }
        }

        patterns
    }

    /// Trace a linear chain from a starting node
    fn trace_linear_chain(
        graph: &FxGraph,
        start: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> Vec<NodeIndex> {
        let mut chain = vec![start];
        visited.insert(start);
        let mut current = start;

        loop {
            let neighbors: Vec<_> = graph.graph.neighbors(current).collect();
            if neighbors.len() != 1 {
                break; // Not a linear chain
            }

            let next = neighbors[0];
            if visited.contains(&next) {
                break; // Already processed
            }

            let incoming: Vec<_> = graph
                .graph
                .neighbors_directed(next, petgraph::Incoming)
                .collect();
            if incoming.len() != 1 {
                break; // Next node has multiple inputs
            }

            chain.push(next);
            visited.insert(next);
            current = next;
        }

        chain
    }

    /// Detect fan-out patterns
    fn detect_fanout_patterns(graph: &FxGraph) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        for (idx, _node) in graph.nodes() {
            let fanout = graph.graph.neighbors(idx).count();
            if fanout > 5 {
                // High fan-out threshold
                let _neighbors: Vec<_> = graph.graph.neighbors(idx).collect();
                patterns.push(DetectedPattern {
                    pattern_type: "high_fanout".to_string(),
                    description: format!("High fan-out node with {} outputs", fanout),
                    nodes: vec![idx],
                    confidence: 1.0,
                    optimization_potential: "Consider broadcast optimization or result caching"
                        .to_string(),
                });
            }
        }

        patterns
    }

    /// Detect potential bottlenecks
    fn detect_bottlenecks(graph: &FxGraph) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        for (idx, _node) in graph.nodes() {
            let incoming = graph
                .graph
                .neighbors_directed(idx, petgraph::Incoming)
                .count();
            let _outgoing = graph
                .graph
                .neighbors_directed(idx, petgraph::Outgoing)
                .count();

            // High incoming connections might indicate a bottleneck
            if incoming > 5 {
                patterns.push(DetectedPattern {
                    pattern_type: "potential_bottleneck".to_string(),
                    description: format!("Node with {} incoming connections", incoming),
                    nodes: vec![idx],
                    confidence: 0.7,
                    optimization_potential: "Consider parallelization or input batching"
                        .to_string(),
                });
            }
        }

        patterns
    }
}

/// Calculate comprehensive graph metrics
pub fn calculate_graph_metrics(graph: &FxGraph) -> GraphMetrics {
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    let input_count = graph.inputs().len();
    let output_count = graph.outputs().len();

    // Calculate max depth using DFS
    let max_depth = calculate_max_depth(graph);

    // Calculate average fanout
    let total_fanout: usize = graph
        .nodes()
        .map(|(idx, _)| graph.graph.neighbors(idx).count())
        .sum();
    let average_fanout = if node_count > 0 {
        total_fanout as f64 / node_count as f64
    } else {
        0.0
    };

    // Calculate connectivity ratio
    let max_possible_edges = if node_count > 1 {
        node_count * (node_count - 1)
    } else {
        1
    };
    let connectivity_ratio = edge_count as f64 / max_possible_edges as f64;

    // Calculate complexity score (heuristic)
    let complexity_score =
        (node_count as f64).ln() * (1.0 + connectivity_ratio) * (1.0 + average_fanout);

    // Calculate operation distribution
    let mut operation_distribution = HashMap::new();
    for (_, node) in graph.nodes() {
        let op_type = match node {
            Node::Input(_) => "input",
            Node::Call(op, _) => op,
            Node::Output => "output",
            Node::Conditional { .. } => "conditional",
            Node::Loop { .. } => "loop",
            Node::Merge { .. } => "merge",
            Node::GetAttr { .. } => "getattr",
        };
        *operation_distribution
            .entry(op_type.to_string())
            .or_insert(0) += 1;
    }

    // Calculate critical path length (simplified)
    let critical_path_length = max_depth;

    GraphMetrics {
        node_count,
        edge_count,
        input_count,
        output_count,
        max_depth,
        average_fanout,
        connectivity_ratio,
        complexity_score,
        operation_distribution,
        critical_path_length,
    }
}

/// Calculate maximum depth of the graph
fn calculate_max_depth(graph: &FxGraph) -> usize {
    let mut max_depth = 0;
    let mut visited = HashSet::new();

    for &input_idx in graph.inputs() {
        let depth = calculate_depth_from_node(graph, input_idx, &mut visited, 0);
        max_depth = max_depth.max(depth);
    }

    max_depth
}

/// Calculate depth from a specific node using DFS
fn calculate_depth_from_node(
    graph: &FxGraph,
    node: NodeIndex,
    visited: &mut HashSet<NodeIndex>,
    current_depth: usize,
) -> usize {
    if visited.contains(&node) {
        return current_depth;
    }

    visited.insert(node);
    let mut max_child_depth = current_depth;

    for neighbor in graph.graph.neighbors(node) {
        let child_depth = calculate_depth_from_node(graph, neighbor, visited, current_depth + 1);
        max_child_depth = max_child_depth.max(child_depth);
    }

    max_child_depth
}

impl Default for GraphLinter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, FxGraph, Node};

    #[test]
    fn test_graph_linter() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        let linter = GraphLinter::new();
        let report = linter.lint_graph(&graph);

        assert_eq!(report.total_issues, 0); // Should be a clean graph
        assert!(report.overall_score > 0.8); // Should have a good score
    }

    #[test]
    fn test_graph_linter_with_issues() {
        let mut graph = FxGraph::new();
        let _disconnected = graph.graph.add_node(Node::Call("relu".to_string(), vec![]));

        // No inputs or outputs - should trigger warnings

        let linter = GraphLinter::new();
        let report = linter.lint_graph(&graph);

        assert!(report.total_issues > 0);
        assert!(report.overall_score < 1.0);
    }

    #[test]
    fn test_graph_diff() {
        let mut old_graph = FxGraph::new();
        let _input1 = old_graph.graph.add_node(Node::Input("x".to_string()));
        let _relu1 = old_graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));

        let mut new_graph = FxGraph::new();
        let _input2 = new_graph.graph.add_node(Node::Input("x".to_string()));
        let _relu2 = new_graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let _sigmoid = new_graph.graph.add_node(Node::Call(
            "sigmoid".to_string(),
            vec!["relu_out".to_string()],
        ));

        let diff = GraphDiff::diff(&old_graph, &new_graph);

        assert_eq!(diff.added_nodes.len(), 1); // sigmoid node added
        assert_eq!(diff.removed_nodes.len(), 0);
    }

    #[test]
    fn test_pattern_detection() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu1 = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let relu2 = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["relu1".to_string()]));
        let relu3 = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["relu2".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        // Create linear chain
        graph.graph.add_edge(
            input,
            relu1,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu1,
            relu2,
            Edge {
                name: "relu1".to_string(),
            },
        );
        graph.graph.add_edge(
            relu2,
            relu3,
            Edge {
                name: "relu2".to_string(),
            },
        );
        graph.graph.add_edge(
            relu3,
            output,
            Edge {
                name: "relu3".to_string(),
            },
        );

        let patterns = PatternDetector::detect_patterns(&graph);

        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.pattern_type == "linear_chain"));
    }

    #[test]
    fn test_graph_metrics() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        let metrics = calculate_graph_metrics(&graph);

        assert_eq!(metrics.node_count, 3);
        assert_eq!(metrics.edge_count, 2);
        assert_eq!(metrics.input_count, 1);
        assert_eq!(metrics.output_count, 1);
        assert!(metrics.average_fanout > 0.0);
    }
}
