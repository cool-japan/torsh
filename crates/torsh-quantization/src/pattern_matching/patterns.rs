//! Pattern Definition System
//!
//! This module provides the pattern definition system for computational graph matching.
//! It includes pattern specifications, pattern nodes, and common pre-defined patterns
//! used in quantization and optimization.

use crate::QuantConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Pattern Node Definition
// =============================================================================

/// Represents a node in a graph pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    /// Operation type to match
    pub op_type: String,
    /// Optional attributes to match
    pub attributes: HashMap<String, String>,
    /// Whether this node is optional in the pattern
    pub optional: bool,
    /// Custom constraints for pattern matching
    pub constraints: Vec<PatternConstraint>,
}

impl PatternNode {
    /// Create a new pattern node
    pub fn new(op_type: String) -> Self {
        Self {
            op_type,
            attributes: HashMap::new(),
            optional: false,
            constraints: Vec::new(),
        }
    }

    /// Set an attribute requirement
    pub fn with_attribute(mut self, key: String, value: String) -> Self {
        self.attributes.insert(key, value);
        self
    }

    /// Mark this node as optional
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Add a custom constraint
    pub fn with_constraint(mut self, constraint: PatternConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Check if this pattern node matches a graph node
    pub fn matches(&self, node_op_type: &str, node_attributes: &HashMap<String, String>) -> bool {
        // Check operation type
        if self.op_type != "*" && self.op_type != node_op_type {
            return false;
        }

        // Check required attributes
        for (key, expected_value) in &self.attributes {
            match node_attributes.get(key) {
                Some(actual_value) if actual_value == expected_value => continue,
                _ => return false,
            }
        }

        // Check custom constraints
        for constraint in &self.constraints {
            if !constraint.evaluate(node_op_type, node_attributes) {
                return false;
            }
        }

        true
    }

    /// Get a description of this pattern node
    pub fn description(&self) -> String {
        let mut desc = self.op_type.clone();

        if !self.attributes.is_empty() {
            let attrs: Vec<String> = self
                .attributes
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            desc.push_str(&format!("[{}]", attrs.join(", ")));
        }

        if self.optional {
            desc.push_str("?");
        }

        desc
    }
}

// =============================================================================
// Pattern Constraints
// =============================================================================

/// Custom constraints for pattern nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternConstraint {
    /// Attribute must have a specific value
    AttributeEquals { key: String, value: String },
    /// Attribute must exist
    AttributeExists { key: String },
    /// Attribute must match a regex pattern
    AttributeMatches { key: String, pattern: String },
    /// Custom predicate function (represented as string for serialization)
    Custom { name: String, description: String },
}

impl PatternConstraint {
    /// Evaluate the constraint against a node
    pub fn evaluate(&self, _op_type: &str, attributes: &HashMap<String, String>) -> bool {
        match self {
            PatternConstraint::AttributeEquals { key, value } => {
                attributes.get(key).map_or(false, |v| v == value)
            }
            PatternConstraint::AttributeExists { key } => attributes.contains_key(key),
            PatternConstraint::AttributeMatches { key, pattern } => {
                if let Some(attr_value) = attributes.get(key) {
                    // Simple pattern matching - could be enhanced with regex crate
                    attr_value.contains(pattern)
                } else {
                    false
                }
            }
            PatternConstraint::Custom { .. } => {
                // Custom constraints would need to be registered separately
                // For now, always return true
                true
            }
        }
    }

    /// Get a description of this constraint
    pub fn description(&self) -> String {
        match self {
            PatternConstraint::AttributeEquals { key, value } => {
                format!("{} == {}", key, value)
            }
            PatternConstraint::AttributeExists { key } => {
                format!("{} exists", key)
            }
            PatternConstraint::AttributeMatches { key, pattern } => {
                format!("{} matches {}", key, pattern)
            }
            PatternConstraint::Custom { name, description } => {
                format!("{}: {}", name, description)
            }
        }
    }
}

// =============================================================================
// Graph Pattern Definition
// =============================================================================

/// Represents a pattern to match in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Pattern name
    pub name: String,
    /// Description of what this pattern represents
    pub description: String,
    /// Nodes in the pattern
    pub nodes: Vec<PatternNode>,
    /// Edges between pattern nodes
    pub edges: Vec<(usize, usize)>, // (from_node_index, to_node_index)
    /// Quantization config to apply if pattern matches
    pub qconfig: Option<QuantConfig>,
    /// Priority for pattern matching (higher = more important)
    pub priority: i32,
    /// Whether this pattern should be applied repeatedly until no more matches
    pub iterative: bool,
}

impl GraphPattern {
    /// Create a new graph pattern
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            nodes: Vec::new(),
            edges: Vec::new(),
            qconfig: None,
            priority: 0,
            iterative: false,
        }
    }

    /// Add a node to the pattern
    pub fn add_node(mut self, node: PatternNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add an edge between two nodes
    pub fn add_edge(mut self, from_index: usize, to_index: usize) -> Self {
        self.edges.push((from_index, to_index));
        self
    }

    /// Set the quantization config
    pub fn with_qconfig(mut self, qconfig: QuantConfig) -> Self {
        self.qconfig = Some(qconfig);
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as iterative
    pub fn iterative(mut self) -> Self {
        self.iterative = true;
        self
    }

    /// Get the number of nodes in the pattern
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the pattern
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if the pattern is valid (all edges reference valid node indices)
    pub fn is_valid(&self) -> bool {
        let node_count = self.nodes.len();
        self.edges
            .iter()
            .all(|(from, to)| *from < node_count && *to < node_count)
    }

    /// Get a textual representation of the pattern
    pub fn to_string(&self) -> String {
        let mut result = format!("Pattern: {} ({})\n", self.name, self.description);

        result.push_str("Nodes:\n");
        for (i, node) in self.nodes.iter().enumerate() {
            result.push_str(&format!("  {}: {}\n", i, node.description()));
        }

        if !self.edges.is_empty() {
            result.push_str("Edges:\n");
            for (from, to) in &self.edges {
                result.push_str(&format!("  {} -> {}\n", from, to));
            }
        }

        if let Some(ref qconfig) = self.qconfig {
            result.push_str(&format!("Quantization Config: {:?}\n", qconfig));
        }

        result.push_str(&format!("Priority: {}\n", self.priority));
        result.push_str(&format!("Iterative: {}\n", self.iterative));

        result
    }
}

// =============================================================================
// Common Patterns
// =============================================================================

/// Common quantization and optimization patterns
pub struct CommonPatterns;

impl CommonPatterns {
    /// Conv2D + BatchNorm fusion pattern
    pub fn conv_batch_norm() -> GraphPattern {
        GraphPattern::new(
            "conv_bn".to_string(),
            "Conv2D followed by BatchNorm fusion".to_string(),
        )
        .add_node(PatternNode::new("conv2d".to_string()))
        .add_node(PatternNode::new("batch_norm".to_string()))
        .add_edge(0, 1)
        .with_priority(10)
    }

    /// Conv2D + BatchNorm + ReLU fusion pattern
    pub fn conv_batch_norm_relu() -> GraphPattern {
        GraphPattern::new(
            "conv_bn_relu".to_string(),
            "Conv2D + BatchNorm + ReLU fusion".to_string(),
        )
        .add_node(PatternNode::new("conv2d".to_string()))
        .add_node(PatternNode::new("batch_norm".to_string()))
        .add_node(PatternNode::new("relu".to_string()))
        .add_edge(0, 1)
        .add_edge(1, 2)
        .with_priority(15)
    }

    /// Conv2D + ReLU fusion pattern
    pub fn conv_relu() -> GraphPattern {
        GraphPattern::new("conv_relu".to_string(), "Conv2D + ReLU fusion".to_string())
            .add_node(PatternNode::new("conv2d".to_string()))
            .add_node(PatternNode::new("relu".to_string()))
            .add_edge(0, 1)
            .with_priority(8)
    }

    /// Linear + ReLU fusion pattern
    pub fn linear_relu() -> GraphPattern {
        GraphPattern::new(
            "linear_relu".to_string(),
            "Linear + ReLU fusion".to_string(),
        )
        .add_node(PatternNode::new("linear".to_string()))
        .add_node(PatternNode::new("relu".to_string()))
        .add_edge(0, 1)
        .with_priority(8)
    }

    /// Quantize + Dequantize elimination pattern
    pub fn quant_dequant() -> GraphPattern {
        GraphPattern::new(
            "quant_dequant".to_string(),
            "Quantize followed by Dequantize elimination".to_string(),
        )
        .add_node(PatternNode::new("quantize".to_string()))
        .add_node(PatternNode::new("dequantize".to_string()))
        .add_edge(0, 1)
        .with_priority(20)
        .iterative()
    }

    /// Add + ReLU fusion pattern
    pub fn add_relu() -> GraphPattern {
        GraphPattern::new("add_relu".to_string(), "Add + ReLU fusion".to_string())
            .add_node(PatternNode::new("add".to_string()))
            .add_node(PatternNode::new("relu".to_string()))
            .add_edge(0, 1)
            .with_priority(5)
    }

    /// Transpose + Transpose elimination pattern
    pub fn transpose_transpose() -> GraphPattern {
        GraphPattern::new(
            "transpose_transpose".to_string(),
            "Consecutive transpose operations elimination".to_string(),
        )
        .add_node(PatternNode::new("transpose".to_string()))
        .add_node(PatternNode::new("transpose".to_string()))
        .add_edge(0, 1)
        .with_priority(15)
        .iterative()
    }

    /// MatMul + Add (bias) fusion pattern
    pub fn matmul_add() -> GraphPattern {
        GraphPattern::new(
            "matmul_add".to_string(),
            "MatMul + Add (bias) fusion".to_string(),
        )
        .add_node(PatternNode::new("matmul".to_string()))
        .add_node(PatternNode::new("add".to_string()))
        .add_edge(0, 1)
        .with_priority(12)
    }

    /// Squeeze + Unsqueeze elimination pattern
    pub fn squeeze_unsqueeze() -> GraphPattern {
        GraphPattern::new(
            "squeeze_unsqueeze".to_string(),
            "Squeeze followed by Unsqueeze elimination".to_string(),
        )
        .add_node(PatternNode::new("squeeze".to_string()))
        .add_node(PatternNode::new("unsqueeze".to_string()))
        .add_edge(0, 1)
        .with_priority(10)
        .iterative()
    }

    /// Reshape + Reshape elimination pattern
    pub fn reshape_reshape() -> GraphPattern {
        GraphPattern::new(
            "reshape_reshape".to_string(),
            "Consecutive reshape operations elimination".to_string(),
        )
        .add_node(PatternNode::new("reshape".to_string()))
        .add_node(PatternNode::new("reshape".to_string()))
        .add_edge(0, 1)
        .with_priority(10)
        .iterative()
    }

    /// Get all common patterns
    pub fn all_patterns() -> Vec<GraphPattern> {
        vec![
            Self::conv_batch_norm_relu(),
            Self::conv_batch_norm(),
            Self::conv_relu(),
            Self::linear_relu(),
            Self::quant_dequant(),
            Self::add_relu(),
            Self::transpose_transpose(),
            Self::matmul_add(),
            Self::squeeze_unsqueeze(),
            Self::reshape_reshape(),
        ]
    }

    /// Get fusion patterns only
    pub fn fusion_patterns() -> Vec<GraphPattern> {
        vec![
            Self::conv_batch_norm_relu(),
            Self::conv_batch_norm(),
            Self::conv_relu(),
            Self::linear_relu(),
            Self::add_relu(),
            Self::matmul_add(),
        ]
    }

    /// Get elimination patterns only
    pub fn elimination_patterns() -> Vec<GraphPattern> {
        vec![
            Self::quant_dequant(),
            Self::transpose_transpose(),
            Self::squeeze_unsqueeze(),
            Self::reshape_reshape(),
        ]
    }
}

// =============================================================================
// Pattern Collection Management
// =============================================================================

/// A collection of patterns with management utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCollection {
    /// All patterns in the collection
    pub patterns: Vec<GraphPattern>,
    /// Name of the collection
    pub name: String,
    /// Description of the collection
    pub description: String,
}

impl PatternCollection {
    /// Create a new pattern collection
    pub fn new(name: String, description: String) -> Self {
        Self {
            patterns: Vec::new(),
            name,
            description,
        }
    }

    /// Add a pattern to the collection
    pub fn add_pattern(mut self, pattern: GraphPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Get patterns sorted by priority (highest first)
    pub fn get_by_priority(&self) -> Vec<&GraphPattern> {
        let mut sorted: Vec<&GraphPattern> = self.patterns.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }

    /// Get patterns by name
    pub fn get_by_name(&self, name: &str) -> Option<&GraphPattern> {
        self.patterns.iter().find(|p| p.name == name)
    }

    /// Get fusion patterns
    pub fn get_fusion_patterns(&self) -> Vec<&GraphPattern> {
        self.patterns
            .iter()
            .filter(|p| {
                p.name.contains("fusion")
                    || p.description.to_lowercase().contains("fusion")
                    || p.nodes.len() > 1
            })
            .collect()
    }

    /// Get elimination patterns
    pub fn get_elimination_patterns(&self) -> Vec<&GraphPattern> {
        self.patterns
            .iter()
            .filter(|p| {
                p.name.contains("elimination")
                    || p.description.to_lowercase().contains("elimination")
                    || p.iterative
            })
            .collect()
    }

    /// Create a collection with common patterns
    pub fn common() -> Self {
        let mut collection = Self::new(
            "Common Patterns".to_string(),
            "Common quantization and optimization patterns".to_string(),
        );

        for pattern in CommonPatterns::all_patterns() {
            collection = collection.add_pattern(pattern);
        }

        collection
    }

    /// Create a collection with fusion patterns only
    pub fn fusion_only() -> Self {
        let mut collection = Self::new(
            "Fusion Patterns".to_string(),
            "Operation fusion patterns for optimization".to_string(),
        );

        for pattern in CommonPatterns::fusion_patterns() {
            collection = collection.add_pattern(pattern);
        }

        collection
    }

    /// Create a collection with elimination patterns only
    pub fn elimination_only() -> Self {
        let mut collection = Self::new(
            "Elimination Patterns".to_string(),
            "Dead code and redundant operation elimination patterns".to_string(),
        );

        for pattern in CommonPatterns::elimination_patterns() {
            collection = collection.add_pattern(pattern);
        }

        collection
    }
}

impl Default for PatternCollection {
    fn default() -> Self {
        Self::common()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_node_creation() {
        let node = PatternNode::new("conv2d".to_string());
        assert_eq!(node.op_type, "conv2d");
        assert!(node.attributes.is_empty());
        assert!(!node.optional);
        assert!(node.constraints.is_empty());
    }

    #[test]
    fn test_pattern_node_with_attributes() {
        let node = PatternNode::new("conv2d".to_string())
            .with_attribute("kernel_size".to_string(), "3x3".to_string())
            .optional();

        assert!(node.optional);
        assert!(node.has_attribute("kernel_size", "3x3"));
    }

    #[test]
    fn test_pattern_node_matching() {
        let mut attributes = HashMap::new();
        attributes.insert("kernel_size".to_string(), "3x3".to_string());

        let node = PatternNode::new("conv2d".to_string())
            .with_attribute("kernel_size".to_string(), "3x3".to_string());

        assert!(node.matches("conv2d", &attributes));

        // Wrong operation type
        assert!(!node.matches("relu", &attributes));

        // Missing attribute
        let empty_attrs = HashMap::new();
        assert!(!node.matches("conv2d", &empty_attrs));

        // Wildcard pattern
        let wildcard = PatternNode::new("*".to_string());
        assert!(wildcard.matches("conv2d", &attributes));
        assert!(wildcard.matches("relu", &attributes));
    }

    #[test]
    fn test_pattern_constraints() {
        let constraint = PatternConstraint::AttributeExists {
            key: "kernel_size".to_string(),
        };

        let mut attrs = HashMap::new();
        attrs.insert("kernel_size".to_string(), "3x3".to_string());

        assert!(constraint.evaluate("conv2d", &attrs));

        let empty_attrs = HashMap::new();
        assert!(!constraint.evaluate("conv2d", &empty_attrs));
    }

    #[test]
    fn test_graph_pattern_creation() {
        let pattern = GraphPattern::new("test_pattern".to_string(), "Test pattern".to_string())
            .add_node(PatternNode::new("conv2d".to_string()))
            .add_node(PatternNode::new("relu".to_string()))
            .add_edge(0, 1);

        assert_eq!(pattern.name, "test_pattern");
        assert_eq!(pattern.node_count(), 2);
        assert_eq!(pattern.edge_count(), 1);
        assert!(pattern.is_valid());
    }

    #[test]
    fn test_common_patterns() {
        let conv_bn = CommonPatterns::conv_batch_norm();
        assert_eq!(conv_bn.name, "conv_bn");
        assert_eq!(conv_bn.node_count(), 2);
        assert_eq!(conv_bn.edge_count(), 1);

        let patterns = CommonPatterns::all_patterns();
        assert!(!patterns.is_empty());

        let fusion = CommonPatterns::fusion_patterns();
        let elimination = CommonPatterns::elimination_patterns();
        assert!(!fusion.is_empty());
        assert!(!elimination.is_empty());
    }

    #[test]
    fn test_pattern_collection() {
        let collection = PatternCollection::common();
        assert!(!collection.patterns.is_empty());

        let by_priority = collection.get_by_priority();
        assert!(!by_priority.is_empty());

        // Check that patterns are sorted by priority
        for i in 1..by_priority.len() {
            assert!(by_priority[i - 1].priority >= by_priority[i].priority);
        }

        let fusion_collection = PatternCollection::fusion_only();
        let elimination_collection = PatternCollection::elimination_only();

        assert!(!fusion_collection.patterns.is_empty());
        assert!(!elimination_collection.patterns.is_empty());
    }

    #[test]
    fn test_pattern_validation() {
        let mut pattern = GraphPattern::new(
            "invalid_pattern".to_string(),
            "Pattern with invalid edge".to_string(),
        );

        pattern = pattern.add_node(PatternNode::new("conv2d".to_string()));
        assert!(pattern.is_valid());

        // Add invalid edge (references non-existent node)
        pattern = pattern.add_edge(0, 5);
        assert!(!pattern.is_valid());
    }
}
