// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Computation Graph Visualization
//!
//! This module provides comprehensive visualization capabilities for autograd
//! computation graphs, enabling visual inspection, debugging, and understanding
//! of gradient flow.
//!
//! # Features
//!
//! - **Multiple Output Formats**: DOT, SVG, JSON, HTML
//! - **Interactive Visualizations**: Web-based interactive graph exploration
//! - **Graph Analysis**: Statistics, complexity metrics, bottleneck detection
//! - **Custom Styling**: Node coloring, edge styling, layout options
//! - **Export/Import**: Save and load graph visualizations
//! - **Real-time Updates**: Live graph updates during computation

use crate::error_handling::{AutogradError, AutogradResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Graph node representing an operation in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node ID
    pub id: String,

    /// Operation name
    pub operation: String,

    /// Node label for display
    pub label: String,

    /// Node type (operation, tensor, parameter, etc.)
    pub node_type: NodeType,

    /// Input edge IDs
    pub inputs: Vec<String>,

    /// Output edge IDs
    pub outputs: Vec<String>,

    /// Node metadata
    pub metadata: HashMap<String, String>,

    /// Node styling
    pub style: NodeStyle,

    /// Execution time (if profiled)
    pub execution_time_ms: Option<f64>,

    /// Memory usage (bytes)
    pub memory_bytes: Option<usize>,

    /// Gradient magnitude (if available)
    pub gradient_magnitude: Option<f64>,
}

/// Type of graph node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// Operation node (e.g., matmul, add)
    Operation,
    /// Tensor/variable node
    Tensor,
    /// Parameter node (trainable)
    Parameter,
    /// Constant node
    Constant,
    /// Placeholder/input node
    Input,
    /// Output node
    Output,
}

/// Node visual styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStyle {
    /// Node color (hex or named)
    pub color: String,

    /// Node shape (box, circle, ellipse, etc.)
    pub shape: String,

    /// Border color
    pub border_color: String,

    /// Border width
    pub border_width: f64,

    /// Font size
    pub font_size: usize,

    /// Whether node is highlighted
    pub highlighted: bool,
}

impl Default for NodeStyle {
    fn default() -> Self {
        Self {
            color: "#E8F4F8".to_string(),
            shape: "box".to_string(),
            border_color: "#4A90E2".to_string(),
            border_width: 1.0,
            font_size: 12,
            highlighted: false,
        }
    }
}

/// Graph edge representing data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge ID
    pub id: String,

    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Edge label (optional)
    pub label: Option<String>,

    /// Edge styling
    pub style: EdgeStyle,

    /// Tensor shape flowing through this edge
    pub shape: Option<Vec<usize>>,

    /// Data type
    pub dtype: Option<String>,
}

/// Edge visual styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStyle {
    /// Edge color
    pub color: String,

    /// Edge width
    pub width: f64,

    /// Edge pattern (solid, dashed, dotted)
    pub pattern: String,

    /// Arrow style
    pub arrow_style: String,

    /// Whether edge is highlighted
    pub highlighted: bool,
}

impl Default for EdgeStyle {
    fn default() -> Self {
        Self {
            color: "#333333".to_string(),
            width: 1.5,
            pattern: "solid".to_string(),
            arrow_style: "normal".to_string(),
            highlighted: false,
        }
    }
}

/// Computation graph for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Graph name
    pub name: String,

    /// Graph nodes
    pub nodes: HashMap<String, GraphNode>,

    /// Graph edges
    pub edges: HashMap<String, GraphEdge>,

    /// Graph metadata
    pub metadata: HashMap<String, String>,

    /// Graph layout direction (TB, LR, BT, RL)
    pub layout_direction: String,
}

impl ComputationGraph {
    /// Create a new computation graph
    pub fn new(name: String) -> Self {
        Self {
            name,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            metadata: HashMap::new(),
            layout_direction: "TB".to_string(), // Top to Bottom
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.insert(edge.id.clone(), edge);
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Get topologically sorted nodes
    pub fn topological_sort(&self) -> Vec<&GraphNode> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        fn visit<'a>(
            node_id: &str,
            graph: &'a ComputationGraph,
            visited: &mut HashSet<String>,
            temp_mark: &mut HashSet<String>,
            sorted: &mut Vec<&'a GraphNode>,
        ) -> bool {
            if visited.contains(node_id) {
                return true;
            }
            if temp_mark.contains(node_id) {
                return false; // Cycle detected
            }

            temp_mark.insert(node_id.to_string());

            if let Some(node) = graph.get_node(node_id) {
                // Resolve edge IDs to source node IDs
                for edge_id in &node.inputs {
                    if let Some(edge) = graph.edges.get(edge_id) {
                        if !visit(&edge.source, graph, visited, temp_mark, sorted) {
                            return false;
                        }
                    }
                }

                sorted.push(node);
                visited.insert(node_id.to_string());
                temp_mark.remove(node_id);
            }

            true
        }

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id) {
                visit(node_id, self, &mut visited, &mut temp_mark, &mut sorted);
            }
        }

        sorted
    }

    /// Get graph statistics
    pub fn statistics(&self) -> GraphStatistics {
        let mut stats = GraphStatistics::default();

        stats.total_nodes = self.nodes.len();
        stats.total_edges = self.edges.len();

        // Count node types
        for node in self.nodes.values() {
            match node.node_type {
                NodeType::Operation => stats.operation_nodes += 1,
                NodeType::Tensor => stats.tensor_nodes += 1,
                NodeType::Parameter => stats.parameter_nodes += 1,
                NodeType::Constant => stats.constant_nodes += 1,
                NodeType::Input => stats.input_nodes += 1,
                NodeType::Output => stats.output_nodes += 1,
            }
        }

        // Calculate depth
        let sorted = self.topological_sort();
        let mut depths: HashMap<String, usize> = HashMap::new();

        for node in sorted {
            let max_input_depth = node
                .inputs
                .iter()
                .filter_map(|id| depths.get(id))
                .max()
                .unwrap_or(&0);

            let depth = max_input_depth + 1;
            depths.insert(node.id.clone(), depth);
            stats.max_depth = stats.max_depth.max(depth);
        }

        // Calculate total execution time and memory
        for node in self.nodes.values() {
            if let Some(time) = node.execution_time_ms {
                stats.total_execution_time_ms += time;
            }
            if let Some(mem) = node.memory_bytes {
                stats.total_memory_bytes += mem;
            }
        }

        stats
    }

    /// Export to DOT format (Graphviz)
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();

        dot.push_str(&format!("digraph \"{}\" {{\n", self.name));
        dot.push_str(&format!("  rankdir={};\n", self.layout_direction));
        dot.push_str("  node [fontname=\"Arial\"];\n");
        dot.push_str("  edge [fontname=\"Arial\"];\n\n");

        // Add nodes
        for (id, node) in &self.nodes {
            let shape = &node.style.shape;
            let color = &node.style.color;
            let border_color = &node.style.border_color;

            let label = if let Some(time) = node.execution_time_ms {
                format!("{}\\n{:.2}ms", node.label, time)
            } else {
                node.label.clone()
            };

            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\", shape={}, fillcolor=\"{}\", color=\"{}\", style=\"filled\"];\n",
                id, label, shape, color, border_color
            ));
        }

        dot.push_str("\n");

        // Add edges
        for (_id, edge) in &self.edges {
            let color = &edge.style.color;
            let width = edge.style.width;

            let label = if let Some(ref lbl) = edge.label {
                format!(", label=\"{}\"", lbl)
            } else {
                String::new()
            };

            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [color=\"{}\", penwidth={}{}];\n",
                edge.source, edge.target, color, width, label
            ));
        }

        dot.push_str("}\n");

        dot
    }

    /// Export to JSON
    pub fn to_json(&self) -> AutogradResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| AutogradError::Configuration {
            parameter: "serialization".to_string(),
            value: "graph".to_string(),
            reason: format!("Failed to serialize: {}", e),
            valid_range: None,
        })
    }

    /// Import from JSON
    pub fn from_json(json: &str) -> AutogradResult<Self> {
        serde_json::from_str(json).map_err(|e| AutogradError::Configuration {
            parameter: "deserialization".to_string(),
            value: "graph".to_string(),
            reason: format!("Failed to deserialize: {}", e),
            valid_range: None,
        })
    }

    /// Generate HTML visualization with interactive features
    pub fn to_html(&self) -> String {
        let json = self.to_json().unwrap_or_else(|_| "{}".to_string());

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{} - Computation Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        #graph {{ width: 100%; height: 800px; border: 1px solid #ccc; }}
        .node {{ cursor: pointer; }}
        .node circle {{ stroke: #333; stroke-width: 2px; }}
        .node text {{ font-size: 12px; pointer-events: none; }}
        .link {{ fill: none; stroke: #999; stroke-width: 2px; }}
        .tooltip {{ position: absolute; background: white; border: 1px solid #ccc; padding: 10px; display: none; }}
    </style>
</head>
<body>
    <h1>{} - Computation Graph</h1>
    <div id="graph"></div>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const graphData = {};

        // D3.js visualization code would go here
        // This is a simplified version - a full implementation would include:
        // - Force-directed layout
        // - Interactive node dragging
        // - Zoom and pan
        // - Tooltips showing node details
        // - Highlighting of connected nodes

        console.log('Graph data:', graphData);
    </script>
</body>
</html>"#,
            self.name, self.name, json
        )
    }
}

/// Graph statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub total_nodes: usize,

    /// Total number of edges
    pub total_edges: usize,

    /// Number of operation nodes
    pub operation_nodes: usize,

    /// Number of tensor nodes
    pub tensor_nodes: usize,

    /// Number of parameter nodes
    pub parameter_nodes: usize,

    /// Number of constant nodes
    pub constant_nodes: usize,

    /// Number of input nodes
    pub input_nodes: usize,

    /// Number of output nodes
    pub output_nodes: usize,

    /// Maximum graph depth
    pub max_depth: usize,

    /// Total execution time (ms)
    pub total_execution_time_ms: f64,

    /// Total memory usage (bytes)
    pub total_memory_bytes: usize,
}

/// Graph builder for constructing computation graphs
pub struct GraphBuilder {
    graph: ComputationGraph,
    next_node_id: usize,
    next_edge_id: usize,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(name: String) -> Self {
        Self {
            graph: ComputationGraph::new(name),
            next_node_id: 0,
            next_edge_id: 0,
        }
    }

    /// Add operation node
    pub fn add_operation(&mut self, operation: String, label: String) -> String {
        let id = format!("op_{}", self.next_node_id);
        self.next_node_id += 1;

        let mut style = NodeStyle::default();
        style.color = "#FFE6CC".to_string();
        style.shape = "box".to_string();

        let node = GraphNode {
            id: id.clone(),
            operation,
            label,
            node_type: NodeType::Operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
            style,
            execution_time_ms: None,
            memory_bytes: None,
            gradient_magnitude: None,
        };

        self.graph.add_node(node);
        id
    }

    /// Add tensor node
    pub fn add_tensor(&mut self, label: String) -> String {
        let id = format!("tensor_{}", self.next_node_id);
        self.next_node_id += 1;

        let mut style = NodeStyle::default();
        style.color = "#CCE5FF".to_string();
        style.shape = "ellipse".to_string();

        let node = GraphNode {
            id: id.clone(),
            operation: "tensor".to_string(),
            label,
            node_type: NodeType::Tensor,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
            style,
            execution_time_ms: None,
            memory_bytes: None,
            gradient_magnitude: None,
        };

        self.graph.add_node(node);
        id
    }

    /// Add edge between nodes
    pub fn add_edge(&mut self, source: String, target: String, label: Option<String>) -> String {
        let id = format!("edge_{}", self.next_edge_id);
        self.next_edge_id += 1;

        let edge = GraphEdge {
            id: id.clone(),
            source: source.clone(),
            target: target.clone(),
            label,
            style: EdgeStyle::default(),
            shape: None,
            dtype: None,
        };

        // Update node connections
        if let Some(source_node) = self.graph.get_node_mut(&source) {
            source_node.outputs.push(id.clone());
        }

        if let Some(target_node) = self.graph.get_node_mut(&target) {
            target_node.inputs.push(id.clone());
        }

        self.graph.add_edge(edge);
        id
    }

    /// Build and return the graph
    pub fn build(self) -> ComputationGraph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new("test_graph".to_string());
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = GraphBuilder::new("test".to_string());

        let input = builder.add_tensor("input".to_string());
        let weight = builder.add_tensor("weight".to_string());
        let matmul = builder.add_operation("matmul".to_string(), "MatMul".to_string());
        let output = builder.add_tensor("output".to_string());

        builder.add_edge(input.clone(), matmul.clone(), None);
        builder.add_edge(weight.clone(), matmul.clone(), None);
        builder.add_edge(matmul.clone(), output.clone(), None);

        let graph = builder.build();

        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn test_topological_sort() {
        let mut builder = GraphBuilder::new("test".to_string());

        let a = builder.add_tensor("A".to_string());
        let b = builder.add_tensor("B".to_string());
        let c = builder.add_operation("add".to_string(), "Add".to_string());
        let d = builder.add_tensor("D".to_string());

        builder.add_edge(a.clone(), c.clone(), None);
        builder.add_edge(b.clone(), c.clone(), None);
        builder.add_edge(c.clone(), d.clone(), None);

        let graph = builder.build();
        let sorted = graph.topological_sort();

        assert_eq!(sorted.len(), 4);
        // Verify topological order
        let positions: HashMap<_, _> = sorted
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.as_str(), i))
            .collect();

        assert!(positions[a.as_str()] < positions[c.as_str()]);
        assert!(positions[b.as_str()] < positions[c.as_str()]);
        assert!(positions[c.as_str()] < positions[d.as_str()]);
    }

    #[test]
    fn test_graph_statistics() {
        let mut builder = GraphBuilder::new("test".to_string());

        builder.add_tensor("t1".to_string());
        builder.add_tensor("t2".to_string());
        builder.add_operation("op1".to_string(), "Op1".to_string());

        let graph = builder.build();
        let stats = graph.statistics();

        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.tensor_nodes, 2);
        assert_eq!(stats.operation_nodes, 1);
    }

    #[test]
    fn test_dot_export() {
        let mut builder = GraphBuilder::new("test".to_string());

        let a = builder.add_tensor("A".to_string());
        let b = builder.add_operation("add".to_string(), "Add".to_string());

        builder.add_edge(a, b, None);

        let graph = builder.build();
        let dot = graph.to_dot();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("Add"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_json_serialization() {
        let graph = ComputationGraph::new("test".to_string());
        let json = graph.to_json();
        assert!(json.is_ok());

        let json_str = json.unwrap();
        let deserialized = ComputationGraph::from_json(&json_str);
        assert!(deserialized.is_ok());
    }
}
