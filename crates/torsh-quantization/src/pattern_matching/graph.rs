//! Graph Data Structures
//!
//! This module provides the core data structures for representing computational graphs
//! used in pattern matching and optimization. It includes graph nodes, graph containers,
//! and fundamental graph operations like topological sorting.

use crate::TorshResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use torsh_core::TorshError;

// =============================================================================
// Graph Node Implementation
// =============================================================================

/// Represents a node in a computational graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for the node
    pub id: String,
    /// Operation type (e.g., "conv2d", "relu", "add")
    pub op_type: String,
    /// Input node IDs
    pub inputs: Vec<String>,
    /// Output node IDs
    pub outputs: Vec<String>,
    /// Node attributes (optional)
    pub attributes: HashMap<String, String>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: String, op_type: String) -> Self {
        Self {
            id,
            op_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add an input to this node
    pub fn add_input(&mut self, input_id: String) {
        self.inputs.push(input_id);
    }

    /// Add an output to this node
    pub fn add_output(&mut self, output_id: String) {
        self.outputs.push(output_id);
    }

    /// Set an attribute
    pub fn set_attribute(&mut self, key: String, value: String) {
        self.attributes.insert(key, value);
    }

    /// Get an attribute
    pub fn get_attribute(&self, key: &str) -> Option<&String> {
        self.attributes.get(key)
    }

    /// Check if this node has a specific attribute value
    pub fn has_attribute(&self, key: &str, value: &str) -> bool {
        self.attributes.get(key).map_or(false, |v| v == value)
    }

    /// Remove an attribute
    pub fn remove_attribute(&mut self, key: &str) -> Option<String> {
        self.attributes.remove(key)
    }

    /// Get all attribute keys
    pub fn attribute_keys(&self) -> Vec<&String> {
        self.attributes.keys().collect()
    }

    /// Check if node has any inputs
    pub fn has_inputs(&self) -> bool {
        !self.inputs.is_empty()
    }

    /// Check if node has any outputs
    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }

    /// Get the number of inputs
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of outputs
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Check if this node connects to another node
    pub fn connects_to(&self, node_id: &str) -> bool {
        self.outputs.contains(&node_id.to_string())
    }

    /// Check if this node receives input from another node
    pub fn receives_from(&self, node_id: &str) -> bool {
        self.inputs.contains(&node_id.to_string())
    }
}

// =============================================================================
// Computation Graph Implementation
// =============================================================================

/// Represents a computational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// All nodes in the graph
    pub nodes: HashMap<String, GraphNode>,
    /// Topological ordering of nodes
    pub execution_order: Vec<String>,
}

impl ComputationGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id.clone(), node);
        self.update_execution_order();
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, id: &str) -> Option<GraphNode> {
        let removed = self.nodes.remove(id);

        // Remove references to this node from other nodes
        if removed.is_some() {
            for node in self.nodes.values_mut() {
                node.inputs.retain(|input_id| input_id != id);
                node.outputs.retain(|output_id| output_id != id);
            }
            self.update_execution_order();
        }

        removed
    }

    /// Connect two nodes (output of src -> input of dst)
    pub fn connect_nodes(&mut self, src_id: &str, dst_id: &str) -> TorshResult<()> {
        if !self.nodes.contains_key(src_id) || !self.nodes.contains_key(dst_id) {
            return Err(TorshError::InvalidArgument(
                "Source or destination node does not exist".to_string(),
            ));
        }

        if let Some(src_node) = self.nodes.get_mut(src_id) {
            if !src_node.outputs.contains(&dst_id.to_string()) {
                src_node.add_output(dst_id.to_string());
            }
        }

        if let Some(dst_node) = self.nodes.get_mut(dst_id) {
            if !dst_node.inputs.contains(&src_id.to_string()) {
                dst_node.add_input(src_id.to_string());
            }
        }

        self.update_execution_order();
        Ok(())
    }

    /// Disconnect two nodes
    pub fn disconnect_nodes(&mut self, src_id: &str, dst_id: &str) -> TorshResult<()> {
        if !self.nodes.contains_key(src_id) || !self.nodes.contains_key(dst_id) {
            return Err(TorshError::InvalidArgument(
                "Source or destination node does not exist".to_string(),
            ));
        }

        if let Some(src_node) = self.nodes.get_mut(src_id) {
            src_node.outputs.retain(|output_id| output_id != dst_id);
        }

        if let Some(dst_node) = self.nodes.get_mut(dst_id) {
            dst_node.inputs.retain(|input_id| input_id != src_id);
        }

        self.update_execution_order();
        Ok(())
    }

    /// Update the execution order using topological sort
    fn update_execution_order(&mut self) {
        self.execution_order.clear();

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut queue = Vec::new();

        // Calculate in-degrees
        for node in self.nodes.values() {
            in_degree.insert(node.id.clone(), node.inputs.len());
            if node.inputs.is_empty() {
                queue.push(node.id.clone());
            }
        }

        // Topological sort
        while let Some(node_id) = queue.pop() {
            self.execution_order.push(node_id.clone());

            if let Some(node) = self.nodes.get(&node_id) {
                for output_id in &node.outputs {
                    if let Some(degree) = in_degree.get_mut(output_id) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(output_id.clone());
                        }
                    }
                }
            }
        }
    }

    /// Get nodes in execution order
    pub fn get_execution_order(&self) -> &[String] {
        &self.execution_order
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get all nodes of a specific operation type
    pub fn get_nodes_by_op_type(&self, op_type: &str) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|node| node.op_type == op_type)
            .collect()
    }

    /// Check if the graph contains cycles
    pub fn has_cycles(&self) -> bool {
        self.execution_order.len() != self.nodes.len()
    }

    /// Get root nodes (nodes with no inputs)
    pub fn get_root_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|node| node.inputs.is_empty())
            .collect()
    }

    /// Get leaf nodes (nodes with no outputs)
    pub fn get_leaf_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|node| node.outputs.is_empty())
            .collect()
    }

    /// Get the depth of the graph (maximum path length)
    pub fn get_graph_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let mut depths: HashMap<String, usize> = HashMap::new();

        // Initialize root nodes with depth 0
        for node in self.get_root_nodes() {
            depths.insert(node.id.clone(), 0);
        }

        // Calculate depths in execution order
        for node_id in &self.execution_order {
            if let Some(node) = self.nodes.get(node_id) {
                let max_input_depth = node
                    .inputs
                    .iter()
                    .filter_map(|input_id| depths.get(input_id))
                    .max()
                    .unwrap_or(&0);
                depths.insert(node_id.clone(), max_input_depth + 1);
            }
        }

        depths.values().max().copied().unwrap_or(0)
    }

    /// Get nodes at a specific depth level
    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<&GraphNode> {
        let mut result = Vec::new();
        let mut current_depths: HashMap<String, usize> = HashMap::new();

        // Initialize root nodes with depth 0
        for node in self.get_root_nodes() {
            current_depths.insert(node.id.clone(), 0);
        }

        // Calculate depths in execution order
        for node_id in &self.execution_order {
            if let Some(node) = self.nodes.get(node_id) {
                let max_input_depth = node
                    .inputs
                    .iter()
                    .filter_map(|input_id| current_depths.get(input_id))
                    .max()
                    .unwrap_or(&0);
                let node_depth = max_input_depth + 1;
                current_depths.insert(node_id.clone(), node_depth);

                if node_depth == depth {
                    result.push(node);
                }
            }
        }

        result
    }

    /// Check if the graph is valid (no cycles, all references valid)
    pub fn is_valid(&self) -> bool {
        // Check for cycles
        if self.has_cycles() {
            return false;
        }

        // Check that all input/output references are valid
        for node in self.nodes.values() {
            for input_id in &node.inputs {
                if !self.nodes.contains_key(input_id) {
                    return false;
                }
            }
            for output_id in &node.outputs {
                if !self.nodes.contains_key(output_id) {
                    return false;
                }
            }
        }

        true
    }

    /// Get graph statistics
    pub fn get_statistics(&self) -> GraphStatistics {
        let total_nodes = self.nodes.len();
        let total_edges = self.nodes.values().map(|node| node.outputs.len()).sum();

        let op_type_counts = self.get_operation_type_counts();
        let depth = self.get_graph_depth();
        let is_valid = self.is_valid();

        GraphStatistics {
            total_nodes,
            total_edges,
            depth,
            op_type_counts,
            is_valid,
        }
    }

    /// Get counts of different operation types
    pub fn get_operation_type_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for node in self.nodes.values() {
            *counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Create a subgraph containing only the specified nodes
    pub fn create_subgraph(&self, node_ids: &[String]) -> TorshResult<ComputationGraph> {
        let mut subgraph = ComputationGraph::new();
        let node_set: HashSet<String> = node_ids.iter().cloned().collect();

        // Add nodes
        for node_id in node_ids {
            if let Some(node) = self.nodes.get(node_id) {
                let mut sub_node = node.clone();
                // Filter inputs and outputs to only include nodes in the subgraph
                sub_node.inputs.retain(|id| node_set.contains(id));
                sub_node.outputs.retain(|id| node_set.contains(id));
                subgraph.add_node(sub_node);
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Node '{}' does not exist in the graph",
                    node_id
                )));
            }
        }

        Ok(subgraph)
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Graph Statistics
// =============================================================================

/// Statistics about a computational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Total number of edges
    pub total_edges: usize,
    /// Maximum depth of the graph
    pub depth: usize,
    /// Count of each operation type
    pub op_type_counts: HashMap<String, usize>,
    /// Whether the graph is valid
    pub is_valid: bool,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a simple linear graph for testing
pub fn create_linear_graph(op_types: &[&str]) -> ComputationGraph {
    let mut graph = ComputationGraph::new();

    for (i, &op_type) in op_types.iter().enumerate() {
        let node_id = format!("node_{}", i);
        let node = GraphNode::new(node_id.clone(), op_type.to_string());
        graph.add_node(node);

        if i > 0 {
            let prev_id = format!("node_{}", i - 1);
            graph.connect_nodes(&prev_id, &node_id).unwrap();
        }
    }

    graph
}

/// Create a simple branching graph for testing
pub fn create_branching_graph() -> ComputationGraph {
    let mut graph = ComputationGraph::new();

    // Root node
    let root = GraphNode::new("root".to_string(), "input".to_string());
    graph.add_node(root);

    // Branch nodes
    let branch1 = GraphNode::new("branch1".to_string(), "conv2d".to_string());
    let branch2 = GraphNode::new("branch2".to_string(), "conv2d".to_string());
    graph.add_node(branch1);
    graph.add_node(branch2);

    // Merge node
    let merge = GraphNode::new("merge".to_string(), "add".to_string());
    graph.add_node(merge);

    // Connect nodes
    graph.connect_nodes("root", "branch1").unwrap();
    graph.connect_nodes("root", "branch2").unwrap();
    graph.connect_nodes("branch1", "merge").unwrap();
    graph.connect_nodes("branch2", "merge").unwrap();

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::new("test_node".to_string(), "relu".to_string());
        assert_eq!(node.id, "test_node");
        assert_eq!(node.op_type, "relu");
        assert!(node.inputs.is_empty());
        assert!(node.outputs.is_empty());
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_graph_node_attributes() {
        let mut node = GraphNode::new("test".to_string(), "conv2d".to_string());
        node.set_attribute("kernel_size".to_string(), "3x3".to_string());

        assert!(node.has_attribute("kernel_size", "3x3"));
        assert_eq!(node.get_attribute("kernel_size"), Some(&"3x3".to_string()));
        assert_eq!(node.attribute_keys().len(), 1);

        let removed = node.remove_attribute("kernel_size");
        assert_eq!(removed, Some("3x3".to_string()));
        assert!(!node.has_attribute("kernel_size", "3x3"));
    }

    #[test]
    fn test_computation_graph_creation() {
        let graph = ComputationGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.execution_order.is_empty());
    }

    #[test]
    fn test_graph_node_operations() {
        let mut graph = ComputationGraph::new();

        let node1 = GraphNode::new("node1".to_string(), "input".to_string());
        let node2 = GraphNode::new("node2".to_string(), "relu".to_string());

        graph.add_node(node1);
        graph.add_node(node2);

        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.get_node("node1").is_some());
        assert!(graph.get_node("nonexistent").is_none());
    }

    #[test]
    fn test_graph_connections() {
        let mut graph = ComputationGraph::new();

        let node1 = GraphNode::new("node1".to_string(), "input".to_string());
        let node2 = GraphNode::new("node2".to_string(), "relu".to_string());

        graph.add_node(node1);
        graph.add_node(node2);

        assert!(graph.connect_nodes("node1", "node2").is_ok());

        let node1 = graph.get_node("node1").unwrap();
        let node2 = graph.get_node("node2").unwrap();

        assert!(node1.connects_to("node2"));
        assert!(node2.receives_from("node1"));
    }

    #[test]
    fn test_topological_ordering() {
        let mut graph = ComputationGraph::new();

        let node1 = GraphNode::new("1".to_string(), "input".to_string());
        let node2 = GraphNode::new("2".to_string(), "relu".to_string());
        let node3 = GraphNode::new("3".to_string(), "output".to_string());

        graph.add_node(node1);
        graph.add_node(node2);
        graph.add_node(node3);

        graph.connect_nodes("1", "2").unwrap();
        graph.connect_nodes("2", "3").unwrap();

        let order = graph.get_execution_order();
        assert_eq!(order.len(), 3);

        let pos1 = order.iter().position(|x| x == "1").unwrap();
        let pos2 = order.iter().position(|x| x == "2").unwrap();
        let pos3 = order.iter().position(|x| x == "3").unwrap();

        assert!(pos1 < pos2 && pos2 < pos3);
    }

    #[test]
    fn test_graph_statistics() {
        let graph = create_linear_graph(&["input", "conv2d", "relu", "output"]);
        let stats = graph.get_statistics();

        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.total_edges, 3);
        assert_eq!(stats.depth, 4);
        assert!(stats.is_valid);
    }

    #[test]
    fn test_subgraph_creation() {
        let graph = create_linear_graph(&["input", "conv2d", "relu", "output"]);
        let subgraph = graph
            .create_subgraph(&["node_1".to_string(), "node_2".to_string()])
            .unwrap();

        assert_eq!(subgraph.nodes.len(), 2);
        assert!(subgraph.get_node("node_1").is_some());
        assert!(subgraph.get_node("node_2").is_some());
    }

    #[test]
    fn test_utility_graphs() {
        let linear = create_linear_graph(&["input", "relu", "output"]);
        assert_eq!(linear.nodes.len(), 3);
        assert_eq!(linear.get_root_nodes().len(), 1);
        assert_eq!(linear.get_leaf_nodes().len(), 1);

        let branching = create_branching_graph();
        assert_eq!(branching.nodes.len(), 4);
        assert_eq!(branching.get_root_nodes().len(), 1);
        assert_eq!(branching.get_leaf_nodes().len(), 1);
    }
}
