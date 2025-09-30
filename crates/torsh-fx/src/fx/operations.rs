//! Graph manipulation and optimization operations

use crate::fx::types::{Edge, Node};
use crate::graph_analysis::GraphDiff;
use crate::graph_analysis::GraphDifference;
use crate::performance::{GraphCompression, ParallelTraversal};
use crate::{FxGraph, TorshResult};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

impl FxGraph {
    /// Get all nodes of a specific type
    pub fn nodes_of_type(&self, node_type: fn(&Node) -> bool) -> Vec<(NodeIndex, &Node)> {
        self.nodes().filter(|(_, node)| node_type(node)).collect()
    }

    /// Get all input nodes
    pub fn input_nodes(&self) -> Vec<(NodeIndex, &Node)> {
        self.nodes_of_type(|node| matches!(node, Node::Input(_)))
    }

    /// Get all output nodes
    pub fn output_nodes(&self) -> Vec<(NodeIndex, &Node)> {
        self.nodes_of_type(|node| matches!(node, Node::Output))
    }

    /// Get all call nodes (operations)
    pub fn call_nodes(&self) -> Vec<(NodeIndex, &Node)> {
        self.nodes_of_type(|node| matches!(node, Node::Call(_, _)))
    }

    /// Get all conditional nodes
    pub fn conditional_nodes(&self) -> Vec<(NodeIndex, &Node)> {
        self.nodes_of_type(|node| matches!(node, Node::Conditional { .. }))
    }

    /// Get all loop nodes
    pub fn loop_nodes(&self) -> Vec<(NodeIndex, &Node)> {
        self.nodes_of_type(|node| matches!(node, Node::Loop { .. }))
    }

    /// Get all unique operation names in the graph
    pub fn get_operation_names(&self) -> Vec<String> {
        let mut op_names = Vec::new();
        for (_, node) in self.call_nodes() {
            if let Node::Call(op_name, _) = node {
                if !op_names.contains(op_name) {
                    op_names.push(op_name.clone());
                }
            }
        }
        op_names.sort();
        op_names
    }

    /// Check if the graph contains a specific operation
    pub fn contains_operation(&self, op_name: &str) -> bool {
        self.call_nodes().iter().any(|(_, node)| {
            if let Node::Call(name, _) = node {
                name == op_name
            } else {
                false
            }
        })
    }

    /// Get nodes by operation name
    pub fn nodes_by_operation(&self, op_name: &str) -> Vec<(NodeIndex, &Node)> {
        self.call_nodes()
            .into_iter()
            .filter(|(_, node)| {
                if let Node::Call(name, _) = node {
                    name == op_name
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<(NodeIndex, NodeIndex, &Edge)> {
        use petgraph::visit::EdgeRef;
        self.graph
            .edge_references()
            .map(|edge_ref| (edge_ref.source(), edge_ref.target(), edge_ref.weight()))
            .collect()
    }

    /// Get the fan-out degree of a node (number of outgoing edges)
    pub fn node_fanout(&self, node_idx: NodeIndex) -> usize {
        self.graph
            .edges_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
    }

    /// Get the fan-in degree of a node (number of incoming edges)
    pub fn node_fanin(&self, node_idx: NodeIndex) -> usize {
        self.graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
            .count()
    }

    /// Find nodes with the highest fan-out (potential bottlenecks)
    pub fn find_high_fanout_nodes(&self, threshold: usize) -> Vec<(NodeIndex, usize)> {
        let mut high_fanout = Vec::new();

        for (idx, _) in self.nodes() {
            let fanout = self.node_fanout(idx);
            if fanout >= threshold {
                high_fanout.push((idx, fanout));
            }
        }

        // Sort by fanout in descending order
        high_fanout.sort_by(|a, b| b.1.cmp(&a.1));
        high_fanout
    }

    /// Compress the graph by removing redundant operations
    pub fn compress(&self) -> TorshResult<FxGraph> {
        GraphCompression::deduplicate_operations(self)
    }

    /// Remove redundant nodes from the graph
    pub fn optimize_nodes(&self) -> TorshResult<FxGraph> {
        GraphCompression::remove_redundant_nodes(self)
    }

    /// Compare this graph with another graph
    pub fn diff(&self, other: &FxGraph) -> GraphDifference {
        GraphDiff::diff(self, other)
    }

    /// Create a parallel traversal instance for this graph
    pub fn parallel_traversal(self: Arc<Self>) -> ParallelTraversal {
        ParallelTraversal::new(self)
    }

    /// Create a copy of the graph with optimizations applied
    pub fn optimize(&self) -> TorshResult<FxGraph> {
        // Start with compressed graph
        let mut optimized = self.compress()?;

        // Remove orphaned nodes
        let orphaned_nodes = optimized.find_orphaned_nodes();
        for &orphaned_idx in &orphaned_nodes {
            optimized.graph.remove_node(orphaned_idx);
        }

        // Remove dead-end nodes that don't contribute to outputs
        let dead_end_nodes = optimized.find_dead_end_nodes();
        for &dead_end_idx in &dead_end_nodes {
            optimized.graph.remove_node(dead_end_idx);
        }

        // Update input/output lists to remove any invalid indices
        optimized
            .inputs
            .retain(|&idx| optimized.graph.node_weight(idx).is_some());
        optimized
            .outputs
            .retain(|&idx| optimized.graph.node_weight(idx).is_some());

        Ok(optimized)
    }

    /// Create a subgraph containing only the specified nodes and their connections
    pub fn subgraph(&self, node_indices: &[NodeIndex]) -> TorshResult<FxGraph> {
        let mut subgraph = FxGraph::new();
        let mut node_mapping = std::collections::HashMap::new();
        let node_set: std::collections::HashSet<_> = node_indices.iter().collect();

        // Add nodes to subgraph
        for &idx in node_indices {
            if let Some(node) = self.get_node(idx) {
                let new_idx = subgraph.add_node(node.clone());
                node_mapping.insert(idx, new_idx);

                // Mark as input/output if it was in the original graph
                if self.inputs.contains(&idx) {
                    subgraph.add_input(new_idx);
                }
                if self.outputs.contains(&idx) {
                    subgraph.add_output(new_idx);
                }
            }
        }

        // Add edges between included nodes
        for (src_idx, target_idx, edge) in self.edges() {
            if node_set.contains(&src_idx) && node_set.contains(&target_idx) {
                if let (Some(&new_src), Some(&new_target)) =
                    (node_mapping.get(&src_idx), node_mapping.get(&target_idx))
                {
                    subgraph.add_edge(new_src, new_target, edge.clone());
                }
            }
        }

        Ok(subgraph)
    }

    /// Merge another graph into this one
    pub fn merge(&mut self, other: &FxGraph) -> TorshResult<HashMap<NodeIndex, NodeIndex>> {
        let mut node_mapping = HashMap::new();

        // Add all nodes from other graph
        for (idx, node) in other.nodes() {
            let new_idx = self.add_node(node.clone());
            node_mapping.insert(idx, new_idx);
        }

        // Add all edges from other graph
        for (src_idx, target_idx, edge) in other.edges() {
            if let (Some(&new_src), Some(&new_target)) =
                (node_mapping.get(&src_idx), node_mapping.get(&target_idx))
            {
                self.add_edge(new_src, new_target, edge.clone());
            }
        }

        // Merge input/output lists
        for &input_idx in &other.inputs {
            if let Some(&new_idx) = node_mapping.get(&input_idx) {
                self.add_input(new_idx);
            }
        }

        for &output_idx in &other.outputs {
            if let Some(&new_idx) = node_mapping.get(&output_idx) {
                self.add_output(new_idx);
            }
        }

        Ok(node_mapping)
    }

    /// Find all paths from inputs to outputs
    pub fn find_all_paths(&self) -> Vec<Vec<NodeIndex>> {
        let mut all_paths = Vec::new();

        // For each input, find paths to each output
        for &input_idx in &self.inputs {
            for &output_idx in &self.outputs {
                if let Some(path) = self.find_path(input_idx, output_idx) {
                    all_paths.push(path);
                }
            }
        }

        all_paths
    }

    /// Find a path between two nodes using DFS
    pub fn find_path(&self, start: NodeIndex, end: NodeIndex) -> Option<Vec<NodeIndex>> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        if self.dfs_find_path(start, end, &mut visited, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    /// Helper function for DFS path finding
    fn dfs_find_path(
        &self,
        current: NodeIndex,
        target: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        path: &mut Vec<NodeIndex>,
    ) -> bool {
        if current == target {
            path.push(current);
            return true;
        }

        if visited.contains(&current) {
            return false;
        }

        visited.insert(current);
        path.push(current);

        // Check all outgoing edges
        for edge_ref in self
            .graph
            .edges_directed(current, petgraph::Direction::Outgoing)
        {
            if self.dfs_find_path(edge_ref.target(), target, visited, path) {
                return true;
            }
        }

        path.pop();
        false
    }
}
