//! Serialization support for FX graphs

use crate::fx::types::{Edge, Node};
use crate::graph_analysis::GraphMetrics;
use crate::FxGraph;
use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{Result, TorshError};

/// Convenience type alias for Results in this crate
pub type TorshResult<T> = Result<T>;

/// Serializable representation of FxGraph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGraph {
    nodes: Vec<(usize, Node)>,
    edges: Vec<(usize, usize, Edge)>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl SerializableGraph {
    /// Convert FxGraph to serializable format
    pub fn from_graph(graph: &FxGraph) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Collect nodes
        for (idx, node) in graph.nodes() {
            nodes.push((idx.index(), node.clone()));
        }

        // Collect edges
        for edge_ref in graph.graph.edge_references() {
            edges.push((
                edge_ref.source().index(),
                edge_ref.target().index(),
                edge_ref.weight().clone(),
            ));
        }

        Self {
            nodes,
            edges,
            inputs: graph.inputs.iter().map(|idx| idx.index()).collect(),
            outputs: graph.outputs.iter().map(|idx| idx.index()).collect(),
        }
    }

    /// Convert serializable format to FxGraph
    pub fn to_graph(self) -> FxGraph {
        let mut graph = Graph::new();
        let mut node_mapping = std::collections::HashMap::new();

        // Add nodes
        for (original_idx, node) in self.nodes {
            let new_idx = graph.add_node(node);
            node_mapping.insert(original_idx, new_idx);
        }

        // Add edges
        for (src_idx, target_idx, edge) in self.edges {
            if let (Some(&src), Some(&target)) =
                (node_mapping.get(&src_idx), node_mapping.get(&target_idx))
            {
                graph.add_edge(src, target, edge);
            }
        }

        // Map input and output indices
        let inputs = self
            .inputs
            .into_iter()
            .filter_map(|idx| node_mapping.get(&idx).copied())
            .collect();
        let outputs = self
            .outputs
            .into_iter()
            .filter_map(|idx| node_mapping.get(&idx).copied())
            .collect();

        FxGraph {
            graph,
            inputs,
            outputs,
        }
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Basic validation of the graph structure
    pub fn validate(&self) -> TorshResult<()> {
        // Check that all edge endpoints refer to valid nodes
        let node_indices: std::collections::HashSet<usize> =
            self.nodes.iter().map(|(idx, _)| *idx).collect();

        for (src, target, _) in &self.edges {
            if !node_indices.contains(src) {
                return Err(TorshError::InvalidArgument(format!(
                    "Edge source {src} not found in nodes"
                )));
            }
            if !node_indices.contains(target) {
                return Err(TorshError::InvalidArgument(format!(
                    "Edge target {target} not found in nodes"
                )));
            }
        }

        // Check that input/output indices are valid
        for &input_idx in &self.inputs {
            if !node_indices.contains(&input_idx) {
                return Err(TorshError::InvalidArgument(format!(
                    "Input index {input_idx} not found in nodes"
                )));
            }
        }

        for &output_idx in &self.outputs {
            if !node_indices.contains(&output_idx) {
                return Err(TorshError::InvalidArgument(format!(
                    "Output index {output_idx} not found in nodes"
                )));
            }
        }

        Ok(())
    }

    /// Count operations by type
    pub fn operation_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for (_, node) in &self.nodes {
            let op_name = match node {
                Node::Input(_) => "input".to_string(),
                Node::Call(op, _) => op.clone(),
                Node::Output => "output".to_string(),
                Node::Conditional { .. } => "conditional".to_string(),
                Node::Loop { .. } => "loop".to_string(),
                Node::Merge { .. } => "merge".to_string(),
                Node::GetAttr { .. } => "getattr".to_string(),
            };

            *counts.entry(op_name).or_insert(0) += 1;
        }

        counts
    }

    /// Check if the graph is a linear chain
    pub fn is_linear_chain(&self) -> bool {
        if self.nodes.len() <= 1 {
            return true;
        }

        // Build adjacency list
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();

        for (src, target, _) in &self.edges {
            outgoing.entry(*src).or_default().push(*target);
            incoming.entry(*target).or_default().push(*src);
        }

        // Check that each node has at most 1 outgoing and 1 incoming edge
        for (idx, _) in &self.nodes {
            let out_count = outgoing.get(idx).map_or(0, |v| v.len());
            let in_count = incoming.get(idx).map_or(0, |v| v.len());

            if out_count > 1 || in_count > 1 {
                return false;
            }
        }

        true
    }

    /// Check if the graph has cycles
    pub fn has_cycles(&self) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        // Build adjacency list
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (src, target, _) in &self.edges {
            adj.entry(*src).or_default().push(*target);
        }

        fn dfs_has_cycle(
            node: usize,
            adj: &HashMap<usize, Vec<usize>>,
            visited: &mut std::collections::HashSet<usize>,
            rec_stack: &mut std::collections::HashSet<usize>,
        ) -> bool {
            visited.insert(node);
            rec_stack.insert(node);

            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        if dfs_has_cycle(neighbor, adj, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(&neighbor) {
                        return true;
                    }
                }
            }

            rec_stack.remove(&node);
            false
        }

        for (idx, _) in &self.nodes {
            if !visited.contains(idx) && dfs_has_cycle(*idx, &adj, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    /// Get the maximum depth of the graph
    pub fn get_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        // Build adjacency list
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (src, target, _) in &self.edges {
            adj.entry(*src).or_default().push(*target);
        }

        fn dfs_depth(
            node: usize,
            adj: &HashMap<usize, Vec<usize>>,
            visited: &mut std::collections::HashSet<usize>,
        ) -> usize {
            if visited.contains(&node) {
                return 0; // Avoid infinite recursion in cycles
            }
            visited.insert(node);

            let mut max_depth = 0;
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let depth = dfs_depth(neighbor, adj, visited);
                    max_depth = max_depth.max(depth);
                }
            }

            visited.remove(&node);
            max_depth + 1
        }

        let mut max_depth = 0;
        for (idx, _) in &self.nodes {
            let mut visited = std::collections::HashSet::new();
            let depth = dfs_depth(*idx, &adj, &mut visited);
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    /// Find orphaned nodes (nodes with no incoming or outgoing edges)
    pub fn find_orphaned_nodes(&self) -> Vec<usize> {
        let mut connected_nodes = std::collections::HashSet::new();

        for (src, target, _) in &self.edges {
            connected_nodes.insert(*src);
            connected_nodes.insert(*target);
        }

        self.nodes
            .iter()
            .filter_map(|(idx, _)| {
                if !connected_nodes.contains(idx) {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find dead-end nodes (nodes that don't lead to any output)
    pub fn find_dead_end_nodes(&self) -> Vec<usize> {
        if self.outputs.is_empty() {
            return Vec::new();
        }

        // Build reverse adjacency list (incoming edges)
        let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();
        for (src, target, _) in &self.edges {
            incoming.entry(*target).or_default().push(*src);
        }

        // BFS from all output nodes to find reachable nodes
        let mut reachable = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        for &output in &self.outputs {
            queue.push_back(output);
            reachable.insert(output);
        }

        while let Some(node) = queue.pop_front() {
            if let Some(predecessors) = incoming.get(&node) {
                for &pred in predecessors {
                    if !reachable.contains(&pred) {
                        reachable.insert(pred);
                        queue.push_back(pred);
                    }
                }
            }
        }

        // Return nodes that are not reachable from any output
        self.nodes
            .iter()
            .filter_map(|(idx, _)| {
                if !reachable.contains(idx) {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all call nodes
    pub fn call_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter_map(|(idx, node)| match node {
                Node::Call(_, _) => Some(*idx),
                _ => None,
            })
            .collect()
    }

    /// Graph metrics for analysis
    pub fn metrics(&self) -> GraphMetrics {
        let node_count = self.node_count();
        let edge_count = self.edge_count();
        let depth = self.get_depth();
        let has_cycles = self.has_cycles();
        let is_linear = self.is_linear_chain();

        // Simple complexity score based on various factors
        let complexity_score = (node_count as f32 * 0.1)
            + (edge_count as f32 * 0.15)
            + (depth as f32 * 0.2)
            + if has_cycles { 10.0 } else { 0.0 }
            + if is_linear { -2.0 } else { 5.0 };

        GraphMetrics {
            node_count,
            edge_count,
            input_count: self.inputs.len(),
            output_count: self.outputs.len(),
            max_depth: depth,
            average_fanout: if node_count > 0 {
                edge_count as f64 / node_count as f64
            } else {
                0.0
            },
            connectivity_ratio: if node_count > 1 {
                edge_count as f64 / ((node_count * (node_count - 1)) as f64)
            } else {
                0.0
            },
            complexity_score: complexity_score as f64,
            operation_distribution: self
                .operation_counts()
                .into_iter()
                .map(|(k, v)| (k, v as u32))
                .collect(),
            critical_path_length: depth,
        }
    }

    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push((idx, node));
        idx
    }

    /// Add an input node index
    pub fn add_input(&mut self, idx: usize) {
        self.inputs.push(idx);
    }

    /// Add an output node index
    pub fn add_output(&mut self, idx: usize) {
        self.outputs.push(idx);
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, src: usize, target: usize, edge: Edge) {
        self.edges.push((src, target, edge));
    }

    /// Create a sequential chain of operations
    pub fn sequential_ops(ops: &[&str]) -> Self {
        let mut graph = Self::new();

        if ops.is_empty() {
            return graph;
        }

        let input = graph.add_node(Node::Input("x".to_string()));
        graph.add_input(input);

        let mut prev = input;
        for (i, &op) in ops.iter().enumerate() {
            let node = graph.add_node(Node::Call(op.to_string(), vec![format!("arg_{i}")]));
            graph.add_edge(
                prev,
                node,
                Edge {
                    name: format!("edge_{i}"),
                },
            );
            prev = node;
        }

        let output = graph.add_node(Node::Output);
        graph.add_edge(
            prev,
            output,
            Edge {
                name: "final".to_string(),
            },
        );
        graph.add_output(output);

        graph
    }
}

impl FxGraph {
    /// Serialize graph to JSON
    pub fn to_json(&self) -> TorshResult<String> {
        let serializable = SerializableGraph::from_graph(self);
        serde_json::to_string_pretty(&serializable).map_err(|e| {
            torsh_core::error::TorshError::SerializationError(format!(
                "Failed to serialize graph to JSON: {}",
                e
            ))
        })
    }

    /// Deserialize graph from JSON
    pub fn from_json(json: &str) -> TorshResult<Self> {
        let serializable: SerializableGraph = serde_json::from_str(json).map_err(|e| {
            torsh_core::error::TorshError::SerializationError(format!(
                "Failed to deserialize graph from JSON: {}",
                e
            ))
        })?;
        Ok(serializable.to_graph())
    }

    /// Serialize graph to binary format
    pub fn to_binary(&self) -> TorshResult<Vec<u8>> {
        let serializable = SerializableGraph::from_graph(self);
        bincode::serde::encode_to_vec(&serializable, bincode::config::standard()).map_err(|e| {
            torsh_core::error::TorshError::SerializationError(format!(
                "Failed to serialize graph to binary: {}",
                e
            ))
        })
    }

    /// Deserialize graph from binary format
    pub fn from_binary(data: &[u8]) -> TorshResult<Self> {
        let (serializable, _): (SerializableGraph, usize) =
            bincode::serde::decode_from_slice(data, bincode::config::standard()).map_err(|e| {
                torsh_core::error::TorshError::SerializationError(format!(
                    "Failed to deserialize graph from binary: {}",
                    e
                ))
            })?;
        Ok(serializable.to_graph())
    }
}
