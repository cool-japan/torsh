//! Graph Construction and Management
//!
//! This module provides functionality for building and managing computation graphs,
//! including node/edge management, statistics tracking, and basic graph operations.

use super::graph_types::*;
use parking_lot::Mutex;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::error::{Result, TorshError};

impl OptimizedGraph {
    /// Create a new optimized computation graph
    ///
    /// # Arguments
    /// * `config` - Optimization configuration for the graph
    ///
    /// # Returns
    /// * `Self` - New OptimizedGraph instance with default settings
    pub fn new(config: GraphOptConfig) -> Self {
        Self {
            graph: DiGraph::new(),
            node_lookup: HashMap::new(),
            execution_order: Arc::new(RwLock::new(None)),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::default())),
            config,
            stats: Arc::new(RwLock::new(GraphStats::default())),
            compressed_checkpoints: Arc::new(Mutex::new(HashMap::new())),
            compression_stats: Arc::new(RwLock::new(CompressionStats::default())),
            nested_checkpoints: Arc::new(Mutex::new(HashMap::new())),
            nested_config: NestedCheckpointConfig::default(),
        }
    }

    /// Add a node to the computation graph
    ///
    /// Adds a new node to the graph and updates internal data structures.
    /// Invalidates cached execution order to ensure consistency.
    ///
    /// # Arguments
    /// * `node` - GraphNode to add to the graph
    ///
    /// # Returns
    /// * `Result<NodeIndex>` - Index of the added node or error
    pub fn add_node(&mut self, node: GraphNode) -> Result<NodeIndex> {
        let node_id = node.id;
        let graph_index = self.graph.add_node(node);
        self.node_lookup.insert(node_id, graph_index);

        // Invalidate cached execution order
        *self
            .execution_order
            .write()
            .expect("lock should not be poisoned") = None;

        // Update statistics
        self.stats
            .write()
            .expect("lock should not be poisoned")
            .total_nodes += 1;

        Ok(graph_index)
    }

    /// Add an edge between two nodes in the graph
    ///
    /// Creates a dependency relationship from the first node to the second node.
    /// Both nodes must already exist in the graph.
    ///
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error if nodes don't exist
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<()> {
        let from_idx = self
            .node_lookup
            .get(&from)
            .ok_or_else(|| TorshError::AutogradError(format!("Node {from} not found")))?;
        let to_idx = self
            .node_lookup
            .get(&to)
            .ok_or_else(|| TorshError::AutogradError(format!("Node {to} not found")))?;

        self.graph.add_edge(*from_idx, *to_idx, ());

        // Invalidate cached execution order
        *self
            .execution_order
            .write()
            .expect("lock should not be poisoned") = None;

        Ok(())
    }

    /// Update the node lookup table
    ///
    /// Rebuilds the mapping from NodeId to NodeIndex after graph modifications.
    /// This is called internally when nodes are removed or graph structure changes.
    pub fn update_node_lookup(&mut self) {
        let mut new_lookup = HashMap::new();
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                new_lookup.insert(node.id, node_idx);
            }
        }
        self.node_lookup = new_lookup;
    }

    /// Get current graph execution statistics
    ///
    /// Returns a clone of the current statistics including node counts,
    /// execution times, memory usage, and optimization metrics.
    ///
    /// # Returns
    /// * `GraphStats` - Current graph statistics
    pub fn get_stats(&self) -> GraphStats {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Reset all graph statistics to default values
    ///
    /// Clears all accumulated statistics including execution times,
    /// memory usage, and optimization counts.
    pub fn reset_stats(&mut self) {
        *self.stats.write().expect("lock should not be poisoned") = GraphStats::default();
    }

    /// Get current memory usage information
    ///
    /// Returns detailed memory information including current usage,
    /// peak usage, node count, and memory efficiency.
    ///
    /// # Returns
    /// * `MemoryInfo` - Current memory usage information
    pub fn get_memory_info(&self) -> MemoryInfo {
        let memory_tracker = self.memory_tracker.lock();
        MemoryInfo {
            current_usage: memory_tracker.current_memory,
            peak_usage: memory_tracker.peak_memory,
            node_count: memory_tracker.node_memory.len(),
            memory_efficiency: memory_tracker.current_memory as f64
                / self.config.memory_budget as f64,
        }
    }

    /// Get the total number of nodes in the graph
    ///
    /// # Returns
    /// * `usize` - Number of nodes currently in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the total number of edges in the graph
    ///
    /// # Returns
    /// * `usize` - Number of edges currently in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if a node exists in the graph
    ///
    /// # Arguments
    /// * `node_id` - Node ID to check
    ///
    /// # Returns
    /// * `bool` - True if the node exists, false otherwise
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.node_lookup.contains_key(&node_id)
    }

    /// Get a reference to a node by ID
    ///
    /// # Arguments
    /// * `node_id` - Node ID to retrieve
    ///
    /// # Returns
    /// * `Option<&GraphNode>` - Reference to the node if it exists
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.node_lookup
            .get(&node_id)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// Get a mutable reference to a node by ID
    ///
    /// # Arguments
    /// * `node_id` - Node ID to retrieve
    ///
    /// # Returns
    /// * `Option<&mut GraphNode>` - Mutable reference to the node if it exists
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        if let Some(&idx) = self.node_lookup.get(&node_id) {
            self.graph.node_weight_mut(idx)
        } else {
            None
        }
    }

    /// Clear all nodes and edges from the graph
    ///
    /// Resets the graph to an empty state while preserving configuration.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_lookup.clear();
        *self
            .execution_order
            .write()
            .expect("lock should not be poisoned") = None;

        // Reset memory tracker
        {
            let mut tracker = self.memory_tracker.lock();
            *tracker = MemoryTracker::default();
        }

        // Reset statistics
        self.reset_stats();
    }

    /// Get all node IDs in the graph
    ///
    /// # Returns
    /// * `Vec<NodeId>` - Vector of all node IDs in the graph
    pub fn get_all_node_ids(&self) -> Vec<NodeId> {
        self.node_lookup.keys().copied().collect()
    }

    /// Check if the graph is empty
    ///
    /// # Returns
    /// * `bool` - True if the graph contains no nodes
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Compute optimal execution order using topological sort with priority scheduling
    ///
    /// Computes the optimal order for executing graph nodes that respects dependencies
    /// and considers node priorities. Uses Kahn's algorithm with a priority queue.
    ///
    /// # Algorithm
    /// 1. Calculate in-degrees for all nodes
    /// 2. Add nodes with zero in-degree to priority queue
    /// 3. Process nodes in priority order
    /// 4. Update neighbors and add to queue when their in-degree reaches zero
    /// 5. Detect cycles if not all nodes are processed
    ///
    /// # Returns
    /// * `Result<()>` - Ok if computation succeeds, error if cycles detected
    pub fn compute_execution_order(&mut self) -> Result<()> {
        tracing::debug!(
            "Computing execution order for {} nodes",
            self.graph.node_count()
        );

        // Update node lookup to ensure consistency
        self.update_node_lookup();

        // Topological sort with priority scheduling
        let mut in_degree: HashMap<petgraph::graph::NodeIndex, usize> = HashMap::new();
        let mut ready_queue = std::collections::BinaryHeap::new();

        // Initialize in-degrees
        for node_idx in self.graph.node_indices() {
            let degree = self
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .count();
            in_degree.insert(node_idx, degree);

            if degree == 0 {
                let node = &self.graph[node_idx];
                ready_queue.push((node.priority, node.id, node_idx));
                tracing::debug!("Node {} has no dependencies", node.id);
            }
        }

        let mut execution_order = Vec::new();

        while let Some((_, node_id, node_idx)) = ready_queue.pop() {
            execution_order.push(node_id);

            // Update neighbors
            for neighbor_idx in self
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            {
                let neighbor_degree = in_degree.get_mut(&neighbor_idx).unwrap();
                *neighbor_degree -= 1;

                if *neighbor_degree == 0 {
                    let neighbor = &self.graph[neighbor_idx];
                    ready_queue.push((neighbor.priority, neighbor.id, neighbor_idx));
                }
            }
        }

        // Check for cycles - but only if we have nodes left
        if self.graph.node_count() > 0 && execution_order.len() != self.graph.node_count() {
            return Err(TorshError::AutogradError(
                format!("Computation graph contains cycles or invalid nodes. Expected {} nodes, got {} in execution order",
                    self.graph.node_count(), execution_order.len())
            ));
        }

        *self
            .execution_order
            .write()
            .expect("lock should not be poisoned") = Some(execution_order);
        tracing::debug!("Execution order computed successfully");

        Ok(())
    }

    /// Get the current execution order
    ///
    /// Returns the cached execution order. The execution order must be computed
    /// first using `compute_execution_order()` or by calling `optimize()`.
    ///
    /// # Returns
    /// * `Result<Vec<NodeId>>` - Execution order or error if not computed
    pub fn get_execution_order(&self) -> Result<Vec<NodeId>> {
        self.execution_order
            .read()
            .expect("lock should not be poisoned")
            .clone()
            .ok_or_else(|| {
                TorshError::AutogradError(
                    "Execution order not computed. Call optimize() first.".to_string(),
                )
            })
    }
}
