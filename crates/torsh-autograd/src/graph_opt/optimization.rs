//! Core Graph Optimization Algorithms
//!
//! This module implements the fundamental optimization passes for computation graphs,
//! including dead code elimination, common subexpression elimination, and operator fusion.

use super::graph_types::*;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_core::error::Result;

impl OptimizedGraph {
    /// Apply all enabled optimization passes to the graph
    ///
    /// Runs optimization passes in the optimal order based on configuration.
    /// This is the main entry point for graph optimization.
    ///
    /// # Returns
    /// * `Result<()>` - Ok if optimization succeeds, error otherwise
    pub fn optimize(&mut self) -> Result<()> {
        tracing::info!("Starting graph optimization");
        let start_time = std::time::Instant::now();

        if self.config.enable_dce {
            self.eliminate_dead_code()?;
        }

        if self.config.enable_cse {
            self.eliminate_common_subexpressions()?;
        }

        if self.config.enable_fusion {
            self.fuse_operations()?;
        }

        // Compute execution order after optimization passes
        self.compute_execution_order()?;

        let optimization_time = start_time.elapsed().as_millis() as u64;
        self.stats.write().unwrap().total_execution_time_ms += optimization_time;

        tracing::info!("Graph optimization completed in {}ms", optimization_time);
        Ok(())
    }

    /// Eliminate dead code (unused nodes)
    ///
    /// Removes nodes that don't contribute to the final output by performing
    /// backward reachability analysis from nodes that require gradients.
    ///
    /// # Algorithm
    /// 1. Identify output nodes (no outgoing edges, requires_grad = true)
    /// 2. Perform backward traversal to mark reachable nodes as live
    /// 3. Remove all unreachable (dead) nodes from the graph
    ///
    /// # Returns
    /// * `Result<()>` - Ok if elimination succeeds, error otherwise
    pub fn eliminate_dead_code(&mut self) -> Result<()> {
        tracing::debug!("Eliminating dead code");

        // Find output nodes (nodes with no outgoing edges that require gradients)
        let mut live_nodes = HashSet::new();
        let mut queue = VecDeque::new();

        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            let has_outputs = self
                .graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .count()
                > 0;

            if !has_outputs && node.requires_grad {
                live_nodes.insert(node_idx);
                queue.push_back(node_idx);
            }
        }

        // Mark all nodes reachable from outputs as live
        while let Some(node_idx) = queue.pop_front() {
            for neighbor_idx in self.graph.neighbors_directed(node_idx, Direction::Incoming) {
                if live_nodes.insert(neighbor_idx) {
                    queue.push_back(neighbor_idx);
                }
            }
        }

        // Remove dead nodes
        let all_nodes: Vec<_> = self.graph.node_indices().collect();
        let mut eliminated_count = 0;

        for node_idx in all_nodes {
            if !live_nodes.contains(&node_idx) {
                let node = self.graph.remove_node(node_idx).unwrap();
                self.node_lookup.remove(&node.id);
                eliminated_count += 1;
            }
        }

        self.stats.write().unwrap().eliminated_nodes += eliminated_count;
        tracing::debug!("Eliminated {} dead nodes", eliminated_count);

        Ok(())
    }

    /// Eliminate common subexpressions
    ///
    /// Identifies and merges nodes that perform identical operations on
    /// the same inputs, reducing redundant computation.
    ///
    /// # Algorithm
    /// 1. Compute operation signatures for all nodes
    /// 2. Group nodes by identical signatures
    /// 3. For each group with multiple nodes, keep one canonical node
    /// 4. Redirect all uses of duplicate nodes to the canonical node
    /// 5. Remove duplicate nodes
    ///
    /// # Returns
    /// * `Result<()>` - Ok if elimination succeeds, error otherwise
    pub fn eliminate_common_subexpressions(&mut self) -> Result<()> {
        tracing::debug!("Eliminating common subexpressions");

        // Group nodes by operation signature
        let mut operation_groups: HashMap<String, Vec<petgraph::graph::NodeIndex>> = HashMap::new();

        for node_idx in self.graph.node_indices() {
            let signature = self.compute_operation_signature(node_idx)?;
            operation_groups
                .entry(signature)
                .or_default()
                .push(node_idx);
        }

        // Merge equivalent operations
        let mut eliminated_count = 0;
        for (_, group) in operation_groups {
            if group.len() > 1 {
                let canonical_node = group[0];

                // Redirect all uses of duplicate nodes to canonical node
                for &duplicate_node in &group[1..] {
                    let outgoing_edges: Vec<_> = self
                        .graph
                        .neighbors_directed(duplicate_node, Direction::Outgoing)
                        .collect();

                    for target in outgoing_edges {
                        self.graph.add_edge(canonical_node, target, ());
                    }

                    let node = self.graph.remove_node(duplicate_node).unwrap();
                    self.node_lookup.remove(&node.id);
                    eliminated_count += 1;
                }
            }
        }

        self.stats.write().unwrap().eliminated_nodes += eliminated_count;
        tracing::debug!("Eliminated {} common subexpressions", eliminated_count);

        Ok(())
    }

    /// Compute operation signature for common subexpression elimination
    ///
    /// Creates a unique signature that identifies equivalent operations.
    /// Operations with identical signatures can be merged.
    ///
    /// # Arguments
    /// * `node_idx` - Index of the node to compute signature for
    ///
    /// # Returns
    /// * `Result<String>` - Operation signature or error
    pub fn compute_operation_signature(
        &self,
        node_idx: petgraph::graph::NodeIndex,
    ) -> Result<String> {
        let node = &self.graph[node_idx];
        let mut signature = format!("{}:", node.op_name);

        // Add input signatures (sorted for consistency)
        let mut inputs: Vec<_> = self
            .graph
            .neighbors_directed(node_idx, Direction::Incoming)
            .map(|n| self.graph[n].id)
            .collect();
        inputs.sort();

        for input_id in inputs {
            signature.push_str(&format!("_{input_id}"));
        }

        Ok(signature)
    }

    /// Fuse compatible operations
    ///
    /// Combines sequential element-wise operations into fused kernels
    /// to reduce memory bandwidth and improve execution efficiency.
    ///
    /// # Algorithm
    /// 1. Identify fusion candidates (adjacent element-wise operations)
    /// 2. Verify fusion compatibility (shape, operation type)
    /// 3. Create fused operations that combine multiple operations
    /// 4. Update graph structure to use fused operations
    ///
    /// # Returns
    /// * `Result<()>` - Ok if fusion succeeds, error otherwise
    pub fn fuse_operations(&mut self) -> Result<()> {
        tracing::debug!("Fusing operations");

        let mut fused_count = 0;

        // Collect fusion candidates before modifying the graph
        let mut fusion_pairs = Vec::new();
        for node_idx in self.graph.node_indices() {
            if self.graph.node_weight(node_idx).is_some() {
                if let Some(fusion_candidate) = self.find_fusion_candidate(node_idx)? {
                    if self.can_fuse_operations(node_idx, fusion_candidate)? {
                        fusion_pairs.push((node_idx, fusion_candidate));
                    }
                }
            }
        }

        // Apply fusions
        for (node_idx, fusion_candidate) in fusion_pairs {
            // Check if both nodes still exist
            if self.graph.node_weight(node_idx).is_some()
                && self.graph.node_weight(fusion_candidate).is_some()
            {
                self.fuse_two_operations(node_idx, fusion_candidate)?;
                fused_count += 1;
            }
        }

        self.stats.write().unwrap().fused_operations += fused_count;
        tracing::debug!("Fused {} operations", fused_count);

        Ok(())
    }

    /// Find a candidate for fusion with the given node
    ///
    /// Searches immediate neighbors for compatible operations that can be fused.
    ///
    /// # Arguments
    /// * `node_idx` - Index of the node to find fusion candidates for
    ///
    /// # Returns
    /// * `Result<Option<NodeIndex>>` - Fusion candidate or None if no candidate found
    pub fn find_fusion_candidate(
        &self,
        node_idx: petgraph::graph::NodeIndex,
    ) -> Result<Option<petgraph::graph::NodeIndex>> {
        if let Some(node) = self.graph.node_weight(node_idx) {
            // Look for fusable operations in immediate neighbors
            for neighbor_idx in self.graph.neighbors_directed(node_idx, Direction::Outgoing) {
                if let Some(neighbor) = self.graph.node_weight(neighbor_idx) {
                    // Simple fusion rules for element-wise operations
                    if self.is_element_wise_op(&node.op_name)
                        && self.is_element_wise_op(&neighbor.op_name)
                    {
                        return Ok(Some(neighbor_idx));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if two operations can be fused
    ///
    /// Verifies compatibility for fusion including operation types,
    /// shapes, and graph structure constraints.
    ///
    /// # Arguments
    /// * `op1` - First operation index
    /// * `op2` - Second operation index
    ///
    /// # Returns
    /// * `Result<bool>` - True if operations can be fused, false otherwise
    pub fn can_fuse_operations(
        &self,
        op1: petgraph::graph::NodeIndex,
        op2: petgraph::graph::NodeIndex,
    ) -> Result<bool> {
        if let (Some(node1), Some(node2)) =
            (self.graph.node_weight(op1), self.graph.node_weight(op2))
        {
            // Check if both are element-wise operations
            if !self.is_element_wise_op(&node1.op_name) || !self.is_element_wise_op(&node2.op_name)
            {
                return Ok(false);
            }

            // Check if output shapes are compatible
            if node1.output_shape != node2.output_shape {
                return Ok(false);
            }

            // Check if op2 has only one input (op1)
            let op2_inputs = self
                .graph
                .neighbors_directed(op2, Direction::Incoming)
                .count();
            if op2_inputs != 1 {
                return Ok(false);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if operation is element-wise
    ///
    /// Element-wise operations can be safely fused since they operate
    /// independently on each tensor element.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation to check
    ///
    /// # Returns
    /// * `bool` - True if operation is element-wise
    pub fn is_element_wise_op(&self, op_name: &str) -> bool {
        matches!(
            op_name,
            "add"
                | "mul"
                | "sub"
                | "div"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "exp"
                | "log"
                | "sqrt"
                | "abs"
                | "neg"
                | "sin"
                | "cos"
                | "pow"
                | "max"
                | "min"
        )
    }

    /// Fuse two operations into one
    ///
    /// Creates a new fused operation that combines the functionality
    /// of two compatible operations, removing the redundant operation.
    ///
    /// # Arguments
    /// * `op1` - First operation index (will be replaced with fused operation)
    /// * `op2` - Second operation index (will be removed)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if fusion succeeds, error otherwise
    pub fn fuse_two_operations(
        &mut self,
        op1: petgraph::graph::NodeIndex,
        op2: petgraph::graph::NodeIndex,
    ) -> Result<()> {
        let node1 = self.graph[op1].clone();
        let node2 = self.graph[op2].clone();

        // Create fused operation name
        let fused_op_name = format!("fused_{}_{}", node1.op_name, node2.op_name);

        // Create new fused node with combined properties
        let fused_node = GraphNode {
            id: node1.id, // Reuse the first node's ID
            op_name: fused_op_name,
            inputs: node1.inputs.clone(),
            output_shape: node2.output_shape.clone(),
            requires_grad: node1.requires_grad || node2.requires_grad,
            priority: node1.priority.max(node2.priority),
            memory_usage: node1.memory_usage + node2.memory_usage,
            compute_cost: node1.compute_cost + node2.compute_cost,
            can_execute_in_place: node1.can_execute_in_place && node2.can_execute_in_place,
            can_recompute: node1.can_recompute && node2.can_recompute,
        };

        // Replace op1 with fused node
        self.graph[op1] = fused_node;

        // Redirect op2's outputs to op1
        let op2_outputs: Vec<_> = self
            .graph
            .neighbors_directed(op2, Direction::Outgoing)
            .collect();

        for output in op2_outputs {
            self.graph.add_edge(op1, output, ());
        }

        // Remove op2 and update lookup table
        let removed_node = self.graph.remove_node(op2).unwrap();
        self.node_lookup.remove(&removed_node.id);

        Ok(())
    }
}
