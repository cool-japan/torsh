//! Control flow analysis for computation graphs

use crate::graph::core::{ComputationGraph, NodeId};
use crate::graph::operations::{BlockType, Operation};
use crate::{JitError, JitResult};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// Control flow analysis for identifying loops, conditions, and dominance relationships
#[derive(Debug, Clone)]
pub struct ControlFlowAnalysis {
    /// Dominator tree: each node maps to its immediate dominator
    pub dominators: HashMap<NodeId, Option<NodeId>>,

    /// Dominated nodes: each node maps to the set of nodes it dominates
    pub dominated: HashMap<NodeId, HashSet<NodeId>>,

    /// Loop information
    pub loops: Vec<LoopInfo>,

    /// Conditional blocks
    pub conditionals: Vec<ConditionalInfo>,

    /// Statistics about the control flow
    pub stats: ControlFlowStats,
}

impl ControlFlowAnalysis {
    /// Create a new control flow analysis
    pub fn new() -> Self {
        Self {
            dominators: HashMap::new(),
            dominated: HashMap::new(),
            loops: Vec::new(),
            conditionals: Vec::new(),
            stats: ControlFlowStats::default(),
        }
    }

    /// Analyze a computation graph for control flow patterns
    pub fn analyze(graph: &ComputationGraph) -> JitResult<Self> {
        let mut analysis = Self::new();

        // Compute dominator tree
        analysis.compute_dominators(graph)?;

        // Detect loops
        analysis.detect_loops(graph)?;

        // Detect conditionals
        analysis.detect_conditionals(graph)?;

        // Compute statistics
        analysis.compute_statistics(graph);

        Ok(analysis)
    }

    /// Compute dominator relationships
    fn compute_dominators(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Simple dominator computation - in practice would use more sophisticated algorithms
        let nodes: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();

        // Initialize dominators
        for &node in &nodes {
            self.dominators.insert(node, None);
            self.dominated.insert(node, HashSet::new());
        }

        // For each node, find nodes that must be traversed to reach it from any input
        for &node in &nodes {
            let mut dominates = HashSet::new();

            // Simple approximation: a node dominates another if all paths to the second
            // node must pass through the first node
            for &other_node in &nodes {
                if node != other_node && self.dominates_node(graph, node, other_node) {
                    dominates.insert(other_node);

                    // Set immediate dominator if none exists or this is closer
                    if self.dominators.get(&other_node).unwrap().is_none() {
                        self.dominators.insert(other_node, Some(node));
                    }
                }
            }

            self.dominated.insert(node, dominates);
        }

        Ok(())
    }

    /// Check if one node dominates another (simplified check)
    fn dominates_node(&self, graph: &ComputationGraph, dominator: NodeId, node: NodeId) -> bool {
        // This is a simplified domination check
        // In practice, would use proper dominator tree algorithms

        if dominator == node {
            return true;
        }

        // Check if dominator is on all paths from inputs to node
        let inputs = &graph.inputs;
        if inputs.is_empty() {
            return false;
        }

        for &input in inputs {
            if !self.path_contains_node(graph, input, node, dominator) {
                return false;
            }
        }

        true
    }

    /// Check if a path from start to end contains a specific node
    fn path_contains_node(
        &self,
        graph: &ComputationGraph,
        start: NodeId,
        end: NodeId,
        check_node: NodeId,
    ) -> bool {
        if start == end {
            return start == check_node;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if current == end {
                return visited.contains(&check_node);
            }

            for neighbor in graph.get_node_outputs(current) {
                if !visited.contains(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        false
    }

    /// Detect loop structures in the graph
    fn detect_loops(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        let nodes: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();

        for &node in &nodes {
            if let Some(node_data) = graph.get_node(node) {
                match &node_data.operation {
                    Operation::While(while_info) => {
                        let loop_info = LoopInfo {
                            header: node,
                            condition: while_info.condition,
                            body_nodes: self.find_loop_body_nodes(graph, while_info.body),
                            loop_type: LoopType::While,
                            max_iterations: while_info.max_iterations,
                        };
                        self.loops.push(loop_info);
                    }
                    Operation::For(for_info) => {
                        let loop_info = LoopInfo {
                            header: node,
                            condition: for_info.start, // Simplified
                            body_nodes: self.find_loop_body_nodes(graph, for_info.body),
                            loop_type: LoopType::For,
                            max_iterations: None, // Could be computed from for loop bounds
                        };
                        self.loops.push(loop_info);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Find all nodes that belong to a loop body
    fn find_loop_body_nodes(
        &self,
        graph: &ComputationGraph,
        body_start: NodeId,
    ) -> HashSet<NodeId> {
        let mut body_nodes = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(body_start);

        while let Some(node) = queue.pop_front() {
            if body_nodes.contains(&node) {
                continue;
            }
            body_nodes.insert(node);

            // Add successors that are part of the loop body
            for successor in graph.get_node_outputs(node) {
                if let Some(successor_data) = graph.get_node(successor) {
                    match &successor_data.operation {
                        Operation::Break | Operation::Continue => {
                            // Don't traverse beyond loop control statements
                            body_nodes.insert(successor);
                        }
                        _ => {
                            if !body_nodes.contains(&successor) {
                                queue.push_back(successor);
                            }
                        }
                    }
                }
            }
        }

        body_nodes
    }

    /// Detect conditional structures in the graph
    fn detect_conditionals(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        let nodes: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();

        for &node in &nodes {
            if let Some(node_data) = graph.get_node(node) {
                if let Operation::If(if_info) = &node_data.operation {
                    let then_nodes = self.find_branch_nodes(graph, if_info.then_block);
                    let else_nodes = if let Some(else_block) = if_info.else_block {
                        self.find_branch_nodes(graph, else_block)
                    } else {
                        HashSet::new()
                    };

                    let conditional_info = ConditionalInfo {
                        condition_node: if_info.condition,
                        then_nodes,
                        else_nodes,
                        merge_point: if_info.merge_point,
                    };
                    self.conditionals.push(conditional_info);
                }
            }
        }

        Ok(())
    }

    /// Find all nodes that belong to a conditional branch
    fn find_branch_nodes(&self, graph: &ComputationGraph, branch_start: NodeId) -> HashSet<NodeId> {
        let mut branch_nodes = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(branch_start);

        while let Some(node) = queue.pop_front() {
            if branch_nodes.contains(&node) {
                continue;
            }
            branch_nodes.insert(node);

            // Add successors until we reach a merge point or loop back
            for successor in graph.get_node_outputs(node) {
                if let Some(successor_data) = graph.get_node(successor) {
                    match &successor_data.operation {
                        Operation::Merge(_) => {
                            // Stop at merge points
                            break;
                        }
                        _ => {
                            if !branch_nodes.contains(&successor) {
                                queue.push_back(successor);
                            }
                        }
                    }
                }
            }
        }

        branch_nodes
    }

    /// Compute control flow statistics
    fn compute_statistics(&mut self, graph: &ComputationGraph) {
        let mut loop_count = 0;
        let mut conditional_count = 0;
        let mut block_count = 0;

        for (_, node) in graph.nodes() {
            match &node.operation {
                Operation::While(_) | Operation::For(_) => loop_count += 1,
                Operation::If(_) => conditional_count += 1,
                Operation::Block(_) => block_count += 1,
                _ => {}
            }
        }

        self.stats = ControlFlowStats {
            total_nodes: graph.node_count(),
            loop_count,
            conditional_count,
            block_count,
            max_loop_depth: self.compute_max_loop_depth(),
            max_conditional_depth: self.compute_max_conditional_depth(),
        };
    }

    /// Compute maximum loop nesting depth
    fn compute_max_loop_depth(&self) -> usize {
        // Simplified computation - would need more sophisticated analysis for nested loops
        if self.loops.is_empty() {
            0
        } else {
            1 // For now, assume max depth of 1
        }
    }

    /// Compute maximum conditional nesting depth
    fn compute_max_conditional_depth(&self) -> usize {
        // Simplified computation - would need more sophisticated analysis for nested conditionals
        if self.conditionals.is_empty() {
            0
        } else {
            1 // For now, assume max depth of 1
        }
    }

    /// Check if a node is inside a loop
    pub fn is_in_loop(&self, node: NodeId) -> bool {
        self.loops
            .iter()
            .any(|loop_info| loop_info.body_nodes.contains(&node))
    }

    /// Check if a node is inside a conditional branch
    pub fn is_in_conditional(&self, node: NodeId) -> bool {
        self.conditionals.iter().any(|cond_info| {
            cond_info.then_nodes.contains(&node) || cond_info.else_nodes.contains(&node)
        })
    }

    /// Get the loop that contains a given node
    pub fn containing_loop(&self, node: NodeId) -> Option<&LoopInfo> {
        self.loops
            .iter()
            .find(|loop_info| loop_info.body_nodes.contains(&node))
    }

    /// Get the conditional that contains a given node
    pub fn containing_conditional(&self, node: NodeId) -> Option<&ConditionalInfo> {
        self.conditionals.iter().find(|cond_info| {
            cond_info.then_nodes.contains(&node) || cond_info.else_nodes.contains(&node)
        })
    }
}

impl Default for ControlFlowAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a loop in the control flow
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Header node of the loop
    pub header: NodeId,
    /// Condition node
    pub condition: NodeId,
    /// Nodes that are part of the loop body
    pub body_nodes: HashSet<NodeId>,
    /// Type of loop
    pub loop_type: LoopType,
    /// Maximum number of iterations (if known)
    pub max_iterations: Option<usize>,
}

/// Types of loops
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopType {
    While,
    For,
    DoWhile,
}

/// Information about a conditional structure
#[derive(Debug, Clone)]
pub struct ConditionalInfo {
    /// The condition node
    pub condition_node: NodeId,
    /// Nodes in the 'then' branch
    pub then_nodes: HashSet<NodeId>,
    /// Nodes in the 'else' branch (if any)
    pub else_nodes: HashSet<NodeId>,
    /// Merge point where branches reconverge
    pub merge_point: Option<NodeId>,
}

/// Statistics about control flow in the graph
#[derive(Debug, Clone, Default)]
pub struct ControlFlowStats {
    /// Total number of nodes in the graph
    pub total_nodes: usize,
    /// Number of loops
    pub loop_count: usize,
    /// Number of conditionals
    pub conditional_count: usize,
    /// Number of block operations
    pub block_count: usize,
    /// Maximum loop nesting depth
    pub max_loop_depth: usize,
    /// Maximum conditional nesting depth
    pub max_conditional_depth: usize,
}
