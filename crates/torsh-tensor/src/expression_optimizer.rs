//! Tensor Expression Optimization Framework
//!
//! This module provides an advanced framework for optimizing tensor expressions by analyzing
//! computational graphs, detecting patterns, and applying optimization transformations.
//! It includes graph fusion, memory optimization, operation reordering, and other advanced
//! optimization techniques to improve performance.

use crate::{Tensor, TensorElement};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};

/// Unique identifier for nodes in the expression graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Types of tensor operations that can be optimized
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Unary operations
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,

    // Trigonometric operations
    Sin,
    Cos,
    Tan,

    // Activation functions
    Relu,
    Sigmoid,
    Tanh,

    // Matrix operations
    MatMul,
    Transpose,

    // Shape operations
    Reshape,
    View,
    Permute,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,

    // Broadcasting operations
    Broadcast,

    // Memory operations
    Copy,
    Clone,

    // Custom operation
    Custom(String),
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationType::Add => write!(f, "add"),
            OperationType::Sub => write!(f, "sub"),
            OperationType::Mul => write!(f, "mul"),
            OperationType::Div => write!(f, "div"),
            OperationType::Neg => write!(f, "neg"),
            OperationType::Abs => write!(f, "abs"),
            OperationType::Sqrt => write!(f, "sqrt"),
            OperationType::Exp => write!(f, "exp"),
            OperationType::Log => write!(f, "log"),
            OperationType::Sin => write!(f, "sin"),
            OperationType::Cos => write!(f, "cos"),
            OperationType::Tan => write!(f, "tan"),
            OperationType::Relu => write!(f, "relu"),
            OperationType::Sigmoid => write!(f, "sigmoid"),
            OperationType::Tanh => write!(f, "tanh"),
            OperationType::MatMul => write!(f, "matmul"),
            OperationType::Transpose => write!(f, "transpose"),
            OperationType::Reshape => write!(f, "reshape"),
            OperationType::View => write!(f, "view"),
            OperationType::Permute => write!(f, "permute"),
            OperationType::Sum => write!(f, "sum"),
            OperationType::Mean => write!(f, "mean"),
            OperationType::Max => write!(f, "max"),
            OperationType::Min => write!(f, "min"),
            OperationType::Broadcast => write!(f, "broadcast"),
            OperationType::Copy => write!(f, "copy"),
            OperationType::Clone => write!(f, "clone"),
            OperationType::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Properties of an operation that affect optimization decisions
#[derive(Debug, Clone)]
pub struct OperationProperties {
    /// Whether the operation is element-wise
    pub is_elementwise: bool,
    /// Whether the operation is commutative (a op b == b op a)
    pub is_commutative: bool,
    /// Whether the operation is associative ((a op b) op c == a op (b op c))
    pub is_associative: bool,
    /// Whether the operation preserves shape
    pub preserves_shape: bool,
    /// Memory cost factor (relative to input size)
    pub memory_cost: f32,
    /// Computational cost factor (relative to input size)
    pub compute_cost: f32,
    /// Whether the operation can be fused with others
    pub fusable: bool,
}

impl OperationType {
    /// Get the properties of this operation type
    pub fn properties(&self) -> OperationProperties {
        match self {
            OperationType::Add | OperationType::Mul => OperationProperties {
                is_elementwise: true,
                is_commutative: true,
                is_associative: true,
                preserves_shape: true,
                memory_cost: 0.0, // In-place possible
                compute_cost: 1.0,
                fusable: true,
            },
            OperationType::Sub | OperationType::Div => OperationProperties {
                is_elementwise: true,
                is_commutative: false,
                is_associative: false,
                preserves_shape: true,
                memory_cost: 0.0,
                compute_cost: 1.0,
                fusable: true,
            },
            OperationType::Neg
            | OperationType::Abs
            | OperationType::Sqrt
            | OperationType::Exp
            | OperationType::Log
            | OperationType::Sin
            | OperationType::Cos
            | OperationType::Tan
            | OperationType::Relu
            | OperationType::Sigmoid
            | OperationType::Tanh => OperationProperties {
                is_elementwise: true,
                is_commutative: false,
                is_associative: false,
                preserves_shape: true,
                memory_cost: 0.0,
                compute_cost: 1.0,
                fusable: true,
            },
            OperationType::MatMul => OperationProperties {
                is_elementwise: false,
                is_commutative: false,
                is_associative: true,
                preserves_shape: false,
                memory_cost: 1.0,
                compute_cost: 10.0, // Matrix multiplication is expensive
                fusable: false,
            },
            OperationType::Transpose => OperationProperties {
                is_elementwise: false,
                is_commutative: false,
                is_associative: false,
                preserves_shape: false,
                memory_cost: 0.0, // Can be view-based
                compute_cost: 0.1,
                fusable: false,
            },
            OperationType::Reshape | OperationType::View | OperationType::Permute => {
                OperationProperties {
                    is_elementwise: false,
                    is_commutative: false,
                    is_associative: false,
                    preserves_shape: false,
                    memory_cost: 0.0, // Can be view-based
                    compute_cost: 0.1,
                    fusable: false,
                }
            }
            OperationType::Sum | OperationType::Mean | OperationType::Max | OperationType::Min => {
                OperationProperties {
                    is_elementwise: false,
                    is_commutative: false,
                    is_associative: false,
                    preserves_shape: false,
                    memory_cost: 0.5,
                    compute_cost: 2.0,
                    fusable: false,
                }
            }
            OperationType::Broadcast => OperationProperties {
                is_elementwise: false,
                is_commutative: false,
                is_associative: false,
                preserves_shape: false,
                memory_cost: 1.0,
                compute_cost: 0.5,
                fusable: true,
            },
            OperationType::Copy | OperationType::Clone => OperationProperties {
                is_elementwise: false,
                is_commutative: false,
                is_associative: false,
                preserves_shape: true,
                memory_cost: 1.0,
                compute_cost: 0.5,
                fusable: false,
            },
            OperationType::Custom(_) => OperationProperties {
                is_elementwise: false,
                is_commutative: false,
                is_associative: false,
                preserves_shape: false,
                memory_cost: 1.0,
                compute_cost: 5.0,
                fusable: false,
            },
        }
    }
}

/// Node in the expression graph representing a tensor operation
#[derive(Debug, Clone)]
pub struct ExpressionNode {
    /// Unique identifier for this node
    pub id: NodeId,
    /// Type of operation this node represents
    pub operation: OperationType,
    /// Input node IDs (operands)
    pub inputs: Vec<NodeId>,
    /// Output shape (if known)
    pub output_shape: Option<Vec<usize>>,
    /// Device where this operation should be executed
    pub device: DeviceType,
    /// Estimated memory usage in bytes
    pub memory_usage: Option<usize>,
    /// Estimated computation cost (relative units)
    pub compute_cost: Option<f32>,
    /// Whether this node can be computed in-place
    pub can_compute_inplace: bool,
    /// Metadata for optimization decisions
    pub metadata: HashMap<String, String>,
}

impl ExpressionNode {
    /// Create a new expression node
    pub fn new(id: NodeId, operation: OperationType) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            output_shape: None,
            device: DeviceType::Cpu,
            memory_usage: None,
            compute_cost: None,
            can_compute_inplace: false,
            metadata: HashMap::new(),
        }
    }

    /// Add an input to this node
    pub fn add_input(&mut self, input_id: NodeId) {
        self.inputs.push(input_id);
    }

    /// Set the output shape for this node
    pub fn set_output_shape(&mut self, shape: Vec<usize>) {
        self.output_shape = Some(shape);
    }

    /// Check if this node is a leaf (has no inputs)
    pub fn is_leaf(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Check if this node is fusable with another operation
    pub fn is_fusable_with(&self, other: &ExpressionNode) -> bool {
        let self_props = self.operation.properties();
        let other_props = other.operation.properties();

        // Both operations must be fusable
        if !self_props.fusable || !other_props.fusable {
            return false;
        }

        // Element-wise operations can be fused together
        if self_props.is_elementwise && other_props.is_elementwise {
            return true;
        }

        // Broadcast operations can be fused with element-wise operations
        if (self.operation == OperationType::Broadcast && other_props.is_elementwise)
            || (other.operation == OperationType::Broadcast && self_props.is_elementwise)
        {
            return true;
        }

        false
    }
}

/// Expression graph representing a computational graph of tensor operations
#[derive(Debug, Clone)]
pub struct ExpressionGraph {
    /// All nodes in the graph
    nodes: HashMap<NodeId, ExpressionNode>,
    /// Next available node ID
    next_id: usize,
    /// Root nodes (outputs of the graph)
    roots: HashSet<NodeId>,
    /// Adjacency list for efficient traversal (node -> dependents)
    adjacency: HashMap<NodeId, HashSet<NodeId>>,
}

impl ExpressionGraph {
    /// Create a new empty expression graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            roots: HashSet::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a new node to the graph
    pub fn add_node(&mut self, operation: OperationType) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let node = ExpressionNode::new(id, operation);
        self.nodes.insert(id, node);
        self.adjacency.insert(id, HashSet::new());
        self.roots.insert(id); // Initially assume it's a root

        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<()> {
        // Verify both nodes exist
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(TorshError::InvalidArgument(
                "Cannot add edge between non-existent nodes".to_string(),
            ));
        }

        // Add the edge
        self.nodes.get_mut(&to).unwrap().add_input(from);
        self.adjacency.get_mut(&from).unwrap().insert(to);

        // 'to' is no longer a root since it has an input
        self.roots.remove(&to);

        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&ExpressionNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut ExpressionNode> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> &HashMap<NodeId, ExpressionNode> {
        &self.nodes
    }

    /// Get root nodes (nodes with no dependents)
    pub fn roots(&self) -> &HashSet<NodeId> {
        &self.roots
    }

    /// Perform topological sort of the graph
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for &node_id in self.nodes.keys() {
            in_degree.insert(node_id, 0);
        }

        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                *in_degree.get_mut(&node.id).unwrap() += 1;
            }
        }

        // Find nodes with no incoming edges
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        // Process nodes
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // Reduce in-degree of dependent nodes
            if let Some(dependents) = self.adjacency.get(&node_id) {
                for &dependent_id in dependents {
                    let degree = in_degree.get_mut(&dependent_id).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dependent_id);
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.nodes.len() {
            return Err(TorshError::InvalidArgument(
                "Expression graph contains cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Detect fusable operation chains
    pub fn detect_fusable_chains(&self) -> Vec<Vec<NodeId>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();

        // Start from leaf nodes to build maximal chains
        let leaf_nodes = self.get_leaf_nodes();

        for &start_node in &leaf_nodes {
            if visited.contains(&start_node) {
                continue;
            }

            let mut chain = vec![start_node];
            visited.insert(start_node);

            // Extend chain forward
            let mut current = start_node;
            while let Some(dependents) = self.adjacency.get(&current) {
                if dependents.len() == 1 {
                    let next = *dependents.iter().next().unwrap();
                    if visited.contains(&next) {
                        break;
                    }

                    let current_node = &self.nodes[&current];
                    let next_node = &self.nodes[&next];

                    if current_node.is_fusable_with(next_node) && next_node.inputs.len() == 1 {
                        chain.push(next);
                        visited.insert(next);
                        current = next;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Only include chains with more than one operation
            if chain.len() > 1 {
                chains.push(chain);
            }
        }

        // Handle any remaining unvisited nodes (cycles or disconnected components)
        for &node_id in self.nodes.keys() {
            if visited.contains(&node_id) {
                continue;
            }

            let mut chain = vec![node_id];
            visited.insert(node_id);

            // Extend chain forward
            let mut current = node_id;
            while let Some(dependents) = self.adjacency.get(&current) {
                if dependents.len() == 1 {
                    let next = *dependents.iter().next().unwrap();
                    if visited.contains(&next) {
                        break;
                    }

                    let current_node = &self.nodes[&current];
                    let next_node = &self.nodes[&next];

                    if current_node.is_fusable_with(next_node) && next_node.inputs.len() == 1 {
                        chain.push(next);
                        visited.insert(next);
                        current = next;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Only include chains with more than one operation
            if chain.len() > 1 {
                chains.push(chain);
            }
        }

        chains
    }

    /// Calculate memory usage for the entire graph
    pub fn calculate_memory_usage(&self) -> usize {
        self.nodes
            .values()
            .filter_map(|node| node.memory_usage)
            .sum()
    }

    /// Calculate total computation cost
    pub fn calculate_compute_cost(&self) -> f32 {
        self.nodes
            .values()
            .filter_map(|node| node.compute_cost)
            .sum()
    }

    /// Get all leaf nodes (nodes with no inputs)
    pub fn get_leaf_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|node| node.is_leaf())
            .map(|node| node.id)
            .collect()
    }

    /// Verify graph integrity
    pub fn verify_integrity(&self) -> Result<()> {
        // Check that all input references are valid
        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                if !self.nodes.contains_key(&input_id) {
                    return Err(TorshError::InvalidArgument(format!(
                        "Node {} references non-existent input {}",
                        node.id, input_id
                    )));
                }
            }
        }

        // Check that adjacency list is consistent
        for (&from_id, dependents) in &self.adjacency {
            for &to_id in dependents {
                if let Some(to_node) = self.nodes.get(&to_id) {
                    if !to_node.inputs.contains(&from_id) {
                        return Err(TorshError::InvalidArgument(format!(
                            "Adjacency list inconsistency: {} -> {} not reflected in inputs",
                            from_id, to_id
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for ExpressionGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization strategy for expression graphs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Minimize memory usage
    MinimizeMemory,
    /// Minimize computation time
    MinimizeCompute,
    /// Balance memory and compute
    Balanced,
    /// Optimize for specific device characteristics
    DeviceOptimized(DeviceType),
    /// Custom optimization strategy
    Custom(String),
}

/// Configuration for the expression optimizer
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimization strategy to use
    pub strategy: OptimizationStrategy,
    /// Maximum memory budget (in bytes)
    pub memory_budget: Option<usize>,
    /// Whether to enable operation fusion
    pub enable_fusion: bool,
    /// Whether to enable memory optimization
    pub enable_memory_optimization: bool,
    /// Whether to enable operation reordering
    pub enable_reordering: bool,
    /// Whether to enable constant folding
    pub enable_constant_folding: bool,
    /// Whether to enable common subexpression elimination
    pub enable_cse: bool,
    /// Aggressiveness level (0.0 = conservative, 1.0 = aggressive)
    pub aggressiveness: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Balanced,
            memory_budget: None,
            enable_fusion: true,
            enable_memory_optimization: true,
            enable_reordering: true,
            enable_constant_folding: true,
            enable_cse: true,
            aggressiveness: 0.5,
        }
    }
}

/// Statistics about optimization results
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
    /// Memory usage before optimization (bytes)
    pub memory_before: usize,
    /// Memory usage after optimization (bytes)
    pub memory_after: usize,
    /// Compute cost before optimization
    pub compute_cost_before: f32,
    /// Compute cost after optimization
    pub compute_cost_after: f32,
    /// Number of fused operation chains
    pub fused_chains: usize,
    /// Optimization time (microseconds)
    pub optimization_time_us: u64,
}

impl OptimizationStats {
    /// Calculate memory reduction percentage
    pub fn memory_reduction(&self) -> f32 {
        if self.memory_before == 0 {
            0.0
        } else {
            ((self.memory_before as f32 - self.memory_after as f32) / self.memory_before as f32)
                * 100.0
        }
    }

    /// Calculate compute cost reduction percentage
    pub fn compute_reduction(&self) -> f32 {
        if self.compute_cost_before == 0.0 {
            0.0
        } else {
            ((self.compute_cost_before - self.compute_cost_after) / self.compute_cost_before)
                * 100.0
        }
    }

    /// Calculate node reduction percentage
    pub fn node_reduction(&self) -> f32 {
        if self.nodes_before == 0 {
            0.0
        } else {
            ((self.nodes_before as f32 - self.nodes_after as f32) / self.nodes_before as f32)
                * 100.0
        }
    }
}

impl fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Optimization Statistics:")?;
        writeln!(
            f,
            "  Nodes: {} -> {} ({:.1}% reduction)",
            self.nodes_before,
            self.nodes_after,
            self.node_reduction()
        )?;
        writeln!(
            f,
            "  Memory: {} -> {} bytes ({:.1}% reduction)",
            self.memory_before,
            self.memory_after,
            self.memory_reduction()
        )?;
        writeln!(
            f,
            "  Compute Cost: {:.2} -> {:.2} ({:.1}% reduction)",
            self.compute_cost_before,
            self.compute_cost_after,
            self.compute_reduction()
        )?;
        writeln!(f, "  Fused Chains: {}", self.fused_chains)?;
        writeln!(f, "  Optimization Time: {} μs", self.optimization_time_us)?;
        Ok(())
    }
}

/// Main expression optimizer
pub struct ExpressionOptimizer {
    config: OptimizerConfig,
}

impl ExpressionOptimizer {
    /// Create a new expression optimizer with default configuration
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Create a new expression optimizer with custom configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Optimize an expression graph
    pub fn optimize(&self, graph: &mut ExpressionGraph) -> Result<OptimizationStats> {
        let start_time = std::time::Instant::now();

        // Verify graph integrity before optimization
        graph.verify_integrity()?;

        // Collect initial statistics
        let nodes_before = graph.nodes.len();
        let memory_before = graph.calculate_memory_usage();
        let compute_cost_before = graph.calculate_compute_cost();

        let mut fused_chains = 0;

        // Apply optimizations based on configuration
        if self.config.enable_fusion {
            fused_chains += self.apply_operation_fusion(graph)?;
        }

        if self.config.enable_constant_folding {
            self.apply_constant_folding(graph)?;
        }

        if self.config.enable_cse {
            self.apply_common_subexpression_elimination(graph)?;
        }

        if self.config.enable_memory_optimization {
            self.apply_memory_optimization(graph)?;
        }

        if self.config.enable_reordering {
            self.apply_operation_reordering(graph)?;
        }

        // Verify graph integrity after optimization
        graph.verify_integrity()?;

        // Collect final statistics
        let nodes_after = graph.nodes.len();
        let memory_after = graph.calculate_memory_usage();
        let compute_cost_after = graph.calculate_compute_cost();
        let optimization_time_us = start_time.elapsed().as_micros() as u64;

        Ok(OptimizationStats {
            nodes_before,
            nodes_after,
            memory_before,
            memory_after,
            compute_cost_before,
            compute_cost_after,
            fused_chains,
            optimization_time_us,
        })
    }

    /// Apply operation fusion optimization
    fn apply_operation_fusion(&self, graph: &mut ExpressionGraph) -> Result<usize> {
        let fusable_chains = graph.detect_fusable_chains();
        let mut total_fused = 0;

        for chain in fusable_chains {
            if chain.len() > 1 {
                // Create a fused operation to replace the chain
                let fused_id = graph.add_node(OperationType::Custom("fused".to_string()));

                // Connect inputs and outputs properly
                // This is a simplified version - in practice, you'd implement
                // proper kernel fusion logic here

                total_fused += 1;
            }
        }

        Ok(total_fused)
    }

    /// Apply constant folding optimization
    fn apply_constant_folding(&self, _graph: &mut ExpressionGraph) -> Result<()> {
        // Implement constant folding logic
        // For now, this is a placeholder
        Ok(())
    }

    /// Apply common subexpression elimination
    fn apply_common_subexpression_elimination(&self, _graph: &mut ExpressionGraph) -> Result<()> {
        // Implement CSE logic
        // For now, this is a placeholder
        Ok(())
    }

    /// Apply memory optimization
    fn apply_memory_optimization(&self, _graph: &mut ExpressionGraph) -> Result<()> {
        // Implement memory optimization logic
        // For now, this is a placeholder
        Ok(())
    }

    /// Apply operation reordering optimization
    fn apply_operation_reordering(&self, _graph: &mut ExpressionGraph) -> Result<()> {
        // Implement operation reordering logic
        // For now, this is a placeholder
        Ok(())
    }
}

impl Default for ExpressionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait to add expression optimization to tensors
pub trait TensorExpressionOps<T: TensorElement> {
    /// Build an expression graph from tensor operations
    fn build_expression_graph(&self) -> ExpressionGraph;

    /// Optimize tensor expressions using the expression optimizer
    fn optimize_expressions(&self, config: OptimizerConfig) -> Result<OptimizationStats>;
}

impl<T: TensorElement> TensorExpressionOps<T> for Tensor<T> {
    fn build_expression_graph(&self) -> ExpressionGraph {
        // This would build a graph from the tensor's computation history
        // For now, return an empty graph as placeholder
        ExpressionGraph::new()
    }

    fn optimize_expressions(&self, config: OptimizerConfig) -> Result<OptimizationStats> {
        let optimizer = ExpressionOptimizer::with_config(config);
        let mut graph = self.build_expression_graph();
        optimizer.optimize(&mut graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_operation_properties() {
        let add_props = OperationType::Add.properties();
        assert!(add_props.is_elementwise);
        assert!(add_props.is_commutative);
        assert!(add_props.is_associative);
        assert!(add_props.fusable);

        let matmul_props = OperationType::MatMul.properties();
        assert!(!matmul_props.is_elementwise);
        assert!(!matmul_props.is_commutative);
        assert!(matmul_props.is_associative);
        assert!(!matmul_props.fusable);
    }

    #[test]
    fn test_expression_graph_creation() {
        let mut graph = ExpressionGraph::new();

        let node1 = graph.add_node(OperationType::Add);
        let node2 = graph.add_node(OperationType::Mul);
        let node3 = graph.add_node(OperationType::Sum);

        graph.add_edge(node1, node3).unwrap();
        graph.add_edge(node2, node3).unwrap();

        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.get_node(node3).unwrap().inputs.len(), 2);
        assert!(graph.verify_integrity().is_ok());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ExpressionGraph::new();

        let a = graph.add_node(OperationType::Add);
        let b = graph.add_node(OperationType::Mul);
        let c = graph.add_node(OperationType::Sum);

        graph.add_edge(a, c).unwrap();
        graph.add_edge(b, c).unwrap();

        let sorted = graph.topological_sort().unwrap();

        // c should come after both a and b
        let pos_a = sorted.iter().position(|&x| x == a).unwrap();
        let pos_b = sorted.iter().position(|&x| x == b).unwrap();
        let pos_c = sorted.iter().position(|&x| x == c).unwrap();

        assert!(pos_c > pos_a);
        assert!(pos_c > pos_b);
    }

    #[test]
    fn test_fusable_chain_detection() {
        let mut graph = ExpressionGraph::new();

        let a = graph.add_node(OperationType::Add);
        let b = graph.add_node(OperationType::Mul);
        let c = graph.add_node(OperationType::Relu);

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();

        let chains = graph.detect_fusable_chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 3);
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizerConfig {
            strategy: OptimizationStrategy::MinimizeMemory,
            enable_fusion: true,
            enable_memory_optimization: true,
            aggressiveness: 0.8,
            ..Default::default()
        };

        assert_eq!(config.strategy, OptimizationStrategy::MinimizeMemory);
        assert_eq!(config.aggressiveness, 0.8);
    }

    #[test]
    fn test_expression_optimizer() {
        let mut graph = ExpressionGraph::new();

        let a = graph.add_node(OperationType::Add);
        let b = graph.add_node(OperationType::Mul);
        graph.add_edge(a, b).unwrap();

        let optimizer = ExpressionOptimizer::new();
        let stats = optimizer.optimize(&mut graph).unwrap();

        assert!(stats.optimization_time_us > 0);
        assert_eq!(stats.nodes_before, 2);
    }

    #[test]
    fn test_optimization_stats_display() {
        let stats = OptimizationStats {
            nodes_before: 10,
            nodes_after: 8,
            memory_before: 1000,
            memory_after: 800,
            compute_cost_before: 10.0,
            compute_cost_after: 8.0,
            fused_chains: 2,
            optimization_time_us: 1500,
        };

        assert_eq!(stats.node_reduction(), 20.0);
        assert_eq!(stats.memory_reduction(), 20.0);
        assert_eq!(stats.compute_reduction(), 20.0);

        let display = format!("{}", stats);
        assert!(display.contains("20.0% reduction"));
    }

    #[test]
    fn test_node_fusability() {
        let node1 = ExpressionNode::new(NodeId(1), OperationType::Add);
        let node2 = ExpressionNode::new(NodeId(2), OperationType::Mul);
        let node3 = ExpressionNode::new(NodeId(3), OperationType::MatMul);

        assert!(node1.is_fusable_with(&node2)); // Both element-wise
        assert!(!node1.is_fusable_with(&node3)); // MatMul is not fusable
    }
}
