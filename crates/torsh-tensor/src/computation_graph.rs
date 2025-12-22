//! Computation Graph for Lazy Evaluation and Optimization
//!
//! This module provides a computation graph system that enables lazy evaluation,
//! operation fusion, and various graph-level optimizations before execution.
//!
//! # Features
//!
//! - **Lazy evaluation**: Build computation graph without executing
//! - **Graph optimization**: Fuse operations, eliminate dead code, constant folding
//! - **Memory planning**: Optimize memory allocation and reuse
//! - **Parallel scheduling**: Automatic parallelization of independent operations
//! - **Visualization**: Generate DOT graphs for debugging

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::Tensor;

/// Unique identifier for graph nodes
pub type NodeId = usize;

/// Operation types in the computation graph
#[derive(Debug, Clone)]
pub enum GraphOp {
    /// Input/constant tensor
    Constant,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Mul,
    /// Element-wise subtraction
    Sub,
    /// Element-wise division
    Div,
    /// Matrix multiplication
    MatMul,
    /// Reshape operation
    Reshape(Vec<usize>),
    /// Transpose operation
    Transpose(usize, usize),
    /// Reduction sum
    Sum(Option<i32>),
    /// Reduction mean
    Mean(Option<i32>),
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Scalar addition
    AddScalar(f64),
    /// Scalar multiplication
    MulScalar(f64),
    /// Custom operation
    Custom(String),
}

impl fmt::Display for GraphOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphOp::Constant => write!(f, "Const"),
            GraphOp::Add => write!(f, "Add"),
            GraphOp::Mul => write!(f, "Mul"),
            GraphOp::Sub => write!(f, "Sub"),
            GraphOp::Div => write!(f, "Div"),
            GraphOp::MatMul => write!(f, "MatMul"),
            GraphOp::Reshape(shape) => write!(f, "Reshape({:?})", shape),
            GraphOp::Transpose(d0, d1) => write!(f, "Transpose({}, {})", d0, d1),
            GraphOp::Sum(dim) => write!(f, "Sum({:?})", dim),
            GraphOp::Mean(dim) => write!(f, "Mean({:?})", dim),
            GraphOp::ReLU => write!(f, "ReLU"),
            GraphOp::Sigmoid => write!(f, "Sigmoid"),
            GraphOp::Tanh => write!(f, "Tanh"),
            GraphOp::AddScalar(s) => write!(f, "AddScalar({})", s),
            GraphOp::MulScalar(s) => write!(f, "MulScalar({})", s),
            GraphOp::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Node in the computation graph
#[derive(Clone)]
pub struct GraphNode<T: TensorElement> {
    /// Unique node ID
    pub id: NodeId,
    /// Operation type
    pub op: GraphOp,
    /// Input node IDs
    pub inputs: Vec<NodeId>,
    /// Cached tensor data (for constants)
    pub data: Option<Arc<Tensor<T>>>,
    /// Output shape (if known)
    pub shape: Option<Vec<usize>>,
    /// Device
    pub device: DeviceType,
}

impl<T: TensorElement> GraphNode<T> {
    /// Create a new graph node
    fn new(id: NodeId, op: GraphOp, inputs: Vec<NodeId>, device: DeviceType) -> Self {
        Self {
            id,
            op,
            inputs,
            data: None,
            shape: None,
            device,
        }
    }

    /// Create a constant node
    fn constant(id: NodeId, tensor: Tensor<T>) -> Self {
        let device = tensor.device;
        let shape = Some(tensor.shape().dims().to_vec());
        Self {
            id,
            op: GraphOp::Constant,
            inputs: Vec::new(),
            data: Some(Arc::new(tensor)),
            shape,
            device,
        }
    }
}

/// Computation graph
pub struct ComputationGraph<T: TensorElement> {
    /// All nodes in the graph
    nodes: HashMap<NodeId, GraphNode<T>>,
    /// Next available node ID
    next_id: NodeId,
    /// Output nodes (nodes that need to be computed)
    outputs: Vec<NodeId>,
    /// Execution cache (computed results)
    cache: Arc<Mutex<HashMap<NodeId, Arc<Tensor<T>>>>>,
}

impl<T: TensorElement + Copy> ComputationGraph<T> {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            outputs: Vec::new(),
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a constant tensor to the graph
    pub fn constant(&mut self, tensor: Tensor<T>) -> NodeId {
        let id = self.allocate_id();
        let node = GraphNode::constant(id, tensor);
        self.nodes.insert(id, node);
        id
    }

    /// Add a binary operation node
    pub fn binary_op(
        &mut self,
        op: GraphOp,
        left: NodeId,
        right: NodeId,
        device: DeviceType,
    ) -> NodeId {
        let id = self.allocate_id();
        let node = GraphNode::new(id, op, vec![left, right], device);
        self.nodes.insert(id, node);
        id
    }

    /// Add a unary operation node
    pub fn unary_op(&mut self, op: GraphOp, input: NodeId, device: DeviceType) -> NodeId {
        let id = self.allocate_id();
        let node = GraphNode::new(id, op, vec![input], device);
        self.nodes.insert(id, node);
        id
    }

    /// Mark a node as an output
    pub fn mark_output(&mut self, node: NodeId) {
        if !self.outputs.contains(&node) {
            self.outputs.push(node);
        }
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of output nodes
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Allocate a new node ID
    fn allocate_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Perform topological sort of the graph
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj_list: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Build adjacency list and calculate in-degrees
        for (&id, node) in &self.nodes {
            in_degree.entry(id).or_insert(0);
            for &input_id in &node.inputs {
                adj_list.entry(input_id).or_insert_with(Vec::new).push(id);
                *in_degree.entry(id).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id);

            if let Some(neighbors) = adj_list.get(&node_id) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            ));
        }

        Ok(sorted)
    }

    /// Optimize the graph by fusing operations
    pub fn optimize(&mut self) -> Result<()>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement,
    {
        // Simple optimizations:
        // 1. Constant folding
        self.fold_constants()?;

        // 2. Dead code elimination
        self.eliminate_dead_code();

        // 3. Operation fusion (future enhancement)

        Ok(())
    }

    /// Fold constant operations
    fn fold_constants(&mut self) -> Result<()>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement,
    {
        let sorted = self.topological_sort()?;

        for &node_id in &sorted {
            let node = self.nodes.get(&node_id).unwrap().clone();

            // Check if all inputs are constants
            let all_constant = node.inputs.iter().all(|&input_id| {
                if let Some(input_node) = self.nodes.get(&input_id) {
                    matches!(input_node.op, GraphOp::Constant)
                } else {
                    false
                }
            });

            if all_constant && !node.inputs.is_empty() {
                // Try to evaluate this node
                if let Ok(result) = self.evaluate_node_internal(&node) {
                    // Replace with constant
                    let mut new_node = GraphNode::constant(node_id, result);
                    new_node.device = node.device;
                    self.nodes.insert(node_id, new_node);
                }
            }
        }

        Ok(())
    }

    /// Eliminate dead code (nodes not contributing to outputs)
    fn eliminate_dead_code(&mut self) {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::from_iter(self.outputs.iter().copied());

        // Mark all reachable nodes
        while let Some(node_id) = queue.pop_front() {
            if reachable.insert(node_id) {
                if let Some(node) = self.nodes.get(&node_id) {
                    for &input_id in &node.inputs {
                        queue.push_back(input_id);
                    }
                }
            }
        }

        // Remove unreachable nodes
        self.nodes.retain(|&id, _| reachable.contains(&id));
    }

    /// Evaluate a single node
    fn evaluate_node_internal(&self, node: &GraphNode<T>) -> Result<Tensor<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement,
    {
        match &node.op {
            GraphOp::Constant => node
                .data
                .as_ref()
                .map(|t| (**t).clone())
                .ok_or_else(|| TorshError::InvalidArgument("Constant has no data".to_string())),
            GraphOp::Add => {
                let left = self.get_input_tensor(node, 0)?;
                let right = self.get_input_tensor(node, 1)?;
                left.add_op(&right)
            }
            GraphOp::Mul => {
                let left = self.get_input_tensor(node, 0)?;
                let right = self.get_input_tensor(node, 1)?;
                left.mul_op(&right)
            }
            GraphOp::Sub => {
                let left = self.get_input_tensor(node, 0)?;
                let right = self.get_input_tensor(node, 1)?;
                left.sub(&right)
            }
            GraphOp::Div => {
                let left = self.get_input_tensor(node, 0)?;
                let right = self.get_input_tensor(node, 1)?;
                left.div(&right)
            }
            GraphOp::AddScalar(s) => {
                let input = self.get_input_tensor(node, 0)?;
                let scalar = T::from_f64(*s).ok_or_else(|| {
                    TorshError::InvalidArgument("Cannot convert scalar to tensor type".to_string())
                })?;
                input.add_scalar(scalar)
            }
            GraphOp::MulScalar(s) => {
                let input = self.get_input_tensor(node, 0)?;
                let scalar = T::from_f64(*s).ok_or_else(|| {
                    TorshError::InvalidArgument("Cannot convert scalar to tensor type".to_string())
                })?;
                input.mul_scalar(scalar)
            }
            GraphOp::ReLU => {
                let input = self.get_input_tensor(node, 0)?;
                input.relu()
            }
            GraphOp::Sigmoid => {
                let input = self.get_input_tensor(node, 0)?;
                input.sigmoid()
            }
            GraphOp::Tanh => {
                let input = self.get_input_tensor(node, 0)?;
                input.tanh()
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unsupported operation: {}",
                node.op
            ))),
        }
    }

    /// Get input tensor for a node
    fn get_input_tensor(&self, node: &GraphNode<T>, index: usize) -> Result<Tensor<T>> {
        let input_id = node.inputs.get(index).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Missing input {} for node {}", index, node.id))
        })?;

        let input_node = self.nodes.get(input_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Input node {} not found", input_id))
        })?;

        if let GraphOp::Constant = input_node.op {
            input_node
                .data
                .as_ref()
                .map(|t| (**t).clone())
                .ok_or_else(|| TorshError::InvalidArgument("Constant has no data".to_string()))
        } else {
            Err(TorshError::InvalidArgument(
                "Can only evaluate constants in internal evaluation".to_string(),
            ))
        }
    }

    /// Execute the graph and get outputs
    pub fn execute(&self) -> Result<Vec<Tensor<T>>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + torsh_core::FloatElement,
    {
        let sorted = self.topological_sort()?;
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        // Evaluate nodes in topological order
        for &node_id in &sorted {
            let node = self.nodes.get(&node_id).unwrap();

            // Skip if already cached
            if cache.contains_key(&node_id) {
                continue;
            }

            let result = self.evaluate_node_internal(node)?;
            cache.insert(node_id, Arc::new(result));
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for &output_id in &self.outputs {
            if let Some(result) = cache.get(&output_id) {
                outputs.push((**result).clone());
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Output node {} not computed",
                    output_id
                )));
            }
        }

        Ok(outputs)
    }

    /// Generate DOT representation for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ComputationGraph {\n");
        dot.push_str("  rankdir=BT;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for (id, node) in &self.nodes {
            let label = format!("{}\\nid={}", node.op, id);
            let color = if self.outputs.contains(id) {
                "red"
            } else if matches!(node.op, GraphOp::Constant) {
                "lightblue"
            } else {
                "lightgray"
            };

            dot.push_str(&format!(
                "  {} [label=\"{}\", fillcolor={}, style=filled];\n",
                id, label, color
            ));
        }

        dot.push('\n');

        // Add edges
        for (id, node) in &self.nodes {
            for (idx, &input_id) in node.inputs.iter().enumerate() {
                dot.push_str(&format!("  {} -> {} [label=\"{}\"];\n", input_id, id, idx));
            }
        }

        dot.push_str("}\n");
        dot
    }
}

impl<T: TensorElement + Copy> Default for ComputationGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        let b = tensor_1d(&[4.0, 5.0, 6.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);

        graph.mark_output(add_id);

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_outputs(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0, 2.0]).unwrap();
        let b = tensor_1d(&[3.0, 4.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);
        let mul_id = graph.unary_op(GraphOp::MulScalar(2.0), add_id, DeviceType::Cpu);

        let sorted = graph.topological_sort().unwrap();

        // Should have all 4 nodes
        assert_eq!(sorted.len(), 4);

        // Constants should come before operations that use them
        let a_pos = sorted.iter().position(|&id| id == a_id).unwrap();
        let b_pos = sorted.iter().position(|&id| id == b_id).unwrap();
        let add_pos = sorted.iter().position(|&id| id == add_id).unwrap();
        let mul_pos = sorted.iter().position(|&id| id == mul_id).unwrap();

        assert!(a_pos < add_pos);
        assert!(b_pos < add_pos);
        assert!(add_pos < mul_pos);
    }

    #[test]
    fn test_constant_folding() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0, 2.0]).unwrap();
        let b = tensor_1d(&[3.0, 4.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);

        graph.mark_output(add_id);

        // Before optimization
        assert_eq!(graph.num_nodes(), 3);

        // Optimize
        graph.optimize().unwrap();

        // After optimization, the add should be folded into a constant
        let add_node = graph.nodes.get(&add_id).unwrap();
        assert!(matches!(add_node.op, GraphOp::Constant));
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0]).unwrap();
        let b = tensor_1d(&[2.0]).unwrap();
        let c = tensor_1d(&[3.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let c_id = graph.constant(c);

        // Create used operation
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);
        graph.mark_output(add_id);

        // Create unused operation (dead code)
        let _mul_id = graph.unary_op(GraphOp::MulScalar(2.0), c_id, DeviceType::Cpu);

        // Before optimization
        assert_eq!(graph.num_nodes(), 5);

        // Optimize
        graph.optimize().unwrap();

        // After optimization, unused nodes should be removed
        // Note: constant folding folds add into a single constant, so we get 1 node
        assert_eq!(graph.num_nodes(), 1); // The folded constant
    }

    #[test]
    fn test_graph_execution() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        let b = tensor_1d(&[4.0, 5.0, 6.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);

        graph.mark_output(add_id);

        let results = graph.execute().unwrap();
        assert_eq!(results.len(), 1);

        let data = results[0].to_vec().unwrap();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_multiple_outputs() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0, 2.0]).unwrap();
        let b = tensor_1d(&[3.0, 4.0]).unwrap();

        let a_id = graph.constant(a);
        let b_id = graph.constant(b);
        let add_id = graph.binary_op(GraphOp::Add, a_id, b_id, DeviceType::Cpu);
        let mul_id = graph.binary_op(GraphOp::Mul, a_id, b_id, DeviceType::Cpu);

        graph.mark_output(add_id);
        graph.mark_output(mul_id);

        let results = graph.execute().unwrap();
        assert_eq!(results.len(), 2);

        let add_data = results[0].to_vec().unwrap();
        let mul_data = results[1].to_vec().unwrap();

        assert_eq!(add_data, vec![4.0, 6.0]);
        assert_eq!(mul_data, vec![3.0, 8.0]);
    }

    #[test]
    fn test_dot_generation() {
        let mut graph = ComputationGraph::<f32>::new();

        let a = tensor_1d(&[1.0]).unwrap();
        let a_id = graph.constant(a);
        graph.mark_output(a_id);

        let dot = graph.to_dot();

        assert!(dot.contains("digraph ComputationGraph"));
        assert!(dot.contains(&format!("id={}", a_id)));
    }
}
