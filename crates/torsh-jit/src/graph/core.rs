//! Core graph representation structures

use crate::{JitError, JitResult};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{DType, DeviceType, Shape};

pub use crate::graph::metadata::GraphMetadata;
pub use crate::graph::operations::Operation;

pub type NodeId = NodeIndex;

/// Edge in the computation graph representing data flow between nodes
#[derive(Debug, Clone, Default)]
pub struct Edge {
    /// Output index of the source node
    pub src_output: usize,
    /// Input index of the destination node
    pub dst_input: usize,
}

/// Serializable wrapper for NodeIndex
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SerializableNodeIndex(pub u32);

impl From<NodeIndex> for SerializableNodeIndex {
    fn from(node_index: NodeIndex) -> Self {
        SerializableNodeIndex(node_index.index() as u32)
    }
}

impl From<SerializableNodeIndex> for NodeIndex {
    fn from(serializable: SerializableNodeIndex) -> Self {
        NodeIndex::new(serializable.0 as usize)
    }
}

/// A node in the computation graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Operation type
    pub operation: Operation,

    /// Node name/id
    pub name: String,

    /// Input shapes
    pub input_shapes: Vec<Option<Shape>>,

    /// Output shapes
    pub output_shapes: Vec<Option<Shape>>,

    /// Data types for outputs
    pub dtypes: Vec<DType>,

    /// Device information
    pub device: DeviceType,

    /// Additional attributes
    pub attributes: HashMap<String, crate::graph::operations::Attribute>,

    // Compatibility fields for existing code
    /// Operation alias for compatibility
    pub op: Operation,

    /// Single dtype for compatibility (first dtype from dtypes vec)
    pub dtype: DType,

    /// Single output shape for compatibility (first shape from output_shapes vec)
    pub output_shape: Shape,

    /// Attributes alias for compatibility
    pub attrs: HashMap<String, crate::graph::operations::Attribute>,

    /// Input connections (placeholder for compatibility)
    pub inputs: Vec<NodeId>,

    /// Whether this is an output node (placeholder for compatibility)
    pub is_output: bool,
}

impl Node {
    /// Create a new node with the given operation
    pub fn new(operation: Operation, name: String) -> Self {
        let op = operation.clone();
        let dtype = DType::F32; // Default dtype
        let output_shape = Shape::new(vec![1]); // Default shape
        let attributes = HashMap::new();

        Self {
            operation,
            name,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            dtypes: Vec::new(),
            device: DeviceType::Cpu,
            attributes: attributes.clone(),

            // Compatibility fields
            op,
            dtype,
            output_shape,
            attrs: attributes,
            inputs: Vec::new(),
            is_output: false,
        }
    }

    /// Set input shapes
    pub fn with_input_shapes(mut self, shapes: Vec<Option<Shape>>) -> Self {
        self.input_shapes = shapes;
        self.sync_compatibility_fields();
        self
    }

    /// Set output shapes
    pub fn with_output_shapes(mut self, shapes: Vec<Option<Shape>>) -> Self {
        self.output_shapes = shapes;
        self.sync_compatibility_fields();
        self
    }

    /// Set data types
    pub fn with_dtypes(mut self, dtypes: Vec<DType>) -> Self {
        self.dtypes = dtypes;
        self.sync_compatibility_fields();
        self
    }

    /// Set device
    pub fn with_device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Add an attribute
    pub fn with_attribute(
        mut self,
        key: String,
        value: crate::graph::operations::Attribute,
    ) -> Self {
        self.attributes.insert(key, value);
        self.sync_compatibility_fields();
        self
    }

    /// Get the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.input_shapes.len()
    }

    /// Get the number of outputs
    pub fn num_outputs(&self) -> usize {
        self.output_shapes.len().max(1) // At least one output
    }

    /// Get input shape at index
    pub fn input_shape(&self, index: usize) -> Option<&Shape> {
        self.input_shapes.get(index).and_then(|s| s.as_ref())
    }

    /// Get output shape at index
    pub fn output_shape(&self, index: usize) -> Option<&Shape> {
        self.output_shapes.get(index).and_then(|s| s.as_ref())
    }

    /// Get data type at output index
    pub fn dtype(&self, index: usize) -> Option<&DType> {
        self.dtypes.get(index)
    }

    /// Check if this is an input node
    pub fn is_input(&self) -> bool {
        matches!(self.operation, Operation::Input | Operation::Parameter(_))
    }

    /// Check if this is a constant node
    pub fn is_constant(&self) -> bool {
        matches!(self.operation, Operation::Constant(_))
    }

    /// Check if this is a control flow node
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self.operation,
            Operation::If(_)
                | Operation::While(_)
                | Operation::For(_)
                | Operation::Break
                | Operation::Continue
                | Operation::Return(_)
                | Operation::Block(_)
                | Operation::Merge(_)
        )
    }

    /// Get memory estimate in bytes
    pub fn memory_estimate(&self) -> usize {
        let mut total = 0;
        for shape_opt in &self.output_shapes {
            if let Some(shape) = shape_opt {
                let elements = shape.dims().iter().product::<usize>();
                // Assume each element is at least 4 bytes
                total += elements * 4;
            }
        }
        total
    }

    /// Get computational complexity estimate (FLOPs)
    pub fn complexity_estimate(&self) -> usize {
        match &self.operation {
            Operation::MatMul | Operation::BatchMatMul => {
                if self.input_shapes.len() >= 2 {
                    if let (Some(Some(a_shape)), Some(Some(b_shape))) =
                        (self.input_shapes.get(0), self.input_shapes.get(1))
                    {
                        // Matrix multiplication complexity: 2 * m * n * k
                        if a_shape.dims().len() >= 2 && b_shape.dims().len() >= 2 {
                            let m = a_shape.dims()[a_shape.dims().len() - 2];
                            let k = a_shape.dims()[a_shape.dims().len() - 1];
                            let n = b_shape.dims()[b_shape.dims().len() - 1];
                            return 2 * m * n * k;
                        }
                    }
                }
                0
            }
            Operation::Conv2d(_) => {
                // Simplified convolution complexity estimation
                if let Some(Some(output_shape)) = self.output_shapes.get(0) {
                    output_shape.dims().iter().product::<usize>() * 9 // 3x3 kernel approximation
                } else {
                    0
                }
            }
            _ => {
                // For other operations, estimate based on output size
                if let Some(Some(output_shape)) = self.output_shapes.get(0) {
                    output_shape.dims().iter().product::<usize>()
                } else {
                    1
                }
            }
        }
    }

    /// Synchronize compatibility fields with main fields
    pub fn sync_compatibility_fields(&mut self) {
        self.op = self.operation.clone();
        self.dtype = self.dtypes.first().copied().unwrap_or(DType::F32);
        self.output_shape = self
            .output_shapes
            .first()
            .and_then(|s| s.as_ref())
            .cloned()
            .unwrap_or_else(|| Shape::new(vec![1]));
        self.attrs = self.attributes.clone();
    }

    /// Set attribute (compatibility method)
    pub fn set_attribute(&mut self, key: String, value: crate::graph::operations::Attribute) {
        self.attributes.insert(key.clone(), value.clone());
        self.attrs.insert(key, value);
    }

    /// Set optimization hint (compatibility method)
    pub fn set_optimization_hint(&mut self, hint: &str, value: &str) -> crate::JitResult<()> {
        let attr_value = crate::graph::operations::Attribute::String(value.to_string());
        self.set_attribute(hint.to_string(), attr_value);
        Ok(())
    }

    /// Get attribute (compatibility method)
    pub fn get_attribute(&self, key: &str) -> Option<&crate::graph::operations::Attribute> {
        self.attributes.get(key)
    }

    /// Get operation type (compatibility method)
    pub fn operation_type(&self) -> &str {
        self.operation.as_str()
    }

    /// Check if node has side effects (compatibility method)
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self.operation,
            Operation::Custom(_) | Operation::Break | Operation::Continue | Operation::Return(_)
        )
    }

    /// Get operation category for optimization purposes
    pub fn operation_category(&self) -> OperationCategory {
        match &self.operation {
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Neg
            | Operation::Abs
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Sin
            | Operation::Cos
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Relu
            | Operation::Gelu
            | Operation::Silu => OperationCategory::ElementWise,
            Operation::MatMul | Operation::BatchMatMul => OperationCategory::LinearAlgebra,
            Operation::Conv2d(_) | Operation::Linear(_) => OperationCategory::NeuralNetwork,
            Operation::Sum { .. }
            | Operation::Mean { .. }
            | Operation::Max { .. }
            | Operation::Min { .. } => OperationCategory::Reduction,
            Operation::Reshape { .. }
            | Operation::Transpose { .. }
            | Operation::Squeeze { .. }
            | Operation::Unsqueeze { .. }
            | Operation::Slice { .. }
            | Operation::Concat { .. } => OperationCategory::ShapeManipulation,
            Operation::If(_)
            | Operation::While(_)
            | Operation::For(_)
            | Operation::Break
            | Operation::Continue
            | Operation::Return(_)
            | Operation::Block(_)
            | Operation::Merge(_) => OperationCategory::ControlFlow,
            Operation::Input | Operation::Parameter(_) | Operation::Constant(_) => {
                OperationCategory::Input
            }
            _ => OperationCategory::Other,
        }
    }

    /// Check if this operation can be vectorized using SIMD instructions
    pub fn is_vectorizable(&self) -> bool {
        match &self.operation {
            // Element-wise operations are highly vectorizable
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Neg
            | Operation::Abs
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Sin
            | Operation::Cos
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Relu
            | Operation::Gelu
            | Operation::Silu => true,
            // Matrix operations can benefit from vectorization
            Operation::MatMul | Operation::BatchMatMul => true,
            // Reduction operations can be vectorized
            Operation::Sum { .. }
            | Operation::Mean { .. }
            | Operation::Max { .. }
            | Operation::Min { .. } => true,
            // Convolutions are vectorizable
            Operation::Conv2d(_) => true,
            // Other operations are typically not vectorizable
            _ => false,
        }
    }

    /// Check if this operation accesses memory (for cache optimization)
    pub fn has_memory_access(&self) -> bool {
        match &self.operation {
            // Operations that don't access external memory
            Operation::Input | Operation::Parameter(_) | Operation::Constant(_) => false,
            // Control flow operations typically don't access memory directly
            Operation::Break | Operation::Continue | Operation::Return(_) => false,
            // All computation operations access memory
            _ => true,
        }
    }

    /// Estimate the working set size (bytes) for memory access patterns
    pub fn estimate_working_set_size(&self) -> usize {
        let mut working_set = 0;

        // Input working set (data being read)
        for shape_opt in &self.input_shapes {
            if let Some(shape) = shape_opt {
                let elements = shape.dims().iter().product::<usize>();
                // Assume each element is at least 4 bytes (f32)
                working_set += elements * 4;
            }
        }

        // Output working set (data being written)
        for shape_opt in &self.output_shapes {
            if let Some(shape) = shape_opt {
                let elements = shape.dims().iter().product::<usize>();
                working_set += elements * 4;
            }
        }

        // Operation-specific working set adjustments
        match &self.operation {
            Operation::MatMul | Operation::BatchMatMul => {
                // Matrix multiplication has intermediate results
                working_set * 2
            }
            Operation::Conv2d(_) => {
                // Convolution may need workspace for im2col
                working_set * 3
            }
            _ => working_set,
        }
    }
}

/// Categories of operations for optimization and analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationCategory {
    ElementWise,
    LinearAlgebra,
    NeuralNetwork,
    Reduction,
    ShapeManipulation,
    ControlFlow,
    Input,
    Other,
}

/// Computation graph representing a neural network or computation
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// Internal graph representation
    pub(crate) graph: DiGraph<Node, Edge>,

    /// Input nodes
    pub inputs: Vec<NodeId>,

    /// Output nodes
    pub outputs: Vec<NodeId>,

    /// Metadata
    pub metadata: GraphMetadata,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: GraphMetadata::default(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> NodeId {
        self.graph.add_node(node)
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, edge: Edge) {
        self.graph.add_edge(from, to, edge);
    }

    /// Mark a node as input
    pub fn add_input(&mut self, node: NodeId) {
        if !self.inputs.contains(&node) {
            self.inputs.push(node);
        }
    }

    /// Mark a node as output
    pub fn add_output(&mut self, node: NodeId) {
        if !self.outputs.contains(&node) {
            self.outputs.push(node);
        }
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.graph
            .node_indices()
            .map(move |idx| (idx, &self.graph[idx]))
    }

    /// Get all edges
    pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &Edge)> + '_ {
        self.graph.edge_indices().map(move |idx| {
            let (src, dst) = self
                .graph
                .edge_endpoints(idx)
                .expect("edge index should be valid");
            (src, dst, &self.graph[idx])
        })
    }

    /// Get node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.graph.node_weight(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.graph.node_weight_mut(id)
    }

    /// Get node inputs
    pub fn get_node_inputs(&self, id: NodeId) -> Vec<NodeId> {
        self.graph
            .neighbors_directed(id, Direction::Incoming)
            .collect()
    }

    /// Get node outputs
    pub fn get_node_outputs(&self, id: NodeId) -> Vec<NodeId> {
        self.graph
            .neighbors_directed(id, Direction::Outgoing)
            .collect()
    }

    /// Get incoming edges for a node
    pub fn incoming_edges(&self, id: NodeId) -> Vec<(NodeId, NodeId, &Edge)> {
        self.graph
            .edges_directed(id, Direction::Incoming)
            .map(|edge_ref| (edge_ref.source(), edge_ref.target(), edge_ref.weight()))
            .collect()
    }

    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, id: NodeId) -> Vec<(NodeId, NodeId, &Edge)> {
        self.graph
            .edges_directed(id, Direction::Outgoing)
            .map(|edge_ref| (edge_ref.source(), edge_ref.target(), edge_ref.weight()))
            .collect()
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, id: NodeId) -> Option<Node> {
        // Remove from inputs/outputs
        self.inputs.retain(|&x| x != id);
        self.outputs.retain(|&x| x != id);

        self.graph.remove_node(id)
    }

    /// Remove an edge from the graph
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) -> bool {
        if let Some(edge_id) = self.graph.find_edge(from, to) {
            self.graph.remove_edge(edge_id).is_some()
        } else {
            false
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Validate the graph structure
    pub fn validate(&self) -> JitResult<()> {
        // Check that all input/output node IDs exist
        for &input_id in &self.inputs {
            if self.graph.node_weight(input_id).is_none() {
                return Err(JitError::GraphError(format!(
                    "Input node {:?} does not exist in graph",
                    input_id
                )));
            }
        }

        for &output_id in &self.outputs {
            if self.graph.node_weight(output_id).is_none() {
                return Err(JitError::GraphError(format!(
                    "Output node {:?} does not exist in graph",
                    output_id
                )));
            }
        }

        // Check for cycles in non-control-flow subgraph
        self.validate_acyclic()?;

        Ok(())
    }

    /// Check that the graph is acyclic (ignoring control flow edges)
    fn validate_acyclic(&self) -> JitResult<()> {
        use petgraph::algo::is_cyclic_directed;

        if is_cyclic_directed(&self.graph) {
            return Err(JitError::GraphError("Graph contains cycles".to_string()));
        }

        Ok(())
    }

    /// Get topological ordering of nodes
    pub fn topological_sort(&self) -> JitResult<Vec<NodeId>> {
        use petgraph::algo::toposort;

        toposort(&self.graph, None)
            .map_err(|_| JitError::GraphError("Graph contains cycles".to_string()))
    }

    /// Clone with only specified nodes
    pub fn subgraph(&self, node_ids: &[NodeId]) -> JitResult<ComputationGraph> {
        let mut new_graph = ComputationGraph::new();
        let mut node_mapping = HashMap::new();

        // Add nodes
        for &node_id in node_ids {
            if let Some(node) = self.get_node(node_id) {
                let new_id = new_graph.add_node(node.clone());
                node_mapping.insert(node_id, new_id);
            } else {
                return Err(JitError::GraphError(format!(
                    "Node {:?} not found in original graph",
                    node_id
                )));
            }
        }

        // Add edges between included nodes
        for &src_id in node_ids {
            for &dst_id in node_ids {
                if let Some(edge_ref) = self.graph.find_edge(src_id, dst_id) {
                    let edge = self.graph.edge_weight(edge_ref).expect("edge should exist");
                    let new_src = node_mapping[&src_id];
                    let new_dst = node_mapping[&dst_id];
                    new_graph.add_edge(new_src, new_dst, edge.clone());
                }
            }
        }

        // Update inputs/outputs
        for &input_id in &self.inputs {
            if let Some(&new_id) = node_mapping.get(&input_id) {
                new_graph.add_input(new_id);
            }
        }

        for &output_id in &self.outputs {
            if let Some(&new_id) = node_mapping.get(&output_id) {
                new_graph.add_output(new_id);
            }
        }

        new_graph.metadata = self.metadata.clone();

        Ok(new_graph)
    }

    /// Get strongly connected components
    pub fn strongly_connected_components(&self) -> Vec<Vec<NodeId>> {
        use petgraph::algo::tarjan_scc;
        tarjan_scc(&self.graph)
    }

    /// Get memory usage estimate in bytes
    pub fn memory_estimate(&self) -> usize {
        self.graph
            .node_weights()
            .map(|node| node.memory_estimate())
            .sum()
    }

    /// Get computational complexity estimate (FLOPs)
    pub fn complexity_estimate(&self) -> usize {
        self.graph
            .node_weights()
            .map(|node| node.complexity_estimate())
            .sum()
    }

    /// Get predecessors of a node (compatibility method)
    pub fn predecessors(&self, node_id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.neighbors_directed(node_id, Direction::Incoming)
    }

    /// Get successors of a node (compatibility method)
    pub fn successors(&self, node_id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.neighbors_directed(node_id, Direction::Outgoing)
    }

    /// Get node by ID (compatibility method)
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.get_node(id)
    }

    /// Get mutable node by ID (compatibility method)
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.get_node_mut(id)
    }

    /// Get directed edges for a node (compatibility method)
    pub fn edges_directed(
        &self,
        node_id: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = petgraph::graph::EdgeReference<'_, Edge>> {
        self.graph.edges_directed(node_id, direction)
    }

    /// Check if the graph is acyclic (compatibility method)
    pub fn is_acyclic(&self) -> bool {
        use petgraph::algo::is_cyclic_directed;
        !is_cyclic_directed(&self.graph)
    }

    /// Replace a node with one of its inputs (for constant folding and branch elimination)
    ///
    /// This operation:
    /// 1. Redirects all edges coming into `node_id` to `replacement_id`
    /// 2. Redirects all edges going out of `node_id` to come from `replacement_id`
    /// 3. Removes `node_id` from the graph
    ///
    /// # Arguments
    ///
    /// * `node_id` - The node to replace
    /// * `replacement_id` - The input node that will replace it
    ///
    /// # Returns
    ///
    /// * `Ok(())` if successful
    /// * `Err(JitError)` if the replacement would create an invalid graph
    pub fn replace_node_with_input(
        &mut self,
        node_id: NodeId,
        replacement_id: NodeId,
    ) -> crate::JitResult<()> {
        // Validate that replacement_id is actually an input to node_id
        let is_predecessor = self
            .predecessors(node_id)
            .any(|pred| pred == replacement_id);

        if !is_predecessor {
            return Err(crate::JitError::CompilationError(format!(
                "Node {:?} is not a predecessor of node {:?}",
                replacement_id, node_id
            )));
        }

        // Collect all successor edges before modification
        let successors: Vec<(NodeId, Edge)> = self
            .graph
            .edges_directed(node_id, Direction::Outgoing)
            .map(|edge_ref| (edge_ref.target(), edge_ref.weight().clone()))
            .collect();

        // Redirect all outgoing edges to come from replacement_id instead
        for (successor_id, edge) in successors {
            self.graph.add_edge(replacement_id, successor_id, edge);
        }

        // Update outputs list if node_id was an output
        if let Some(pos) = self.outputs.iter().position(|&id| id == node_id) {
            self.outputs[pos] = replacement_id;
        }

        // Remove the replaced node (this also cleans up inputs/outputs lists)
        self.remove_node(node_id);

        Ok(())
    }

    /// Replace a node with a sequence of nodes (for loop unrolling and macro expansion)
    ///
    /// This operation:
    /// 1. Inserts the sequence of nodes into the graph
    /// 2. Connects the first node in the sequence to the inputs of `node_id`
    /// 3. Connects the last node in the sequence to the outputs of `node_id`
    /// 4. Removes `node_id` from the graph
    ///
    /// # Arguments
    ///
    /// * `node_id` - The node to replace
    /// * `sequence` - The sequence of nodes to insert (must not be empty)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if successful
    /// * `Err(JitError)` if the sequence is empty or would create an invalid graph
    pub fn replace_node_with_sequence(
        &mut self,
        node_id: NodeId,
        sequence: &[Node],
    ) -> crate::JitResult<()> {
        if sequence.is_empty() {
            return Err(crate::JitError::CompilationError(
                "Cannot replace node with empty sequence".to_string(),
            ));
        }

        // Add all nodes in the sequence
        let sequence_ids: Vec<NodeId> = sequence
            .iter()
            .map(|node| self.graph.add_node(node.clone()))
            .collect();

        let first_id = sequence_ids[0];
        let last_id = *sequence_ids.last().expect("sequence should not be empty");

        // Connect nodes in the sequence to each other
        for window in sequence_ids.windows(2) {
            let edge = Edge {
                src_output: 0,
                dst_input: 0,
            };
            self.graph.add_edge(window[0], window[1], edge);
        }

        // Collect predecessor edges before modification
        let predecessors: Vec<(NodeId, Edge)> = self
            .graph
            .edges_directed(node_id, Direction::Incoming)
            .map(|edge_ref| (edge_ref.source(), edge_ref.weight().clone()))
            .collect();

        // Redirect incoming edges to the first node in the sequence
        for (pred_id, edge) in predecessors {
            self.graph.add_edge(pred_id, first_id, edge);
        }

        // Collect successor edges before modification
        let successors: Vec<(NodeId, Edge)> = self
            .graph
            .edges_directed(node_id, Direction::Outgoing)
            .map(|edge_ref| (edge_ref.target(), edge_ref.weight().clone()))
            .collect();

        // Redirect outgoing edges to come from the last node in the sequence
        for (succ_id, edge) in successors {
            self.graph.add_edge(last_id, succ_id, edge);
        }

        // Update inputs list if node_id was an input
        if let Some(pos) = self.inputs.iter().position(|&id| id == node_id) {
            self.inputs[pos] = first_id;
        }

        // Update outputs list if node_id was an output
        if let Some(pos) = self.outputs.iter().position(|&id| id == node_id) {
            self.outputs[pos] = last_id;
        }

        // Remove the replaced node (this also cleans up inputs/outputs lists)
        self.remove_node(node_id);

        Ok(())
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to create a Shape from a slice of dimensions
pub fn shape_from_slice(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec())
}
