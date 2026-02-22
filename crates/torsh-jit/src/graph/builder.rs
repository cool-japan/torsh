//! Graph builder for constructing computation graphs

use crate::graph::core::{ComputationGraph, Edge, Node, NodeId};
use crate::graph::metadata::GraphMetadata;
use crate::graph::operations::{ConstantInfo, ConstantValue, Operation, ParameterInfo};
use crate::{JitError, JitResult};
use std::collections::HashMap;
use torsh_core::{DType, DeviceType, Shape};

/// Builder for constructing computation graphs
#[derive(Debug)]
pub struct GraphBuilder {
    graph: ComputationGraph,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
        }
    }

    /// Create a new graph builder with metadata
    pub fn with_metadata(metadata: GraphMetadata) -> Self {
        let mut graph = ComputationGraph::new();
        graph.metadata = metadata;
        Self { graph }
    }

    /// Add an input node
    pub fn add_input(&mut self, name: String, shape: Shape, dtype: DType) -> NodeId {
        let node = Node::new(Operation::Input, name)
            .with_output_shapes(vec![Some(shape)])
            .with_dtypes(vec![dtype]);
        let node_id = self.graph.add_node(node);
        self.graph.add_input(node_id);
        node_id
    }

    /// Add a parameter node
    pub fn add_parameter(
        &mut self,
        name: String,
        shape: Shape,
        dtype: DType,
        trainable: bool,
    ) -> NodeId {
        let param_info = ParameterInfo {
            name: name.clone(),
            trainable,
        };
        let node = Node::new(Operation::Parameter(param_info), name)
            .with_output_shapes(vec![Some(shape)])
            .with_dtypes(vec![dtype]);
        let node_id = self.graph.add_node(node);
        self.graph.add_input(node_id);
        node_id
    }

    /// Add a constant node
    pub fn add_constant(
        &mut self,
        name: String,
        value: ConstantValue,
        shape: Shape,
        dtype: DType,
    ) -> NodeId {
        let const_info = ConstantInfo { value };
        let node = Node::new(Operation::Constant(const_info), name)
            .with_output_shapes(vec![Some(shape)])
            .with_dtypes(vec![dtype]);
        self.graph.add_node(node)
    }

    /// Add a generic operation node
    pub fn add_operation(&mut self, name: String, operation: Operation) -> NodeId {
        let node = Node::new(operation, name);
        self.graph.add_node(node)
    }

    /// Add an operation node with shape information
    pub fn add_operation_with_shapes(
        &mut self,
        name: String,
        operation: Operation,
        input_shapes: Vec<Option<Shape>>,
        output_shapes: Vec<Option<Shape>>,
        dtypes: Vec<DType>,
    ) -> NodeId {
        let node = Node::new(operation, name)
            .with_input_shapes(input_shapes)
            .with_output_shapes(output_shapes)
            .with_dtypes(dtypes);
        self.graph.add_node(node)
    }

    /// Connect two nodes with an edge
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> JitResult<()> {
        self.connect_with_ports(from, 0, to, 0)
    }

    /// Connect two nodes with specific ports
    pub fn connect_with_ports(
        &mut self,
        from: NodeId,
        from_output: usize,
        to: NodeId,
        to_input: usize,
    ) -> JitResult<()> {
        // Validate nodes exist
        if self.graph.get_node(from).is_none() {
            return Err(JitError::GraphError(format!(
                "Source node {:?} does not exist",
                from
            )));
        }
        if self.graph.get_node(to).is_none() {
            return Err(JitError::GraphError(format!(
                "Destination node {:?} does not exist",
                to
            )));
        }

        let edge = Edge {
            src_output: from_output,
            dst_input: to_input,
        };
        self.graph.add_edge(from, to, edge);
        Ok(())
    }

    /// Mark a node as output
    pub fn mark_output(&mut self, node_id: NodeId) -> JitResult<()> {
        if self.graph.get_node(node_id).is_none() {
            return Err(JitError::GraphError(format!(
                "Node {:?} does not exist",
                node_id
            )));
        }
        self.graph.add_output(node_id);
        Ok(())
    }

    /// Set node shape information
    pub fn set_node_shapes(
        &mut self,
        node_id: NodeId,
        input_shapes: Vec<Option<Shape>>,
        output_shapes: Vec<Option<Shape>>,
    ) -> JitResult<()> {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.input_shapes = input_shapes;
            node.output_shapes = output_shapes;
            Ok(())
        } else {
            Err(JitError::GraphError(format!(
                "Node {:?} does not exist",
                node_id
            )))
        }
    }

    /// Set node data types
    pub fn set_node_dtypes(&mut self, node_id: NodeId, dtypes: Vec<DType>) -> JitResult<()> {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.dtypes = dtypes;
            Ok(())
        } else {
            Err(JitError::GraphError(format!(
                "Node {:?} does not exist",
                node_id
            )))
        }
    }

    /// Set node device
    pub fn set_node_device(&mut self, node_id: NodeId, device: DeviceType) -> JitResult<()> {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.device = device;
            Ok(())
        } else {
            Err(JitError::GraphError(format!(
                "Node {:?} does not exist",
                node_id
            )))
        }
    }

    /// Build and return the computation graph
    pub fn build(self) -> JitResult<ComputationGraph> {
        self.graph.validate()?;
        Ok(self.graph)
    }

    /// Build without validation (for testing purposes)
    pub fn build_unchecked(self) -> ComputationGraph {
        self.graph
    }

    /// Get a reference to the current graph being built
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }

    /// Get a mutable reference to the current graph being built
    pub fn graph_mut(&mut self) -> &mut ComputationGraph {
        &mut self.graph
    }

    /// Clone the current state of the graph
    pub fn clone_graph(&self) -> ComputationGraph {
        self.graph.clone()
    }

    // Convenience methods for common operations

    /// Add an element-wise binary operation
    pub fn add_binary_op(
        &mut self,
        name: String,
        operation: Operation,
        left: NodeId,
        right: NodeId,
    ) -> JitResult<NodeId> {
        let node_id = self.add_operation(name, operation);
        self.connect(left, node_id)?;
        self.connect(right, node_id)?;
        Ok(node_id)
    }

    /// Add an element-wise unary operation
    pub fn add_unary_op(
        &mut self,
        name: String,
        operation: Operation,
        input: NodeId,
    ) -> JitResult<NodeId> {
        let node_id = self.add_operation(name, operation);
        self.connect(input, node_id)?;
        Ok(node_id)
    }

    /// Add a reduction operation
    pub fn add_reduction_op(
        &mut self,
        name: String,
        operation: Operation,
        input: NodeId,
    ) -> JitResult<NodeId> {
        let node_id = self.add_operation(name, operation);
        self.connect(input, node_id)?;
        Ok(node_id)
    }

    /// Create a linear chain of operations
    pub fn create_linear_chain(
        &mut self,
        operations: Vec<(String, Operation)>,
        input: NodeId,
    ) -> JitResult<NodeId> {
        let mut current = input;
        for (name, op) in operations {
            current = self.add_unary_op(name, op, current)?;
        }
        Ok(current)
    }

    /// Create a residual connection (input + f(input))
    pub fn create_residual_connection(
        &mut self,
        input: NodeId,
        transform_ops: Vec<(String, Operation)>,
    ) -> JitResult<NodeId> {
        let transformed = self.create_linear_chain(transform_ops, input)?;
        self.add_binary_op(
            "residual_add".to_string(),
            Operation::Add,
            input,
            transformed,
        )
    }

    /// Validate the current graph state
    pub fn validate(&self) -> JitResult<()> {
        self.graph.validate()
    }

    /// Get statistics about the current graph
    pub fn statistics(&self) -> GraphStatistics {
        GraphStatistics::from_graph(&self.graph)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a computation graph
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub operation_counts: HashMap<String, usize>,
    pub memory_estimate: usize,
    pub complexity_estimate: usize,
}

impl GraphStatistics {
    /// Generate statistics from a computation graph
    pub fn from_graph(graph: &ComputationGraph) -> Self {
        let mut operation_counts = HashMap::new();

        for (_, node) in graph.nodes() {
            let op_name = node.operation.as_str().to_string();
            *operation_counts.entry(op_name).or_insert(0) += 1;
        }

        Self {
            node_count: graph.node_count(),
            edge_count: graph.edge_count(),
            input_count: graph.inputs.len(),
            output_count: graph.outputs.len(),
            operation_counts,
            memory_estimate: graph.memory_estimate(),
            complexity_estimate: graph.complexity_estimate(),
        }
    }
}
