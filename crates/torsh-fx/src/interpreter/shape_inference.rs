//! Shape Inference System for FX Graph Analysis
//!
//! This module provides comprehensive shape inference capabilities for FX graphs.
//! It analyzes graph structure, propagates shape information through operations,
//! and validates shape compatibility for all tensor operations.

use crate::interpreter::operations::{global_registry, is_operation_registered};
use crate::{FxGraph, Node, TorshResult};
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use torsh_core::{dtype::DType, error::TorshError, shape::Shape};
use torsh_tensor::Tensor;

/// Shape information for a tensor
///
/// Combines shape and data type information for comprehensive tensor analysis.
/// Used throughout the shape inference system to track tensor properties.
#[derive(Debug, Clone)]
pub struct ShapeInfo {
    /// The tensor shape (dimensions)
    pub shape: Shape,
    /// The tensor data type
    pub dtype: DType,
}

impl ShapeInfo {
    /// Create new shape information
    ///
    /// # Arguments
    /// * `shape` - Tensor shape
    /// * `dtype` - Tensor data type
    ///
    /// # Returns
    /// * `Self` - New shape information instance
    pub fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    /// Create shape information from a tensor
    ///
    /// # Arguments
    /// * `tensor` - Tensor to extract shape information from
    ///
    /// # Returns
    /// * `Self` - Shape information extracted from tensor
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
        }
    }
}

/// Shape inference context
///
/// Manages shape inference for an entire graph, tracking shape information
/// for each node and providing methods for propagating shapes through operations.
pub struct ShapeInferenceContext {
    /// Shape information for each node
    shapes: HashMap<NodeIndex, ShapeInfo>,
}

impl ShapeInferenceContext {
    /// Create a new shape inference context
    ///
    /// # Returns
    /// * `Self` - New empty shape inference context
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    /// Set shape information for a node
    ///
    /// # Arguments
    /// * `node` - Node index to set shape for
    /// * `shape_info` - Shape information to associate with the node
    pub fn set_shape(&mut self, node: NodeIndex, shape_info: ShapeInfo) {
        self.shapes.insert(node, shape_info);
    }

    /// Get shape information for a node
    ///
    /// # Arguments
    /// * `node` - Node index to get shape for
    ///
    /// # Returns
    /// * `Option<&ShapeInfo>` - Shape information if available
    pub fn get_shape(&self, node: NodeIndex) -> Option<&ShapeInfo> {
        self.shapes.get(&node)
    }

    /// Infer shapes for the entire graph
    ///
    /// Performs a topological traversal of the graph and infers shapes for all nodes
    /// based on input shapes and operation-specific shape inference rules.
    ///
    /// # Arguments
    /// * `graph` - FX graph to perform shape inference on
    /// * `input_shapes` - Map of input node names to their shape information
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if inference succeeds, error otherwise
    pub fn infer_shapes(
        &mut self,
        graph: &FxGraph,
        input_shapes: HashMap<String, ShapeInfo>,
    ) -> TorshResult<()> {
        // Set shapes for input nodes
        for &input_idx in graph.inputs() {
            if let Some(Node::Input(name)) = graph.get_node(input_idx) {
                if let Some(shape_info) = input_shapes.get(name) {
                    self.set_shape(input_idx, shape_info.clone());
                }
            }
        }

        // Perform topological traversal and infer shapes
        let execution_order = self.compute_execution_order(graph)?;

        for node_idx in execution_order {
            if let Some(node) = graph.get_node(node_idx) {
                match node {
                    Node::Input(_) => {
                        // Already handled above
                    }
                    Node::Call(op_name, _) => {
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        let output_shape = self.infer_operation_shape(op_name, &input_shapes)?;
                        self.set_shape(node_idx, output_shape);
                    }
                    Node::Conditional { .. } => {
                        // For conditionals, we need to merge shapes from both branches
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        let output_shape = if let Some(first_shape) = input_shapes.first() {
                            first_shape.clone()
                        } else {
                            ShapeInfo::new(Shape::new(vec![1]), DType::F32)
                        };
                        self.set_shape(node_idx, output_shape);
                    }
                    Node::Loop { .. } => {
                        // For loops, use the shape of loop variables
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        let output_shape = if let Some(first_shape) = input_shapes.first() {
                            first_shape.clone()
                        } else {
                            ShapeInfo::new(Shape::new(vec![1]), DType::F32)
                        };
                        self.set_shape(node_idx, output_shape);
                    }
                    Node::Output => {
                        // Output nodes inherit shape from their input
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        if let Some(input_shape) = input_shapes.first() {
                            self.set_shape(node_idx, input_shape.clone());
                        }
                    }
                    Node::Merge { .. } => {
                        // Merge nodes use the shape of their first input
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        let output_shape = if let Some(first_shape) = input_shapes.first() {
                            first_shape.clone()
                        } else {
                            ShapeInfo::new(Shape::new(vec![1]), DType::F32)
                        };
                        self.set_shape(node_idx, output_shape);
                    }
                    Node::GetAttr { .. } => {
                        // GetAttr nodes inherit shape from their target
                        let input_shapes = self.get_input_shapes_for_node(graph, node_idx)?;
                        let output_shape = if let Some(first_shape) = input_shapes.first() {
                            first_shape.clone()
                        } else {
                            ShapeInfo::new(Shape::new(vec![1]), DType::F32)
                        };
                        self.set_shape(node_idx, output_shape);
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute execution order for shape inference
    ///
    /// Performs topological sort to determine the order in which nodes should be processed
    /// for shape inference. This ensures dependencies are processed before dependent nodes.
    ///
    /// # Arguments
    /// * `graph` - FX graph to compute execution order for
    ///
    /// # Returns
    /// * `TorshResult<Vec<NodeIndex>>` - Topologically sorted node indices or error if graph has cycles
    fn compute_execution_order(&self, graph: &FxGraph) -> TorshResult<Vec<NodeIndex>> {
        match toposort(&graph.graph, None) {
            Ok(order) => Ok(order),
            Err(_) => Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            )),
        }
    }

    /// Get input shapes for a specific node
    ///
    /// Collects shape information from all input nodes to the specified node.
    /// This is used during shape inference to determine the inputs available
    /// for operation-specific shape inference.
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the node
    /// * `node_idx` - Index of the node to get input shapes for
    ///
    /// # Returns
    /// * `TorshResult<Vec<ShapeInfo>>` - Vector of input shape information
    fn get_input_shapes_for_node(
        &self,
        graph: &FxGraph,
        node_idx: NodeIndex,
    ) -> TorshResult<Vec<ShapeInfo>> {
        let mut input_shapes = Vec::new();

        // Get all predecessor nodes
        let predecessors: Vec<_> = graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .collect();

        for pred_idx in predecessors {
            if let Some(shape_info) = self.get_shape(pred_idx) {
                input_shapes.push(shape_info.clone());
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Missing shape information for predecessor node {:?}",
                    pred_idx
                )));
            }
        }

        Ok(input_shapes)
    }

    /// Infer shape for a specific operation
    ///
    /// Uses operation-specific rules to infer the output shape based on input shapes.
    /// Supports both built-in operations and custom registered operations.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    /// * `input_shapes` - Vector of input shape information
    ///
    /// # Returns
    /// * `TorshResult<ShapeInfo>` - Inferred output shape information
    fn infer_operation_shape(
        &self,
        op_name: &str,
        input_shapes: &[ShapeInfo],
    ) -> TorshResult<ShapeInfo> {
        if input_shapes.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No input shapes provided for operation".to_string(),
            ));
        }

        // Handle built-in operations
        if self.is_builtin_operation(op_name) {
            return self.infer_builtin_operation_shape(op_name, input_shapes);
        }

        // Handle custom registered operations
        if is_operation_registered(op_name) {
            if let Ok(operation) = global_registry().get(op_name) {
                let input_shapes_only: Vec<Shape> =
                    input_shapes.iter().map(|si| si.shape.clone()).collect();
                let input_dtypes: Vec<DType> = input_shapes.iter().map(|si| si.dtype).collect();

                let output_shape = operation.infer_shape(&input_shapes_only)?;
                let output_dtype = operation.infer_type(&input_dtypes)?;

                return Ok(ShapeInfo::new(output_shape, output_dtype));
            }
        }

        // Default: use first input shape and type
        Ok(input_shapes[0].clone())
    }

    /// Check if an operation is a built-in operation
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation to check
    ///
    /// # Returns
    /// * `bool` - True if operation is built-in, false otherwise
    fn is_builtin_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name,
            "add"
                | "sub"
                | "mul"
                | "div"
                | "matmul"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "gelu"
                | "softmax"
                | "layer_norm"
                | "batch_norm"
                | "conv2d"
        )
    }

    /// Infer shape for built-in operations
    ///
    /// Implements shape inference rules for all built-in operations.
    /// Each operation has specific rules for how output shapes are determined
    /// from input shapes.
    ///
    /// # Arguments
    /// * `op_name` - Name of the built-in operation
    /// * `input_shapes` - Vector of input shape information
    ///
    /// # Returns
    /// * `TorshResult<ShapeInfo>` - Inferred output shape information
    fn infer_builtin_operation_shape(
        &self,
        op_name: &str,
        input_shapes: &[ShapeInfo],
    ) -> TorshResult<ShapeInfo> {
        match op_name {
            "add" | "sub" | "mul" | "div" => {
                // Element-wise operations: output shape is broadcast result
                if input_shapes.len() >= 2 {
                    let broadcast_shape =
                        self.broadcast_shapes(&input_shapes[0].shape, &input_shapes[1].shape)?;
                    // Use the higher precision dtype
                    let output_dtype =
                        self.promote_dtype(input_shapes[0].dtype, input_shapes[1].dtype);
                    Ok(ShapeInfo::new(broadcast_shape, output_dtype))
                } else {
                    Ok(input_shapes[0].clone())
                }
            }
            "matmul" => {
                // Matrix multiplication: [a, b] @ [b, c] -> [a, c]
                if input_shapes.len() >= 2 {
                    let shape1 = &input_shapes[0].shape;
                    let shape2 = &input_shapes[1].shape;

                    if shape1.ndim() >= 2 && shape2.ndim() >= 2 {
                        let mut output_dims = shape1.dims().to_vec();
                        let shape2_dims = shape2.dims();

                        // Update last dimension of first tensor with last dimension of second tensor
                        let last_idx = output_dims.len() - 1;
                        output_dims[last_idx] = shape2_dims[shape2_dims.len() - 1];

                        let output_shape = Shape::new(output_dims);
                        let output_dtype =
                            self.promote_dtype(input_shapes[0].dtype, input_shapes[1].dtype);
                        Ok(ShapeInfo::new(output_shape, output_dtype))
                    } else {
                        Err(TorshError::InvalidArgument(
                            "Matrix multiplication requires at least 2D tensors".to_string(),
                        ))
                    }
                } else {
                    Err(TorshError::InvalidArgument(
                        "Matrix multiplication requires two inputs".to_string(),
                    ))
                }
            }
            "relu" | "sigmoid" | "tanh" | "gelu" => {
                // Activation functions: preserve input shape and type
                Ok(input_shapes[0].clone())
            }
            "softmax" => {
                // Softmax: preserve input shape, output is always float
                let output_dtype = match input_shapes[0].dtype {
                    DType::F16 => DType::F16,
                    DType::F32 => DType::F32,
                    DType::F64 => DType::F64,
                    _ => DType::F32, // Default to F32 for integer inputs
                };
                Ok(ShapeInfo::new(input_shapes[0].shape.clone(), output_dtype))
            }
            "layer_norm" | "batch_norm" => {
                // Normalization: preserve input shape and type
                Ok(input_shapes[0].clone())
            }
            "conv2d" => {
                // Convolution: complex shape calculation based on kernel, stride, padding
                // For now, preserve input shape (simplified)
                Ok(input_shapes[0].clone())
            }
            _ => {
                // Unknown operation: use first input shape
                Ok(input_shapes[0].clone())
            }
        }
    }

    /// Broadcast two shapes according to NumPy broadcasting rules
    ///
    /// # Arguments
    /// * `shape1` - First shape to broadcast
    /// * `shape2` - Second shape to broadcast
    ///
    /// # Returns
    /// * `TorshResult<Shape>` - Broadcast result shape or error if incompatible
    fn broadcast_shapes(&self, shape1: &Shape, shape2: &Shape) -> TorshResult<Shape> {
        let dims1 = shape1.dims();
        let dims2 = shape2.dims();

        let max_ndim = dims1.len().max(dims2.len());
        let mut result_dims = vec![1; max_ndim];

        for i in 0..max_ndim {
            let dim1 = if i < dims1.len() {
                dims1[dims1.len() - 1 - i]
            } else {
                1
            };
            let dim2 = if i < dims2.len() {
                dims2[dims2.len() - 1 - i]
            } else {
                1
            };

            if dim1 == 1 {
                result_dims[max_ndim - 1 - i] = dim2;
            } else if dim2 == 1 {
                result_dims[max_ndim - 1 - i] = dim1;
            } else if dim1 == dim2 {
                result_dims[max_ndim - 1 - i] = dim1;
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    dims1, dims2
                )));
            }
        }

        Ok(Shape::new(result_dims))
    }

    /// Promote data types according to standard promotion rules
    ///
    /// # Arguments
    /// * `dtype1` - First data type
    /// * `dtype2` - Second data type
    ///
    /// # Returns
    /// * `DType` - Promoted data type
    fn promote_dtype(&self, dtype1: DType, dtype2: DType) -> DType {
        use DType::*;

        match (dtype1, dtype2) {
            // If types are the same, return that type
            (a, b) if a == b => a,

            // Float types take precedence
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,

            // Integer promotion
            (I64, _) | (_, I64) => I64,
            (I32, _) | (_, I32) => I32,
            (I16, _) | (_, I16) => I16,

            // Boolean operations default to the non-boolean type
            (Bool, other) | (other, Bool) => other,

            // Default case
            _ => dtype1,
        }
    }

    /// Get all inferred shapes
    ///
    /// # Returns
    /// * `&HashMap<NodeIndex, ShapeInfo>` - Reference to all inferred shapes
    pub fn get_all_shapes(&self) -> &HashMap<NodeIndex, ShapeInfo> {
        &self.shapes
    }

    /// Validate that all required shapes have been inferred
    ///
    /// # Arguments
    /// * `graph` - FX graph to validate
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if all shapes are available, error otherwise
    pub fn validate_complete_inference(&self, graph: &FxGraph) -> TorshResult<()> {
        for node_idx in graph.graph.node_indices() {
            if self.get_shape(node_idx).is_none() {
                return Err(TorshError::InvalidArgument(format!(
                    "Missing shape information for node {:?}",
                    node_idx
                )));
            }
        }
        Ok(())
    }
}

impl Default for ShapeInferenceContext {
    fn default() -> Self {
        Self::new()
    }
}
