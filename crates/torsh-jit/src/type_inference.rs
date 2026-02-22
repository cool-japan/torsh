//! Type inference and shape propagation for JIT compilation
//!
//! This module provides automatic type and shape inference for computation graphs,
//! allowing dynamic determination of tensor shapes and data types.

use crate::graph::{ComputationGraph, Conv2dInfo, Node, NodeId, Operation};
use crate::{JitError, JitResult};
use std::collections::HashMap;
use torsh_core::{DType, Shape};

/// Type inference engine
pub struct TypeInference {
    /// Inferred types for each node
    node_types: HashMap<NodeId, DType>,

    /// Inferred shapes for each node
    node_shapes: HashMap<NodeId, Shape>,

    /// Type constraints from the graph
    constraints: Vec<TypeConstraint>,
}

/// Type constraint between nodes
#[derive(Debug, Clone)]
pub enum TypeConstraint {
    /// Two nodes must have the same type
    SameType(NodeId, NodeId),

    /// Node must have specific type
    ExactType(NodeId, DType),

    /// Node type must be compatible with operation
    OperationCompat(NodeId, OperationTypeReq),

    /// Broadcasting constraint
    Broadcast(NodeId, NodeId, NodeId), // left, right, result
}

/// Type requirements for operations
#[derive(Debug, Clone)]
pub enum OperationTypeReq {
    /// Must be floating point
    FloatingPoint,

    /// Must be numeric (int or float)
    Numeric,

    /// Must be boolean
    Boolean,

    /// Must match specific type
    Exact(DType),

    /// Must be one of several types
    OneOf(Vec<DType>),
}

/// Shape inference engine
pub struct ShapeInference {
    /// Known shapes
    node_shapes: HashMap<NodeId, Shape>,

    /// Shape constraints
    constraints: Vec<ShapeConstraint>,
}

/// Shape constraint
#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// Exact shape
    Exact(NodeId, Shape),

    /// Same shape
    SameShape(NodeId, NodeId),

    /// Broadcasting compatible
    BroadcastCompat(NodeId, NodeId, NodeId),

    /// Matmul compatible
    MatmulCompat(NodeId, NodeId, NodeId),

    /// Convolution compatible
    ConvCompat(NodeId, NodeId, NodeId, Conv2dInfo),
}

impl TypeInference {
    /// Create new type inference engine
    pub fn new() -> Self {
        Self {
            node_types: HashMap::new(),
            node_shapes: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Infer types for all nodes in the graph
    pub fn infer_types(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // First pass: collect explicit types and constraints
        self.collect_constraints(graph)?;

        // Second pass: propagate types
        self.propagate_types(graph)?;

        // Third pass: validate all types are resolved
        self.validate_types(graph)?;

        Ok(())
    }

    /// Collect type constraints from the graph
    fn collect_constraints(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            // Store explicit types
            self.node_types.insert(node_id, node.dtype);
            self.node_shapes.insert(node_id, node.output_shape.clone());

            // Add operation-specific constraints
            self.add_operation_constraints(graph, node_id, node)?;
        }

        Ok(())
    }

    /// Add constraints based on operation type
    fn add_operation_constraints(
        &mut self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> JitResult<()> {
        match &node.op {
            // Element-wise operations preserve input types
            Operation::Neg
            | Operation::Abs
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Sin
            | Operation::Cos
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Relu
            | Operation::Gelu => {
                // Input and output must have same type
                for pred_id in graph.predecessors(node_id) {
                    self.constraints
                        .push(TypeConstraint::SameType(pred_id, node_id));
                }
                self.constraints.push(TypeConstraint::OperationCompat(
                    node_id,
                    OperationTypeReq::FloatingPoint,
                ));
            }

            // Binary operations need compatible types
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Pow
            | Operation::Maximum
            | Operation::Minimum => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    self.constraints
                        .push(TypeConstraint::Broadcast(preds[0], preds[1], node_id));
                }
            }

            // Matrix operations
            Operation::MatMul | Operation::BatchMatMul => {
                self.constraints.push(TypeConstraint::OperationCompat(
                    node_id,
                    OperationTypeReq::FloatingPoint,
                ));
            }

            // Reduction operations preserve type
            Operation::Sum { .. }
            | Operation::Mean { .. }
            | Operation::Max { .. }
            | Operation::Min { .. } => {
                for pred_id in graph.predecessors(node_id) {
                    self.constraints
                        .push(TypeConstraint::SameType(pred_id, node_id));
                }
            }

            // Convolution
            Operation::Conv2d(_) => {
                self.constraints.push(TypeConstraint::OperationCompat(
                    node_id,
                    OperationTypeReq::FloatingPoint,
                ));
            }

            // Constants have fixed types
            Operation::Constant(const_info) => {
                let dtype = match &const_info.value {
                    crate::graph::ConstantValue::Bool(_) => DType::Bool,
                    crate::graph::ConstantValue::Int(_) => DType::I64,
                    crate::graph::ConstantValue::UInt(_) => DType::U64,
                    crate::graph::ConstantValue::Float(_) => DType::F32,
                    crate::graph::ConstantValue::String(_) => DType::F32, // Default
                    crate::graph::ConstantValue::FloatArray(_) => DType::F32,
                    crate::graph::ConstantValue::IntArray(_) => DType::I64,
                    crate::graph::ConstantValue::Array(_) => DType::F32, // Default
                    crate::graph::ConstantValue::Tensor { dtype, .. } => {
                        // Try to parse dtype string, default to F32
                        match dtype.as_str() {
                            "f32" | "float32" => DType::F32,
                            "f64" | "float64" => DType::F64,
                            "i32" | "int32" => DType::I32,
                            "i64" | "int64" => DType::I64,
                            _ => DType::F32,
                        }
                    }
                    crate::graph::ConstantValue::Complex { .. } => DType::F32,
                    crate::graph::ConstantValue::None => DType::F32, // Default
                    crate::graph::ConstantValue::Undefined => DType::F32, // Default
                    crate::graph::ConstantValue::Scalar(_) => DType::F32, // Legacy alias
                    crate::graph::ConstantValue::IntScalar(_) => DType::I64, // Legacy alias
                };
                self.constraints
                    .push(TypeConstraint::ExactType(node_id, dtype));
            }

            _ => {} // No special constraints
        }

        Ok(())
    }

    /// Propagate types through the graph
    fn propagate_types(&mut self, _graph: &ComputationGraph) -> JitResult<()> {
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for constraint in &self.constraints.clone() {
                if self.apply_constraint(constraint)? {
                    changed = true;
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            return Err(JitError::GraphError(
                "Type inference did not converge".to_string(),
            ));
        }

        Ok(())
    }

    /// Apply a single type constraint
    fn apply_constraint(&mut self, constraint: &TypeConstraint) -> JitResult<bool> {
        match constraint {
            TypeConstraint::SameType(node1, node2) => {
                let type1 = self.node_types.get(node1).copied();
                let type2 = self.node_types.get(node2).copied();

                match (type1, type2) {
                    (Some(t1), Some(t2)) => {
                        if t1 != t2 {
                            return Err(JitError::GraphError(format!(
                                "Type mismatch: {:?} != {:?}",
                                t1, t2
                            )));
                        }
                        Ok(false)
                    }
                    (Some(t), None) => {
                        self.node_types.insert(*node2, t);
                        Ok(true)
                    }
                    (None, Some(t)) => {
                        self.node_types.insert(*node1, t);
                        Ok(true)
                    }
                    (None, None) => Ok(false),
                }
            }

            TypeConstraint::ExactType(node, dtype) => {
                if let Some(existing) = self.node_types.get(node) {
                    if existing != dtype {
                        return Err(JitError::GraphError(format!(
                            "Type conflict: expected {:?}, got {:?}",
                            dtype, existing
                        )));
                    }
                    Ok(false)
                } else {
                    self.node_types.insert(*node, *dtype);
                    Ok(true)
                }
            }

            TypeConstraint::OperationCompat(node, req) => {
                if let Some(dtype) = self.node_types.get(node) {
                    if !self.check_type_requirement(*dtype, req) {
                        return Err(JitError::GraphError(format!(
                            "Type {:?} incompatible with requirement {:?}",
                            dtype, req
                        )));
                    }
                }
                Ok(false)
            }

            TypeConstraint::Broadcast(left, right, result) => {
                // For simplicity, use the "higher precision" type
                let left_type = self.node_types.get(left).copied();
                let right_type = self.node_types.get(right).copied();

                if let (Some(l), Some(r)) = (left_type, right_type) {
                    let result_type = self.resolve_broadcast_type(l, r)?;
                    if let Some(existing) = self.node_types.get(result) {
                        if *existing != result_type {
                            return Err(JitError::GraphError(format!(
                                "Broadcast type conflict: expected {:?}, got {:?}",
                                result_type, existing
                            )));
                        }
                        Ok(false)
                    } else {
                        self.node_types.insert(*result, result_type);
                        Ok(true)
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Check if a type satisfies a requirement
    fn check_type_requirement(&self, dtype: DType, req: &OperationTypeReq) -> bool {
        match req {
            OperationTypeReq::FloatingPoint => {
                matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
            }
            OperationTypeReq::Numeric => matches!(
                dtype,
                DType::I8
                    | DType::I16
                    | DType::I32
                    | DType::I64
                    | DType::U8
                    | DType::F16
                    | DType::BF16
                    | DType::F32
                    | DType::F64
            ),
            OperationTypeReq::Boolean => matches!(dtype, DType::Bool),
            OperationTypeReq::Exact(expected) => dtype == *expected,
            OperationTypeReq::OneOf(types) => types.contains(&dtype),
        }
    }

    /// Resolve the result type of broadcasting two types
    fn resolve_broadcast_type(&self, left: DType, right: DType) -> JitResult<DType> {
        use DType::*;

        // Type promotion rules (simplified)
        let result = match (left, right) {
            (a, b) if a == b => a,
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,
            (I64, _) | (_, I64) => I64,
            (I32, _) | (_, I32) => I32,
            (I16, _) | (_, I16) => I16,
            (I8, _) | (_, I8) => I8,
            _ => {
                return Err(JitError::GraphError(format!(
                    "Cannot broadcast types {:?} and {:?}",
                    left, right
                )))
            }
        };

        Ok(result)
    }

    /// Validate that all types are resolved
    fn validate_types(&self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, _) in graph.nodes() {
            if !self.node_types.contains_key(&node_id) {
                return Err(JitError::GraphError(format!(
                    "Unresolved type for node {:?}",
                    node_id
                )));
            }
        }
        Ok(())
    }

    /// Get inferred type for a node
    pub fn get_type(&self, node_id: NodeId) -> Option<DType> {
        self.node_types.get(&node_id).copied()
    }

    /// Get all inferred types
    pub fn get_all_types(&self) -> &HashMap<NodeId, DType> {
        &self.node_types
    }
}

impl ShapeInference {
    /// Create new shape inference engine
    pub fn new() -> Self {
        Self {
            node_shapes: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Infer shapes for all nodes in the graph
    pub fn infer_shapes(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        // Collect shape constraints
        self.collect_shape_constraints(graph)?;

        // Propagate shapes
        self.propagate_shapes()?;

        // Validate all shapes are resolved
        self.validate_shapes(graph)?;

        Ok(())
    }

    /// Collect shape constraints from the graph
    fn collect_shape_constraints(&mut self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, node) in graph.nodes() {
            // Store explicit shapes
            self.node_shapes.insert(node_id, node.output_shape.clone());

            // Add operation-specific constraints
            self.add_shape_constraints(graph, node_id, node)?;
        }

        Ok(())
    }

    /// Add shape constraints based on operation
    fn add_shape_constraints(
        &mut self,
        graph: &ComputationGraph,
        node_id: NodeId,
        node: &Node,
    ) -> JitResult<()> {
        match &node.op {
            // Element-wise operations preserve shape
            Operation::Neg
            | Operation::Abs
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Sin
            | Operation::Cos
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Relu
            | Operation::Gelu => {
                for pred_id in graph.predecessors(node_id) {
                    self.constraints
                        .push(ShapeConstraint::SameShape(pred_id, node_id));
                }
            }

            // Binary operations with broadcasting
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    self.constraints.push(ShapeConstraint::BroadcastCompat(
                        preds[0], preds[1], node_id,
                    ));
                }
            }

            // Matrix multiplication
            Operation::MatMul => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if preds.len() == 2 {
                    self.constraints
                        .push(ShapeConstraint::MatmulCompat(preds[0], preds[1], node_id));
                }
            }

            // Reshape operations
            Operation::Reshape { shape } => {
                let new_shape = Shape::new(
                    shape
                        .iter()
                        .map(|&dim| {
                            if dim == -1 {
                                // Infer dimension
                                // For now, just use 1 as placeholder
                                1
                            } else {
                                dim as usize
                            }
                        })
                        .collect(),
                );
                self.constraints
                    .push(ShapeConstraint::Exact(node_id, new_shape));
            }

            // Convolution
            Operation::Conv2d(conv_info) => {
                let preds: Vec<_> = graph.predecessors(node_id).collect();
                if !preds.is_empty() {
                    self.constraints.push(ShapeConstraint::ConvCompat(
                        preds[0], // input
                        node_id,  // weight is implicit
                        node_id,  // output
                        conv_info.clone(),
                    ));
                }
            }

            _ => {} // No special shape constraints
        }

        Ok(())
    }

    /// Propagate shapes through constraints
    fn propagate_shapes(&mut self) -> JitResult<()> {
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        // Track which constraints have been applied successfully to avoid redundant work
        let mut applied_constraints = std::collections::HashSet::new();

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;
            let mut constraints_to_apply = Vec::new();

            // Only re-apply constraints that might produce new information
            for (idx, constraint) in self.constraints.iter().enumerate() {
                if !applied_constraints.contains(&idx) || self.constraint_might_change(constraint) {
                    constraints_to_apply.push((idx, constraint.clone()));
                }
            }

            for (idx, constraint) in constraints_to_apply {
                if self.apply_shape_constraint(&constraint)? {
                    changed = true;
                    applied_constraints.insert(idx);
                }
            }

            // If no constraints made changes, we've reached a fixed point
            if !changed {
                break;
            }
        }

        if iterations >= MAX_ITERATIONS {
            return Err(JitError::GraphError(
                "Shape inference did not converge".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if a constraint might still produce changes
    fn constraint_might_change(&self, constraint: &ShapeConstraint) -> bool {
        match constraint {
            ShapeConstraint::Exact(node, _) => {
                // Only apply if the node doesn't have a shape yet
                !self.node_shapes.contains_key(node)
            }
            ShapeConstraint::SameShape(node1, node2) => {
                // Only apply if at least one node has a shape and the other doesn't,
                // or if both have shapes but they're different
                let has_shape1 = self.node_shapes.contains_key(node1);
                let has_shape2 = self.node_shapes.contains_key(node2);
                (!has_shape1 && has_shape2) || (has_shape1 && !has_shape2)
            }
            ShapeConstraint::BroadcastCompat(left, right, result) => {
                // Only apply if inputs have shapes but result doesn't
                self.node_shapes.contains_key(left)
                    && self.node_shapes.contains_key(right)
                    && !self.node_shapes.contains_key(result)
            }
            ShapeConstraint::MatmulCompat(left, right, result) => {
                // Only apply if inputs have shapes but result doesn't
                self.node_shapes.contains_key(left)
                    && self.node_shapes.contains_key(right)
                    && !self.node_shapes.contains_key(result)
            }
            ShapeConstraint::ConvCompat(input, _, output, _) => {
                // Only apply if input has shape but output doesn't
                self.node_shapes.contains_key(input) && !self.node_shapes.contains_key(output)
            }
        }
    }

    /// Apply a single shape constraint
    fn apply_shape_constraint(&mut self, constraint: &ShapeConstraint) -> JitResult<bool> {
        match constraint {
            ShapeConstraint::Exact(node, shape) => {
                self.node_shapes.insert(*node, shape.clone());
                Ok(true)
            }

            ShapeConstraint::SameShape(node1, node2) => {
                let shape1 = self.node_shapes.get(node1).cloned();
                let shape2 = self.node_shapes.get(node2).cloned();

                match (shape1, shape2) {
                    (Some(s1), Some(s2)) => {
                        if s1.dims() != s2.dims() {
                            return Err(JitError::GraphError(format!(
                                "Shape mismatch: {:?} != {:?}",
                                s1.dims(),
                                s2.dims()
                            )));
                        }
                        Ok(false)
                    }
                    (Some(s), None) => {
                        self.node_shapes.insert(*node2, s);
                        Ok(true)
                    }
                    (None, Some(s)) => {
                        self.node_shapes.insert(*node1, s);
                        Ok(true)
                    }
                    (None, None) => Ok(false),
                }
            }

            ShapeConstraint::BroadcastCompat(left, right, result) => {
                if let (Some(left_shape), Some(right_shape)) =
                    (self.node_shapes.get(left), self.node_shapes.get(right))
                {
                    let result_shape = self.broadcast_shapes(left_shape, right_shape)?;
                    self.node_shapes.insert(*result, result_shape);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }

            ShapeConstraint::MatmulCompat(left, right, result) => {
                if let (Some(left_shape), Some(right_shape)) =
                    (self.node_shapes.get(left), self.node_shapes.get(right))
                {
                    let result_shape = self.matmul_output_shape(left_shape, right_shape)?;
                    self.node_shapes.insert(*result, result_shape);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }

            ShapeConstraint::ConvCompat(input, _weight, output, conv_info) => {
                if let Some(input_shape) = self.node_shapes.get(input) {
                    let output_shape = self.conv_output_shape(input_shape, conv_info)?;
                    self.node_shapes.insert(*output, output_shape);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Helper function to get dimension for broadcasting (from right to left)
    fn get_broadcast_dim(dims: &[usize], index: usize) -> usize {
        if index < dims.len() {
            dims[dims.len() - 1 - index]
        } else {
            1
        }
    }

    /// Compute broadcast shape
    fn broadcast_shapes(&self, left: &Shape, right: &Shape) -> JitResult<Shape> {
        let left_dims = left.dims();
        let right_dims = right.dims();

        let max_ndim = left_dims.len().max(right_dims.len());
        let mut result_dims = vec![1; max_ndim];

        // Align dimensions from the right
        for i in 0..max_ndim {
            let left_dim = Self::get_broadcast_dim(left_dims, i);
            let right_dim = Self::get_broadcast_dim(right_dims, i);

            let result_dim = if left_dim == 1 {
                right_dim
            } else if right_dim == 1 || left_dim == right_dim {
                left_dim
            } else {
                return Err(JitError::GraphError(format!(
                    "Cannot broadcast dimensions {} and {}",
                    left_dim, right_dim
                )));
            };

            result_dims[max_ndim - 1 - i] = result_dim;
        }

        Ok(Shape::new(result_dims))
    }

    /// Compute matrix multiplication output shape
    fn matmul_output_shape(&self, left: &Shape, right: &Shape) -> JitResult<Shape> {
        let left_dims = left.dims();
        let right_dims = right.dims();

        if left_dims.len() < 2 || right_dims.len() < 2 {
            return Err(JitError::GraphError(
                "MatMul requires at least 2D tensors".to_string(),
            ));
        }

        let left_rows = left_dims[left_dims.len() - 2];
        let left_cols = left_dims[left_dims.len() - 1];
        let right_rows = right_dims[right_dims.len() - 2];
        let right_cols = right_dims[right_dims.len() - 1];

        if left_cols != right_rows {
            return Err(JitError::GraphError(format!(
                "MatMul dimension mismatch: {} != {}",
                left_cols, right_rows
            )));
        }

        // Result shape: batch dims + [left_rows, right_cols]
        let mut result_dims = left_dims[..left_dims.len() - 2].to_vec();
        result_dims.push(left_rows);
        result_dims.push(right_cols);

        Ok(Shape::new(result_dims))
    }

    /// Compute convolution output shape
    fn conv_output_shape(&self, input: &Shape, conv_info: &Conv2dInfo) -> JitResult<Shape> {
        let input_dims = input.dims();
        if input_dims.len() != 4 {
            return Err(JitError::GraphError(
                "Conv2d requires 4D input [N, C, H, W]".to_string(),
            ));
        }

        let batch_size = input_dims[0];
        let _in_channels = input_dims[1];
        let in_height = input_dims[2];
        let in_width = input_dims[3];

        let out_height = (in_height + 2 * conv_info.padding.0 - conv_info.kernel_size.0)
            / conv_info.stride.0
            + 1;
        let out_width =
            (in_width + 2 * conv_info.padding.1 - conv_info.kernel_size.1) / conv_info.stride.1 + 1;

        Ok(Shape::new(vec![
            batch_size,
            conv_info.out_channels,
            out_height,
            out_width,
        ]))
    }

    /// Validate all shapes are resolved
    fn validate_shapes(&self, graph: &ComputationGraph) -> JitResult<()> {
        for (node_id, _) in graph.nodes() {
            if !self.node_shapes.contains_key(&node_id) {
                return Err(JitError::GraphError(format!(
                    "Unresolved shape for node {:?}",
                    node_id
                )));
            }
        }
        Ok(())
    }

    /// Get inferred shape for a node
    pub fn get_shape(&self, node_id: NodeId) -> Option<&Shape> {
        self.node_shapes.get(&node_id)
    }

    /// Get all inferred shapes
    pub fn get_all_shapes(&self) -> &HashMap<NodeId, Shape> {
        &self.node_shapes
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ShapeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputationGraph, Edge, Node};
    use torsh_core::DeviceType;

    #[test]
    fn test_type_inference_simple() {
        let mut graph = ComputationGraph::new();

        let input = graph.add_node(
            Node::new(Operation::Input, "input".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        let relu = graph.add_node(
            Node::new(Operation::Relu, "relu".to_string())
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu),
        );

        graph.add_edge(input, relu, Edge::default());
        graph.add_input(input);
        graph.add_output(relu);

        let mut type_inference = TypeInference::new();
        assert!(type_inference.infer_types(&graph).is_ok());

        assert_eq!(type_inference.get_type(input), Some(DType::F32));
        assert_eq!(type_inference.get_type(relu), Some(DType::F32));
    }

    #[test]
    fn test_shape_inference_broadcast() {
        let shape_inference = ShapeInference::new();

        let shape1 = Shape::new(vec![1, 3, 1]);
        let shape2 = Shape::new(vec![2, 1, 4]);

        let result = shape_inference.broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_inference_matmul() {
        let shape_inference = ShapeInference::new();

        let left = Shape::new(vec![32, 128]);
        let right = Shape::new(vec![128, 64]);

        let result = shape_inference.matmul_output_shape(&left, &right).unwrap();
        assert_eq!(result.dims(), &[32, 64]);
    }
}
