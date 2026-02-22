//! Type Checking System for FX Graph Validation
//!
//! This module provides comprehensive type checking capabilities for FX graphs.
//! It validates type compatibility across operations, performs type inference,
//! and ensures type safety throughout graph execution.

use crate::interpreter::operations::{global_registry, is_operation_registered};
use crate::{FxGraph, Node, TorshResult};
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use torsh_core::{dtype::DType, error::TorshError};

/// Type checking context
///
/// Manages type validation for an entire graph, tracking type information
/// for each node and providing methods for propagating types through operations.
pub struct TypeCheckingContext {
    /// Type information for each node
    types: HashMap<NodeIndex, DType>,
}

impl TypeCheckingContext {
    /// Create a new type checking context
    ///
    /// # Returns
    /// * `Self` - New empty type checking context
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    /// Set type for a node
    ///
    /// # Arguments
    /// * `node` - Node index to set type for
    /// * `dtype` - Data type to associate with the node
    pub fn set_type(&mut self, node: NodeIndex, dtype: DType) {
        self.types.insert(node, dtype);
    }

    /// Get type for a node
    ///
    /// # Arguments
    /// * `node` - Node index to get type for
    ///
    /// # Returns
    /// * `Option<DType>` - Data type if available
    pub fn get_type(&self, node: NodeIndex) -> Option<DType> {
        self.types.get(&node).copied()
    }

    /// Perform type checking for the entire graph
    ///
    /// Performs a topological traversal of the graph and validates types for all nodes
    /// based on input types and operation-specific type checking rules.
    ///
    /// # Arguments
    /// * `graph` - FX graph to perform type checking on
    /// * `input_types` - Map of input node names to their data types
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if type checking succeeds, error otherwise
    pub fn check_types(
        &mut self,
        graph: &FxGraph,
        input_types: HashMap<String, DType>,
    ) -> TorshResult<()> {
        // Set types for input nodes
        for &input_idx in graph.inputs() {
            if let Some(Node::Input(name)) = graph.get_node(input_idx) {
                if let Some(&dtype) = input_types.get(name) {
                    self.set_type(input_idx, dtype);
                }
            }
        }

        // Perform topological traversal and check types
        let execution_order = self.compute_execution_order(graph)?;

        for node_idx in execution_order {
            if let Some(node) = graph.get_node(node_idx) {
                match node {
                    Node::Input(_) => {
                        // Already handled above
                    }
                    Node::Call(op_name, _) => {
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        self.validate_operation_types(op_name, &input_types)?;
                        let output_type = self.infer_operation_type(op_name, &input_types)?;
                        self.set_type(node_idx, output_type);
                    }
                    Node::Conditional { .. } => {
                        // For conditionals, validate that all branches have compatible types
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        let output_type = self.validate_conditional_types(&input_types)?;
                        self.set_type(node_idx, output_type);
                    }
                    Node::Loop { .. } => {
                        // For loops, validate loop variable types
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        let output_type = if let Some(&first_type) = input_types.first() {
                            first_type
                        } else {
                            DType::F32 // Default type
                        };
                        self.set_type(node_idx, output_type);
                    }
                    Node::Output => {
                        // Output nodes inherit type from their input
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        if let Some(&input_type) = input_types.first() {
                            self.set_type(node_idx, input_type);
                        }
                    }
                    Node::Merge { .. } => {
                        // Merge nodes use the type of their first input
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        let output_type = if let Some(&first_type) = input_types.first() {
                            first_type
                        } else {
                            DType::F32 // Default type
                        };
                        self.set_type(node_idx, output_type);
                    }
                    Node::GetAttr { .. } => {
                        // GetAttr nodes inherit type from their target
                        let input_types = self.get_input_types_for_node(graph, node_idx)?;
                        let output_type = if let Some(&first_type) = input_types.first() {
                            first_type
                        } else {
                            DType::F32 // Default type
                        };
                        self.set_type(node_idx, output_type);
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute execution order for type checking
    ///
    /// Performs topological sort to determine the order in which nodes should be processed
    /// for type checking. This ensures dependencies are processed before dependent nodes.
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

    /// Get input types for a specific node
    ///
    /// Collects type information from all input nodes to the specified node.
    /// This is used during type checking to determine the types available
    /// for operation-specific type validation.
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the node
    /// * `node_idx` - Index of the node to get input types for
    ///
    /// # Returns
    /// * `TorshResult<Vec<DType>>` - Vector of input data types
    fn get_input_types_for_node(
        &self,
        graph: &FxGraph,
        node_idx: NodeIndex,
    ) -> TorshResult<Vec<DType>> {
        let mut input_types = Vec::new();

        // Get all predecessor nodes
        let predecessors: Vec<_> = graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .collect();

        for pred_idx in predecessors {
            if let Some(dtype) = self.get_type(pred_idx) {
                input_types.push(dtype);
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Missing type information for predecessor node {:?}",
                    pred_idx
                )));
            }
        }

        Ok(input_types)
    }

    /// Validate types for a specific operation
    ///
    /// Checks that the input types are compatible with the specified operation.
    /// This includes both built-in operations and custom registered operations.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    /// * `input_types` - Vector of input data types
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if types are valid, error otherwise
    fn validate_operation_types(&self, op_name: &str, input_types: &[DType]) -> TorshResult<()> {
        if input_types.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No input types provided for operation".to_string(),
            ));
        }

        // Handle built-in operations
        if self.is_builtin_operation(op_name) {
            return self.validate_builtin_operation_types(op_name, input_types);
        }

        // Handle custom registered operations
        if is_operation_registered(op_name) {
            if let Ok(operation) = global_registry().get(op_name) {
                return operation.validate_types(input_types);
            }
        }

        // Unknown operation - accept any types
        Ok(())
    }

    /// Infer output type for a specific operation
    ///
    /// Uses operation-specific rules to infer the output type based on input types.
    /// Supports both built-in operations and custom registered operations.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    /// * `input_types` - Vector of input data types
    ///
    /// # Returns
    /// * `TorshResult<DType>` - Inferred output data type
    fn infer_operation_type(&self, op_name: &str, input_types: &[DType]) -> TorshResult<DType> {
        if input_types.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No input types provided for operation".to_string(),
            ));
        }

        // Handle built-in operations
        if self.is_builtin_operation(op_name) {
            return self.infer_builtin_operation_type(op_name, input_types);
        }

        // Handle custom registered operations
        if is_operation_registered(op_name) {
            if let Ok(operation) = global_registry().get(op_name) {
                return operation.infer_type(input_types);
            }
        }

        // Default: use first input type
        Ok(input_types[0])
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

    /// Validate types for built-in operations
    ///
    /// Implements type validation rules for all built-in operations.
    /// Each operation has specific requirements for input type compatibility.
    ///
    /// # Arguments
    /// * `op_name` - Name of the built-in operation
    /// * `input_types` - Vector of input data types
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if types are valid, error otherwise
    fn validate_builtin_operation_types(
        &self,
        op_name: &str,
        input_types: &[DType],
    ) -> TorshResult<()> {
        match op_name {
            "add" | "sub" | "mul" | "div" => {
                // Arithmetic operations: require at least one input, prefer numeric types
                if input_types.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "Arithmetic operations require at least one input".to_string(),
                    ));
                }

                // Check that types are numeric
                for &dtype in input_types {
                    if matches!(dtype, DType::Bool) {
                        return Err(TorshError::InvalidArgument(
                            "Arithmetic operations do not support boolean types".to_string(),
                        ));
                    }
                }

                // For binary operations, check compatibility
                if input_types.len() >= 2 {
                    self.validate_arithmetic_compatibility(input_types[0], input_types[1])?;
                }

                Ok(())
            }
            "matmul" => {
                // Matrix multiplication: requires exactly two numeric inputs
                if input_types.len() != 2 {
                    return Err(TorshError::InvalidArgument(
                        "Matrix multiplication requires exactly two inputs".to_string(),
                    ));
                }

                for &dtype in input_types {
                    if !self.is_numeric_type(dtype) {
                        return Err(TorshError::InvalidArgument(
                            "Matrix multiplication requires numeric types".to_string(),
                        ));
                    }
                }

                Ok(())
            }
            "relu" | "sigmoid" | "tanh" | "gelu" => {
                // Activation functions: require single numeric input
                if input_types.len() != 1 {
                    return Err(TorshError::InvalidArgument(
                        "Activation functions require exactly one input".to_string(),
                    ));
                }

                if !self.is_numeric_type(input_types[0]) {
                    return Err(TorshError::InvalidArgument(
                        "Activation functions require numeric types".to_string(),
                    ));
                }

                Ok(())
            }
            "softmax" => {
                // Softmax: requires single numeric input, outputs float
                if input_types.len() != 1 {
                    return Err(TorshError::InvalidArgument(
                        "Softmax requires exactly one input".to_string(),
                    ));
                }

                if !self.is_numeric_type(input_types[0]) {
                    return Err(TorshError::InvalidArgument(
                        "Softmax requires numeric types".to_string(),
                    ));
                }

                Ok(())
            }
            "layer_norm" | "batch_norm" => {
                // Normalization: requires single numeric input
                if input_types.len() < 1 {
                    return Err(TorshError::InvalidArgument(
                        "Normalization operations require at least one input".to_string(),
                    ));
                }

                if !self.is_numeric_type(input_types[0]) {
                    return Err(TorshError::InvalidArgument(
                        "Normalization operations require numeric types".to_string(),
                    ));
                }

                Ok(())
            }
            "conv2d" => {
                // Convolution: requires at least input and kernel (numeric)
                if input_types.len() < 2 {
                    return Err(TorshError::InvalidArgument(
                        "Convolution requires at least input and kernel".to_string(),
                    ));
                }

                for &dtype in input_types.iter().take(2) {
                    if !self.is_numeric_type(dtype) {
                        return Err(TorshError::InvalidArgument(
                            "Convolution requires numeric types".to_string(),
                        ));
                    }
                }

                Ok(())
            }
            _ => {
                // Unknown operation: accept any types
                Ok(())
            }
        }
    }

    /// Infer output type for built-in operations
    ///
    /// Implements type inference rules for all built-in operations.
    /// Each operation has specific rules for how output types are determined
    /// from input types.
    ///
    /// # Arguments
    /// * `op_name` - Name of the built-in operation
    /// * `input_types` - Vector of input data types
    ///
    /// # Returns
    /// * `TorshResult<DType>` - Inferred output data type
    fn infer_builtin_operation_type(
        &self,
        op_name: &str,
        input_types: &[DType],
    ) -> TorshResult<DType> {
        match op_name {
            "add" | "sub" | "mul" | "div" => {
                // Arithmetic operations: use type promotion rules
                if input_types.len() >= 2 {
                    Ok(self.promote_dtype(input_types[0], input_types[1]))
                } else {
                    Ok(input_types[0])
                }
            }
            "matmul" => {
                // Matrix multiplication: use type promotion rules
                if input_types.len() >= 2 {
                    Ok(self.promote_dtype(input_types[0], input_types[1]))
                } else {
                    Ok(input_types[0])
                }
            }
            "relu" | "sigmoid" | "tanh" | "gelu" => {
                // Activation functions: preserve input type
                Ok(input_types[0])
            }
            "softmax" => {
                // Softmax: always outputs float type
                match input_types[0] {
                    DType::F16 => Ok(DType::F16),
                    DType::F32 => Ok(DType::F32),
                    DType::F64 => Ok(DType::F64),
                    _ => Ok(DType::F32), // Default to F32 for integer inputs
                }
            }
            "layer_norm" | "batch_norm" => {
                // Normalization: preserve input type (but promote integers to float)
                match input_types[0] {
                    DType::F16 | DType::F32 | DType::F64 => Ok(input_types[0]),
                    _ => Ok(DType::F32), // Promote integers to F32
                }
            }
            "conv2d" => {
                // Convolution: use type promotion of input and kernel
                if input_types.len() >= 2 {
                    Ok(self.promote_dtype(input_types[0], input_types[1]))
                } else {
                    Ok(input_types[0])
                }
            }
            _ => {
                // Unknown operation: use first input type
                Ok(input_types[0])
            }
        }
    }

    /// Validate types for conditional operations
    ///
    /// Ensures that all branches of a conditional have compatible types.
    ///
    /// # Arguments
    /// * `input_types` - Vector of input data types from all branches
    ///
    /// # Returns
    /// * `TorshResult<DType>` - Common output type or error if incompatible
    fn validate_conditional_types(&self, input_types: &[DType]) -> TorshResult<DType> {
        if input_types.is_empty() {
            return Ok(DType::F32); // Default type
        }

        let first_type = input_types[0];

        // All branches should have the same type or compatible types
        for &dtype in input_types.iter().skip(1) {
            if !self.are_types_compatible(first_type, dtype) {
                return Err(TorshError::InvalidArgument(format!(
                    "Incompatible types in conditional: {:?} and {:?}",
                    first_type, dtype
                )));
            }
        }

        // Return the promoted type
        Ok(input_types
            .iter()
            .fold(first_type, |acc, &dtype| self.promote_dtype(acc, dtype)))
    }

    /// Check if a data type is numeric
    ///
    /// # Arguments
    /// * `dtype` - Data type to check
    ///
    /// # Returns
    /// * `bool` - True if type is numeric, false otherwise
    fn is_numeric_type(&self, dtype: DType) -> bool {
        matches!(
            dtype,
            DType::F16
                | DType::F32
                | DType::F64
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
        )
    }

    /// Check if two types are compatible
    ///
    /// # Arguments
    /// * `type1` - First data type
    /// * `type2` - Second data type
    ///
    /// # Returns
    /// * `bool` - True if types are compatible, false otherwise
    fn are_types_compatible(&self, type1: DType, type2: DType) -> bool {
        // Same types are always compatible
        if type1 == type2 {
            return true;
        }

        // Numeric types are generally compatible
        self.is_numeric_type(type1) && self.is_numeric_type(type2)
    }

    /// Validate arithmetic compatibility between two types
    ///
    /// # Arguments
    /// * `type1` - First data type
    /// * `type2` - Second data type
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if compatible, error otherwise
    fn validate_arithmetic_compatibility(&self, type1: DType, type2: DType) -> TorshResult<()> {
        if self.are_types_compatible(type1, type2) {
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Incompatible types for arithmetic operation: {:?} and {:?}",
                type1, type2
            )))
        }
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

            // Unsigned integers
            (U8, I8) | (I8, U8) => I16, // Promote to larger signed type
            (U8, _) | (_, U8) => dtype_max(dtype1, dtype2),

            // Boolean operations default to the non-boolean type
            (Bool, other) | (other, Bool) => other,

            // Default case
            _ => dtype1,
        }
    }

    /// Get all inferred types
    ///
    /// # Returns
    /// * `&HashMap<NodeIndex, DType>` - Reference to all inferred types
    pub fn get_all_types(&self) -> &HashMap<NodeIndex, DType> {
        &self.types
    }

    /// Validate that all required types have been inferred
    ///
    /// # Arguments
    /// * `graph` - FX graph to validate
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if all types are available, error otherwise
    pub fn validate_complete_inference(&self, graph: &FxGraph) -> TorshResult<()> {
        for node_idx in graph.graph.node_indices() {
            if self.get_type(node_idx).is_none() {
                return Err(TorshError::InvalidArgument(format!(
                    "Missing type information for node {:?}",
                    node_idx
                )));
            }
        }
        Ok(())
    }
}

impl Default for TypeCheckingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to compare data types for promotion
fn dtype_max(dtype1: DType, dtype2: DType) -> DType {
    use DType::*;
    match (dtype1, dtype2) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (F16, _) | (_, F16) => F16,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (I16, _) | (_, I16) => I16,
        (I8, _) | (_, I8) => I8,
        (U8, _) | (_, U8) => U8,
        _ => dtype1,
    }
}
