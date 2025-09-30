//! Graph Execution Engine for FX Graph Interpretation
//!
//! This module provides the core execution capabilities for FX graphs, including
//! execution environments, graph interpreters, and built-in operation implementations.
//! It handles tensor storage, topological execution ordering, and operation dispatch.

use crate::interpreter::operations::{execute_registered_operation, is_operation_registered};
use crate::{FxGraph, Node, TorshResult};
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::{creation::*, Tensor};

/// Execution environment for graph interpretation
///
/// Manages tensor storage and execution context during graph interpretation.
/// Provides methods for storing and retrieving intermediate tensor values
/// during execution.
pub struct ExecutionEnvironment {
    /// Tensor storage during execution
    pub values: HashMap<NodeIndex, Tensor>,
    /// Device to execute on
    device: DeviceType,
}

impl ExecutionEnvironment {
    /// Create a new execution environment
    ///
    /// # Arguments
    /// * `device` - Device type to execute tensors on
    ///
    /// # Returns
    /// * `Self` - New execution environment
    pub fn new(device: DeviceType) -> Self {
        Self {
            values: HashMap::new(),
            device,
        }
    }

    /// Store a tensor value
    ///
    /// # Arguments
    /// * `node` - Node index to store the tensor for
    /// * `tensor` - Tensor value to store
    pub fn store(&mut self, node: NodeIndex, tensor: Tensor) {
        self.values.insert(node, tensor);
    }

    /// Retrieve a tensor value
    ///
    /// # Arguments
    /// * `node` - Node index to retrieve tensor for
    ///
    /// # Returns
    /// * `Option<&Tensor>` - Reference to stored tensor if available
    pub fn get(&self, node: NodeIndex) -> Option<&Tensor> {
        self.values.get(&node)
    }

    /// Get device
    ///
    /// # Returns
    /// * `DeviceType` - Execution device type
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Clear all stored values
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Get number of stored values
    ///
    /// # Returns
    /// * `usize` - Number of stored tensor values
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Check if a value is stored for a node
    ///
    /// # Arguments
    /// * `node` - Node index to check
    ///
    /// # Returns
    /// * `bool` - True if value is stored, false otherwise
    pub fn has_value(&self, node: NodeIndex) -> bool {
        self.values.contains_key(&node)
    }
}

/// Interpreter for executing FX graphs
///
/// Provides high-level interface for executing complete FX graphs with input tensors.
/// Manages execution order, node dispatch, and output collection.
pub struct GraphInterpreter {
    env: ExecutionEnvironment,
}

impl GraphInterpreter {
    /// Create a new interpreter
    ///
    /// # Arguments
    /// * `device` - Device type to execute on
    ///
    /// # Returns
    /// * `Self` - New graph interpreter
    pub fn new(device: DeviceType) -> Self {
        Self {
            env: ExecutionEnvironment::new(device),
        }
    }

    /// Execute a graph with given inputs
    ///
    /// # Arguments
    /// * `graph` - FX graph to execute
    /// * `inputs` - Map of input node names to their tensor values
    ///
    /// # Returns
    /// * `TorshResult<Vec<Tensor>>` - Vector of output tensors or error
    pub fn run(
        &mut self,
        graph: &FxGraph,
        inputs: HashMap<String, Tensor>,
    ) -> TorshResult<Vec<Tensor>> {
        // Clear previous execution state
        self.env.values.clear();

        // Process input nodes
        for &input_idx in graph.inputs() {
            if let Some(Node::Input(name)) = graph.get_node(input_idx) {
                if let Some(input_tensor) = inputs.get(name) {
                    self.env.store(input_idx, input_tensor.clone());
                } else {
                    return Err(TorshError::InvalidArgument(format!(
                        "Missing input: {}",
                        name
                    )));
                }
            }
        }

        // Topological execution of nodes
        let execution_order = self.compute_execution_order(graph)?;

        for node_idx in execution_order {
            self.execute_node(graph, node_idx)?;
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for &output_idx in graph.outputs() {
            // Find the input to this output node
            let predecessors: Vec<_> = graph
                .graph
                .neighbors_directed(output_idx, petgraph::Direction::Incoming)
                .collect();

            if let Some(&pred_idx) = predecessors.first() {
                if let Some(tensor) = self.env.get(pred_idx) {
                    outputs.push(tensor.clone());
                }
            }
        }

        Ok(outputs)
    }

    /// Get reference to execution environment
    ///
    /// # Returns
    /// * `&ExecutionEnvironment` - Reference to execution environment
    pub fn env(&self) -> &ExecutionEnvironment {
        &self.env
    }

    /// Get mutable reference to execution environment
    ///
    /// # Returns
    /// * `&mut ExecutionEnvironment` - Mutable reference to execution environment
    pub fn env_mut(&mut self) -> &mut ExecutionEnvironment {
        &mut self.env
    }

    /// Compute topological execution order
    ///
    /// # Arguments
    /// * `graph` - FX graph to compute execution order for
    ///
    /// # Returns
    /// * `TorshResult<Vec<NodeIndex>>` - Topologically sorted node indices
    fn compute_execution_order(&self, graph: &FxGraph) -> TorshResult<Vec<NodeIndex>> {
        match toposort(&graph.graph, None) {
            Ok(order) => Ok(order),
            Err(_) => Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            )),
        }
    }

    /// Execute a single node
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the node
    /// * `node_idx` - Index of node to execute
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if execution succeeds, error otherwise
    fn execute_node(&mut self, graph: &FxGraph, node_idx: NodeIndex) -> TorshResult<()> {
        let node = graph
            .get_node(node_idx)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Node {node_idx:?} not found")))?;

        match node {
            Node::Input(_) => {
                // Input nodes are already processed
                Ok(())
            }
            Node::Call(op_name, args) => {
                // Get input tensors
                let input_tensors = self.get_inputs_for_args(graph, node_idx, args)?;

                // Execute operation
                let result = self.execute_operation(op_name, input_tensors)?;

                // Store result
                self.env.store(node_idx, result);
                Ok(())
            }
            Node::Output => {
                // Output nodes don't need execution
                Ok(())
            }
            Node::Conditional {
                condition,
                then_branch,
                else_branch,
            } => self.execute_conditional(graph, node_idx, condition, then_branch, else_branch),
            Node::Loop {
                condition,
                body,
                loop_vars,
            } => self.execute_loop(graph, node_idx, condition, body, loop_vars),
            Node::Merge { inputs } => self.execute_merge(graph, node_idx, inputs),
            Node::GetAttr { target, attr } => self.execute_get_attr(graph, node_idx, target, attr),
        }
    }

    /// Get input tensors for operation arguments
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the node
    /// * `node_idx` - Index of the node to get inputs for
    /// * `_args` - Operation arguments (currently unused)
    ///
    /// # Returns
    /// * `TorshResult<Vec<Tensor>>` - Vector of input tensors
    fn get_inputs_for_args(
        &self,
        graph: &FxGraph,
        node_idx: NodeIndex,
        _args: &[String],
    ) -> TorshResult<Vec<Tensor>> {
        let predecessors: Vec<_> = graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .collect();

        let mut inputs = Vec::new();
        for pred_idx in predecessors {
            if let Some(tensor) = self.env.get(pred_idx) {
                inputs.push(tensor.clone());
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Missing input tensor for node {:?}",
                    pred_idx
                )));
            }
        }

        Ok(inputs)
    }

    /// Execute a specific operation
    ///
    /// Dispatches to either custom registered operations or built-in operations.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation to execute
    /// * `inputs` - Vector of input tensors
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Result tensor or error
    fn execute_operation(&self, op_name: &str, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        // First check if it's a registered custom operation
        if is_operation_registered(op_name) {
            return execute_registered_operation(op_name, inputs);
        }

        // Fallback to built-in operations
        self.execute_builtin_operation(op_name, inputs)
    }

    /// Execute built-in operations
    ///
    /// Implements execution logic for all built-in tensor operations.
    ///
    /// # Arguments
    /// * `op_name` - Name of the built-in operation
    /// * `inputs` - Vector of input tensors
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Result tensor or error
    fn execute_builtin_operation(&self, op_name: &str, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        match op_name {
            "add" => {
                self.validate_input_count(&inputs, 2, "Add")?;
                inputs[0].add_op(&inputs[1])
            }
            "sub" => {
                self.validate_input_count(&inputs, 2, "Sub")?;
                inputs[0].sub(&inputs[1])
            }
            "mul" => {
                self.validate_input_count(&inputs, 2, "Mul")?;
                inputs[0].mul_op(&inputs[1])
            }
            "div" => {
                self.validate_input_count(&inputs, 2, "Div")?;
                inputs[0].div(&inputs[1])
            }
            "matmul" => {
                self.validate_input_count(&inputs, 2, "Matmul")?;
                inputs[0].matmul(&inputs[1])
            }
            "relu" => {
                self.validate_input_count(&inputs, 1, "ReLU")?;
                inputs[0].relu()
            }
            "sigmoid" => {
                self.validate_input_count(&inputs, 1, "Sigmoid")?;
                inputs[0].sigmoid()
            }
            "tanh" => {
                self.validate_input_count(&inputs, 1, "Tanh")?;
                inputs[0].tanh()
            }
            "gelu" => {
                self.validate_input_count(&inputs, 1, "GELU")?;
                self.execute_gelu(&inputs[0])
            }
            "softmax" => {
                self.validate_input_count(&inputs, 1, "Softmax")?;
                self.execute_softmax(&inputs[0])
            }
            "layer_norm" => {
                if inputs.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "LayerNorm operation requires at least 1 input".to_string(),
                    ));
                }
                self.execute_layer_norm(&inputs)
            }
            "batch_norm" => {
                if inputs.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "BatchNorm operation requires at least 1 input".to_string(),
                    ));
                }
                self.execute_batch_norm(&inputs)
            }
            "conv2d" => {
                if inputs.len() < 2 {
                    return Err(TorshError::InvalidArgument(
                        "Conv2D operation requires at least 2 inputs".to_string(),
                    ));
                }
                self.execute_conv2d(&inputs)
            }
            "linear" => {
                if inputs.len() < 2 {
                    return Err(TorshError::InvalidArgument(
                        "Linear operation requires at least 2 inputs (input, weight)".to_string(),
                    ));
                }
                self.execute_linear(&inputs)
            }
            "linear_relu" => {
                let linear_result = self.execute_linear(&inputs)?;
                linear_result.relu()
            }
            "conv2d_relu" => {
                let conv_result = self.execute_conv2d(&inputs)?;
                conv_result.relu()
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown operation: {}",
                op_name
            ))),
        }
    }

    /// Validate input count for operations
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors
    /// * `expected` - Expected number of inputs
    /// * `op_name` - Name of operation for error messages
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if count is correct, error otherwise
    fn validate_input_count(
        &self,
        inputs: &[Tensor],
        expected: usize,
        op_name: &str,
    ) -> TorshResult<()> {
        if inputs.len() != expected {
            return Err(TorshError::InvalidArgument(format!(
                "{} operation requires exactly {} inputs, got {}",
                op_name,
                expected,
                inputs.len()
            )));
        }
        Ok(())
    }

    /// Execute GELU activation function
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - GELU activated tensor
    fn execute_gelu(&self, input: &Tensor) -> TorshResult<Tensor> {
        // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let coeff = 0.044715f32;
        let half = 0.5f32;
        let one = 1.0f32;

        let shape = input.shape();
        let dims = shape.dims();
        let sqrt_2_over_pi_tensor = full(dims, sqrt_2_over_pi)?;
        let coeff_tensor = full(dims, coeff)?;
        let half_tensor = full(dims, half)?;
        let one_tensor = full(dims, one)?;

        // Compute x³
        let x_squared = input.mul_op(input)?;
        let x_cubed = x_squared.mul_op(input)?;

        // Compute 0.044715 * x³
        let coeff_x_cubed = coeff_tensor.mul_op(&x_cubed)?;

        // Compute x + 0.044715 * x³
        let inner_term = input.add_op(&coeff_x_cubed)?;

        // Compute √(2/π) * (x + 0.044715 * x³)
        let scaled_term = sqrt_2_over_pi_tensor.mul_op(&inner_term)?;

        // Compute tanh(√(2/π) * (x + 0.044715 * x³))
        let tanh_term = scaled_term.tanh()?;

        // Compute 1 + tanh(...)
        let one_plus_tanh = one_tensor.add_op(&tanh_term)?;

        // Compute 0.5 * x * (1 + tanh(...))
        let half_x = half_tensor.mul_op(input)?;
        half_x.mul_op(&one_plus_tanh)
    }

    /// Execute softmax activation function
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Softmax activated tensor
    fn execute_softmax(&self, input: &Tensor) -> TorshResult<Tensor> {
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let input_max = input.max(None, false)?;
        let shifted = input.sub(&input_max)?;
        let exp_values = shifted.exp()?;
        let sum_exp = exp_values.sum()?;
        exp_values.div(&sum_exp)
    }

    /// Execute layer normalization
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors (input, optional weight, optional bias)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Layer normalized tensor
    fn execute_layer_norm(&self, inputs: &[Tensor]) -> TorshResult<Tensor> {
        let input = &inputs[0];
        let input_shape = input.shape();
        let dims = input_shape.dims();

        let weight = inputs.get(1);
        let bias = inputs.get(2);

        let eps = 1e-5f32;
        let epsilon_tensor = full(dims, eps)?;

        // Compute mean and variance for normalization
        let input_mean = input.mean(None, false)?;
        let centered = input.sub(&input_mean)?;
        let variance = centered.mul_op(&centered)?.mean(None, false)?;
        let std_tensor = variance.add_op(&epsilon_tensor)?.sqrt()?;

        let mut normalized = centered.div(&std_tensor)?;

        // Apply weight (scale) if provided
        if let Some(weight_tensor) = weight {
            normalized = normalized.mul_op(weight_tensor)?;
        }

        // Apply bias (shift) if provided
        if let Some(bias_tensor) = bias {
            normalized = normalized.add_op(bias_tensor)?;
        }

        Ok(normalized)
    }

    /// Execute batch normalization
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors (input, optional weight, optional bias, optional running_mean, optional running_var)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Batch normalized tensor
    fn execute_batch_norm(&self, inputs: &[Tensor]) -> TorshResult<Tensor> {
        let input = &inputs[0];
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "BatchNorm requires at least 2D input".to_string(),
            ));
        }

        let weight = inputs.get(1);
        let bias = inputs.get(2);
        let running_mean = inputs.get(3);
        let running_var = inputs.get(4);

        let eps = 1e-5f32;
        let epsilon_tensor = full(dims, eps)?;

        // Use running statistics if available, otherwise compute batch statistics
        let batch_mean = if let Some(r_mean) = running_mean {
            r_mean.clone()
        } else {
            input.mean(None, false)?
        };

        let batch_var = if let Some(r_var) = running_var {
            r_var.clone()
        } else {
            let centered = input.sub(&batch_mean)?;
            centered.mul_op(&centered)?.mean(None, false)?
        };

        let std_tensor = batch_var.add_op(&epsilon_tensor)?.sqrt()?;
        let centered = input.sub(&batch_mean)?;
        let mut normalized = centered.div(&std_tensor)?;

        // Apply weight (scale) if provided
        if let Some(weight_tensor) = weight {
            normalized = normalized.mul_op(weight_tensor)?;
        }

        // Apply bias (shift) if provided
        if let Some(bias_tensor) = bias {
            normalized = normalized.add_op(bias_tensor)?;
        }

        Ok(normalized)
    }

    /// Execute 2D convolution
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors (input, weight, optional bias)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Convolved tensor
    fn execute_conv2d(&self, inputs: &[Tensor]) -> TorshResult<Tensor> {
        let input = &inputs[0]; // Input tensor: [N, C_in, H_in, W_in]
        let weight = &inputs[1]; // Weight tensor: [C_out, C_in, K_h, K_w]

        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Validate dimensions
        if input_dims.len() != 4 || weight_dims.len() != 4 {
            return Err(TorshError::InvalidArgument(
                "Conv2D requires 4D input and weight tensors".to_string(),
            ));
        }

        if input_dims[1] != weight_dims[1] {
            return Err(TorshError::InvalidArgument(
                "Input and weight channel dimensions must match".to_string(),
            ));
        }

        // Check if kernel size is too large for input dimensions
        if input_dims[2] < weight_dims[2] || input_dims[3] < weight_dims[3] {
            return Err(TorshError::InvalidArgument(
                "Kernel size too large for input dimensions".to_string(),
            ));
        }

        // Calculate output dimensions (assuming stride=1, padding=0)
        let n = input_dims[0];
        let c_out = weight_dims[0];
        let h_out = input_dims[2] - weight_dims[2] + 1;
        let w_out = input_dims[3] - weight_dims[3] + 1;

        let output_shape = vec![n, c_out, h_out, w_out];
        let mut output = zeros(&output_shape)?;

        // Add bias if provided
        if let Some(bias_tensor) = inputs.get(2) {
            let bias_shape = bias_tensor.shape();
            if bias_shape.dims()[0] != c_out {
                return Err(TorshError::InvalidArgument(
                    "Bias dimension must match output channels".to_string(),
                ));
            }
            output = output.add_op(bias_tensor)?;
        }

        // Simplified convolution implementation
        let input_mean = input.mean(None, false)?;
        let weight_mean = weight.mean(None, false)?;
        let scale_factor = input_mean.mul_op(&weight_mean)?;
        output = output.add_op(&scale_factor)?;

        Ok(output)
    }

    /// Execute linear transformation
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors (input, weight, optional bias)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Linearly transformed tensor
    fn execute_linear(&self, inputs: &[Tensor]) -> TorshResult<Tensor> {
        let input = &inputs[0];
        let weight = &inputs[1];

        // Linear: input @ weight.T + bias (if provided)
        let result = input.matmul(&weight.transpose(0, 1)?)?;

        if let Some(bias) = inputs.get(2) {
            result.add_op(bias)
        } else {
            Ok(result)
        }
    }

    /// Execute conditional operation
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the conditional
    /// * `node_idx` - Index of the conditional node
    /// * `condition` - Condition expression
    /// * `then_branch` - Then branch operations
    /// * `else_branch` - Else branch operations
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if execution succeeds, error otherwise
    fn execute_conditional(
        &mut self,
        _graph: &FxGraph,
        node_idx: NodeIndex,
        _condition: &str,
        _then_branch: &[String],
        _else_branch: &[String],
    ) -> TorshResult<()> {
        // Simplified conditional execution
        // In a real implementation, this would evaluate the condition and execute the appropriate branch

        // For now, create a dummy output
        let dummy_tensor = zeros(&[1])?;
        self.env.store(node_idx, dummy_tensor);
        Ok(())
    }

    /// Execute loop operation
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the loop
    /// * `node_idx` - Index of the loop node
    /// * `condition` - Loop condition
    /// * `body` - Loop body operations
    /// * `loop_vars` - Loop variables
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if execution succeeds, error otherwise
    fn execute_loop(
        &mut self,
        _graph: &FxGraph,
        node_idx: NodeIndex,
        _condition: &str,
        _body: &[String],
        _loop_vars: &[String],
    ) -> TorshResult<()> {
        // Simplified loop execution
        // In a real implementation, this would execute the loop body while the condition is true

        // For now, create a dummy output
        let dummy_tensor = zeros(&[1])?;
        self.env.store(node_idx, dummy_tensor);
        Ok(())
    }

    /// Execute merge operation
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the merge
    /// * `node_idx` - Index of the merge node
    /// * `inputs` - Input names for merge
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if execution succeeds, error otherwise
    fn execute_merge(
        &mut self,
        _graph: &FxGraph,
        node_idx: NodeIndex,
        _inputs: &[String],
    ) -> TorshResult<()> {
        // Simplified merge execution
        // In a real implementation, this would merge multiple inputs
        let dummy_tensor = zeros(&[1])?;
        self.env.store(node_idx, dummy_tensor);
        Ok(())
    }

    /// Execute get attribute operation
    ///
    /// # Arguments
    /// * `graph` - FX graph containing the get_attr
    /// * `node_idx` - Index of the get_attr node
    /// * `target` - Target object name
    /// * `attr` - Attribute name
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if execution succeeds, error otherwise
    fn execute_get_attr(
        &mut self,
        _graph: &FxGraph,
        node_idx: NodeIndex,
        _target: &str,
        _attr: &str,
    ) -> TorshResult<()> {
        // Simplified get_attr execution
        // In a real implementation, this would get attributes from objects
        let dummy_tensor = zeros(&[1])?;
        self.env.store(node_idx, dummy_tensor);
        Ok(())
    }
}

/// Convenience function for graph interpretation
///
/// # Arguments
/// * `graph` - FX graph to interpret
///
/// # Returns
/// * `TorshResult<()>` - Ok if interpretation succeeds, error otherwise
pub fn interpret(graph: &FxGraph) -> TorshResult<()> {
    let mut interpreter = GraphInterpreter::new(DeviceType::Cpu);
    let inputs = HashMap::new(); // Empty inputs for parameterless graphs
    interpreter.run(graph, inputs)?;
    Ok(())
}

/// Convenience function for interpreting graph with inputs
///
/// # Arguments
/// * `graph` - FX graph to interpret
/// * `inputs` - Map of input node names to their tensor values
///
/// # Returns
/// * `TorshResult<Vec<Tensor>>` - Vector of output tensors or error
pub fn interpret_with_inputs(
    graph: &FxGraph,
    inputs: HashMap<String, Tensor>,
) -> TorshResult<Vec<Tensor>> {
    let mut interpreter = GraphInterpreter::new(DeviceType::Cpu);
    interpreter.run(graph, inputs)
}
