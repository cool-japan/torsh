//! Debug execution engine for JIT debugging
//!
//! This module provides the execution engine specifically designed for debugging,
//! with instrumentation and step-by-step execution capabilities.

use super::core::{
    DebugStatistics, DebugValue, DebuggerConfig, ExecutionState, InstructionExecutionResult,
    NodeExecutionResult,
};
use crate::{
    graph::Node,
    ir::{Instruction, IrModule, IrOpcode},
    ComputationGraph, JitError, JitResult, NodeId,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::{DType, Shape};

/// Debug execution engine
///
/// Provides specialized execution capabilities for debugging including
/// instrumentation, step-by-step execution, and state tracking.
pub struct DebugExecutionEngine {
    config: DebuggerConfig,
    execution_count: usize,
    total_execution_time: Duration,
    instruction_timings: HashMap<String, Vec<Duration>>,
    operation_stats: HashMap<String, OperationStatistics>,
}

/// Statistics for individual operations
#[derive(Debug, Clone)]
pub struct OperationStatistics {
    pub count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl DebugExecutionEngine {
    /// Create a new debug execution engine
    ///
    /// # Arguments
    /// * `config` - Configuration for the debug execution engine
    pub fn new(config: DebuggerConfig) -> Self {
        Self {
            config,
            execution_count: 0,
            total_execution_time: Duration::new(0, 0),
            instruction_timings: HashMap::new(),
            operation_stats: HashMap::new(),
        }
    }

    /// Execute a graph node with debugging instrumentation
    ///
    /// # Arguments
    /// * `node` - The node to execute
    /// * `graph` - The computation graph containing the node
    /// * `node_id` - The ID of the node being executed
    ///
    /// # Returns
    /// The result of node execution with timing information
    pub fn execute_node_debug(
        &mut self,
        node: &Node,
        graph: &ComputationGraph,
        node_id: NodeId,
    ) -> JitResult<NodeExecutionResult> {
        let start_time = Instant::now();

        // Execute the actual node operation
        let result = self.execute_node_operation(node, graph, node_id)?;

        let execution_time = start_time.elapsed();
        self.record_operation_timing(node.op.as_str(), execution_time);
        self.execution_count += 1;
        self.total_execution_time += execution_time;

        Ok(result)
    }

    /// Execute an IR instruction with debugging instrumentation
    ///
    /// # Arguments
    /// * `instruction` - The instruction to execute
    /// * `ir_module` - The IR module containing the instruction
    /// * `execution_state` - Current execution state
    ///
    /// # Returns
    /// The result of instruction execution
    pub fn execute_instruction_debug(
        &mut self,
        instruction: &Instruction,
        ir_module: &IrModule,
        execution_state: &mut ExecutionState,
    ) -> JitResult<InstructionExecutionResult> {
        let start_time = Instant::now();

        // Execute the actual instruction
        let result = self.execute_ir_instruction(instruction, ir_module, execution_state)?;

        let execution_time = start_time.elapsed();
        let instruction_name = format!("{:?}", instruction.opcode);
        self.record_operation_timing(&instruction_name, execution_time);
        self.execution_count += 1;
        self.total_execution_time += execution_time;

        Ok(result)
    }

    /// Execute a node operation (implementation)
    fn execute_node_operation(
        &self,
        node: &Node,
        graph: &ComputationGraph,
        node_id: NodeId,
    ) -> JitResult<NodeExecutionResult> {
        // Get input values from predecessor nodes
        let inputs = self.get_node_inputs(graph, node_id)?;

        // Execute based on operation type
        match node.op.as_str() {
            "add" => self.execute_add_operation(&inputs),
            "mul" => self.execute_multiply_operation(&inputs),
            "sub" => self.execute_subtract_operation(&inputs),
            "div" => self.execute_divide_operation(&inputs),
            "relu" => self.execute_relu_operation(&inputs),
            "sigmoid" => self.execute_sigmoid_operation(&inputs),
            "tanh" => self.execute_tanh_operation(&inputs),
            "matmul" => self.execute_matmul_operation(&inputs),
            "reshape" => self.execute_reshape_operation(&inputs, &node.attrs),
            "transpose" => self.execute_transpose_operation(&inputs),
            "concat" => self.execute_concat_operation(&inputs, &node.attrs),
            "split" => self.execute_split_operation(&inputs, &node.attrs),
            _ => {
                // Default implementation for unknown operations
                Ok(NodeExecutionResult {
                    data: vec![0.0],
                    shape: Shape::new(vec![1]),
                    dtype: DType::F32,
                })
            }
        }
    }

    /// Execute an IR instruction (implementation)
    fn execute_ir_instruction(
        &self,
        instruction: &Instruction,
        ir_module: &IrModule,
        execution_state: &mut ExecutionState,
    ) -> JitResult<InstructionExecutionResult> {
        match instruction.opcode {
            IrOpcode::Add => {
                let result = self.execute_ir_add(instruction, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Mul => {
                let result = self.execute_ir_multiply(instruction, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Sub => {
                let result = self.execute_ir_subtract(instruction, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Div => {
                let result = self.execute_ir_divide(instruction, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Const => {
                let result = self.execute_ir_const(instruction)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Load => {
                let result = self.execute_ir_load(instruction, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            IrOpcode::Store => {
                self.execute_ir_store(instruction, execution_state)?;
                Ok(InstructionExecutionResult::SideEffect)
            }
            IrOpcode::Return => Ok(InstructionExecutionResult::Return),
            IrOpcode::Call => {
                let result = self.execute_ir_call(instruction, ir_module, execution_state)?;
                Ok(InstructionExecutionResult::Value(result))
            }
            _ => Ok(InstructionExecutionResult::NoOp),
        }
    }

    // Node operation implementations

    fn execute_add_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 2 {
            return Err(JitError::RuntimeError(
                "Add operation requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape != b.shape {
            return Err(JitError::RuntimeError(
                "Shape mismatch in add operation".to_string(),
            ));
        }

        let result_data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    fn execute_multiply_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 2 {
            return Err(JitError::RuntimeError(
                "Multiply operation requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape != b.shape {
            return Err(JitError::RuntimeError(
                "Shape mismatch in multiply operation".to_string(),
            ));
        }

        let result_data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    fn execute_subtract_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 2 {
            return Err(JitError::RuntimeError(
                "Subtract operation requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape != b.shape {
            return Err(JitError::RuntimeError(
                "Shape mismatch in subtract operation".to_string(),
            ));
        }

        let result_data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    fn execute_divide_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 2 {
            return Err(JitError::RuntimeError(
                "Divide operation requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape != b.shape {
            return Err(JitError::RuntimeError(
                "Shape mismatch in divide operation".to_string(),
            ));
        }

        let result_data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| {
                if y.abs() < f32::EPSILON {
                    f32::INFINITY
                } else {
                    x / y
                }
            })
            .collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    fn execute_relu_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "ReLU operation requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];
        let result_data: Vec<f32> = input.data.iter().map(|&x| x.max(0.0)).collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: input.shape.clone(),
            dtype: input.dtype,
        })
    }

    fn execute_sigmoid_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "Sigmoid operation requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];
        let result_data: Vec<f32> = input
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: input.shape.clone(),
            dtype: input.dtype,
        })
    }

    fn execute_tanh_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "Tanh operation requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];
        let result_data: Vec<f32> = input.data.iter().map(|&x| x.tanh()).collect();

        Ok(NodeExecutionResult {
            data: result_data,
            shape: input.shape.clone(),
            dtype: input.dtype,
        })
    }

    fn execute_matmul_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 2 {
            return Err(JitError::RuntimeError(
                "MatMul operation requires exactly 2 inputs".to_string(),
            ));
        }

        // Simplified matrix multiplication - in practice this would be more complex
        let a = &inputs[0];
        let b = &inputs[1];

        // For simplicity, assume both are 2D matrices
        if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
            return Err(JitError::RuntimeError(
                "MatMul requires 2D matrices".to_string(),
            ));
        }

        let (m, k) = (a.shape.dims()[0], a.shape.dims()[1]);
        let (k2, n) = (b.shape.dims()[0], b.shape.dims()[1]);

        if k != k2 {
            return Err(JitError::RuntimeError(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result_data[i * n + j] += a.data[i * k + l] * b.data[l * n + j];
                }
            }
        }

        Ok(NodeExecutionResult {
            data: result_data,
            shape: Shape::new(vec![m, n]),
            dtype: a.dtype,
        })
    }

    fn execute_reshape_operation(
        &self,
        inputs: &[NodeExecutionResult],
        attributes: &HashMap<String, crate::graph::Attribute>,
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "Reshape operation requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];

        // Parse new shape from attributes
        let shape_attr = attributes.get("shape").ok_or_else(|| {
            JitError::RuntimeError("Reshape operation missing shape attribute".to_string())
        })?;

        // Extract string value from Attribute enum
        let new_shape_str = match shape_attr {
            crate::graph::Attribute::String(s) => s,
            _ => {
                return Err(JitError::RuntimeError(
                    "Reshape shape attribute must be a string".to_string(),
                ))
            }
        };

        // Simplified shape parsing - in practice this would be more robust
        let new_dims: Result<Vec<usize>, _> = new_shape_str
            .trim_matches(['[', ']'])
            .split(',')
            .map(|s| s.trim().parse())
            .collect();

        let new_dims =
            new_dims.map_err(|_| JitError::RuntimeError("Invalid shape format".to_string()))?;
        let new_shape = Shape::new(new_dims);

        // Verify that total elements remain the same
        if input.shape.numel() != new_shape.numel() {
            return Err(JitError::RuntimeError(
                "Reshape: total elements must remain constant".to_string(),
            ));
        }

        Ok(NodeExecutionResult {
            data: input.data.clone(),
            shape: new_shape,
            dtype: input.dtype,
        })
    }

    fn execute_transpose_operation(
        &self,
        inputs: &[NodeExecutionResult],
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "Transpose operation requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];

        // Simplified transpose for 2D matrices
        if input.shape.ndim() != 2 {
            return Err(JitError::RuntimeError(
                "Transpose currently supports only 2D matrices".to_string(),
            ));
        }

        let (rows, cols) = (input.shape.dims()[0], input.shape.dims()[1]);
        let mut result_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = input.data[i * cols + j];
            }
        }

        Ok(NodeExecutionResult {
            data: result_data,
            shape: Shape::new(vec![cols, rows]),
            dtype: input.dtype,
        })
    }

    fn execute_concat_operation(
        &self,
        inputs: &[NodeExecutionResult],
        attributes: &HashMap<String, crate::graph::Attribute>,
    ) -> JitResult<NodeExecutionResult> {
        if inputs.is_empty() {
            return Err(JitError::RuntimeError(
                "Concat operation requires at least 1 input".to_string(),
            ));
        }

        // Parse axis from attributes
        let axis = attributes
            .get("axis")
            .and_then(|attr| match attr {
                crate::graph::Attribute::String(s) => s.parse::<usize>().ok(),
                crate::graph::Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(0);

        // Simplified concatenation along axis 0
        if axis != 0 {
            return Err(JitError::RuntimeError(
                "Concat currently supports only axis 0".to_string(),
            ));
        }

        let first_input = &inputs[0];
        let mut total_size = first_input.shape.dims()[0];
        let mut result_data = first_input.data.clone();

        for input in &inputs[1..] {
            if input.shape.ndim() != first_input.shape.ndim() {
                return Err(JitError::RuntimeError(
                    "All inputs must have same number of dimensions".to_string(),
                ));
            }

            // Check that all dimensions except axis 0 match
            for (i, (&dim1, &dim2)) in first_input.shape.dims()[1..]
                .iter()
                .zip(input.shape.dims()[1..].iter())
                .enumerate()
            {
                if dim1 != dim2 {
                    return Err(JitError::RuntimeError(format!(
                        "Dimension mismatch at axis {}",
                        i + 1
                    )));
                }
            }

            total_size += input.shape.dims()[0];
            result_data.extend_from_slice(&input.data);
        }

        let mut new_dims = first_input.shape.dims().to_vec();
        new_dims[0] = total_size;

        Ok(NodeExecutionResult {
            data: result_data,
            shape: Shape::new(new_dims),
            dtype: first_input.dtype,
        })
    }

    fn execute_split_operation(
        &self,
        inputs: &[NodeExecutionResult],
        attributes: &HashMap<String, crate::graph::Attribute>,
    ) -> JitResult<NodeExecutionResult> {
        if inputs.len() != 1 {
            return Err(JitError::RuntimeError(
                "Split operation requires exactly 1 input".to_string(),
            ));
        }

        // For simplicity, return the first split only
        // In practice, this would return multiple outputs
        Ok(inputs[0].clone())
    }

    // IR instruction implementations

    fn execute_ir_add(
        &self,
        instruction: &Instruction,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Simplified IR add - would access operands from instruction
        Ok(DebugValue::Scalar(42.0))
    }

    fn execute_ir_multiply(
        &self,
        instruction: &Instruction,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Simplified IR multiply
        Ok(DebugValue::Scalar(84.0))
    }

    fn execute_ir_subtract(
        &self,
        instruction: &Instruction,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Simplified IR subtract
        Ok(DebugValue::Scalar(21.0))
    }

    fn execute_ir_divide(
        &self,
        instruction: &Instruction,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Simplified IR divide
        Ok(DebugValue::Scalar(2.0))
    }

    fn execute_ir_const(&self, instruction: &Instruction) -> JitResult<DebugValue> {
        // In a real implementation, we'd extract the constant value from instruction attributes
        Ok(DebugValue::Scalar(1.0))
    }

    fn execute_ir_load(
        &self,
        instruction: &Instruction,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Load from memory - simplified
        Ok(DebugValue::Scalar(std::f64::consts::PI))
    }

    fn execute_ir_store(
        &self,
        instruction: &Instruction,
        execution_state: &mut ExecutionState,
    ) -> JitResult<()> {
        // Store to memory - simplified
        Ok(())
    }

    fn execute_ir_call(
        &self,
        instruction: &Instruction,
        ir_module: &IrModule,
        execution_state: &ExecutionState,
    ) -> JitResult<DebugValue> {
        // Function call - simplified
        Ok(DebugValue::Scalar(100.0))
    }

    // Helper methods

    fn get_node_inputs(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
    ) -> JitResult<Vec<NodeExecutionResult>> {
        // Simplified - in practice this would get actual computed values from predecessor nodes
        Ok(vec![NodeExecutionResult {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::new(vec![3]),
            dtype: DType::F32,
        }])
    }

    fn record_operation_timing(&mut self, operation: &str, duration: Duration) {
        self.instruction_timings
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);

        // Update operation statistics
        let timings = &self.instruction_timings[operation];
        let count = timings.len();
        let total_time: Duration = timings.iter().sum();
        let average_time = total_time / count as u32;
        let min_time = *timings.iter().min().expect("timings should not be empty");
        let max_time = *timings.iter().max().expect("timings should not be empty");

        self.operation_stats.insert(
            operation.to_string(),
            OperationStatistics {
                count,
                total_time,
                average_time,
                min_time,
                max_time,
            },
        );
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> DebugStatistics {
        DebugStatistics {
            total_steps: self.execution_count,
            total_execution_time: self.total_execution_time,
            breakpoints_hit: 0,   // Would be tracked separately
            watches_triggered: 0, // Would be tracked separately
        }
    }

    /// Get detailed operation statistics
    pub fn get_operation_statistics(&self) -> &HashMap<String, OperationStatistics> {
        &self.operation_stats
    }

    /// Get timing information for a specific operation
    pub fn get_operation_timings(&self, operation: &str) -> Option<&Vec<Duration>> {
        self.instruction_timings.get(operation)
    }

    /// Reset all statistics and timing information
    pub fn reset_statistics(&mut self) {
        self.execution_count = 0;
        self.total_execution_time = Duration::new(0, 0);
        self.instruction_timings.clear();
        self.operation_stats.clear();
    }

    /// Get the configuration
    pub fn config(&self) -> &DebuggerConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: DebuggerConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_execution_engine_creation() {
        let config = DebuggerConfig::default();
        let engine = DebugExecutionEngine::new(config);

        assert_eq!(engine.execution_count, 0);
        assert_eq!(engine.total_execution_time, Duration::new(0, 0));
        assert!(engine.instruction_timings.is_empty());
        assert!(engine.operation_stats.is_empty());
    }

    #[test]
    fn test_operation_timing_recording() {
        let config = DebuggerConfig::default();
        let mut engine = DebugExecutionEngine::new(config);

        let duration = Duration::from_millis(10);
        engine.record_operation_timing("add", duration);

        assert_eq!(engine.instruction_timings.get("add").unwrap().len(), 1);
        assert!(engine.operation_stats.contains_key("add"));

        let stats = &engine.operation_stats["add"];
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_time, duration);
        assert_eq!(stats.min_time, duration);
        assert_eq!(stats.max_time, duration);
    }

    #[test]
    fn test_add_operation() {
        let config = DebuggerConfig::default();
        let engine = DebugExecutionEngine::new(config);

        let input1 = NodeExecutionResult {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::new(vec![3]),
            dtype: DType::F32,
        };

        let input2 = NodeExecutionResult {
            data: vec![4.0, 5.0, 6.0],
            shape: Shape::new(vec![3]),
            dtype: DType::F32,
        };

        let result = engine.execute_add_operation(&[input1, input2]).unwrap();
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
        assert_eq!(result.shape.dims(), &[3]);
    }

    #[test]
    fn test_relu_operation() {
        let config = DebuggerConfig::default();
        let engine = DebugExecutionEngine::new(config);

        let input = NodeExecutionResult {
            data: vec![-1.0, 0.0, 1.0, -2.0, 3.0],
            shape: Shape::new(vec![5]),
            dtype: DType::F32,
        };

        let result = engine.execute_relu_operation(&[input]).unwrap();
        assert_eq!(result.data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_statistics_reset() {
        let config = DebuggerConfig::default();
        let mut engine = DebugExecutionEngine::new(config);

        engine.record_operation_timing("test", Duration::from_millis(10));
        engine.execution_count = 5;
        engine.total_execution_time = Duration::from_millis(50);

        engine.reset_statistics();

        assert_eq!(engine.execution_count, 0);
        assert_eq!(engine.total_execution_time, Duration::new(0, 0));
        assert!(engine.instruction_timings.is_empty());
        assert!(engine.operation_stats.is_empty());
    }
}
