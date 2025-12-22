//! Debug session management for JIT debugging
//!
//! This module provides the core debug session functionality including
//! step-by-step execution, state management, and debugging operations.

use super::core::{
    ContinueResult, DebugState, DebugStatistics, DebugValue, DebuggerConfig,
    DisassemblyInstruction, DisassemblyView, EvaluationResult, ExecutionLocation, ExecutionState,
    ExecutionStep, InspectionResult, InspectionTarget, InstructionExecutionResult, MemoryView,
    NodeExecutionResult, NodeMetadata, StepResult, TypeInfo,
};
use super::execution::DebugExecutionEngine;
use super::state::{CallStack, MemoryState};
use crate::{ir::IrModule, ComputationGraph, JitError, JitResult, NodeId};
use std::collections::HashMap;
use std::time::SystemTime;
use torsh_core::{DType, Shape};

/// Debug session managing the state of debugging
pub struct DebugSession {
    graph: Option<ComputationGraph>,
    ir_module: Option<IrModule>,
    current_location: ExecutionLocation,
    execution_state: ExecutionState,
    execution_trace: Vec<ExecutionStep>,
    call_stack: CallStack,
    variable_bindings: HashMap<String, DebugValue>,
    memory_state: MemoryState,
    statistics: DebugStatistics,
    config: DebuggerConfig,
    execution_engine: DebugExecutionEngine,
}

impl DebugSession {
    /// Create a new debug session for a computation graph
    ///
    /// # Arguments
    /// * `graph` - The computation graph to debug
    /// * `config` - Configuration for the debug session
    pub fn new(graph: ComputationGraph, config: DebuggerConfig) -> Self {
        let initial_location = ExecutionLocation::GraphNode(
            graph
                .nodes()
                .next()
                .map(|(id, _)| id)
                .unwrap_or(NodeId::new(0)),
        );

        let execution_engine = DebugExecutionEngine::new(config.clone());

        Self {
            graph: Some(graph),
            ir_module: None,
            current_location: initial_location,
            execution_state: ExecutionState::new(),
            execution_trace: Vec::new(),
            call_stack: CallStack::new(),
            variable_bindings: HashMap::new(),
            memory_state: MemoryState::new(),
            statistics: DebugStatistics::new(),
            config,
            execution_engine,
        }
    }

    /// Create a new debug session for an IR module
    ///
    /// # Arguments
    /// * `ir_module` - The IR module to debug
    /// * `config` - Configuration for the debug session
    pub fn from_ir(ir_module: IrModule, config: DebuggerConfig) -> Self {
        let initial_location = ExecutionLocation::Instruction {
            function: "main".to_string(),
            instruction_index: 0,
        };

        let execution_engine = DebugExecutionEngine::new(config.clone());

        Self {
            graph: None,
            ir_module: Some(ir_module),
            current_location: initial_location,
            execution_state: ExecutionState::new(),
            execution_trace: Vec::new(),
            call_stack: CallStack::new(),
            variable_bindings: HashMap::new(),
            memory_state: MemoryState::new(),
            statistics: DebugStatistics::new(),
            config,
            execution_engine,
        }
    }

    /// Execute a single step
    ///
    /// # Returns
    /// The result of the step execution
    pub fn step(&mut self) -> JitResult<StepResult> {
        if self.is_execution_complete() {
            return Err(JitError::AnalysisError(
                "Execution already completed".to_string(),
            ));
        }

        let step_start = std::time::Instant::now();

        let result = match self.current_location.clone() {
            ExecutionLocation::GraphNode(node_id) => self.step_graph_node(node_id),
            ExecutionLocation::Instruction {
                function,
                instruction_index,
            } => self.step_ir_instruction(&function, instruction_index),
            ExecutionLocation::Completed => {
                return Err(JitError::AnalysisError(
                    "Execution already completed".to_string(),
                ));
            }
        };

        let step_duration = step_start.elapsed();
        self.statistics.total_steps += 1;
        self.statistics.total_execution_time += step_duration;

        match result {
            Ok(_) => {
                if self.is_execution_complete() {
                    Ok(StepResult::Completed)
                } else {
                    Ok(StepResult::Success)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Execute a step over (don't step into function calls)
    ///
    /// # Returns
    /// The result of the step-over execution
    pub fn step_over(&mut self) -> JitResult<StepResult> {
        let current_call_depth = self.call_stack.depth();

        loop {
            let result = self.step()?;

            match result {
                StepResult::Completed => return Ok(StepResult::Completed),
                StepResult::Success => {
                    // Stop if we're at the same or shallower call depth
                    if self.call_stack.depth() <= current_call_depth {
                        break;
                    }
                }
            }
        }

        Ok(StepResult::Success)
    }

    /// Execute a step into (step into function calls)
    ///
    /// # Returns
    /// The result of the step-into execution
    pub fn step_into(&mut self) -> JitResult<StepResult> {
        // Step into is the same as regular step
        self.step()
    }

    /// Execute a step out (continue until returning from current function)
    ///
    /// # Returns
    /// The result of the step-out execution
    pub fn step_out(&mut self) -> JitResult<StepResult> {
        let target_depth = self.call_stack.depth().saturating_sub(1);

        while self.call_stack.depth() > target_depth {
            let result = self.step()?;
            if matches!(result, StepResult::Completed) {
                return Ok(StepResult::Completed);
            }
        }

        Ok(StepResult::Success)
    }

    /// Continue execution until breakpoint or completion
    ///
    /// # Returns
    /// The result of continue execution
    pub fn continue_execution(&mut self) -> JitResult<ContinueResult> {
        loop {
            // Check for breakpoints at current location
            if self.should_break_at_current_location() {
                return Ok(ContinueResult::Breakpoint);
            }

            // Execute next step
            match self.step() {
                Ok(StepResult::Success) => {
                    // Check if execution is complete
                    if self.is_execution_complete() {
                        return Ok(ContinueResult::Completed);
                    }
                }
                Ok(StepResult::Completed) => {
                    return Ok(ContinueResult::Completed);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Step through a graph node
    fn step_graph_node(&mut self, node_id: NodeId) -> JitResult<()> {
        if let Some(graph) = &self.graph {
            if let Some(node) = graph.node(node_id) {
                // Record execution step
                let step = ExecutionStep {
                    location: self.current_location.clone(),
                    timestamp: SystemTime::now(),
                    operation: node.operation_type().to_string(),
                    inputs: self.get_node_inputs(node_id),
                    outputs: Vec::new(), // Will be filled after execution
                    state_changes: HashMap::new(),
                };

                // Execute node operation using the execution engine
                let result = self
                    .execution_engine
                    .execute_node_debug(node, graph, node_id)?;

                // Update variable bindings
                self.variable_bindings.insert(
                    format!("node_{:?}", node_id),
                    DebugValue::Tensor {
                        data: result.data.clone(),
                        shape: result.shape.clone(),
                        dtype: result.dtype,
                    },
                );

                // Move to next node
                self.advance_to_next_graph_node(node_id)?;

                // Complete the execution step record
                let mut completed_step = step;
                completed_step.outputs = vec![result];
                self.execution_trace.push(completed_step);
            }
        }

        Ok(())
    }

    /// Step through an IR instruction
    fn step_ir_instruction(&mut self, function: &str, instruction_index: usize) -> JitResult<()> {
        if let Some(ir_module) = self.ir_module.clone() {
            if let Some(func) = ir_module.get_function(function) {
                if let Some(instruction) = func.instructions().get(instruction_index) {
                    // Record execution step
                    let step = ExecutionStep {
                        location: self.current_location.clone(),
                        timestamp: SystemTime::now(),
                        operation: format!("{:?}", instruction),
                        inputs: self.get_instruction_inputs(instruction),
                        outputs: Vec::new(),
                        state_changes: HashMap::new(),
                    };

                    // Execute instruction using the execution engine
                    let result = self.execution_engine.execute_instruction_debug(
                        instruction,
                        &ir_module,
                        &mut self.execution_state,
                    )?;

                    // Update execution state
                    self.update_execution_state_from_instruction(instruction, result)?;

                    // Move to next instruction
                    self.advance_to_next_instruction(function, instruction_index)?;

                    // Complete execution step record
                    self.execution_trace.push(step);
                }
            }
        }

        Ok(())
    }

    /// Get inputs for a node
    fn get_node_inputs(&self, node_id: NodeId) -> Vec<NodeExecutionResult> {
        // In a real implementation, this would get actual computed values from predecessor nodes
        if let Some(graph) = &self.graph {
            let inputs = graph.get_node_inputs(node_id);
            inputs
                .iter()
                .map(|&input_id| {
                    // Try to get the computed value from variable bindings
                    let var_name = format!("node_{:?}", input_id);
                    if let Some(DebugValue::Tensor { data, shape, dtype }) =
                        self.variable_bindings.get(&var_name)
                    {
                        NodeExecutionResult {
                            data: data.clone(),
                            shape: shape.clone(),
                            dtype: *dtype,
                        }
                    } else {
                        // Default placeholder
                        NodeExecutionResult {
                            data: vec![1.0, 2.0, 3.0],
                            shape: Shape::new(vec![3]),
                            dtype: DType::F32,
                        }
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get inputs for an instruction
    fn get_instruction_inputs(
        &self,
        instruction: &crate::ir::Instruction,
    ) -> Vec<NodeExecutionResult> {
        // Placeholder implementation - in practice would extract from instruction operands
        Vec::new()
    }

    /// Advance to the next graph node
    fn advance_to_next_graph_node(&mut self, current_node: NodeId) -> JitResult<()> {
        if let Some(graph) = &self.graph {
            // Find successors of current node
            let successors: Vec<_> = graph
                .edges()
                .filter(|(source, _target, _edge)| *source == current_node)
                .map(|(_source, target, _edge)| target)
                .collect();

            if let Some(next_node) = successors.first() {
                self.current_location = ExecutionLocation::GraphNode(*next_node);
            } else {
                // No more nodes - execution complete
                self.current_location = ExecutionLocation::Completed;
            }
        }
        Ok(())
    }

    /// Advance to the next instruction
    fn advance_to_next_instruction(
        &mut self,
        function: &str,
        current_index: usize,
    ) -> JitResult<()> {
        if let Some(ir_module) = &self.ir_module {
            if let Some(func) = ir_module.get_function(function) {
                let next_index = current_index + 1;
                if next_index < func.instructions().len() {
                    self.current_location = ExecutionLocation::Instruction {
                        function: function.to_string(),
                        instruction_index: next_index,
                    };
                } else {
                    // End of function
                    if !self.call_stack.is_empty() {
                        self.current_location = self.call_stack.pop();
                    } else {
                        self.current_location = ExecutionLocation::Completed;
                    }
                }
            }
        }
        Ok(())
    }

    /// Update execution state from instruction result
    fn update_execution_state_from_instruction(
        &mut self,
        instruction: &crate::ir::Instruction,
        result: InstructionExecutionResult,
    ) -> JitResult<()> {
        match result {
            InstructionExecutionResult::Value(value) => {
                // Update variable bindings
                let var_name = format!("temp_{}", self.execution_trace.len());
                self.variable_bindings.insert(var_name, value);
            }
            InstructionExecutionResult::SideEffect => {
                // Handle side effects (memory writes, etc.)
                // This could update memory state or other side effects
            }
            InstructionExecutionResult::Return => {
                // Handle function return
                if !self.call_stack.is_empty() {
                    let return_location = self.call_stack.pop();
                    self.current_location = return_location;
                } else {
                    self.current_location = ExecutionLocation::Completed;
                }
            }
            InstructionExecutionResult::NoOp => {
                // No state change
            }
        }
        Ok(())
    }

    /// Check if execution should break at current location
    fn should_break_at_current_location(&self) -> bool {
        // This would check against a global breakpoint manager
        // For now, just return false as a placeholder
        false
    }

    /// Check if execution is complete
    pub fn is_execution_complete(&self) -> bool {
        matches!(self.current_location, ExecutionLocation::Completed)
    }

    /// Inspect a target (variable, memory location, etc.)
    ///
    /// # Arguments
    /// * `target` - The target to inspect
    ///
    /// # Returns
    /// The inspection result
    pub fn inspect_target(&self, target: &InspectionTarget) -> JitResult<InspectionResult> {
        match target {
            InspectionTarget::Variable(name) => {
                if let Some(value) = self.variable_bindings.get(name) {
                    Ok(InspectionResult::Variable {
                        name: name.clone(),
                        value: value.clone(),
                        type_info: self.get_type_info_for_value(value),
                    })
                } else {
                    Err(JitError::RuntimeError(format!(
                        "Variable '{}' not found",
                        name
                    )))
                }
            }
            InspectionTarget::Node(node_id) => {
                let var_name = format!("node_{:?}", node_id);
                if let Some(value) = self.variable_bindings.get(&var_name) {
                    Ok(InspectionResult::Node {
                        node_id: *node_id,
                        value: value.clone(),
                        metadata: self.get_node_metadata(*node_id),
                    })
                } else {
                    Err(JitError::RuntimeError(format!(
                        "Node {:?} not executed yet",
                        node_id
                    )))
                }
            }
            InspectionTarget::Memory(address) => {
                let memory_content = self.memory_state.read_memory(*address, 16)?;
                Ok(InspectionResult::Memory {
                    address: *address,
                    content: memory_content,
                    size: 16,
                })
            }
        }
    }

    /// Get type information for a debug value
    fn get_type_info_for_value(&self, value: &DebugValue) -> TypeInfo {
        match value {
            DebugValue::Scalar(_) => TypeInfo {
                type_name: "f64".to_string(),
                size_bytes: 8,
                alignment: 8,
            },
            DebugValue::Integer(_) => TypeInfo {
                type_name: "i64".to_string(),
                size_bytes: 8,
                alignment: 8,
            },
            DebugValue::Boolean(_) => TypeInfo {
                type_name: "bool".to_string(),
                size_bytes: 1,
                alignment: 1,
            },
            DebugValue::Tensor { dtype, shape, .. } => TypeInfo {
                type_name: format!("Tensor<{:?}>", dtype),
                size_bytes: shape.size(0).unwrap_or(1) * dtype.size_bytes(),
                alignment: dtype.size_bytes(),
            },
        }
    }

    /// Get metadata for a node
    fn get_node_metadata(&self, node_id: NodeId) -> NodeMetadata {
        if let Some(graph) = &self.graph {
            if let Some(node) = graph.node(node_id) {
                let input_count = graph.get_node_inputs(node_id).len();
                return NodeMetadata {
                    operation: node.operation_type().to_string(),
                    input_count,
                    output_shape: node.output_shape.clone(),
                    dtype: node.dtype,
                };
            }
        }

        NodeMetadata {
            operation: "unknown".to_string(),
            input_count: 0,
            output_shape: Shape::new(vec![]),
            dtype: DType::F32,
        }
    }

    /// Evaluate an expression in the current context
    ///
    /// # Arguments
    /// * `expression` - The expression to evaluate
    ///
    /// # Returns
    /// The evaluation result
    pub fn evaluate_expression(&self, expression: &str) -> JitResult<EvaluationResult> {
        // Simple expression evaluator
        if let Some(value) = self.variable_bindings.get(expression) {
            Ok(EvaluationResult {
                expression: expression.to_string(),
                result: value.clone(),
                success: true,
                error_message: None,
            })
        } else if expression.starts_with("node_") {
            // Try to parse as node reference
            if let Ok(node_index) = expression[5..].parse::<usize>() {
                let var_name = format!("node_{}", node_index);
                if let Some(value) = self.variable_bindings.get(&var_name) {
                    Ok(EvaluationResult {
                        expression: expression.to_string(),
                        result: value.clone(),
                        success: true,
                        error_message: None,
                    })
                } else {
                    Ok(EvaluationResult {
                        expression: expression.to_string(),
                        result: DebugValue::Scalar(0.0),
                        success: false,
                        error_message: Some("Node not executed yet".to_string()),
                    })
                }
            } else {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Scalar(0.0),
                    success: false,
                    error_message: Some("Invalid node reference".to_string()),
                })
            }
        } else {
            // Try to parse as literal value - try integer first, then float
            if let Ok(value) = expression.parse::<i64>() {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Integer(value),
                    success: true,
                    error_message: None,
                })
            } else if let Ok(value) = expression.parse::<f64>() {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Scalar(value),
                    success: true,
                    error_message: None,
                })
            } else if let Ok(value) = expression.parse::<bool>() {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Boolean(value),
                    success: true,
                    error_message: None,
                })
            } else {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Scalar(0.0),
                    success: false,
                    error_message: Some("Expression not found or invalid".to_string()),
                })
            }
        }
    }

    /// Get current debug state
    pub fn get_current_state(&self) -> DebugState {
        DebugState {
            location: self.current_location.clone(),
            call_stack: self.call_stack.clone(),
            variables: self.variable_bindings.clone(),
            execution_step: self.execution_trace.len(),
            is_running: !self.is_execution_complete(),
        }
    }

    /// Get execution trace
    pub fn get_execution_trace(&self) -> Vec<ExecutionStep> {
        self.execution_trace.clone()
    }

    /// Get call stack
    pub fn get_call_stack(&self) -> CallStack {
        self.call_stack.clone()
    }

    /// Get local variables
    pub fn get_local_variables(&self) -> HashMap<String, DebugValue> {
        // In a real implementation, this would return variables in the current scope
        self.variable_bindings.clone()
    }

    /// Get memory view
    pub fn get_memory_view(&self, address: u64) -> JitResult<MemoryView> {
        let content = self.memory_state.read_memory(address, 64)?;
        Ok(MemoryView {
            start_address: address,
            content,
            size: 64,
        })
    }

    /// Disassemble at location
    pub fn disassemble_at(&self, location: ExecutionLocation) -> JitResult<DisassemblyView> {
        match location {
            ExecutionLocation::GraphNode(node_id) => {
                if let Some(graph) = &self.graph {
                    if let Some(node) = graph.node(node_id) {
                        Ok(DisassemblyView {
                            location,
                            instructions: vec![DisassemblyInstruction {
                                address: node_id.index() as u64,
                                opcode: node.operation_type().to_string(),
                                operands: format!(
                                    "inputs: {}",
                                    graph.get_node_inputs(node_id).len()
                                ),
                                comment: Some(format!("Output shape: {:?}", node.output_shape)),
                            }],
                        })
                    } else {
                        Err(JitError::RuntimeError("Node not found".to_string()))
                    }
                } else {
                    Err(JitError::RuntimeError("No graph available".to_string()))
                }
            }
            ExecutionLocation::Instruction {
                ref function,
                instruction_index,
            } => {
                if let Some(ir_module) = &self.ir_module {
                    if let Some(func) = ir_module.get_function(function) {
                        if let Some(instruction) = func.instructions().get(instruction_index) {
                            Ok(DisassemblyView {
                                location,
                                instructions: vec![DisassemblyInstruction {
                                    address: instruction_index as u64,
                                    opcode: format!("{:?}", instruction.opcode),
                                    operands: format!("operands: {}", instruction.operands.len()),
                                    comment: Some(format!("result: {:?}", instruction.result)),
                                }],
                            })
                        } else {
                            Err(JitError::RuntimeError("Instruction not found".to_string()))
                        }
                    } else {
                        Err(JitError::RuntimeError("Function not found".to_string()))
                    }
                } else {
                    Err(JitError::RuntimeError("No IR module available".to_string()))
                }
            }
            ExecutionLocation::Completed => {
                Err(JitError::RuntimeError("Execution completed".to_string()))
            }
        }
    }

    /// Get debug statistics
    pub fn get_statistics(&self) -> DebugStatistics {
        // Combine session statistics with execution engine statistics
        let mut stats = self.statistics.clone();
        let engine_stats = self.execution_engine.get_statistics();

        stats.total_steps = stats.total_steps.max(engine_stats.total_steps);
        stats.total_execution_time = stats
            .total_execution_time
            .max(engine_stats.total_execution_time);

        stats
    }

    /// Reset the debug session
    pub fn reset(&mut self) {
        self.execution_trace.clear();
        self.call_stack.clear();
        self.variable_bindings.clear();
        self.memory_state.clear();
        self.statistics = DebugStatistics::new();
        self.execution_state = ExecutionState::new();

        // Reset to initial location
        if let Some(graph) = &self.graph {
            self.current_location = ExecutionLocation::GraphNode(
                graph
                    .nodes()
                    .next()
                    .map(|(id, _)| id)
                    .unwrap_or(NodeId::new(0)),
            );
        } else if self.ir_module.is_some() {
            self.current_location = ExecutionLocation::Instruction {
                function: "main".to_string(),
                instruction_index: 0,
            };
        } else {
            self.current_location = ExecutionLocation::Completed;
        }
    }

    /// Get configuration
    pub fn config(&self) -> &DebuggerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DebuggerConfig) {
        self.config = config.clone();
        self.execution_engine.update_config(config);
    }

    /// Set a variable value (for testing/debugging purposes)
    pub fn set_variable(&mut self, name: String, value: DebugValue) {
        self.variable_bindings.insert(name, value);
    }

    /// Get variable value
    pub fn get_variable(&self, name: &str) -> Option<&DebugValue> {
        self.variable_bindings.get(name)
    }

    /// Get execution engine statistics
    pub fn get_execution_engine_statistics(
        &self,
    ) -> &std::collections::HashMap<String, super::execution::OperationStatistics> {
        self.execution_engine.get_operation_statistics()
    }

    /// Get memory state
    pub fn memory_state(&self) -> &MemoryState {
        &self.memory_state
    }

    /// Get mutable memory state
    pub fn memory_state_mut(&mut self) -> &mut MemoryState {
        &mut self.memory_state
    }
}

// Implement the ExpressionEvaluator trait for DebugSession
impl super::watch::ExpressionEvaluator for DebugSession {
    fn evaluate_expression(&self, expression: &str) -> JitResult<EvaluationResult> {
        self.evaluate_expression(expression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_session() -> DebugSession {
        let config = DebuggerConfig::default();
        // Create a minimal graph for testing
        let graph = ComputationGraph::new(); // Assuming this creates an empty graph
        DebugSession::new(graph, config)
    }

    #[test]
    fn test_session_creation() {
        let session = create_test_session();
        assert!(!session.is_execution_complete());
        assert_eq!(session.execution_trace.len(), 0);
        assert_eq!(session.variable_bindings.len(), 0);
    }

    #[test]
    fn test_variable_management() {
        let mut session = create_test_session();

        let value = DebugValue::Scalar(42.0);
        session.set_variable("test_var".to_string(), value.clone());

        assert_eq!(session.get_variable("test_var"), Some(&value));
        assert_eq!(session.get_variable("nonexistent"), None);
    }

    #[test]
    fn test_expression_evaluation() {
        let mut session = create_test_session();

        // Test literal values
        let result = session.evaluate_expression("42.5").unwrap();
        assert!(result.success);
        assert!(matches!(result.result, DebugValue::Scalar(42.5)));

        let result = session.evaluate_expression("123").unwrap();
        assert!(result.success);
        assert!(matches!(result.result, DebugValue::Integer(123)));

        let result = session.evaluate_expression("true").unwrap();
        assert!(result.success);
        assert!(matches!(result.result, DebugValue::Boolean(true)));

        // Test variable lookup
        session.set_variable("x".to_string(), DebugValue::Scalar(3.14));
        let result = session.evaluate_expression("x").unwrap();
        assert!(result.success);
        assert!(matches!(result.result, DebugValue::Scalar(3.14)));

        // Test unknown variable
        let result = session.evaluate_expression("unknown_var").unwrap();
        assert!(!result.success);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_execution_state() {
        let session = create_test_session();
        let state = session.get_current_state();

        assert!(state.is_running);
        assert_eq!(state.execution_step, 0);
        assert!(state.call_stack.is_empty());
        assert!(state.variables.is_empty());
    }

    #[test]
    fn test_memory_operations() {
        let mut session = create_test_session();

        let memory = session.memory_state_mut();
        memory.write_memory(0x1000, &[1, 2, 3, 4]).unwrap();

        let memory_view = session.get_memory_view(0x1000).unwrap();
        assert_eq!(memory_view.start_address, 0x1000);
        assert_eq!(&memory_view.content[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_session_reset() {
        let mut session = create_test_session();

        // Add some state
        session.set_variable("test".to_string(), DebugValue::Scalar(1.0));
        session.statistics.total_steps = 10;

        // Reset and verify
        session.reset();

        assert_eq!(session.variable_bindings.len(), 0);
        assert_eq!(session.execution_trace.len(), 0);
        assert_eq!(session.statistics.total_steps, 0);
        assert!(session.call_stack.is_empty());
    }

    #[test]
    fn test_configuration_update() {
        let mut session = create_test_session();

        let mut new_config = DebuggerConfig::default();
        new_config.max_trace_length = 5000;

        session.update_config(new_config.clone());
        assert_eq!(session.config().max_trace_length, 5000);
    }

    #[test]
    fn test_inspection_targets() {
        let mut session = create_test_session();

        // Test variable inspection
        session.set_variable("test_var".to_string(), DebugValue::Scalar(42.0));
        let result = session.inspect_target(&InspectionTarget::Variable("test_var".to_string()));
        assert!(result.is_ok());

        // Test memory inspection
        session
            .memory_state_mut()
            .write_memory(0x1000, &[1, 2, 3, 4])
            .unwrap();
        let result = session.inspect_target(&InspectionTarget::Memory(0x1000));
        assert!(result.is_ok());

        // Test unknown variable
        let result = session.inspect_target(&InspectionTarget::Variable("unknown".to_string()));
        assert!(result.is_err());
    }
}
