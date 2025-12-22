//! Core types and data structures for JIT debugging
//!
//! This module provides fundamental types, enums, and configurations
//! used throughout the JIT debugging system.

use crate::NodeId;
use std::collections::HashMap;
use torsh_core::{DType, Shape};

/// Debugger configuration
#[derive(Debug, Clone)]
pub struct DebuggerConfig {
    pub enable_single_step: bool,
    pub enable_breakpoints: bool,
    pub enable_watches: bool,
    pub enable_memory_view: bool,
    pub enable_disassembly: bool,
    pub max_trace_length: usize,
    pub ui_mode: UiMode,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            enable_single_step: true,
            enable_breakpoints: true,
            enable_watches: true,
            enable_memory_view: true,
            enable_disassembly: true,
            max_trace_length: 10000,
            ui_mode: UiMode::Interactive,
        }
    }
}

/// UI mode for the debugger
#[derive(Debug, Clone)]
pub enum UiMode {
    Interactive,
    Batch,
    Remote,
}

/// Debug commands
#[derive(Debug, Clone)]
pub enum DebugCommand {
    Step,
    StepOver,
    StepInto,
    StepOut,
    Continue,
    SetBreakpoint { location: BreakpointLocation },
    RemoveBreakpoint { id: BreakpointId },
    ListBreakpoints,
    Watch { expression: String },
    Unwatch { id: WatchId },
    ListWatches,
    Inspect { target: InspectionTarget },
    CallStack,
    Locals,
    Memory { address: u64 },
    Disassemble { location: ExecutionLocation },
    Help,
    Quit,
}

/// Execution locations
#[derive(Debug, Clone)]
pub enum ExecutionLocation {
    GraphNode(NodeId),
    Instruction {
        function: String,
        instruction_index: usize,
    },
    Completed,
}

/// Breakpoint locations
#[derive(Debug, Clone)]
pub enum BreakpointLocation {
    GraphNode(NodeId),
    Instruction {
        function: String,
        instruction: usize,
    },
}

/// Inspection targets
#[derive(Debug, Clone)]
pub enum InspectionTarget {
    Variable(String),
    Node(NodeId),
    Memory(u64),
}

/// Debug values that can be inspected
#[derive(Debug, Clone, PartialEq)]
pub enum DebugValue {
    Scalar(f64),
    Integer(i64),
    Boolean(bool),
    Tensor {
        data: Vec<f32>,
        shape: Shape,
        dtype: DType,
    },
}

/// Breakpoint representation
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: BreakpointId,
    pub location: BreakpointLocation,
    pub condition: Option<String>,
    pub enabled: bool,
    pub hit_count: u32,
}

/// Watch representation
#[derive(Debug, Clone)]
pub struct Watch {
    pub id: WatchId,
    pub expression: String,
    pub enabled: bool,
    pub last_value: Option<DebugValue>,
}

/// Call frame for call stack
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub location: ExecutionLocation,
    pub return_location: ExecutionLocation,
    pub local_variables: HashMap<String, DebugValue>,
}

/// Current debug state
#[derive(Debug, Clone)]
pub struct DebugState {
    pub location: ExecutionLocation,
    pub call_stack: crate::debugger::state::CallStack,
    pub variables: HashMap<String, DebugValue>,
    pub execution_step: usize,
    pub is_running: bool,
}

/// Execution step in the trace
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub location: ExecutionLocation,
    pub timestamp: std::time::SystemTime,
    pub operation: String,
    pub inputs: Vec<NodeExecutionResult>,
    pub outputs: Vec<NodeExecutionResult>,
    pub state_changes: HashMap<String, DebugValue>,
}

/// Execution state tracking
#[derive(Debug, Clone)]
pub struct ExecutionState {
    pub registers: HashMap<u32, DebugValue>,
    pub memory: crate::debugger::state::MemoryState,
    pub flags: HashMap<String, bool>,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            memory: crate::debugger::state::MemoryState::new(),
            flags: HashMap::new(),
        }
    }
}

/// Node execution result
#[derive(Debug, Clone)]
pub struct NodeExecutionResult {
    pub data: Vec<f32>,
    pub shape: Shape,
    pub dtype: DType,
}

/// Instruction execution result
#[derive(Debug, Clone)]
pub enum InstructionExecutionResult {
    Value(DebugValue),
    SideEffect,
    Return,
    NoOp,
}

/// Debug statistics
#[derive(Debug, Clone)]
pub struct DebugStatistics {
    pub total_steps: usize,
    pub total_execution_time: std::time::Duration,
    pub breakpoints_hit: usize,
    pub watches_triggered: usize,
}

impl DebugStatistics {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            total_execution_time: std::time::Duration::from_millis(0),
            breakpoints_hit: 0,
            watches_triggered: 0,
        }
    }
}

/// Type information for debug values
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub type_name: String,
    pub size_bytes: usize,
    pub alignment: usize,
}

/// Node metadata for debugging
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub operation: String,
    pub input_count: usize,
    pub output_shape: Shape,
    pub dtype: DType,
}

/// Memory view for debugging
#[derive(Debug, Clone)]
pub struct MemoryView {
    pub start_address: u64,
    pub content: Vec<u8>,
    pub size: usize,
}

/// Disassembly view
#[derive(Debug, Clone)]
pub struct DisassemblyView {
    pub location: ExecutionLocation,
    pub instructions: Vec<DisassemblyInstruction>,
}

/// Disassembly instruction
#[derive(Debug, Clone)]
pub struct DisassemblyInstruction {
    pub address: u64,
    pub opcode: String,
    pub operands: String,
    pub comment: Option<String>,
}

/// Watch update notification
#[derive(Debug, Clone)]
pub struct WatchUpdate {
    pub watch_id: WatchId,
    pub old_value: Option<DebugValue>,
    pub new_value: DebugValue,
}

/// Evaluation result for expressions
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub expression: String,
    pub result: DebugValue,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Inspection result
#[derive(Debug, Clone)]
pub enum InspectionResult {
    Variable {
        name: String,
        value: DebugValue,
        type_info: TypeInfo,
    },
    Node {
        node_id: NodeId,
        value: DebugValue,
        metadata: NodeMetadata,
    },
    Memory {
        address: u64,
        content: Vec<u8>,
        size: usize,
    },
}

/// Result types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreakpointId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WatchId(pub u64);

#[derive(Debug)]
pub enum DebugCommandResult {
    Continue,
    Exit,
    ExecutionComplete,
}

#[derive(Debug)]
pub enum StepResult {
    Success,
    Completed,
}

#[derive(Debug)]
pub enum ContinueResult {
    Breakpoint,
    Completed,
}

/// Debug session result
#[derive(Debug)]
pub struct DebugSessionResult {
    pub execution_trace: Vec<ExecutionStep>,
    pub final_state: DebugState,
    pub command_history: Vec<DebugCommand>,
    pub statistics: DebugStatistics,
}
