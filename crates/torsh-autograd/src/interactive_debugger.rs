// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interactive Gradient Computation Debugger
//!
//! This module provides an interactive debugging interface for gradient computations,
//! allowing step-by-step execution, tensor inspection, and computation graph analysis.
//!
//! # Features
//!
//! - **Step-by-step Execution**: Execute operations one at a time
//! - **Breakpoints**: Set breakpoints on specific operations or conditions
//! - **Tensor Inspection**: Examine tensor values and gradients
//! - **Call Stack**: View operation call stack
//! - **Watchpoints**: Monitor specific tensors for changes
//! - **Time Travel**: Step backward through execution history

use crate::error_handling::{AutogradError, AutogradResult};
use crate::gradient_tracer::{EventType, PathId, TraceEvent};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Debugger state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebuggerState {
    /// Debugger is inactive
    Inactive,
    /// Debugger is paused (waiting for user input)
    Paused,
    /// Debugger is running
    Running,
    /// Debugger is stepping (will pause after next operation)
    Stepping,
    /// Debugger is continuing to next breakpoint
    Continuing,
}

/// Breakpoint condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointCondition {
    /// Break on specific operation name
    OperationName(String),

    /// Break on specific tensor ID
    TensorId(String),

    /// Break on anomaly detection
    Anomaly,

    /// Break on memory threshold
    MemoryThreshold(usize),

    /// Break after N operations
    OperationCount(usize),

    /// Break on gradient explosion (norm exceeds threshold)
    GradientExplosion(f64),

    /// Break on gradient vanishing (norm below threshold)
    GradientVanishing(f64),

    /// Custom condition (evaluates to true if should break)
    Custom(String),
}

/// Breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    /// Breakpoint ID
    pub id: u64,

    /// Breakpoint condition
    pub condition: BreakpointCondition,

    /// Whether breakpoint is enabled
    pub enabled: bool,

    /// Number of times this breakpoint has been hit
    pub hit_count: usize,

    /// Description
    pub description: String,
}

impl Breakpoint {
    /// Create a new breakpoint
    pub fn new(id: u64, condition: BreakpointCondition, description: String) -> Self {
        Self {
            id,
            condition,
            enabled: true,
            hit_count: 0,
            description,
        }
    }

    /// Check if this breakpoint should trigger
    pub fn should_trigger(&mut self, event: &TraceEvent, _context: &DebugContext) -> bool {
        if !self.enabled {
            return false;
        }

        let triggered = match &self.condition {
            BreakpointCondition::OperationName(name) => &event.operation == name,
            BreakpointCondition::TensorId(id) => {
                event.input_ids.contains(id) || event.output_ids.contains(id)
            }
            BreakpointCondition::Anomaly => {
                matches!(event.event_type, EventType::Custom)
            }
            BreakpointCondition::MemoryThreshold(threshold) => {
                event.memory_allocated.unwrap_or(0) > *threshold
            }
            BreakpointCondition::OperationCount(count) => self.hit_count >= *count,
            BreakpointCondition::GradientExplosion(_threshold) => {
                // TODO: Implement gradient norm checking
                false
            }
            BreakpointCondition::GradientVanishing(_threshold) => {
                // TODO: Implement gradient norm checking
                false
            }
            BreakpointCondition::Custom(_expr) => {
                // TODO: Implement custom expression evaluation
                false
            }
        };

        if triggered {
            self.hit_count += 1;
        }

        triggered
    }
}

/// Watchpoint for monitoring tensor changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watchpoint {
    /// Watchpoint ID
    pub id: u64,

    /// Tensor ID to watch
    pub tensor_id: String,

    /// Whether to break on read
    pub break_on_read: bool,

    /// Whether to break on write
    pub break_on_write: bool,

    /// Whether to break on gradient update
    pub break_on_gradient: bool,

    /// Number of times this watchpoint has been triggered
    pub trigger_count: usize,
}

/// Debug context containing current execution state
#[derive(Debug, Clone)]
pub struct DebugContext {
    /// Current operation index
    pub operation_index: usize,

    /// Total operations
    pub total_operations: usize,

    /// Current call stack
    pub call_stack: Vec<String>,

    /// Current memory usage
    pub memory_usage: usize,

    /// Tensor values (tensor_id -> description)
    pub tensor_values: HashMap<String, String>,

    /// Gradient values (tensor_id -> gradient_description)
    pub gradient_values: HashMap<String, String>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new() -> Self {
        Self {
            operation_index: 0,
            total_operations: 0,
            call_stack: Vec::new(),
            memory_usage: 0,
            tensor_values: HashMap::new(),
            gradient_values: HashMap::new(),
        }
    }
}

impl Default for DebugContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Interactive debugger for gradient computations
pub struct InteractiveDebugger {
    /// Debugger state
    state: Arc<RwLock<DebuggerState>>,

    /// Breakpoints
    breakpoints: Arc<Mutex<HashMap<u64, Breakpoint>>>,

    /// Watchpoints
    watchpoints: Arc<Mutex<HashMap<u64, Watchpoint>>>,

    /// Debug context
    context: Arc<Mutex<DebugContext>>,

    /// Execution history
    history: Arc<Mutex<VecDeque<TraceEvent>>>,

    /// Next breakpoint ID
    next_breakpoint_id: Arc<Mutex<u64>>,

    /// Next watchpoint ID
    next_watchpoint_id: Arc<Mutex<u64>>,

    /// Current path being debugged
    current_path: Arc<Mutex<Option<PathId>>>,

    /// Maximum history size
    max_history_size: usize,

    /// Command queue (for programmatic control)
    #[allow(dead_code)]
    command_queue: Arc<Mutex<VecDeque<DebugCommand>>>,
}

/// Debug command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebugCommand {
    /// Step to next operation
    Step,

    /// Continue execution
    Continue,

    /// Step over (skip into children)
    StepOver,

    /// Step out (return to parent)
    StepOut,

    /// Run to completion
    Run,

    /// Pause execution
    Pause,

    /// Restart from beginning
    Restart,

    /// Inspect tensor
    InspectTensor(String),

    /// Inspect gradient
    InspectGradient(String),

    /// Show call stack
    ShowCallStack,

    /// Show memory usage
    ShowMemory,

    /// List breakpoints
    ListBreakpoints,

    /// List watchpoints
    ListWatchpoints,
}

impl InteractiveDebugger {
    /// Create a new interactive debugger
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DebuggerState::Inactive)),
            breakpoints: Arc::new(Mutex::new(HashMap::new())),
            watchpoints: Arc::new(Mutex::new(HashMap::new())),
            context: Arc::new(Mutex::new(DebugContext::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            next_breakpoint_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            next_watchpoint_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            current_path: Arc::new(Mutex::new(None)),
            max_history_size: 1000,
            command_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Start debugging a gradient path
    pub fn start_debugging(&self, path_id: PathId) -> AutogradResult<()> {
        *self.state.write() = DebuggerState::Paused;
        *self.current_path.lock() = Some(path_id);
        self.context.lock().operation_index = 0;

        Ok(())
    }

    /// Stop debugging
    pub fn stop_debugging(&self) {
        *self.state.write() = DebuggerState::Inactive;
        *self.current_path.lock() = None;
        self.history.lock().clear();
    }

    /// Get current state
    pub fn state(&self) -> DebuggerState {
        *self.state.read()
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&self, condition: BreakpointCondition, description: String) -> u64 {
        let id = {
            let mut next_id = self.next_breakpoint_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let breakpoint = Breakpoint::new(id, condition, description);
        self.breakpoints.lock().insert(id, breakpoint);

        id
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&self, id: u64) -> AutogradResult<()> {
        self.breakpoints.lock().remove(&id);
        Ok(())
    }

    /// Enable a breakpoint
    pub fn enable_breakpoint(&self, id: u64) -> AutogradResult<()> {
        if let Some(bp) = self.breakpoints.lock().get_mut(&id) {
            bp.enabled = true;
            Ok(())
        } else {
            Err(AutogradError::Configuration {
                parameter: "breakpoint_id".to_string(),
                value: id.to_string(),
                reason: "Breakpoint not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// Disable a breakpoint
    pub fn disable_breakpoint(&self, id: u64) -> AutogradResult<()> {
        if let Some(bp) = self.breakpoints.lock().get_mut(&id) {
            bp.enabled = false;
            Ok(())
        } else {
            Err(AutogradError::Configuration {
                parameter: "breakpoint_id".to_string(),
                value: id.to_string(),
                reason: "Breakpoint not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// List all breakpoints
    pub fn list_breakpoints(&self) -> Vec<Breakpoint> {
        self.breakpoints.lock().values().cloned().collect()
    }

    /// Add a watchpoint
    pub fn add_watchpoint(&self, tensor_id: String) -> u64 {
        let id = {
            let mut next_id = self.next_watchpoint_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let watchpoint = Watchpoint {
            id,
            tensor_id,
            break_on_read: true,
            break_on_write: true,
            break_on_gradient: true,
            trigger_count: 0,
        };

        self.watchpoints.lock().insert(id, watchpoint);

        id
    }

    /// Remove a watchpoint
    pub fn remove_watchpoint(&self, id: u64) -> AutogradResult<()> {
        self.watchpoints.lock().remove(&id);
        Ok(())
    }

    /// List all watchpoints
    pub fn list_watchpoints(&self) -> Vec<Watchpoint> {
        self.watchpoints.lock().values().cloned().collect()
    }

    /// Process a trace event (called during execution)
    pub fn process_event(&self, event: &TraceEvent) -> AutogradResult<bool> {
        // Add to history
        {
            let mut history = self.history.lock();
            history.push_back(event.clone());

            while history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        // Update context
        {
            let mut context = self.context.lock();
            context.operation_index += 1;

            if let Some(mem) = event.memory_allocated {
                context.memory_usage += mem;
            }

            if let Some(mem) = event.memory_deallocated {
                context.memory_usage = context.memory_usage.saturating_sub(mem);
            }
        }

        // Check breakpoints
        let should_break = {
            let mut breakpoints = self.breakpoints.lock();
            let context = self.context.lock();

            breakpoints
                .values_mut()
                .any(|bp| bp.should_trigger(event, &context))
        };

        if should_break {
            *self.state.write() = DebuggerState::Paused;
            return Ok(true);
        }

        // Check state
        match *self.state.read() {
            DebuggerState::Stepping => {
                *self.state.write() = DebuggerState::Paused;
                Ok(true)
            }
            DebuggerState::Paused => Ok(true),
            _ => Ok(false),
        }
    }

    /// Execute a debug command
    pub fn execute_command(&self, command: DebugCommand) -> AutogradResult<String> {
        match command {
            DebugCommand::Step => {
                *self.state.write() = DebuggerState::Stepping;
                Ok("Stepping to next operation...".to_string())
            }

            DebugCommand::Continue => {
                *self.state.write() = DebuggerState::Continuing;
                Ok("Continuing execution...".to_string())
            }

            DebugCommand::Run => {
                *self.state.write() = DebuggerState::Running;
                Ok("Running to completion...".to_string())
            }

            DebugCommand::Pause => {
                *self.state.write() = DebuggerState::Paused;
                Ok("Paused".to_string())
            }

            DebugCommand::ShowCallStack => {
                let context = self.context.lock();
                let mut output = String::from("Call Stack:\n");

                for (i, op) in context.call_stack.iter().enumerate() {
                    output.push_str(&format!("  #{}: {}\n", i, op));
                }

                Ok(output)
            }

            DebugCommand::ShowMemory => {
                let context = self.context.lock();
                Ok(format!(
                    "Current memory usage: {} bytes",
                    context.memory_usage
                ))
            }

            DebugCommand::InspectTensor(tensor_id) => {
                let context = self.context.lock();
                if let Some(desc) = context.tensor_values.get(&tensor_id) {
                    Ok(format!("Tensor {}: {}", tensor_id, desc))
                } else {
                    Ok(format!("Tensor {} not found in current context", tensor_id))
                }
            }

            DebugCommand::InspectGradient(tensor_id) => {
                let context = self.context.lock();
                if let Some(desc) = context.gradient_values.get(&tensor_id) {
                    Ok(format!("Gradient for {}: {}", tensor_id, desc))
                } else {
                    Ok(format!("Gradient for {} not found", tensor_id))
                }
            }

            DebugCommand::ListBreakpoints => {
                let breakpoints = self.list_breakpoints();
                let mut output = String::from("Breakpoints:\n");

                for bp in breakpoints {
                    output.push_str(&format!(
                        "  #{}: {} [{}] (hits: {})\n",
                        bp.id,
                        bp.description,
                        if bp.enabled { "enabled" } else { "disabled" },
                        bp.hit_count
                    ));
                }

                Ok(output)
            }

            DebugCommand::ListWatchpoints => {
                let watchpoints = self.list_watchpoints();
                let mut output = String::from("Watchpoints:\n");

                for wp in watchpoints {
                    output.push_str(&format!(
                        "  #{}: {} (triggers: {})\n",
                        wp.id, wp.tensor_id, wp.trigger_count
                    ));
                }

                Ok(output)
            }

            DebugCommand::Restart => {
                self.stop_debugging();
                Ok("Debugger restarted".to_string())
            }

            DebugCommand::StepOver | DebugCommand::StepOut => {
                // TODO: Implement step over/out logic
                Ok("Not yet implemented".to_string())
            }
        }
    }

    /// Get debug context
    pub fn context(&self) -> DebugContext {
        self.context.lock().clone()
    }

    /// Get execution history
    pub fn history(&self) -> Vec<TraceEvent> {
        self.history.lock().iter().cloned().collect()
    }

    /// Generate debug summary
    pub fn summary(&self) -> String {
        let context = self.context.lock();
        let breakpoints = self.breakpoints.lock();
        let watchpoints = self.watchpoints.lock();

        let mut output = String::new();

        output.push_str("=== Interactive Debugger Summary ===\n\n");
        output.push_str(&format!("State: {:?}\n", *self.state.read()));
        output.push_str(&format!(
            "Progress: {}/{} operations\n",
            context.operation_index, context.total_operations
        ));
        output.push_str(&format!("Memory usage: {} bytes\n", context.memory_usage));
        output.push_str(&format!(
            "Breakpoints: {} ({} enabled)\n",
            breakpoints.len(),
            breakpoints.values().filter(|b| b.enabled).count()
        ));
        output.push_str(&format!("Watchpoints: {}\n", watchpoints.len()));

        output
    }
}

impl Default for InteractiveDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Global interactive debugger instance
static GLOBAL_DEBUGGER: once_cell::sync::Lazy<InteractiveDebugger> =
    once_cell::sync::Lazy::new(InteractiveDebugger::new);

/// Get the global debugger
pub fn global_debugger() -> &'static InteractiveDebugger {
    &GLOBAL_DEBUGGER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = InteractiveDebugger::new();
        assert_eq!(debugger.state(), DebuggerState::Inactive);
    }

    #[test]
    fn test_breakpoint_management() {
        let debugger = InteractiveDebugger::new();

        let bp_id = debugger.add_breakpoint(
            BreakpointCondition::OperationName("matmul".to_string()),
            "Break on matmul".to_string(),
        );

        assert!(bp_id > 0);

        let breakpoints = debugger.list_breakpoints();
        assert_eq!(breakpoints.len(), 1);

        debugger.disable_breakpoint(bp_id).unwrap();

        let breakpoints = debugger.list_breakpoints();
        assert!(!breakpoints[0].enabled);

        debugger.remove_breakpoint(bp_id).unwrap();

        let breakpoints = debugger.list_breakpoints();
        assert_eq!(breakpoints.len(), 0);
    }

    #[test]
    fn test_watchpoint_management() {
        let debugger = InteractiveDebugger::new();

        let wp_id = debugger.add_watchpoint("tensor_1".to_string());
        assert!(wp_id > 0);

        let watchpoints = debugger.list_watchpoints();
        assert_eq!(watchpoints.len(), 1);

        debugger.remove_watchpoint(wp_id).unwrap();

        let watchpoints = debugger.list_watchpoints();
        assert_eq!(watchpoints.len(), 0);
    }

    #[test]
    fn test_command_execution() {
        let debugger = InteractiveDebugger::new();

        let result = debugger.execute_command(DebugCommand::ShowMemory);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.contains("memory usage"));
    }

    #[test]
    fn test_state_transitions() {
        let debugger = InteractiveDebugger::new();

        debugger.execute_command(DebugCommand::Step).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Stepping);

        debugger.execute_command(DebugCommand::Continue).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Continuing);

        debugger.execute_command(DebugCommand::Pause).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Paused);
    }
}
