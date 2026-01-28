//! Interactive JIT debugger with breakpoints, stepping, and introspection
//!
//! This module provides comprehensive debugging capabilities for JIT compilation including:
//! - Interactive debugging sessions with step-by-step execution
//! - Breakpoint management and conditional breakpoints
//! - State introspection and variable watching
//! - Call stack analysis and navigation
//! - Memory state inspection and analysis
//! - Performance profiling and statistics
//!
//! # Architecture
//!
//! The debugger is organized into focused modules:
//!
//! - **core**: Core types, enums, and data structures
//! - **breakpoints**: Breakpoint management system
//! - **watch**: Watch expression management
//! - **execution**: Debug execution engine with instrumentation
//! - **state**: Call stack and memory state management
//! - **interface**: User interface and command parsing
//! - **session**: Debug session management and execution logic
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
//! use torsh_jit::ComputationGraph;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a debugger with default configuration
//! let mut debugger = JitDebugger::new(DebuggerConfig::default());
//!
//! // Debug a computation graph
//! let graph = ComputationGraph::new();
//! // let result = debugger.debug_graph(graph)?;
//! // println!("Debug session completed with {} steps", result.execution_trace.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Interactive Debugging
//!
//! ```rust
//! use torsh_jit::debugger::{JitDebugger, DebuggerConfig, BreakpointLocation};
//! use torsh_jit::{ComputationGraph, NodeId};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut debugger = JitDebugger::new(DebuggerConfig::default());
//!
//! // Set breakpoints
//! let bp_id = debugger.set_breakpoint(BreakpointLocation::GraphNode(NodeId::new(5)))?;
//!
//! // Add watch expressions
//! let watch_id = debugger.add_watch("node_output".to_string())?;
//!
//! // Start debugging session (commented for doctest)
//! let graph = ComputationGraph::new();
//! // let result = debugger.debug_graph(graph)?;
//! # Ok(())
//! # }
//! ```

pub mod breakpoints;
pub mod core;
pub mod execution;
pub mod interface;
pub mod session;
pub mod state;
pub mod watch;

// Re-export core types for convenience
pub use core::{
    Breakpoint, BreakpointId, BreakpointLocation, CallFrame, ContinueResult, DebugCommand,
    DebugCommandResult, DebugSessionResult, DebugState, DebugStatistics, DebugValue,
    DebuggerConfig, DisassemblyInstruction, DisassemblyView, EvaluationResult, ExecutionLocation,
    ExecutionStep, InspectionResult, InspectionTarget, InstructionExecutionResult, MemoryView,
    NodeExecutionResult, NodeMetadata, StepResult, TypeInfo, UiMode, Watch, WatchId, WatchUpdate,
};

// Re-export main components
pub use breakpoints::BreakpointManager;
pub use execution::{DebugExecutionEngine, OperationStatistics};
pub use interface::DebuggerInterface;
pub use session::DebugSession;
pub use state::{CallStack, CallStackSummary, MemoryRegion, MemoryState, MemoryStats};
pub use watch::{ExpressionEvaluator, WatchManager};

use crate::{ir::IrModule, ComputationGraph, JitError, JitResult};
use std::sync::{Arc, Mutex};

/// Interactive JIT debugger
///
/// The main orchestrator that coordinates all debugging components to provide
/// a comprehensive debugging experience for JIT-compiled code.
///
/// # Features
///
/// - **Step-by-step execution**: Execute code one instruction/node at a time
/// - **Breakpoint management**: Set conditional and unconditional breakpoints
/// - **Watch expressions**: Monitor variable and expression values
/// - **State inspection**: Examine variables, memory, and call stack
/// - **Performance profiling**: Track execution timing and statistics
/// - **Interactive interface**: Command-line interface with rich formatting
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
/// use torsh_jit::ComputationGraph;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut debugger = JitDebugger::new(DebuggerConfig::default());
/// let graph = ComputationGraph::new();
/// // let result = debugger.debug_graph(graph)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Advanced Configuration
///
/// ```rust
/// use torsh_jit::debugger::{JitDebugger, DebuggerConfig, UiMode};
///
/// let config = DebuggerConfig {
///     enable_single_step: true,
///     enable_breakpoints: true,
///     enable_watches: true,
///     enable_memory_view: true,
///     enable_disassembly: true,
///     max_trace_length: 10000,
///     ui_mode: UiMode::Interactive,
/// };
///
/// let mut debugger = JitDebugger::new(config);
/// ```
pub struct JitDebugger {
    config: DebuggerConfig,
    session: Option<Arc<Mutex<DebugSession>>>,
    breakpoints: BreakpointManager,
    watch_manager: WatchManager,
    call_stack: CallStack,
    execution_engine: DebugExecutionEngine,
    ui_interface: DebuggerInterface,
}

impl JitDebugger {
    /// Create a new JIT debugger
    ///
    /// # Arguments
    /// * `config` - Configuration for the debugger
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
    ///
    /// let debugger = JitDebugger::new(DebuggerConfig::default());
    /// ```
    pub fn new(config: DebuggerConfig) -> Self {
        Self {
            breakpoints: BreakpointManager::new(),
            watch_manager: WatchManager::new(),
            call_stack: CallStack::new(),
            execution_engine: DebugExecutionEngine::new(config.clone()),
            ui_interface: DebuggerInterface::new(config.clone()),
            session: None,
            config,
        }
    }

    /// Start a debugging session for a computation graph
    ///
    /// This method starts an interactive debugging session that allows step-by-step
    /// execution through the computation graph with full debugging capabilities.
    ///
    /// # Arguments
    /// * `graph` - The computation graph to debug
    ///
    /// # Returns
    /// A `DebugSessionResult` containing the execution trace, final state,
    /// command history, and statistics from the debugging session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
    /// use torsh_jit::ComputationGraph;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut debugger = JitDebugger::new(DebuggerConfig::default());
    /// let graph = ComputationGraph::new();
    /// // let result = debugger.debug_graph(graph)?;
    /// // println!("Executed {} steps", result.execution_trace.len());
    /// // println!("Final state: {:?}", result.final_state);
    /// # Ok(())
    /// # }
    /// ```
    pub fn debug_graph(&mut self, graph: ComputationGraph) -> JitResult<DebugSessionResult> {
        let session = DebugSession::new(graph, self.config.clone());
        let session_arc = Arc::new(Mutex::new(session));
        self.session = Some(session_arc.clone());

        // Start interactive debugging loop
        self.interactive_debug_loop(session_arc)
    }

    /// Start a debugging session for an IR module
    ///
    /// This method starts an interactive debugging session for IR (Intermediate Representation)
    /// code, allowing instruction-level debugging with full state inspection.
    ///
    /// # Arguments
    /// * `ir_module` - The IR module to debug
    ///
    /// # Returns
    /// A `DebugSessionResult` containing the execution trace, final state,
    /// command history, and statistics from the debugging session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
    /// use torsh_jit::ir::IrModule;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut debugger = JitDebugger::new(DebuggerConfig::default());
    /// let ir_module = IrModule::new("main".to_string());
    /// // let result = debugger.debug_ir(ir_module)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn debug_ir(&mut self, ir_module: IrModule) -> JitResult<DebugSessionResult> {
        let session = DebugSession::from_ir(ir_module, self.config.clone());
        let session_arc = Arc::new(Mutex::new(session));
        self.session = Some(session_arc.clone());

        // Start interactive debugging loop
        self.interactive_debug_loop(session_arc)
    }

    /// Interactive debugging loop
    ///
    /// The main loop that handles user interaction, command processing,
    /// and execution control during a debugging session.
    fn interactive_debug_loop(
        &mut self,
        session: Arc<Mutex<DebugSession>>,
    ) -> JitResult<DebugSessionResult> {
        let mut command_history = Vec::new();
        let mut continue_execution = true;

        self.ui_interface.show_welcome_message();

        while continue_execution {
            // Get current state
            let current_state = {
                let session_guard = session.lock().expect("lock should not be poisoned");
                session_guard.get_current_state()
            };

            // Display current state
            self.ui_interface.display_current_state(&current_state);

            // Get user command
            let command = self.ui_interface.get_user_command()?;
            command_history.push(command.clone());

            // Process command
            match self.process_debug_command(command, session.clone())? {
                DebugCommandResult::Continue => {}
                DebugCommandResult::Exit => continue_execution = false,
                DebugCommandResult::ExecutionComplete => {
                    self.ui_interface.show_execution_complete();
                    continue_execution = false;
                }
            }
        }

        // Generate session result
        let session_guard = session.lock().expect("lock should not be poisoned");
        Ok(DebugSessionResult {
            execution_trace: session_guard.get_execution_trace(),
            final_state: session_guard.get_current_state(),
            command_history,
            statistics: session_guard.get_statistics(),
        })
    }

    /// Process a debug command
    ///
    /// Handles the execution of user commands during debugging sessions.
    fn process_debug_command(
        &mut self,
        command: DebugCommand,
        session: Arc<Mutex<DebugSession>>,
    ) -> JitResult<DebugCommandResult> {
        match command {
            DebugCommand::Step => {
                let mut session_guard = session.lock().expect("lock should not be poisoned");
                session_guard.step()?;
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::StepOver => {
                let mut session_guard = session.lock().expect("lock should not be poisoned");
                session_guard.step_over()?;
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::StepInto => {
                let mut session_guard = session.lock().expect("lock should not be poisoned");
                session_guard.step_into()?;
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::StepOut => {
                let mut session_guard = session.lock().expect("lock should not be poisoned");
                session_guard.step_out()?;
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Continue => {
                let mut session_guard = session.lock().expect("lock should not be poisoned");
                let result = session_guard.continue_execution()?;
                match result {
                    ContinueResult::Breakpoint => Ok(DebugCommandResult::Continue),
                    ContinueResult::Completed => Ok(DebugCommandResult::ExecutionComplete),
                }
            }
            DebugCommand::SetBreakpoint { location } => {
                self.breakpoints.set_breakpoint(location.clone())?;
                self.ui_interface.show_breakpoint_set(location);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::RemoveBreakpoint { id } => {
                self.breakpoints.remove_breakpoint(id)?;
                self.ui_interface.show_breakpoint_removed(id);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::ListBreakpoints => {
                let breakpoints = self.breakpoints.list_breakpoints();
                self.ui_interface.show_breakpoints(&breakpoints);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Watch { expression } => {
                let watch_id = self.watch_manager.add_watch(expression.clone())?;
                self.ui_interface.show_watch_added(watch_id, &expression);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Unwatch { id } => {
                self.watch_manager.remove_watch(id)?;
                self.ui_interface.show_watch_removed(id);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::ListWatches => {
                let watches = self.watch_manager.list_watches();
                self.ui_interface.show_watches(&watches);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Inspect { target } => {
                let session_guard = session.lock().expect("lock should not be poisoned");
                let inspection_result = session_guard.inspect_target(&target)?;
                self.ui_interface.show_inspection_result(&inspection_result);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::CallStack => {
                let session_guard = session.lock().expect("lock should not be poisoned");
                let call_stack = session_guard.get_call_stack();
                self.ui_interface.show_call_stack(&call_stack);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Locals => {
                let session_guard = session.lock().expect("lock should not be poisoned");
                let locals = session_guard.get_local_variables();
                self.ui_interface.show_local_variables(&locals);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Memory { address } => {
                let session_guard = session.lock().expect("lock should not be poisoned");
                let memory_view = session_guard.get_memory_view(address)?;
                self.ui_interface.show_memory_view(&memory_view);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Disassemble { location } => {
                let session_guard = session.lock().expect("lock should not be poisoned");
                let disassembly = session_guard.disassemble_at(location)?;
                self.ui_interface.show_disassembly(&disassembly);
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Help => {
                self.ui_interface.show_help();
                Ok(DebugCommandResult::Continue)
            }
            DebugCommand::Quit => Ok(DebugCommandResult::Exit),
        }
    }

    /// Set a breakpoint at a specific location
    ///
    /// # Arguments
    /// * `location` - The location where the breakpoint should be set
    ///
    /// # Returns
    /// The ID of the newly created breakpoint
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig, BreakpointLocation};
    /// use torsh_jit::NodeId;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut debugger = JitDebugger::new(DebuggerConfig::default());
    /// let location = BreakpointLocation::GraphNode(NodeId::new(5));
    /// let bp_id = debugger.set_breakpoint(location)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_breakpoint(&mut self, location: BreakpointLocation) -> JitResult<BreakpointId> {
        self.breakpoints.set_breakpoint(location)
    }

    /// Remove a breakpoint
    ///
    /// # Arguments
    /// * `id` - The ID of the breakpoint to remove
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig, BreakpointLocation};
    /// use torsh_jit::NodeId;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut debugger = JitDebugger::new(DebuggerConfig::default());
    /// let location = BreakpointLocation::GraphNode(NodeId::new(0));
    /// let bp_id = debugger.set_breakpoint(location)?;
    /// debugger.remove_breakpoint(bp_id)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_breakpoint(&mut self, id: BreakpointId) -> JitResult<()> {
        self.breakpoints.remove_breakpoint(id)
    }

    /// Add a watch expression
    ///
    /// # Arguments
    /// * `expression` - The expression to watch
    ///
    /// # Returns
    /// The ID of the newly created watch
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_jit::debugger::{JitDebugger, DebuggerConfig};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut debugger = JitDebugger::new(DebuggerConfig::default());
    /// let watch_id = debugger.add_watch("variable_name".to_string())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_watch(&mut self, expression: String) -> JitResult<WatchId> {
        self.watch_manager.add_watch(expression)
    }

    /// Remove a watch expression
    ///
    /// # Arguments
    /// * `id` - The ID of the watch to remove
    pub fn remove_watch(&mut self, id: WatchId) -> JitResult<()> {
        self.watch_manager.remove_watch(id)
    }

    /// Get current debugging state
    ///
    /// # Returns
    /// The current debug state if a session is active, None otherwise
    pub fn get_current_state(&self) -> Option<DebugState> {
        if let Some(session) = &self.session {
            let session_guard = session.lock().expect("lock should not be poisoned");
            Some(session_guard.get_current_state())
        } else {
            None
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
        if let Some(session) = &self.session {
            let session_guard = session.lock().expect("lock should not be poisoned");
            session_guard.evaluate_expression(expression)
        } else {
            Err(JitError::RuntimeError(
                "No active debug session".to_string(),
            ))
        }
    }

    /// Get breakpoint manager
    pub fn breakpoints(&self) -> &BreakpointManager {
        &self.breakpoints
    }

    /// Get mutable breakpoint manager
    pub fn breakpoints_mut(&mut self) -> &mut BreakpointManager {
        &mut self.breakpoints
    }

    /// Get watch manager
    pub fn watch_manager(&self) -> &WatchManager {
        &self.watch_manager
    }

    /// Get mutable watch manager
    pub fn watch_manager_mut(&mut self) -> &mut WatchManager {
        &mut self.watch_manager
    }

    /// Get debugger interface
    pub fn interface(&self) -> &DebuggerInterface {
        &self.ui_interface
    }

    /// Get mutable debugger interface
    pub fn interface_mut(&mut self) -> &mut DebuggerInterface {
        &mut self.ui_interface
    }

    /// Get execution engine
    pub fn execution_engine(&self) -> &DebugExecutionEngine {
        &self.execution_engine
    }

    /// Get mutable execution engine
    pub fn execution_engine_mut(&mut self) -> &mut DebugExecutionEngine {
        &mut self.execution_engine
    }

    /// Get configuration
    pub fn config(&self) -> &DebuggerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DebuggerConfig) {
        self.config = config.clone();
        self.execution_engine.update_config(config.clone());
        self.ui_interface.update_config(config);
    }

    /// Check if there is an active debug session
    pub fn has_active_session(&self) -> bool {
        self.session.is_some()
    }

    /// Clear the current debug session
    pub fn clear_session(&mut self) {
        self.session = None;
    }

    /// Get debug statistics from the current session
    pub fn get_session_statistics(&self) -> Option<DebugStatistics> {
        if let Some(session) = &self.session {
            let session_guard = session.lock().expect("lock should not be poisoned");
            Some(session_guard.get_statistics())
        } else {
            None
        }
    }
}

/// Convenience function to create a debugger with default configuration
///
/// # Examples
///
/// ```rust
/// use torsh_jit::debugger::create_debugger;
///
/// let debugger = create_debugger();
/// ```
pub fn create_debugger() -> JitDebugger {
    JitDebugger::new(DebuggerConfig::default())
}

/// Convenience function to create a debugger with custom configuration
///
/// # Arguments
/// * `config` - Custom debugger configuration
///
/// # Examples
///
/// ```rust
/// use torsh_jit::debugger::{create_debugger_with_config, DebuggerConfig, UiMode};
///
/// let config = DebuggerConfig {
///     ui_mode: UiMode::Batch,
///     max_trace_length: 5000,
///     ..DebuggerConfig::default()
/// };
/// let debugger = create_debugger_with_config(config);
/// ```
pub fn create_debugger_with_config(config: DebuggerConfig) -> JitDebugger {
    JitDebugger::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = create_debugger();
        assert!(!debugger.has_active_session());
        assert_eq!(debugger.config().max_trace_length, 10000);
    }

    #[test]
    fn test_debugger_with_custom_config() {
        let config = DebuggerConfig {
            max_trace_length: 5000,
            enable_memory_view: false,
            ..DebuggerConfig::default()
        };
        let debugger = create_debugger_with_config(config);
        assert_eq!(debugger.config().max_trace_length, 5000);
        assert!(!debugger.config().enable_memory_view);
    }

    #[test]
    fn test_breakpoint_management() {
        let mut debugger = create_debugger();

        let location = BreakpointLocation::GraphNode(crate::NodeId::new(0));
        let bp_id = debugger.set_breakpoint(location).unwrap();

        assert_eq!(debugger.breakpoints().count(), 1);
        assert!(debugger.remove_breakpoint(bp_id).is_ok());
        assert_eq!(debugger.breakpoints().count(), 0);
    }

    #[test]
    fn test_watch_management() {
        let mut debugger = create_debugger();

        let watch_id = debugger.add_watch("test_expression".to_string()).unwrap();
        assert_eq!(debugger.watch_manager().count(), 1);

        assert!(debugger.remove_watch(watch_id).is_ok());
        assert_eq!(debugger.watch_manager().count(), 0);
    }

    #[test]
    fn test_configuration_update() {
        let mut debugger = create_debugger();

        let new_config = DebuggerConfig {
            max_trace_length: 8000,
            ..DebuggerConfig::default()
        };

        debugger.update_config(new_config.clone());
        assert_eq!(debugger.config().max_trace_length, 8000);
    }

    #[test]
    fn test_session_management() {
        let debugger = create_debugger();

        assert!(!debugger.has_active_session());
        assert!(debugger.get_current_state().is_none());
        assert!(debugger.get_session_statistics().is_none());
    }

    #[test]
    fn test_expression_evaluation_without_session() {
        let debugger = create_debugger();

        let result = debugger.evaluate_expression("test");
        assert!(result.is_err());
    }
}
