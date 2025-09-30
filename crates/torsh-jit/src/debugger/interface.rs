//! User interface for JIT debugging
//!
//! This module provides the interactive user interface for debugging including
//! command parsing, user interaction, and display formatting.

use super::core::{
    Breakpoint, BreakpointId, BreakpointLocation, DebugCommand, DebugState, DebugValue,
    DebuggerConfig, DisassemblyView, ExecutionLocation, InspectionResult, InspectionTarget,
    MemoryView, Watch, WatchId,
};
use super::state::CallStack;
use crate::{JitError, JitResult, NodeId};
use std::collections::HashMap;
use std::io::{self, Write};

/// Debugger user interface
pub struct DebuggerInterface {
    config: DebuggerConfig,
    command_history: Vec<String>,
    command_aliases: HashMap<String, String>,
}

impl DebuggerInterface {
    /// Create a new debugger interface
    ///
    /// # Arguments
    /// * `config` - Configuration for the debugger interface
    pub fn new(config: DebuggerConfig) -> Self {
        let mut aliases = HashMap::new();
        // Setup default command aliases
        aliases.insert("s".to_string(), "step".to_string());
        aliases.insert("n".to_string(), "next".to_string());
        aliases.insert("i".to_string(), "into".to_string());
        aliases.insert("o".to_string(), "out".to_string());
        aliases.insert("c".to_string(), "continue".to_string());
        aliases.insert("b".to_string(), "break".to_string());
        aliases.insert("d".to_string(), "delete".to_string());
        aliases.insert("w".to_string(), "watch".to_string());
        aliases.insert("p".to_string(), "inspect".to_string());
        aliases.insert("mem".to_string(), "memory".to_string());
        aliases.insert("dis".to_string(), "disasm".to_string());
        aliases.insert("h".to_string(), "help".to_string());
        aliases.insert("q".to_string(), "quit".to_string());

        Self {
            config,
            command_history: Vec::new(),
            command_aliases: aliases,
        }
    }

    /// Show welcome message
    pub fn show_welcome_message(&self) {
        println!("🦀 ToRSh JIT Debugger v1.0");
        println!("Type 'help' for available commands");
        println!("=====================================");
    }

    /// Display current debug state
    ///
    /// # Arguments
    /// * `state` - The current debug state to display
    pub fn display_current_state(&self, state: &DebugState) {
        println!("\n--- Current State ---");
        println!("📍 Location: {:?}", state.location);
        println!("🔢 Step: {}", state.execution_step);
        println!("▶️  Running: {}", state.is_running);

        if !state.call_stack.is_empty() {
            println!("📞 Call depth: {}", state.call_stack.depth());
        }

        if !state.variables.is_empty() {
            println!("🔍 Variables:");
            for (name, value) in &state.variables {
                println!("  {} = {:?}", name, value);
            }
        }
    }

    /// Get user command from input
    ///
    /// # Returns
    /// The parsed debug command
    pub fn get_user_command(&mut self) -> JitResult<DebugCommand> {
        print!("(torsh-debug) ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| JitError::RuntimeError(format!("Failed to read input: {}", e)))?;

        let input = input.trim();

        // Add to command history if not empty
        if !input.is_empty() {
            self.command_history.push(input.to_string());
        }

        self.parse_command(input)
    }

    /// Parse a command string into a DebugCommand
    ///
    /// # Arguments
    /// * `input` - The input string to parse
    ///
    /// # Returns
    /// The parsed debug command
    pub fn parse_command(&self, input: &str) -> JitResult<DebugCommand> {
        let parts: Vec<&str> = input.split_whitespace().collect();

        if parts.is_empty() {
            return Err(JitError::RuntimeError("Empty command".to_string()));
        }

        // Resolve aliases
        let binding = parts[0].to_string();
        let command = self.command_aliases.get(parts[0]).unwrap_or(&binding);

        match command.as_str() {
            "step" => Ok(DebugCommand::Step),
            "next" => Ok(DebugCommand::StepOver),
            "into" => Ok(DebugCommand::StepInto),
            "out" => Ok(DebugCommand::StepOut),
            "continue" => Ok(DebugCommand::Continue),
            "break" => {
                if let Some(&location_str) = parts.get(1) {
                    let location = self.parse_breakpoint_location(location_str)?;
                    Ok(DebugCommand::SetBreakpoint { location })
                } else {
                    Err(JitError::RuntimeError(
                        "Breakpoint location required".to_string(),
                    ))
                }
            }
            "delete" => {
                if let Some(&id_str) = parts.get(1) {
                    let id = id_str
                        .parse::<u64>()
                        .map_err(|_| JitError::RuntimeError("Invalid breakpoint ID".to_string()))?;
                    Ok(DebugCommand::RemoveBreakpoint {
                        id: BreakpointId(id),
                    })
                } else {
                    Err(JitError::RuntimeError("Breakpoint ID required".to_string()))
                }
            }
            "breakpoints" => Ok(DebugCommand::ListBreakpoints),
            "watch" => {
                if parts.len() > 1 {
                    let expression = parts[1..].join(" ");
                    Ok(DebugCommand::Watch { expression })
                } else {
                    Err(JitError::RuntimeError(
                        "Watch expression required".to_string(),
                    ))
                }
            }
            "unwatch" => {
                if let Some(&id_str) = parts.get(1) {
                    let id = id_str
                        .parse::<u64>()
                        .map_err(|_| JitError::RuntimeError("Invalid watch ID".to_string()))?;
                    Ok(DebugCommand::Unwatch { id: WatchId(id) })
                } else {
                    Err(JitError::RuntimeError("Watch ID required".to_string()))
                }
            }
            "watches" => Ok(DebugCommand::ListWatches),
            "inspect" => {
                if let Some(&target_str) = parts.get(1) {
                    let target = self.parse_inspection_target(target_str)?;
                    Ok(DebugCommand::Inspect { target })
                } else {
                    Err(JitError::RuntimeError(
                        "Inspection target required".to_string(),
                    ))
                }
            }
            "stack" => Ok(DebugCommand::CallStack),
            "locals" => Ok(DebugCommand::Locals),
            "memory" => {
                if let Some(&addr_str) = parts.get(1) {
                    let address = self.parse_memory_address(addr_str)?;
                    Ok(DebugCommand::Memory { address })
                } else {
                    Err(JitError::RuntimeError(
                        "Memory address required".to_string(),
                    ))
                }
            }
            "disasm" => {
                let location = if parts.len() > 1 {
                    self.parse_execution_location(&parts[1..])?
                } else {
                    // Use current location placeholder
                    ExecutionLocation::GraphNode(NodeId::new(0))
                };
                Ok(DebugCommand::Disassemble { location })
            }
            "help" => Ok(DebugCommand::Help),
            "quit" | "exit" => Ok(DebugCommand::Quit),
            "history" => {
                self.show_command_history();
                Ok(DebugCommand::Help) // Just show help after history
            }
            "clear" => {
                self.clear_screen();
                Ok(DebugCommand::Help)
            }
            _ => Err(JitError::RuntimeError(format!(
                "Unknown command: {}. Type 'help' for available commands.",
                command
            ))),
        }
    }

    /// Parse a breakpoint location string
    fn parse_breakpoint_location(&self, location_str: &str) -> JitResult<BreakpointLocation> {
        if location_str.starts_with("node_") {
            let node_index = location_str[5..]
                .parse::<usize>()
                .map_err(|_| JitError::RuntimeError("Invalid node index".to_string()))?;
            Ok(BreakpointLocation::GraphNode(NodeId::new(node_index)))
        } else if location_str.contains(':') {
            let parts: Vec<&str> = location_str.split(':').collect();
            if parts.len() == 2 {
                let function = parts[0].to_string();
                let instruction = parts[1]
                    .parse::<usize>()
                    .map_err(|_| JitError::RuntimeError("Invalid instruction index".to_string()))?;
                Ok(BreakpointLocation::Instruction {
                    function,
                    instruction,
                })
            } else {
                Err(JitError::RuntimeError(
                    "Invalid breakpoint location format. Use 'node_N' or 'function:instruction'"
                        .to_string(),
                ))
            }
        } else {
            Err(JitError::RuntimeError(
                "Invalid breakpoint location format. Use 'node_N' or 'function:instruction'"
                    .to_string(),
            ))
        }
    }

    /// Parse an inspection target string
    fn parse_inspection_target(&self, target_str: &str) -> JitResult<InspectionTarget> {
        if target_str.starts_with("node_") {
            let node_index = target_str[5..]
                .parse::<usize>()
                .map_err(|_| JitError::RuntimeError("Invalid node index".to_string()))?;
            Ok(InspectionTarget::Node(NodeId::new(node_index)))
        } else if target_str.starts_with("0x") {
            let address = u64::from_str_radix(&target_str[2..], 16)
                .map_err(|_| JitError::RuntimeError("Invalid memory address".to_string()))?;
            Ok(InspectionTarget::Memory(address))
        } else {
            Ok(InspectionTarget::Variable(target_str.to_string()))
        }
    }

    /// Parse memory address from string
    fn parse_memory_address(&self, addr_str: &str) -> JitResult<u64> {
        if addr_str.starts_with("0x") {
            u64::from_str_radix(&addr_str[2..], 16)
                .map_err(|_| JitError::RuntimeError("Invalid hexadecimal address".to_string()))
        } else {
            addr_str
                .parse::<u64>()
                .map_err(|_| JitError::RuntimeError("Invalid decimal address".to_string()))
        }
    }

    /// Parse an execution location from command parts
    fn parse_execution_location(&self, parts: &[&str]) -> JitResult<ExecutionLocation> {
        if parts.is_empty() {
            return Err(JitError::RuntimeError("Location required".to_string()));
        }

        if parts[0].starts_with("node_") {
            let node_index = parts[0][5..]
                .parse::<usize>()
                .map_err(|_| JitError::RuntimeError("Invalid node index".to_string()))?;
            Ok(ExecutionLocation::GraphNode(NodeId::new(node_index)))
        } else if parts.len() >= 2 {
            let function = parts[0].to_string();
            let instruction_index = parts[1]
                .parse::<usize>()
                .map_err(|_| JitError::RuntimeError("Invalid instruction index".to_string()))?;
            Ok(ExecutionLocation::Instruction {
                function,
                instruction_index,
            })
        } else {
            Err(JitError::RuntimeError(
                "Invalid location format. Use 'node_N' or 'function instruction_index'".to_string(),
            ))
        }
    }

    /// Show that a breakpoint was set
    pub fn show_breakpoint_set(&self, location: BreakpointLocation) {
        println!("✅ Breakpoint set at {:?}", location);
    }

    /// Show that a breakpoint was removed
    pub fn show_breakpoint_removed(&self, id: BreakpointId) {
        println!("❌ Breakpoint {} removed", id.0);
    }

    /// Show list of breakpoints
    pub fn show_breakpoints(&self, breakpoints: &[&Breakpoint]) {
        if breakpoints.is_empty() {
            println!("📍 No breakpoints set");
        } else {
            println!("📍 Breakpoints:");
            for bp in breakpoints {
                let status = if bp.enabled {
                    "✅ enabled"
                } else {
                    "❌ disabled"
                };
                let condition = if let Some(ref cond) = bp.condition {
                    format!(" [condition: {}]", cond)
                } else {
                    String::new()
                };
                println!(
                    "  {} - {:?} ({}) [hits: {}]{}",
                    bp.id.0, bp.location, status, bp.hit_count, condition
                );
            }
        }
    }

    /// Show that a watch was added
    pub fn show_watch_added(&self, id: WatchId, expression: &str) {
        println!("👁️  Watch {} added: {}", id.0, expression);
    }

    /// Show that a watch was removed
    pub fn show_watch_removed(&self, id: WatchId) {
        println!("❌ Watch {} removed", id.0);
    }

    /// Show list of watch expressions
    pub fn show_watches(&self, watches: &[&Watch]) {
        if watches.is_empty() {
            println!("👁️  No watches set");
        } else {
            println!("👁️  Watches:");
            for watch in watches {
                let status = if watch.enabled {
                    "✅ enabled"
                } else {
                    "❌ disabled"
                };
                let value = if let Some(ref val) = watch.last_value {
                    format!(" = {:?}", val)
                } else {
                    " = <not evaluated>".to_string()
                };
                println!(
                    "  {} - {} ({}){}",
                    watch.id.0, watch.expression, status, value
                );
            }
        }
    }

    /// Show inspection result
    pub fn show_inspection_result(&self, result: &InspectionResult) {
        match result {
            InspectionResult::Variable {
                name,
                value,
                type_info,
            } => {
                println!(
                    "🔍 Variable '{}': {:?} (type: {}, {} bytes)",
                    name, value, type_info.type_name, type_info.size_bytes
                );
            }
            InspectionResult::Node {
                node_id,
                value,
                metadata,
            } => {
                println!("🔍 Node {:?}: {:?}", node_id, value);
                println!("  Operation: {}", metadata.operation);
                println!("  Inputs: {}", metadata.input_count);
                println!("  Output shape: {:?}", metadata.output_shape);
                println!("  Data type: {:?}", metadata.dtype);
            }
            InspectionResult::Memory {
                address,
                content,
                size,
            } => {
                println!("🔍 Memory at 0x{:x} (size: {}):", address, size);
                self.format_memory_dump(*address, content);
            }
        }
    }

    /// Show call stack
    pub fn show_call_stack(&self, call_stack: &CallStack) {
        if call_stack.is_empty() {
            println!("📞 Call stack is empty");
        } else {
            println!("📞 Call stack (depth: {}):", call_stack.depth());
            for (i, frame) in call_stack.frames().iter().rev().enumerate() {
                let marker = if i == 0 { "▶️ " } else { "  " };
                println!(
                    "{}#{} - {} at {:?}",
                    marker, i, frame.function_name, frame.location
                );
                if !frame.local_variables.is_empty() {
                    println!("      locals: {} variables", frame.local_variables.len());
                }
            }
        }
    }

    /// Show local variables
    pub fn show_local_variables(&self, locals: &HashMap<String, DebugValue>) {
        if locals.is_empty() {
            println!("🔢 No local variables");
        } else {
            println!("🔢 Local variables ({}):", locals.len());
            for (name, value) in locals {
                println!("  {} = {:?}", name, value);
            }
        }
    }

    /// Show memory view
    pub fn show_memory_view(&self, memory_view: &MemoryView) {
        println!(
            "🧠 Memory view starting at 0x{:x}:",
            memory_view.start_address
        );
        self.format_memory_dump(memory_view.start_address, &memory_view.content);
    }

    /// Format memory dump for display
    fn format_memory_dump(&self, start_address: u64, content: &[u8]) {
        for (i, chunk) in content.chunks(16).enumerate() {
            print!("  {:08x}: ", start_address + (i * 16) as u64);

            // Hex bytes
            for (j, byte) in chunk.iter().enumerate() {
                if j == 8 {
                    print!(" "); // Extra space in the middle
                }
                print!("{:02x} ", byte);
            }

            // Pad if less than 16 bytes
            for j in chunk.len()..16 {
                if j == 8 {
                    print!(" ");
                }
                print!("   ");
            }

            print!(" |");

            // ASCII representation
            for byte in chunk {
                let ch = if byte.is_ascii_graphic() || *byte == b' ' {
                    *byte as char
                } else {
                    '.'
                };
                print!("{}", ch);
            }

            println!("|");
        }
    }

    /// Show disassembly
    pub fn show_disassembly(&self, disassembly: &DisassemblyView) {
        println!("📋 Disassembly at {:?}:", disassembly.location);
        for instruction in &disassembly.instructions {
            print!(
                "  {:08x}: {} {}",
                instruction.address, instruction.opcode, instruction.operands
            );
            if let Some(comment) = &instruction.comment {
                print!(" ; {}", comment);
            }
            println!();
        }
    }

    /// Show help information
    pub fn show_help(&self) {
        println!("🆘 Available commands:");
        println!("  step, s            - Execute one step");
        println!("  next, n            - Step over function calls");
        println!("  into, i            - Step into function calls");
        println!("  out, o             - Step out of current function");
        println!("  continue, c        - Continue execution");
        println!("  break <loc>, b     - Set breakpoint at location");
        println!("    Examples: break node_5, break main:10");
        println!("  delete <id>, d     - Remove breakpoint by ID");
        println!("  breakpoints        - List all breakpoints");
        println!("  watch <expr>, w    - Watch expression");
        println!("    Examples: watch variable_name, watch node_3");
        println!("  unwatch <id>       - Remove watch by ID");
        println!("  watches            - List all watches");
        println!("  inspect <target>, p - Inspect variable/node/memory");
        println!("    Examples: inspect var, inspect node_1, inspect 0x1000");
        println!("  stack              - Show call stack");
        println!("  locals             - Show local variables");
        println!("  memory <addr>, mem - Show memory contents");
        println!("    Examples: memory 0x1000, memory 4096");
        println!("  disasm <loc>, dis  - Disassemble at location");
        println!("  history            - Show command history");
        println!("  clear              - Clear screen");
        println!("  help, h            - Show this help");
        println!("  quit, q            - Exit debugger");
    }

    /// Show execution complete message
    pub fn show_execution_complete(&self) {
        println!("✅ Execution completed successfully.");
    }

    /// Show command history
    pub fn show_command_history(&self) {
        if self.command_history.is_empty() {
            println!("📜 No command history");
        } else {
            println!("📜 Command history:");
            for (i, cmd) in self.command_history.iter().enumerate() {
                println!("  {}: {}", i + 1, cmd);
            }
        }
    }

    /// Clear the screen
    pub fn clear_screen(&self) {
        // Clear screen using ANSI escape sequences
        print!("\x1B[2J\x1B[1;1H");
        io::stdout().flush().unwrap();
    }

    /// Show error message
    pub fn show_error(&self, error: &str) {
        println!("❌ Error: {}", error);
    }

    /// Show warning message
    pub fn show_warning(&self, warning: &str) {
        println!("⚠️  Warning: {}", warning);
    }

    /// Show information message
    pub fn show_info(&self, info: &str) {
        println!("ℹ️  {}", info);
    }

    /// Show success message
    pub fn show_success(&self, message: &str) {
        println!("✅ {}", message);
    }

    /// Set a command alias
    pub fn set_alias(&mut self, alias: String, command: String) {
        self.command_aliases.insert(alias, command);
    }

    /// Remove a command alias
    pub fn remove_alias(&mut self, alias: &str) {
        self.command_aliases.remove(alias);
    }

    /// Get command history
    pub fn get_command_history(&self) -> &[String] {
        &self.command_history
    }

    /// Clear command history
    pub fn clear_command_history(&mut self) {
        self.command_history.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &DebuggerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DebuggerConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_interface() -> DebuggerInterface {
        DebuggerInterface::new(DebuggerConfig::default())
    }

    #[test]
    fn test_interface_creation() {
        let interface = create_test_interface();
        assert!(!interface.command_aliases.is_empty());
        assert!(interface.command_history.is_empty());
    }

    #[test]
    fn test_command_parsing_basic() {
        let interface = create_test_interface();

        assert!(matches!(
            interface.parse_command("step"),
            Ok(DebugCommand::Step)
        ));
        assert!(matches!(
            interface.parse_command("s"),
            Ok(DebugCommand::Step)
        ));
        assert!(matches!(
            interface.parse_command("continue"),
            Ok(DebugCommand::Continue)
        ));
        assert!(matches!(
            interface.parse_command("c"),
            Ok(DebugCommand::Continue)
        ));
        assert!(matches!(
            interface.parse_command("help"),
            Ok(DebugCommand::Help)
        ));
        assert!(matches!(
            interface.parse_command("quit"),
            Ok(DebugCommand::Quit)
        ));
    }

    #[test]
    fn test_breakpoint_location_parsing() {
        let interface = create_test_interface();

        let result = interface.parse_breakpoint_location("node_5");
        assert!(matches!(result, Ok(BreakpointLocation::GraphNode(_))));

        let result = interface.parse_breakpoint_location("main:10");
        assert!(matches!(result, Ok(BreakpointLocation::Instruction { .. })));

        let result = interface.parse_breakpoint_location("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_inspection_target_parsing() {
        let interface = create_test_interface();

        let result = interface.parse_inspection_target("node_3");
        assert!(matches!(result, Ok(InspectionTarget::Node(_))));

        let result = interface.parse_inspection_target("0x1000");
        assert!(matches!(result, Ok(InspectionTarget::Memory(4096))));

        let result = interface.parse_inspection_target("variable_name");
        assert!(matches!(result, Ok(InspectionTarget::Variable(_))));
    }

    #[test]
    fn test_memory_address_parsing() {
        let interface = create_test_interface();

        assert_eq!(interface.parse_memory_address("0x1000").unwrap(), 4096);
        assert_eq!(interface.parse_memory_address("1000").unwrap(), 1000);
        assert!(interface.parse_memory_address("invalid").is_err());
    }

    #[test]
    fn test_command_with_arguments() {
        let interface = create_test_interface();

        let result = interface.parse_command("break node_5");
        assert!(matches!(result, Ok(DebugCommand::SetBreakpoint { .. })));

        let result = interface.parse_command("watch variable_name");
        assert!(matches!(result, Ok(DebugCommand::Watch { .. })));

        let result = interface.parse_command("inspect node_3");
        assert!(matches!(result, Ok(DebugCommand::Inspect { .. })));
    }

    #[test]
    fn test_alias_management() {
        let mut interface = create_test_interface();

        interface.set_alias("test".to_string(), "step".to_string());
        assert!(matches!(
            interface.parse_command("test"),
            Ok(DebugCommand::Step)
        ));

        interface.remove_alias("test");
        assert!(interface.parse_command("test").is_err());
    }

    #[test]
    fn test_invalid_commands() {
        let interface = create_test_interface();

        assert!(interface.parse_command("").is_err());
        assert!(interface.parse_command("invalid_command").is_err());
        assert!(interface.parse_command("break").is_err()); // Missing argument
        assert!(interface.parse_command("watch").is_err()); // Missing argument
    }
}
