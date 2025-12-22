//! Debug and Development Tools for FX Graph Interpretation
//!
//! This module provides comprehensive debugging capabilities for FX graph execution,
//! including debug execution environments, step-by-step execution logging,
//! and graph validation utilities.

use crate::interpreter::execution::ExecutionEnvironment;
use crate::interpreter::metrics::ExecutionMetrics;
use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::Tensor;

/// Enhanced execution environment with debugging capabilities
///
/// Extends the basic execution environment with detailed logging, step-by-step execution,
/// and comprehensive debugging information for development and troubleshooting.
pub struct DebugExecutionEnvironment {
    /// Base execution environment
    env: ExecutionEnvironment,
    /// Execution metrics
    metrics: ExecutionMetrics,
    /// Debug mode flag
    debug_mode: bool,
    /// Step-by-step execution log
    execution_log: Vec<String>,
}

impl DebugExecutionEnvironment {
    /// Create a new debug execution environment
    ///
    /// # Arguments
    /// * `device` - Device type to execute on
    /// * `debug_mode` - Whether to enable detailed debugging
    ///
    /// # Returns
    /// * `Self` - New debug execution environment
    pub fn new(device: DeviceType, debug_mode: bool) -> Self {
        Self {
            env: ExecutionEnvironment::new(device),
            metrics: ExecutionMetrics::new(),
            debug_mode,
            execution_log: Vec::new(),
        }
    }

    /// Get reference to base execution environment
    ///
    /// # Returns
    /// * `&ExecutionEnvironment` - Reference to underlying execution environment
    pub fn env(&self) -> &ExecutionEnvironment {
        &self.env
    }

    /// Get mutable reference to base execution environment
    ///
    /// # Returns
    /// * `&mut ExecutionEnvironment` - Mutable reference to execution environment
    pub fn env_mut(&mut self) -> &mut ExecutionEnvironment {
        &mut self.env
    }

    /// Get execution metrics
    ///
    /// # Returns
    /// * `&ExecutionMetrics` - Reference to collected execution metrics
    pub fn metrics(&self) -> &ExecutionMetrics {
        &self.metrics
    }

    /// Check if debug mode is enabled
    ///
    /// # Returns
    /// * `bool` - True if debug mode is enabled
    pub fn is_debug_mode(&self) -> bool {
        self.debug_mode
    }

    /// Set debug mode
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable debug mode
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Add entry to execution log
    ///
    /// # Arguments
    /// * `message` - Log message to add
    pub fn log(&mut self, message: String) {
        if self.debug_mode {
            self.execution_log.push(message);
        }
    }

    /// Get execution log
    ///
    /// # Returns
    /// * `&[String]` - Reference to execution log entries
    pub fn get_log(&self) -> &[String] {
        &self.execution_log
    }

    /// Clear execution log
    pub fn clear_log(&mut self) {
        self.execution_log.clear();
    }

    /// Log node execution start
    ///
    /// # Arguments
    /// * `node_idx` - Index of node being executed
    /// * `node` - Node information
    pub fn log_node_start(&mut self, node_idx: NodeIndex, node: &Node) {
        let message = match node {
            Node::Input(name) => format!("Executing input node {:?}: '{}'", node_idx, name),
            Node::Call(op_name, args) => format!(
                "Executing call node {:?}: operation '{}' with {} args",
                node_idx,
                op_name,
                args.len()
            ),
            Node::Output => format!("Executing output node {:?}", node_idx),
            Node::Conditional { condition, .. } => format!(
                "Executing conditional node {:?}: condition '{}'",
                node_idx, condition
            ),
            Node::Loop { condition, .. } => format!(
                "Executing loop node {:?}: condition '{}'",
                node_idx, condition
            ),
            Node::Merge { inputs, .. } => format!(
                "Executing merge node {:?}: {} inputs",
                node_idx,
                inputs.len()
            ),
            Node::GetAttr { target, attr } => format!(
                "Executing get_attr node {:?}: {}.{}",
                node_idx, target, attr
            ),
        };
        self.log(message);
    }

    /// Log node execution completion
    ///
    /// # Arguments
    /// * `node_idx` - Index of completed node
    /// * `duration_ms` - Execution duration in milliseconds
    pub fn log_node_completion(&mut self, node_idx: NodeIndex, duration_ms: f64) {
        let message = format!("Completed node {:?} in {:.2} ms", node_idx, duration_ms);
        self.log(message);
    }

    /// Log tensor information
    ///
    /// # Arguments
    /// * `node_idx` - Index of node associated with tensor
    /// * `tensor` - Tensor to log information about
    pub fn log_tensor_info(&mut self, node_idx: NodeIndex, tensor: &Tensor) {
        if self.debug_mode {
            let message = format!(
                "Node {:?} tensor: shape={:?}, dtype={:?}, device={:?}",
                node_idx,
                tensor.shape().dims(),
                tensor.dtype(),
                tensor.device()
            );
            self.log(message);
        }
    }

    /// Log operation execution
    ///
    /// # Arguments
    /// * `op_name` - Name of operation
    /// * `input_count` - Number of input tensors
    /// * `duration_ms` - Operation execution time
    pub fn log_operation(&mut self, op_name: &str, input_count: usize, duration_ms: f64) {
        if self.debug_mode {
            let message = format!(
                "Operation '{}' with {} inputs completed in {:.2} ms",
                op_name, input_count, duration_ms
            );
            self.log(message);
        }
        self.metrics.add_operation_time(op_name, duration_ms);
    }

    /// Log error information
    ///
    /// # Arguments
    /// * `node_idx` - Index of node where error occurred
    /// * `error` - Error that occurred
    pub fn log_error(&mut self, node_idx: NodeIndex, error: &TorshError) {
        let message = format!("Error at node {:?}: {}", node_idx, error);
        self.log(message);
    }

    /// Generate debug report
    ///
    /// # Returns
    /// * `String` - Comprehensive debug report
    pub fn generate_debug_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Debug Execution Report ===\n\n");

        // Environment information
        report.push_str(&format!("Device: {:?}\n", self.env.device()));
        report.push_str(&format!("Debug Mode: {}\n", self.debug_mode));
        report.push_str(&format!("Stored Values: {}\n", self.env.value_count()));
        report.push_str(&format!("Log Entries: {}\n\n", self.execution_log.len()));

        // Performance metrics
        if !self.metrics.is_empty() {
            report.push_str("=== Performance Metrics ===\n");
            report.push_str(&self.metrics.generate_report());
            report.push_str("\n\n");
        }

        // Execution log
        if !self.execution_log.is_empty() {
            report.push_str("=== Execution Log ===\n");
            for (i, entry) in self.execution_log.iter().enumerate() {
                report.push_str(&format!("{:4}: {}\n", i + 1, entry));
            }
        }

        report
    }

    /// Print debug information to console
    pub fn print_debug_info(&self) {
        if self.debug_mode {
            println!("{}", self.generate_debug_report());
        }
    }

    /// Save debug report to string
    ///
    /// # Returns
    /// * `String` - Complete debug report
    pub fn save_debug_report(&self) -> String {
        self.generate_debug_report()
    }

    /// Reset debug state
    pub fn reset(&mut self) {
        self.env.clear();
        self.metrics.clear();
        self.execution_log.clear();
    }

    /// Get summary of execution state
    ///
    /// # Returns
    /// * `String` - Brief execution state summary
    pub fn execution_summary(&self) -> String {
        format!(
            "Debug Environment: {} values stored, {} operations, {:.2}ms total, {} log entries",
            self.env.value_count(),
            self.metrics.operation_count,
            self.metrics.total_time_ms,
            self.execution_log.len()
        )
    }
}

/// Utility functions for interpreter debugging and validation
pub mod utils {
    use super::*;
    use crate::interpreter::operations::is_operation_registered;

    /// Validate that a graph can be executed (all required operations are available)
    ///
    /// Checks that all operations referenced in the graph are either built-in
    /// or registered as custom operations.
    ///
    /// # Arguments
    /// * `graph` - FX graph to validate
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if graph is executable, error with missing operations
    pub fn validate_graph_executability(graph: &FxGraph) -> TorshResult<()> {
        let mut missing_ops = Vec::new();

        for (_, node) in graph.call_nodes() {
            if let Node::Call(op_name, _) = node {
                if !is_operation_registered(op_name) && !is_builtin_operation(op_name) {
                    missing_ops.push(op_name.clone());
                }
            }
        }

        if !missing_ops.is_empty() {
            return Err(TorshError::InvalidArgument(format!(
                "Missing operations: {}",
                missing_ops.join(", ")
            )));
        }

        Ok(())
    }

    /// Check if an operation is a built-in operation
    ///
    /// # Arguments
    /// * `op_name` - Name of operation to check
    ///
    /// # Returns
    /// * `bool` - True if operation is built-in
    pub fn is_builtin_operation(op_name: &str) -> bool {
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
                | "linear"
                | "linear_relu"
                | "conv2d_relu"
        )
    }

    /// Estimate execution complexity of a graph
    ///
    /// Provides a rough estimate of computational complexity based on
    /// operation types and counts.
    ///
    /// # Arguments
    /// * `graph` - FX graph to analyze
    ///
    /// # Returns
    /// * `usize` - Estimated complexity score
    pub fn estimate_execution_complexity(graph: &FxGraph) -> usize {
        let mut complexity = 0;

        for (_, node) in graph.call_nodes() {
            if let Node::Call(op_name, _) = node {
                complexity += match op_name.as_str() {
                    "add" | "sub" | "mul" | "div" => 1,
                    "matmul" | "linear" => 10,
                    "conv2d" => 20,
                    "relu" | "sigmoid" | "tanh" => 2,
                    "gelu" | "softmax" | "layer_norm" | "batch_norm" => 5,
                    "linear_relu" => 12, // linear + relu
                    "conv2d_relu" => 22, // conv2d + relu
                    _ => 3,              // Default complexity for unknown ops
                };
            }
        }

        complexity
    }

    /// Generate execution summary for a graph
    ///
    /// Creates a detailed summary of graph structure and expected execution
    /// characteristics.
    ///
    /// # Arguments
    /// * `graph` - FX graph to analyze
    ///
    /// # Returns
    /// * `String` - Formatted execution summary
    pub fn generate_execution_summary(graph: &FxGraph) -> String {
        let call_nodes = graph.call_nodes();
        let op_counts = graph.operation_counts();
        let complexity = estimate_execution_complexity(graph);

        let mut summary = format!(
            "Graph Execution Summary:\n\
             Total Operations: {}\n\
             Estimated Complexity: {}\n\
             Operation Types: {}\n\
             Input Nodes: {}\n\
             Output Nodes: {}\n\n\
             Operation Distribution:",
            call_nodes.len(),
            complexity,
            op_counts.len(),
            graph.inputs().len(),
            graph.outputs().len()
        );

        let mut sorted_ops: Vec<_> = op_counts.iter().collect();
        sorted_ops.sort_by(|a, b| b.1.cmp(a.1));

        for (op_name, count) in sorted_ops {
            let op_complexity = match op_name.as_str() {
                "add" | "sub" | "mul" | "div" => 1,
                "matmul" | "linear" => 10,
                "conv2d" => 20,
                "relu" | "sigmoid" | "tanh" => 2,
                "gelu" | "softmax" | "layer_norm" | "batch_norm" => 5,
                _ => 3,
            };
            summary.push_str(&format!(
                "\n  {}: {} instances (complexity: {} each)",
                op_name, count, op_complexity
            ));
        }

        // Add recommendations
        summary.push_str("\n\nRecommendations:");
        if complexity > 1000 {
            summary.push_str("\n  - High complexity graph: consider optimization");
        }
        if op_counts.len() > 50 {
            summary.push_str("\n  - Many operation types: verify all are available");
        }
        if call_nodes.len() > 500 {
            summary.push_str("\n  - Large graph: consider batching or partitioning");
        }

        summary
    }

    /// Validate graph structure integrity
    ///
    /// Performs comprehensive validation of graph structure, checking for
    /// common issues and inconsistencies.
    ///
    /// # Arguments
    /// * `graph` - FX graph to validate
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if graph structure is valid, error otherwise
    pub fn validate_graph_structure(graph: &FxGraph) -> TorshResult<()> {
        // Check for empty graph
        if graph.graph.node_count() == 0 {
            return Err(TorshError::InvalidArgument("Graph is empty".to_string()));
        }

        // Check for inputs and outputs
        if graph.inputs().is_empty() {
            return Err(TorshError::InvalidArgument(
                "Graph has no input nodes".to_string(),
            ));
        }

        if graph.outputs().is_empty() {
            return Err(TorshError::InvalidArgument(
                "Graph has no output nodes".to_string(),
            ));
        }

        // Check for cycles (this would be caught during execution, but good to check early)
        use petgraph::algo::is_cyclic_directed;
        if is_cyclic_directed(&graph.graph) {
            return Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            ));
        }

        // Check that all nodes are reachable from inputs
        // (This is a simplified check - a full implementation would use graph traversal)
        let node_count = graph.graph.node_count();
        let input_count = graph.inputs().len();
        let output_count = graph.outputs().len();

        if node_count < input_count + output_count {
            return Err(TorshError::InvalidArgument(
                "Invalid node count relationship".to_string(),
            ));
        }

        Ok(())
    }

    /// Create debug-friendly graph description
    ///
    /// # Arguments
    /// * `graph` - FX graph to describe
    ///
    /// # Returns
    /// * `String` - Human-readable graph description
    pub fn describe_graph(graph: &FxGraph) -> String {
        let mut description = String::new();

        description.push_str("=== FX Graph Description ===\n\n");

        // Basic statistics
        description.push_str(&format!("Nodes: {}\n", graph.graph.node_count()));
        description.push_str(&format!("Edges: {}\n", graph.graph.edge_count()));
        description.push_str(&format!("Inputs: {}\n", graph.inputs().len()));
        description.push_str(&format!("Outputs: {}\n", graph.outputs().len()));

        // Input nodes
        description.push_str("\nInput Nodes:\n");
        for &input_idx in graph.inputs() {
            if let Some(Node::Input(name)) = graph.get_node(input_idx) {
                description.push_str(&format!("  {:?}: '{}'\n", input_idx, name));
            }
        }

        // Output nodes
        description.push_str("\nOutput Nodes:\n");
        for &output_idx in graph.outputs() {
            if let Some(Node::Output) = graph.get_node(output_idx) {
                description.push_str(&format!("  {:?}\n", output_idx));
            }
        }

        // Operation summary
        let op_counts = graph.operation_counts();
        if !op_counts.is_empty() {
            description.push_str("\nOperations:\n");
            let mut sorted_ops: Vec<_> = op_counts.iter().collect();
            sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
            for (op_name, count) in sorted_ops {
                description.push_str(&format!("  {}: {} instances\n", op_name, count));
            }
        }

        description
    }
}
