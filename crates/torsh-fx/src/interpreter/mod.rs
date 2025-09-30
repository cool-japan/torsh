//! Graph interpreter
//!
//! This module provides a comprehensive, high-performance graph interpretation system for
//! FX graphs. The modular architecture allows for sophisticated operation execution,
//! shape inference, type checking, performance monitoring, and debugging capabilities
//! while maintaining backward compatibility.
//!
//! The interpreter system has been refactored into specialized modules for better
//! maintainability and extensibility:
//!
//! - **operations**: Custom operation registry and management
//! - **shape_inference**: Graph shape analysis and inference
//! - **type_checking**: Graph type validation and checking
//! - **execution**: Core graph interpretation engine
//! - **metrics**: Performance monitoring and metrics collection
//! - **debug**: Debugging capabilities and development tools
//!
//! ## Migration Guide
//!
//! All existing functionality is preserved through comprehensive re-exports.
//! No changes are required to existing code. The modular structure is available
//! for advanced users who want to use individual components.
//!
//! ### Backward Compatibility
//!
//! All previous imports and usage patterns continue to work:
//!
//! ```rust
//! use torsh_fx::interpreter::{CustomOperation, GraphInterpreter, ExecutionMetrics};
//! ```
//!
//! ### Advanced Usage
//!
//! Advanced users can now access specialized modules directly:
//!
//! ```rust
//! use torsh_fx::interpreter::operations::OperationRegistry;
//! use torsh_fx::interpreter::shape_inference::ShapeInferenceContext;
//! use torsh_fx::interpreter::debug::utils::validate_graph_executability;
//! ```
//!
//! # Architecture
//!
//! The interpreter system follows a modular design where each module handles a specific
//! aspect of graph interpretation. This allows for:
//!
//! - **Separation of concerns**: Each module has a clear, focused responsibility
//! - **Maintainability**: Code is organized into logical, manageable units
//! - **Extensibility**: New features can be added to specific modules without affecting others
//! - **Testability**: Each module can be tested independently
//! - **Reusability**: Modules can be used independently or in combination
//!
//! # Usage
//!
//! ## Basic Graph Execution
//!
//! ```rust
//! use torsh_fx::interpreter::{GraphInterpreter, interpret_with_inputs};
//! use torsh_core::device::DeviceType;
//! use torsh_tensor::Tensor;
//! use std::collections::HashMap;
//!
//! // Create an interpreter
//! let mut interpreter = GraphInterpreter::new(DeviceType::Cpu);
//!
//! // Execute a graph with inputs
//! let inputs: HashMap<String, Tensor> = HashMap::new(); // Add your input tensors here
//! // let outputs = interpreter.run(&graph, inputs)?;
//!
//! // Or use the convenience function
//! // let outputs = interpret_with_inputs(&graph, inputs)?;
//! ```
//!
//! ## Custom Operations
//!
//! ```rust
//! use torsh_fx::interpreter::{CustomOperation, register_operation};
//!
//! // Define your custom operation
//! struct MyCustomOp;
//!
//! impl CustomOperation for MyCustomOp {
//!     fn execute(&self, inputs: Vec<torsh_tensor::Tensor>) -> torsh_fx::TorshResult<torsh_tensor::Tensor> {
//!         // Your operation implementation
//!         Ok(inputs[0].clone())
//!     }
//!
//!     fn name(&self) -> &str {
//!         "my_custom_op"
//!     }
//!
//!     fn clone_operation(&self) -> Box<dyn CustomOperation> {
//!         Box::new(MyCustomOp)
//!     }
//! }
//!
//! // Register the operation
//! // register_operation(MyCustomOp)?;
//! ```
//!
//! ## Shape Inference
//!
//! ```rust
//! use torsh_fx::interpreter::{ShapeInferenceContext, ShapeInfo};
//! use std::collections::HashMap;
//!
//! let mut context = ShapeInferenceContext::new();
//! let input_shapes: HashMap<String, ShapeInfo> = HashMap::new(); // Add your input shapes here
//! // context.infer_shapes(&graph, input_shapes)?;
//! ```
//!
//! ## Type Checking
//!
//! ```rust
//! use torsh_fx::interpreter::{TypeCheckingContext};
//! use torsh_core::dtype::DType;
//! use std::collections::HashMap;
//!
//! let mut context = TypeCheckingContext::new();
//! let input_types: HashMap<String, DType> = HashMap::new(); // Add your input types here
//! // context.check_types(&graph, input_types)?;
//! ```
//!
//! ## Performance Monitoring
//!
//! ```rust
//! use torsh_fx::interpreter::{ExecutionMetrics, MetricsCollector};
//!
//! let mut metrics = ExecutionMetrics::new();
//! metrics.add_operation_time("matmul", 15.5);
//! let report = metrics.generate_report();
//! println!("{}", report);
//! ```
//!
//! ## Debugging
//!
//! ```rust
//! use torsh_fx::interpreter::{DebugExecutionEnvironment, debug};
//! use torsh_core::device::DeviceType;
//!
//! let mut debug_env = DebugExecutionEnvironment::new(DeviceType::Cpu, true);
//! debug_env.log("Starting graph execution".to_string());
//!
//! // Validate graph before execution
//! // debug::utils::validate_graph_executability(&graph)?;
//! // let summary = debug::utils::generate_execution_summary(&graph);
//! ```

// Import all specialized modules
pub mod debug;
pub mod execution;
pub mod metrics;
pub mod operations;
pub mod shape_inference;
pub mod type_checking;

// Re-export all public functionality for backward compatibility and ease of use

// === Operations Module Re-exports ===
pub use operations::{
    execute_registered_operation, global_registry, is_operation_registered, register_operation,
    CustomOperation, OperationRegistry,
};

// === Shape Inference Module Re-exports ===
pub use shape_inference::{ShapeInferenceContext, ShapeInfo};

// === Type Checking Module Re-exports ===
pub use type_checking::TypeCheckingContext;

// === Execution Module Re-exports ===
pub use execution::{interpret, interpret_with_inputs, ExecutionEnvironment, GraphInterpreter};

// === Metrics Module Re-exports ===
pub use metrics::{ExecutionMetrics, ExecutionTimer, MetricsCollector};

// === Debug Module Re-exports ===
pub use debug::{utils, DebugExecutionEnvironment};

// === Convenience Functions ===

use crate::{FxGraph, TorshResult};
use std::collections::HashMap;
use torsh_core::dtype::DType;
use torsh_tensor::Tensor;

/// Perform shape inference on a graph
///
/// Convenience function that creates a shape inference context and performs
/// shape inference for the entire graph.
///
/// # Arguments
/// * `graph` - FX graph to perform shape inference on
/// * `input_shapes` - Map of input node names to their shape information
///
/// # Returns
/// * `TorshResult<HashMap<petgraph::graph::NodeIndex, ShapeInfo>>` - Map of node indices to inferred shapes
pub fn infer_graph_shapes(
    graph: &FxGraph,
    input_shapes: HashMap<String, ShapeInfo>,
) -> TorshResult<HashMap<petgraph::graph::NodeIndex, ShapeInfo>> {
    let mut context = ShapeInferenceContext::new();
    context.infer_shapes(graph, input_shapes)?;
    Ok(context.get_all_shapes().clone())
}

/// Perform type checking on a graph
///
/// Convenience function that creates a type checking context and performs
/// type checking for the entire graph.
///
/// # Arguments
/// * `graph` - FX graph to perform type checking on
/// * `input_types` - Map of input node names to their data types
///
/// # Returns
/// * `TorshResult<HashMap<petgraph::graph::NodeIndex, DType>>` - Map of node indices to inferred types
pub fn check_graph_types(
    graph: &FxGraph,
    input_types: HashMap<String, DType>,
) -> TorshResult<HashMap<petgraph::graph::NodeIndex, DType>> {
    let mut context = TypeCheckingContext::new();
    context.check_types(graph, input_types)?;
    Ok(context.get_all_types().clone())
}

/// Comprehensive graph validation
///
/// Performs complete validation of a graph including structure, executability,
/// and provides detailed analysis.
///
/// # Arguments
/// * `graph` - FX graph to validate
///
/// # Returns
/// * `TorshResult<String>` - Validation report or error if validation fails
pub fn validate_graph(graph: &FxGraph) -> TorshResult<String> {
    // Validate graph structure
    debug::utils::validate_graph_structure(graph)?;

    // Validate executability
    debug::utils::validate_graph_executability(graph)?;

    // Generate comprehensive summary
    let summary = debug::utils::generate_execution_summary(graph);
    let description = debug::utils::describe_graph(graph);

    Ok(format!(
        "Graph Validation: PASSED\n\n{}\n\n{}",
        description, summary
    ))
}

/// Get interpreter system information
///
/// Returns information about the interpreter system including available
/// operations, module status, and system capabilities.
///
/// # Returns
/// * `String` - System information report
pub fn system_info() -> String {
    let registry = global_registry();
    let operation_count = registry.operation_count();
    let registered_operations = registry.list_operations();

    let builtin_ops = [
        "add",
        "sub",
        "mul",
        "div",
        "matmul",
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "softmax",
        "layer_norm",
        "batch_norm",
        "conv2d",
        "linear",
        "linear_relu",
        "conv2d_relu",
    ];

    format!(
        "ToRSh FX Graph Interpreter System\n\
         ===================================\n\
         \n\
         Modules:\n\
         - Operations: Custom operation registry and management\n\
         - Shape Inference: Graph shape analysis and inference\n\
         - Type Checking: Graph type validation and checking\n\
         - Execution: Core graph interpretation engine\n\
         - Metrics: Performance monitoring and metrics collection\n\
         - Debug: Debugging capabilities and development tools\n\
         \n\
         Built-in Operations: {} available\n\
         {}\n\
         \n\
         Registered Custom Operations: {}\n\
         {}\n\
         \n\
         Capabilities:\n\
         - Graph execution with topological ordering\n\
         - Custom operation registration and execution\n\
         - Comprehensive shape inference\n\
         - Type checking and validation\n\
         - Performance monitoring and profiling\n\
         - Debug execution environments\n\
         - Graph structure validation\n\
         - Backward compatibility maintained",
        builtin_ops.len(),
        builtin_ops.join(", "),
        operation_count,
        if registered_operations.is_empty() {
            "None".to_string()
        } else {
            registered_operations.join(", ")
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_system_info() {
        let info = system_info();
        assert!(info.contains("ToRSh FX Graph Interpreter System"));
        assert!(info.contains("Built-in Operations"));
    }

    #[test]
    fn test_execution_metrics() {
        let mut metrics = ExecutionMetrics::new();
        metrics.add_operation_time("add", 1.5);
        metrics.add_operation_time("mul", 2.3);

        assert_eq!(metrics.operation_count, 2);
        assert_eq!(metrics.get_operation_time("add"), 1.5);
        assert_eq!(metrics.get_operation_time("mul"), 2.3);

        let report = metrics.generate_report();
        assert!(report.contains("Performance Report"));
    }

    #[test]
    fn test_debug_environment() {
        let mut debug_env = DebugExecutionEnvironment::new(DeviceType::Cpu, true);
        debug_env.log("Test message".to_string());

        assert_eq!(debug_env.get_log().len(), 1);
        assert_eq!(debug_env.get_log()[0], "Test message");

        let summary = debug_env.execution_summary();
        assert!(summary.contains("Debug Environment"));
    }

    #[test]
    fn test_shape_info() {
        use torsh_core::dtype::DType;
        use torsh_core::shape::Shape;

        let shape = Shape::new(vec![2, 3, 4]);
        let dtype = DType::F32;
        let shape_info = ShapeInfo::new(shape.clone(), dtype);

        assert_eq!(shape_info.shape.dims(), shape.dims());
        assert_eq!(shape_info.dtype, dtype);
    }

    #[test]
    fn test_execution_environment() {
        use petgraph::graph::NodeIndex;

        let mut env = ExecutionEnvironment::new(DeviceType::Cpu);
        let tensor = zeros(&[2, 2]).unwrap();
        let node_idx = NodeIndex::new(0);

        env.store(node_idx, tensor.clone());
        assert!(env.has_value(node_idx));
        assert_eq!(env.value_count(), 1);

        let retrieved = env.get(node_idx).unwrap();
        assert_eq!(retrieved.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        let mut metrics1 = ExecutionMetrics::new();
        metrics1.add_operation_time("add", 10.0);
        metrics1.set_total_time(10.0);

        let mut metrics2 = ExecutionMetrics::new();
        metrics2.add_operation_time("mul", 20.0);
        metrics2.set_total_time(20.0);

        collector.add_run(metrics1);
        collector.add_run(metrics2);

        assert_eq!(collector.run_count(), 2);
        assert_eq!(collector.average_execution_time(), 15.0);
        assert_eq!(collector.fastest_execution(), Some(10.0));
        assert_eq!(collector.slowest_execution(), Some(20.0));
    }
}
