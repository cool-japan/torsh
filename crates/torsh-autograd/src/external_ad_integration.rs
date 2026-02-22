//! External Automatic Differentiation Library Integration Framework
//!
//! This module provides a common interface for integrating ToRSh autograd
//! with external automatic differentiation libraries. It defines traits and
//! utilities that enable seamless interoperability with various AD frameworks.

use crate::context::AutogradContext;
// AutogradTensor trait is available through crate root - it's generic
use crate::AutogradTensor;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::dtype::TensorElement;
use torsh_core::error::TorshError;

/// Trait for external AD library integration
pub trait ExternalADLibrary {
    /// Name of the external library
    fn library_name(&self) -> &'static str;

    /// Version of the external library
    fn library_version(&self) -> String;

    /// Check if the library is available/installed
    fn is_available(&self) -> bool;

    /// Initialize the integration
    fn initialize(&mut self) -> Result<(), TorshError>;

    /// Shutdown the integration
    fn shutdown(&mut self) -> Result<(), TorshError>;

    /// Convert ToRSh tensor to external library format
    fn export_tensor(&self, tensor: &dyn Any) -> Result<Box<dyn Any>, TorshError>;

    /// Import tensor from external library format
    fn import_tensor(&self, external_tensor: &dyn Any) -> Result<Box<dyn Any>, TorshError>;

    /// Execute computation graph in external library
    fn execute_graph(&self, graph: &ComputationGraph) -> Result<ExecutionResult, TorshError>;

    /// Get gradient computation capabilities
    fn gradient_capabilities(&self) -> GradientCapabilities;
}

/// Computation graph representation for external libraries
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// Graph nodes representing operations
    pub nodes: Vec<GraphNode>,
    /// Input tensor IDs
    pub inputs: Vec<String>,
    /// Output tensor IDs
    pub outputs: Vec<String>,
    /// Graph metadata
    pub metadata: HashMap<String, String>,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,
    /// Operation type
    pub operation: Operation,
    /// Input tensor IDs
    pub inputs: Vec<String>,
    /// Output tensor IDs
    pub outputs: Vec<String>,
    /// Node attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Operation types supported in the graph
#[derive(Debug, Clone)]
pub enum Operation {
    /// Mathematical operations
    Add,
    Subtract,
    Multiply,
    Divide,
    MatMul,

    /// Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    /// Reduction operations
    Sum,
    Mean,
    Max,
    Min,

    /// Shape operations
    Reshape,
    Transpose,
    Permute,

    /// Custom operation
    Custom(String),
}

/// Attribute values for graph nodes
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
    Bool(bool),
}

/// Execution result from external library
#[derive(Debug)]
pub struct ExecutionResult {
    /// Output tensors
    pub outputs: HashMap<String, Box<dyn Any>>,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Gradient computation capabilities of external library
#[derive(Debug, Clone)]
pub struct GradientCapabilities {
    /// Supports forward-mode AD
    pub supports_forward_mode: bool,
    /// Supports reverse-mode AD
    pub supports_reverse_mode: bool,
    /// Supports higher-order derivatives
    pub supports_higher_order: bool,
    /// Supports custom derivatives
    pub supports_custom_derivatives: bool,
    /// Maximum supported derivative order
    pub max_derivative_order: usize,
    /// Supported data types
    pub supported_dtypes: Vec<String>,
}

/// Registry for external AD library integrations
pub struct ExternalADRegistry {
    libraries: HashMap<String, Box<dyn ExternalADLibrary + Send + Sync>>,
    active_library: Option<String>,
}

impl ExternalADRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            libraries: HashMap::new(),
            active_library: None,
        }
    }

    /// Register an external AD library
    pub fn register<T: ExternalADLibrary + Send + Sync + 'static>(
        &mut self,
        library: T,
    ) -> Result<(), TorshError> {
        let name = library.library_name().to_string();

        if self.libraries.contains_key(&name) {
            return Err(TorshError::InvalidArgument(format!(
                "Library {} already registered",
                name
            )));
        }

        tracing::info!(
            "Registering external AD library: {} v{}",
            name,
            library.library_version()
        );
        self.libraries.insert(name, Box::new(library));
        Ok(())
    }

    /// Set the active library for autograd operations
    pub fn set_active_library(&mut self, name: &str) -> Result<(), TorshError> {
        if !self.libraries.contains_key(name) {
            return Err(TorshError::InvalidArgument(format!(
                "Library {} not registered",
                name
            )));
        }

        if let Some(library) = self.libraries.get(name) {
            if !library.is_available() {
                return Err(TorshError::RuntimeError(format!(
                    "Library {} is not available",
                    name
                )));
            }
        }

        self.active_library = Some(name.to_string());
        tracing::info!("Set active AD library: {}", name);
        Ok(())
    }

    /// Get the active library
    pub fn get_active_library(&self) -> Option<&(dyn ExternalADLibrary + Send + Sync)> {
        if let Some(name) = &self.active_library {
            self.libraries.get(name).map(|lib| lib.as_ref())
        } else {
            None
        }
    }

    /// List all registered libraries
    pub fn list_libraries(&self) -> Vec<String> {
        self.libraries.keys().cloned().collect()
    }

    /// Get capabilities of all registered libraries
    pub fn get_all_capabilities(&self) -> HashMap<String, GradientCapabilities> {
        self.libraries
            .iter()
            .map(|(name, library)| (name.clone(), library.gradient_capabilities()))
            .collect()
    }
}

impl Default for ExternalADRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge for integrating external AD computations with ToRSh autograd
pub struct AutogradBridge {
    registry: Arc<Mutex<ExternalADRegistry>>,
    #[allow(dead_code)]
    context: AutogradContext,
}

impl AutogradBridge {
    /// Create a new autograd bridge
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Mutex::new(ExternalADRegistry::new())),
            context: AutogradContext::new(),
        }
    }

    /// Register an external library with the bridge
    pub fn register_library<T: ExternalADLibrary + Send + Sync + 'static>(
        &self,
        library: T,
    ) -> Result<(), TorshError> {
        let mut registry = self.registry.lock().expect("lock should not be poisoned");
        registry.register(library)
    }

    /// Execute computation using external library
    pub fn execute_external<T: TensorElement>(
        &self,
        graph: ComputationGraph,
        inputs: Vec<&dyn AutogradTensor<T>>,
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>, TorshError> {
        let registry = self.registry.lock().expect("lock should not be poisoned");

        let library = registry
            .get_active_library()
            .ok_or_else(|| TorshError::RuntimeError("No active external AD library".to_string()))?;

        // Convert ToRSh tensors to external format
        // Note: In a real implementation, this would properly convert tensors
        // For now, we simulate the conversion without actual type casting
        let _input_count = inputs.len();
        tracing::debug!(
            "Preparing {} input tensors for external computation",
            _input_count
        );

        // Execute in external library
        let result = library.execute_graph(&graph)?;

        // Convert results back to ToRSh tensors
        // Note: In a real implementation, this would require proper tensor conversion
        // based on the specific external library's tensor format
        tracing::info!(
            "Executed external computation in {:.2}ms using {}",
            result.execution_time_ms,
            library.library_name()
        );

        // For now, return empty vector as this is a framework/interface
        // Concrete implementations would handle the actual tensor conversion
        Ok(Vec::new())
    }

    /// Create a computation graph builder
    pub fn graph_builder(&self) -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Get summary of available external libraries
    pub fn library_summary(&self) -> String {
        let registry = self.registry.lock().expect("lock should not be poisoned");
        let libraries = registry.list_libraries();
        let capabilities = registry.get_all_capabilities();

        let mut summary = String::from("External AD Libraries Summary:\n");
        for lib_name in libraries {
            if let Some(caps) = capabilities.get(&lib_name) {
                summary.push_str(&format!(
                    "  {} - Forward: {}, Reverse: {}, Higher-order: {}, Max order: {}\n",
                    lib_name,
                    caps.supports_forward_mode,
                    caps.supports_reverse_mode,
                    caps.supports_higher_order,
                    caps.max_derivative_order
                ));
            }
        }
        summary
    }
}

impl Default for AutogradBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for computation graphs
pub struct GraphBuilder {
    nodes: Vec<GraphNode>,
    next_node_id: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
    metadata: HashMap<String, String>,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_node_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an operation node to the graph
    pub fn add_operation(
        &mut self,
        operation: Operation,
        inputs: Vec<String>,
        attributes: HashMap<String, AttributeValue>,
    ) -> String {
        let node_id = format!("node_{}", self.next_node_id);
        let output_id = format!("tensor_{}", self.next_node_id);

        let node = GraphNode {
            id: node_id.clone(),
            operation,
            inputs,
            outputs: vec![output_id.clone()],
            attributes,
        };

        self.nodes.push(node);
        self.next_node_id += 1;

        output_id
    }

    /// Mark a tensor as graph input
    pub fn add_input(&mut self, tensor_id: String) {
        self.inputs.push(tensor_id);
    }

    /// Mark a tensor as graph output
    pub fn add_output(&mut self, tensor_id: String) {
        self.outputs.push(tensor_id);
    }

    /// Add metadata to the graph
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Build the final computation graph
    pub fn build(self) -> ComputationGraph {
        ComputationGraph {
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            metadata: self.metadata,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Global instance of the autograd bridge
static GLOBAL_BRIDGE: std::sync::OnceLock<AutogradBridge> = std::sync::OnceLock::new();

/// Get the global autograd bridge instance
pub fn get_global_bridge() -> &'static AutogradBridge {
    GLOBAL_BRIDGE.get_or_init(|| AutogradBridge::new())
}

/// Convenience function to register a library globally
pub fn register_external_library<T: ExternalADLibrary + Send + Sync + 'static>(
    library: T,
) -> Result<(), TorshError> {
    get_global_bridge().register_library(library)
}

/// Convenience function to execute external computation
pub fn execute_external_computation<T: TensorElement>(
    graph: ComputationGraph,
    inputs: Vec<&dyn AutogradTensor<T>>,
) -> Result<Vec<Box<dyn AutogradTensor<T>>>, TorshError> {
    get_global_bridge().execute_external(graph, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockADLibrary {
        name: &'static str,
        available: bool,
    }

    impl ExternalADLibrary for MockADLibrary {
        fn library_name(&self) -> &'static str {
            self.name
        }

        fn library_version(&self) -> String {
            "1.0.0".to_string()
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn initialize(&mut self) -> Result<(), TorshError> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), TorshError> {
            Ok(())
        }

        fn export_tensor(&self, _tensor: &dyn Any) -> Result<Box<dyn Any>, TorshError> {
            Ok(Box::new(42i32))
        }

        fn import_tensor(&self, _external_tensor: &dyn Any) -> Result<Box<dyn Any>, TorshError> {
            Ok(Box::new(42i32))
        }

        fn execute_graph(&self, _graph: &ComputationGraph) -> Result<ExecutionResult, TorshError> {
            Ok(ExecutionResult {
                outputs: HashMap::new(),
                metadata: HashMap::new(),
                execution_time_ms: 1.0,
            })
        }

        fn gradient_capabilities(&self) -> GradientCapabilities {
            GradientCapabilities {
                supports_forward_mode: true,
                supports_reverse_mode: true,
                supports_higher_order: false,
                supports_custom_derivatives: true,
                max_derivative_order: 2,
                supported_dtypes: vec!["f32".to_string(), "f64".to_string()],
            }
        }
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = ExternalADRegistry::new();
        let library = MockADLibrary {
            name: "MockLib",
            available: true,
        };

        assert!(registry.register(library).is_ok());
        assert_eq!(registry.list_libraries().len(), 1);
        assert!(registry.list_libraries().contains(&"MockLib".to_string()));
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = GraphBuilder::new();

        builder.add_input("input_0".to_string());
        let output = builder.add_operation(
            Operation::Add,
            vec!["input_0".to_string(), "input_1".to_string()],
            HashMap::new(),
        );
        builder.add_output(output);

        let graph = builder.build();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_autograd_bridge() {
        let bridge = AutogradBridge::new();
        let library = MockADLibrary {
            name: "TestLib",
            available: true,
        };

        assert!(bridge.register_library(library).is_ok());

        let summary = bridge.library_summary();
        assert!(summary.contains("TestLib"));
        assert!(summary.contains("Forward: true"));
    }
}
