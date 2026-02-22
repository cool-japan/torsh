//! Custom Operation Registry and Management System
//!
//! This module provides a comprehensive system for registering, managing, and executing
//! custom operations in the FX graph interpreter. It includes thread-safe operation
//! registries, operation validation, and global operation management.

use crate::TorshResult;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::{dtype::DType, error::TorshError, shape::Shape};
use torsh_tensor::Tensor;

/// Trait for custom operations
///
/// This trait defines the interface for all custom operations that can be executed
/// within the FX graph interpreter. Operations must be thread-safe and provide
/// methods for execution, validation, and type/shape inference.
pub trait CustomOperation: Send + Sync {
    /// Execute the custom operation
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors for the operation
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Result tensor or error
    fn execute(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor>;

    /// Get the operation name
    ///
    /// # Returns
    /// * `&str` - Unique identifier for this operation
    fn name(&self) -> &str;

    /// Clone this operation
    ///
    /// # Returns
    /// * `Box<dyn CustomOperation>` - Boxed clone of this operation
    fn clone_operation(&self) -> Box<dyn CustomOperation>;

    /// Get operation metadata (optional)
    ///
    /// # Returns
    /// * `Option<HashMap<String, String>>` - Optional metadata dictionary
    fn metadata(&self) -> Option<HashMap<String, String>> {
        None
    }

    /// Validate inputs (optional)
    ///
    /// # Arguments
    /// * `inputs` - Slice of input tensors to validate
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if inputs are valid, error otherwise
    fn validate_inputs(&self, _inputs: &[Tensor]) -> TorshResult<()> {
        Ok(())
    }

    /// Infer output shape from input shapes (optional)
    ///
    /// # Arguments
    /// * `input_shapes` - Slice of input shapes
    ///
    /// # Returns
    /// * `TorshResult<Shape>` - Inferred output shape or error
    fn infer_shape(&self, input_shapes: &[Shape]) -> TorshResult<Shape> {
        // Default implementation: use first input shape
        input_shapes
            .first()
            .cloned()
            .ok_or_else(|| TorshError::InvalidArgument("No input shapes provided".to_string()))
    }

    /// Validate types for the operation (optional)
    ///
    /// # Arguments
    /// * `input_types` - Slice of input data types
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if types are valid, error otherwise
    fn validate_types(&self, _input_types: &[DType]) -> TorshResult<()> {
        // Default implementation: accept any types
        Ok(())
    }

    /// Infer output type from input types (optional)
    ///
    /// # Arguments
    /// * `input_types` - Slice of input data types
    ///
    /// # Returns
    /// * `TorshResult<DType>` - Inferred output type or error
    fn infer_type(&self, input_types: &[DType]) -> TorshResult<DType> {
        // Default implementation: use first input type
        input_types
            .first()
            .copied()
            .ok_or_else(|| TorshError::InvalidArgument("No input types provided".to_string()))
    }
}

/// Registry for custom operations
///
/// Thread-safe registry that manages custom operations. Operations can be registered,
/// retrieved, and executed through this registry. Each registry maintains its own
/// independent set of operations.
#[derive(Default)]
pub struct OperationRegistry {
    operations: Arc<RwLock<HashMap<String, Box<dyn CustomOperation>>>>,
}

impl OperationRegistry {
    /// Create a new operation registry
    ///
    /// # Returns
    /// * `Self` - New empty operation registry
    pub fn new() -> Self {
        Self {
            operations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a custom operation
    ///
    /// # Arguments
    /// * `operation` - Operation to register
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if registered successfully, error if name already exists
    pub fn register<T: CustomOperation + 'static>(&self, operation: T) -> TorshResult<()> {
        let name = operation.name().to_string();
        let mut ops = self.operations.write().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire write lock on operations".to_string())
        })?;

        if ops.contains_key(&name) {
            return Err(TorshError::InvalidArgument(format!(
                "Operation '{}' already registered",
                name
            )));
        }

        ops.insert(name, Box::new(operation));
        Ok(())
    }

    /// Get a registered operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation to retrieve
    ///
    /// # Returns
    /// * `TorshResult<Box<dyn CustomOperation>>` - Cloned operation or error if not found
    pub fn get(&self, name: &str) -> TorshResult<Box<dyn CustomOperation>> {
        let ops = self.operations.read().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire read lock on operations".to_string())
        })?;

        if let Some(operation) = ops.get(name) {
            Ok(operation.clone_operation())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Operation '{}' not found in registry",
                name
            )))
        }
    }

    /// Check if an operation is registered
    ///
    /// # Arguments
    /// * `name` - Name of the operation to check
    ///
    /// # Returns
    /// * `bool` - True if operation is registered, false otherwise
    pub fn is_registered(&self, name: &str) -> bool {
        if let Ok(ops) = self.operations.read() {
            ops.contains_key(name)
        } else {
            false
        }
    }

    /// List all registered operations
    ///
    /// # Returns
    /// * `Vec<String>` - Vector of all registered operation names
    pub fn list_operations(&self) -> Vec<String> {
        if let Ok(ops) = self.operations.read() {
            ops.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Execute a registered operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation to execute
    /// * `inputs` - Input tensors for the operation
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Result tensor or error
    pub fn execute(&self, name: &str, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        let ops = self.operations.read().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire read lock on operations".to_string())
        })?;

        if let Some(operation) = ops.get(name) {
            operation.validate_inputs(&inputs)?;
            operation.execute(inputs)
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Operation '{}' not found in registry",
                name
            )))
        }
    }

    /// Get operation metadata
    ///
    /// # Arguments
    /// * `name` - Name of the operation
    ///
    /// # Returns
    /// * `TorshResult<Option<HashMap<String, String>>>` - Operation metadata or error
    pub fn get_operation_metadata(
        &self,
        name: &str,
    ) -> TorshResult<Option<HashMap<String, String>>> {
        let ops = self.operations.read().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire read lock on operations".to_string())
        })?;

        if let Some(operation) = ops.get(name) {
            Ok(operation.metadata())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Operation '{}' not found in registry",
                name
            )))
        }
    }

    /// Clear all registered operations
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if cleared successfully, error on lock failure
    pub fn clear(&self) -> TorshResult<()> {
        let mut ops = self.operations.write().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire write lock on operations".to_string())
        })?;
        ops.clear();
        Ok(())
    }

    /// Get the number of registered operations
    ///
    /// # Returns
    /// * `usize` - Number of registered operations
    pub fn operation_count(&self) -> usize {
        if let Ok(ops) = self.operations.read() {
            ops.len()
        } else {
            0
        }
    }

    /// Validate an operation without executing it
    ///
    /// # Arguments
    /// * `name` - Name of the operation to validate
    /// * `inputs` - Input tensors to validate against
    ///
    /// # Returns
    /// * `TorshResult<()>` - Ok if validation passes, error otherwise
    pub fn validate_operation(&self, name: &str, inputs: &[Tensor]) -> TorshResult<()> {
        let ops = self.operations.read().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire read lock on operations".to_string())
        })?;

        if let Some(operation) = ops.get(name) {
            operation.validate_inputs(inputs)
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Operation '{}' not found in registry",
                name
            )))
        }
    }
}

/// Global operation registry
static GLOBAL_REGISTRY: std::sync::OnceLock<OperationRegistry> = std::sync::OnceLock::new();

/// Get the global operation registry
///
/// # Returns
/// * `&'static OperationRegistry` - Reference to the global registry
pub fn global_registry() -> &'static OperationRegistry {
    GLOBAL_REGISTRY.get_or_init(|| OperationRegistry::new())
}

/// Register a custom operation globally
///
/// # Arguments
/// * `operation` - Operation to register in the global registry
///
/// # Returns
/// * `TorshResult<()>` - Ok if registered successfully, error otherwise
pub fn register_operation<T: CustomOperation + 'static>(operation: T) -> TorshResult<()> {
    global_registry().register(operation)
}

/// Check if an operation is registered globally
///
/// # Arguments
/// * `name` - Name of the operation to check
///
/// # Returns
/// * `bool` - True if operation is registered globally, false otherwise
pub fn is_operation_registered(name: &str) -> bool {
    global_registry().is_registered(name)
}

/// Execute a globally registered operation
///
/// # Arguments
/// * `name` - Name of the operation to execute
/// * `inputs` - Input tensors for the operation
///
/// # Returns
/// * `TorshResult<Tensor>` - Result tensor or error
pub fn execute_registered_operation(name: &str, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
    global_registry().execute(name, inputs)
}
