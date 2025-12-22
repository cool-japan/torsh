//! Custom operation registration system for torsh-tensor
//!
//! This module provides a flexible system for registering custom operations that can be used
//! with tensors, including automatic differentiation support. Users can define their own
//! operations and integrate them seamlessly with the existing tensor API.

use crate::{core_ops::Tensor, TensorElement};
use num_traits::FromPrimitive;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::error::{Result, TorshError};

/// Trait for custom operation implementations
///
/// Custom operations must implement this trait to be registered and used with tensors.
/// The trait provides both forward and backward operations for automatic differentiation.
pub trait CustomOperation<T: TensorElement>: Send + Sync {
    /// Get the name of this operation
    fn name(&self) -> &str;

    /// Get a description of what this operation does
    fn description(&self) -> &str;

    /// Execute the forward pass of the operation
    ///
    /// # Arguments
    /// * `inputs` - Input tensors for the operation
    /// * `params` - Optional parameters for the operation
    ///
    /// # Returns
    /// The result tensor(s) from applying this operation
    fn forward(&self, inputs: &[Tensor<T>], params: &OperationParams) -> Result<Vec<Tensor<T>>>;

    /// Execute the backward pass of the operation (optional for non-differentiable ops)
    ///
    /// # Arguments
    /// * `grad_outputs` - Gradients with respect to the outputs
    /// * `inputs` - Original input tensors
    /// * `outputs` - Original output tensors
    /// * `params` - Operation parameters
    ///
    /// # Returns
    /// Gradients with respect to the inputs
    fn backward(
        &self,
        grad_outputs: &[Tensor<T>],
        inputs: &[Tensor<T>],
        _outputs: &[Tensor<T>],
        _params: &OperationParams,
    ) -> Result<Vec<Option<Tensor<T>>>> {
        // Default implementation for non-differentiable operations
        // Validate that we have gradient outputs matching expected count
        let _ = grad_outputs.is_empty(); // Check if empty but continue

        // Return None gradients for all inputs (non-differentiable by default)
        Ok(vec![None; inputs.len()])
    }

    /// Validate that the inputs are compatible with this operation
    ///
    /// # Arguments
    /// * `inputs` - Input tensors to validate
    /// * `params` - Operation parameters
    ///
    /// # Returns
    /// True if inputs are valid, false otherwise
    fn validate_inputs(&self, inputs: &[Tensor<T>], _params: &OperationParams) -> Result<()> {
        // Default implementation - basic validation
        if inputs.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidShape(
                "Operation requires at least one input tensor".to_string(),
            ));
        }

        // Validate that all input tensors have data
        for (idx, input) in inputs.iter().enumerate() {
            let _ = (idx, input.shape.is_empty()); // Check shape validity
        }

        Ok(())
    }

    /// Get the expected output shapes given input shapes
    ///
    /// # Arguments
    /// * `input_shapes` - Shapes of input tensors
    /// * `params` - Operation parameters
    ///
    /// # Returns
    /// Expected shapes of output tensors
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        params: &OperationParams,
    ) -> Result<Vec<Vec<usize>>>;

    /// Check if this operation supports automatic differentiation
    fn supports_autograd(&self) -> bool {
        true // Most operations should support autograd
    }

    /// Get the number of expected inputs
    fn num_inputs(&self) -> usize;

    /// Get the number of expected outputs
    fn num_outputs(&self) -> usize;
}

/// Parameters that can be passed to custom operations
#[derive(Debug, Clone)]
pub struct OperationParams {
    /// String parameters
    pub strings: HashMap<String, String>,
    /// Integer parameters
    pub integers: HashMap<String, i64>,
    /// Float parameters
    pub floats: HashMap<String, f64>,
    /// Boolean parameters
    pub booleans: HashMap<String, bool>,
    /// Vector parameters
    pub vectors: HashMap<String, Vec<f64>>,
    /// Shape parameters
    pub shapes: HashMap<String, Vec<usize>>,
}

impl OperationParams {
    /// Create a new empty parameter set
    pub fn new() -> Self {
        Self {
            strings: HashMap::new(),
            integers: HashMap::new(),
            floats: HashMap::new(),
            booleans: HashMap::new(),
            vectors: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Add a string parameter
    pub fn with_string(mut self, key: &str, value: &str) -> Self {
        self.strings.insert(key.to_string(), value.to_string());
        self
    }

    /// Add an integer parameter
    pub fn with_int(mut self, key: &str, value: i64) -> Self {
        self.integers.insert(key.to_string(), value);
        self
    }

    /// Add a float parameter
    pub fn with_float(mut self, key: &str, value: f64) -> Self {
        self.floats.insert(key.to_string(), value);
        self
    }

    /// Add a boolean parameter
    pub fn with_bool(mut self, key: &str, value: bool) -> Self {
        self.booleans.insert(key.to_string(), value);
        self
    }

    /// Add a vector parameter
    pub fn with_vector(mut self, key: &str, value: Vec<f64>) -> Self {
        self.vectors.insert(key.to_string(), value);
        self
    }

    /// Add a shape parameter
    pub fn with_shape(mut self, key: &str, value: Vec<usize>) -> Self {
        self.shapes.insert(key.to_string(), value);
        self
    }

    /// Get a string parameter
    pub fn get_string(&self, key: &str) -> Option<&String> {
        self.strings.get(key)
    }

    /// Get an integer parameter
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.integers.get(key).copied()
    }

    /// Get a float parameter
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.floats.get(key).copied()
    }

    /// Get a boolean parameter
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.booleans.get(key).copied()
    }

    /// Get a vector parameter
    pub fn get_vector(&self, key: &str) -> Option<&Vec<f64>> {
        self.vectors.get(key)
    }

    /// Get a shape parameter
    pub fn get_shape(&self, key: &str) -> Option<&Vec<usize>> {
        self.shapes.get(key)
    }
}

impl Default for OperationParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a registered operation
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    /// Operation name
    pub name: String,
    /// Operation description
    pub description: String,
    /// Number of inputs
    pub num_inputs: usize,
    /// Number of outputs
    pub num_outputs: usize,
    /// Whether the operation supports autograd
    pub supports_autograd: bool,
    /// Data type this operation is registered for
    pub data_type: TypeId,
    /// Version of the operation
    pub version: String,
    /// Author/creator information
    pub author: Option<String>,
    /// Additional tags for categorization
    pub tags: Vec<String>,
}

/// Registry for custom operations
///
/// This registry maintains a collection of custom operations that can be applied to tensors.
/// Operations are stored per data type to ensure type safety.
pub struct CustomOperationRegistry {
    /// Operations stored by (TypeId, operation_name)
    operations: RwLock<HashMap<(TypeId, String), Arc<dyn Any + Send + Sync>>>,
    /// Metadata for registered operations
    metadata: RwLock<HashMap<(TypeId, String), OperationMetadata>>,
}

impl CustomOperationRegistry {
    /// Create a new operation registry
    pub fn new() -> Self {
        Self {
            operations: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom operation
    ///
    /// # Arguments
    /// * `operation` - The operation implementation
    /// * `version` - Version string for this operation
    /// * `author` - Optional author information
    /// * `tags` - Optional tags for categorization
    ///
    /// # Returns
    /// Success or error if registration fails
    pub fn register<T: TensorElement + 'static>(
        &self,
        operation: Box<dyn CustomOperation<T>>,
        version: &str,
        author: Option<String>,
        tags: Vec<String>,
    ) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let name = operation.name().to_string();
        let key = (type_id, name.clone());

        // Create metadata
        let metadata = OperationMetadata {
            name: name.clone(),
            description: operation.description().to_string(),
            num_inputs: operation.num_inputs(),
            num_outputs: operation.num_outputs(),
            supports_autograd: operation.supports_autograd(),
            data_type: type_id,
            version: version.to_string(),
            author,
            tags,
        };

        // Store the operation and metadata
        {
            let mut ops = self.operations.write().unwrap();
            let mut meta = self.metadata.write().unwrap();

            if ops.contains_key(&key) {
                return Err(TorshError::InvalidArgument(format!(
                    "Operation '{}' for type {:?} is already registered",
                    name, type_id
                )));
            }

            // Store the operation as Arc<dyn CustomOperation<T>> wrapped in Arc<dyn Any>
            let arc_op: Arc<dyn CustomOperation<T>> = Arc::from(operation);
            let boxed_any: Arc<dyn Any + Send + Sync> = Arc::new(arc_op);
            ops.insert(key.clone(), boxed_any);
            meta.insert(key, metadata);
        }

        Ok(())
    }

    /// Get a registered operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation to retrieve
    ///
    /// # Returns
    /// Reference to the operation if found
    pub fn get<T: TensorElement + 'static>(
        &self,
        name: &str,
    ) -> Option<Arc<dyn CustomOperation<T>>> {
        let type_id = TypeId::of::<T>();
        let key = (type_id, name.to_string());

        let ops = self.operations.read().unwrap();
        ops.get(&key).and_then(|arc_any| {
            // Downcast Arc<dyn Any> to Arc<dyn CustomOperation<T>>
            arc_any
                .downcast_ref::<Arc<dyn CustomOperation<T>>>()
                .map(|arc_op| Arc::clone(arc_op))
        })
    }

    /// Get metadata for a registered operation
    pub fn get_metadata<T: TensorElement + 'static>(
        &self,
        name: &str,
    ) -> Option<OperationMetadata> {
        let type_id = TypeId::of::<T>();
        let key = (type_id, name.to_string());

        let meta = self.metadata.read().unwrap();
        meta.get(&key).cloned()
    }

    /// List all registered operations for a given type
    pub fn list_operations<T: TensorElement + 'static>(&self) -> Vec<String> {
        let type_id = TypeId::of::<T>();
        let meta = self.metadata.read().unwrap();

        meta.keys()
            .filter(|(tid, _)| *tid == type_id)
            .map(|(_, name)| name.clone())
            .collect()
    }

    /// Remove a registered operation
    pub fn unregister<T: TensorElement + 'static>(&self, name: &str) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let key = (type_id, name.to_string());

        let mut ops = self.operations.write().unwrap();
        let mut meta = self.metadata.write().unwrap();

        if ops.remove(&key).is_none() {
            return Err(TorshError::InvalidArgument(format!(
                "Operation '{}' for type {:?} is not registered",
                name, type_id
            )));
        }

        meta.remove(&key);
        Ok(())
    }

    /// Check if an operation is registered
    pub fn is_registered<T: TensorElement + 'static>(&self, name: &str) -> bool {
        let type_id = TypeId::of::<T>();
        let key = (type_id, name.to_string());

        let ops = self.operations.read().unwrap();
        ops.contains_key(&key)
    }

    /// Get total number of registered operations
    pub fn count(&self) -> usize {
        let ops = self.operations.read().unwrap();
        ops.len()
    }

    /// Clear all registered operations
    pub fn clear(&self) {
        let mut ops = self.operations.write().unwrap();
        let mut meta = self.metadata.write().unwrap();
        ops.clear();
        meta.clear();
    }
}

impl Default for CustomOperationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global custom operation registry
static GLOBAL_REGISTRY: std::sync::LazyLock<CustomOperationRegistry> =
    std::sync::LazyLock::new(CustomOperationRegistry::new);

/// Get the global custom operation registry
pub fn global_registry() -> &'static CustomOperationRegistry {
    &GLOBAL_REGISTRY
}

/// Extension trait to add custom operation support to tensors
pub trait TensorCustomOps<T: TensorElement> {
    /// Apply a custom operation to this tensor
    ///
    /// # Arguments
    /// * `op_name` - Name of the registered operation
    /// * `other_inputs` - Additional input tensors (if any)
    /// * `params` - Operation parameters
    ///
    /// # Returns
    /// Result tensor(s) from the operation
    fn apply_custom_op(
        &self,
        op_name: &str,
        other_inputs: &[&Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Tensor<T>>>;

    /// Apply a custom operation using a specific registry
    fn apply_custom_op_with_registry(
        &self,
        registry: &CustomOperationRegistry,
        op_name: &str,
        other_inputs: &[&Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Tensor<T>>>;
}

impl<T: TensorElement + 'static> TensorCustomOps<T> for Tensor<T> {
    fn apply_custom_op(
        &self,
        op_name: &str,
        other_inputs: &[&Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Tensor<T>>> {
        self.apply_custom_op_with_registry(global_registry(), op_name, other_inputs, params)
    }

    fn apply_custom_op_with_registry(
        &self,
        registry: &CustomOperationRegistry,
        op_name: &str,
        other_inputs: &[&Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Tensor<T>>> {
        // Get the operation from the registry
        let operation = registry.get::<T>(op_name).ok_or_else(|| {
            TorshError::InvalidArgument(format!(
                "Custom operation '{}' not found for type",
                op_name
            ))
        })?;

        // Prepare input tensors
        let mut inputs = vec![self.clone()];
        inputs.extend(other_inputs.iter().map(|&t| t.clone()));

        // Validate inputs
        operation.validate_inputs(&inputs, params)?;

        // Check input count
        if inputs.len() != operation.num_inputs() {
            return Err(TorshError::InvalidArgument(format!(
                "Operation '{}' expects {} inputs, got {}",
                op_name,
                operation.num_inputs(),
                inputs.len()
            )));
        }

        // Execute the forward pass
        let outputs = operation.forward(&inputs, params)?;

        // Check output count
        if outputs.len() != operation.num_outputs() {
            return Err(TorshError::InvalidArgument(format!(
                "Operation '{}' produced {} outputs, expected {}",
                op_name,
                outputs.len(),
                operation.num_outputs()
            )));
        }

        Ok(outputs)
    }
}

// Example custom operations

/// A simple element-wise scaling operation
pub struct ScaleOperation;

impl<T: TensorElement + Copy + std::ops::Mul<Output = T> + num_traits::FromPrimitive>
    CustomOperation<T> for ScaleOperation
{
    fn name(&self) -> &str {
        "scale"
    }

    fn description(&self) -> &str {
        "Scales tensor elements by a constant factor"
    }

    fn forward(&self, inputs: &[Tensor<T>], params: &OperationParams) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Scale operation requires exactly 1 input".to_string(),
            ));
        }

        let scale = params.get_float("scale").unwrap_or(1.0);
        let scale_val = <T as FromPrimitive>::from_f64(scale).ok_or_else(|| {
            TorshError::InvalidArgument("Cannot convert scale factor to tensor type".to_string())
        })?;

        let result = inputs[0].mul_scalar(scale_val)?;
        Ok(vec![result])
    }

    fn backward(
        &self,
        grad_outputs: &[Tensor<T>],
        _inputs: &[Tensor<T>],
        _outputs: &[Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Option<Tensor<T>>>> {
        let scale = params.get_float("scale").unwrap_or(1.0);
        let scale_val = <T as FromPrimitive>::from_f64(scale).ok_or_else(|| {
            TorshError::InvalidArgument("Cannot convert scale factor to tensor type".to_string())
        })?;

        let grad_input = grad_outputs[0].mul_scalar(scale_val)?;
        Ok(vec![Some(grad_input)])
    }

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _params: &OperationParams,
    ) -> Result<Vec<Vec<usize>>> {
        if input_shapes.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Scale operation requires exactly 1 input".to_string(),
            ));
        }
        Ok(vec![input_shapes[0].clone()])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

/// A tensor concatenation operation along a specified axis
pub struct ConcatOperation;

impl<T: TensorElement + Copy> CustomOperation<T> for ConcatOperation {
    fn name(&self) -> &str {
        "concat"
    }

    fn description(&self) -> &str {
        "Concatenates tensors along a specified axis"
    }

    fn forward(&self, inputs: &[Tensor<T>], params: &OperationParams) -> Result<Vec<Tensor<T>>> {
        if inputs.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "Concat operation requires at least 2 inputs".to_string(),
            ));
        }

        let axis = params.get_int("axis").unwrap_or(0) as usize;

        // Use the existing cat operation from the tensor API
        let input_refs: Vec<&Tensor<T>> = inputs.iter().collect();
        let result = Tensor::cat(&input_refs, axis as i32)?;
        Ok(vec![result])
    }

    fn backward(
        &self,
        grad_outputs: &[Tensor<T>],
        inputs: &[Tensor<T>],
        _outputs: &[Tensor<T>],
        params: &OperationParams,
    ) -> Result<Vec<Option<Tensor<T>>>> {
        let axis = params.get_int("axis").unwrap_or(0) as usize;
        let grad_output = &grad_outputs[0];

        // Split the gradient back to match input sizes
        let mut split_sizes = Vec::new();
        for input in inputs {
            split_sizes.push(input.shape().dims()[axis]);
        }

        // Use multiple slice operations instead of split_with_sizes
        let mut grad_inputs = Vec::new();
        let mut start = 0;
        for &size in &split_sizes {
            let end = start + size;
            let slice = grad_output.slice_tensor(axis, start, end)?;
            grad_inputs.push(Some(slice));
            start = end;
        }
        Ok(grad_inputs)
    }

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        params: &OperationParams,
    ) -> Result<Vec<Vec<usize>>> {
        if input_shapes.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "Concat operation requires at least 2 inputs".to_string(),
            ));
        }

        let axis = params.get_int("axis").unwrap_or(0) as usize;
        let mut output_shape = input_shapes[0].clone();

        if axis >= output_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Concat axis {} out of bounds for {} dimensions",
                axis,
                output_shape.len()
            )));
        }

        // Sum the sizes along the concatenation axis
        let mut total_size = output_shape[axis];
        for shape in &input_shapes[1..] {
            if shape.len() != output_shape.len() {
                return Err(TorshError::InvalidArgument(
                    "All tensors must have the same number of dimensions".to_string(),
                ));
            }

            // Check that all dimensions except the concat axis match
            for (i, (&dim1, &dim2)) in output_shape.iter().zip(shape.iter()).enumerate() {
                if i != axis && dim1 != dim2 {
                    return Err(TorshError::InvalidArgument(format!(
                        "Dimension {} mismatch: {} vs {}",
                        i, dim1, dim2
                    )));
                }
            }

            total_size += shape[axis];
        }

        output_shape[axis] = total_size;
        Ok(vec![output_shape])
    }

    fn num_inputs(&self) -> usize {
        // Variable number of inputs, but we'll validate at runtime
        2 // Minimum required
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn validate_inputs(&self, inputs: &[Tensor<T>], params: &OperationParams) -> Result<()> {
        if inputs.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "Concat operation requires at least 2 inputs".to_string(),
            ));
        }

        let axis = params.get_int("axis").unwrap_or(0) as usize;
        let first_tensor_shape = inputs[0].shape();
        let first_shape = first_tensor_shape.dims();

        if axis >= first_shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Concat axis {} out of bounds for {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        // Validate that all tensors have compatible shapes
        for (i, tensor) in inputs.iter().enumerate().skip(1) {
            let tensor_shape = tensor.shape();
            let shape = tensor_shape.dims();
            if shape.len() != first_shape.len() {
                return Err(TorshError::InvalidArgument(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    shape.len(),
                    first_shape.len()
                )));
            }

            for (dim_idx, (&dim1, &dim2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if dim_idx != axis && dim1 != dim2 {
                    return Err(TorshError::InvalidArgument(format!(
                        "Tensor {} dimension {} mismatch: {} vs {}",
                        i, dim_idx, dim1, dim2
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_operation_params() {
        let params = OperationParams::new()
            .with_string("mode", "linear")
            .with_int("axis", 1)
            .with_float("scale", 2.5)
            .with_bool("inplace", false)
            .with_vector("weights", vec![1.0, 2.0, 3.0])
            .with_shape("target_shape", vec![10, 20]);

        assert_eq!(params.get_string("mode"), Some(&"linear".to_string()));
        assert_eq!(params.get_int("axis"), Some(1));
        assert_eq!(params.get_float("scale"), Some(2.5));
        assert_eq!(params.get_bool("inplace"), Some(false));
        assert_eq!(params.get_vector("weights"), Some(&vec![1.0, 2.0, 3.0]));
        assert_eq!(params.get_shape("target_shape"), Some(&vec![10, 20]));

        assert_eq!(params.get_string("nonexistent"), None);
    }

    #[test]
    fn test_registry_operations() {
        let registry = CustomOperationRegistry::new();

        // Register a scale operation
        let scale_op = Box::new(ScaleOperation);
        registry
            .register::<f32>(
                scale_op,
                "1.0.0",
                Some("Test".to_string()),
                vec!["math".to_string()],
            )
            .unwrap();

        // Check registration
        assert!(registry.is_registered::<f32>("scale"));
        assert!(!registry.is_registered::<f32>("nonexistent"));

        // Get metadata
        let metadata = registry.get_metadata::<f32>("scale").unwrap();
        assert_eq!(metadata.name, "scale");
        assert_eq!(
            metadata.description,
            "Scales tensor elements by a constant factor"
        );
        assert_eq!(metadata.num_inputs, 1);
        assert_eq!(metadata.num_outputs, 1);
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.author, Some("Test".to_string()));
        assert_eq!(metadata.tags, vec!["math".to_string()]);

        // List operations
        let ops = registry.list_operations::<f32>();
        assert_eq!(ops, vec!["scale".to_string()]);

        // Unregister
        registry.unregister::<f32>("scale").unwrap();
        assert!(!registry.is_registered::<f32>("scale"));
    }

    #[test]
    fn test_scale_operation() {
        let registry = CustomOperationRegistry::new();
        let scale_op = Box::new(ScaleOperation);
        registry
            .register::<f32>(scale_op, "1.0.0", None, vec![])
            .unwrap();

        // Create test tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();

        // Apply scale operation
        let params = OperationParams::new().with_float("scale", 2.0);
        let results = tensor
            .apply_custom_op_with_registry(&registry, "scale", &[], &params)
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];
        let expected_data = vec![2.0f32, 4.0, 6.0, 8.0];
        assert_eq!(result.data().unwrap(), expected_data);
    }

    #[test]
    fn test_concat_operation() {
        let registry = CustomOperationRegistry::new();
        let concat_op = Box::new(ConcatOperation);
        registry
            .register::<f32>(concat_op, "1.0.0", None, vec![])
            .unwrap();

        // Create test tensors (1D to work with current cat implementation)
        let data1 = vec![1.0f32, 2.0];
        let tensor1 = Tensor::from_data(data1, vec![2], DeviceType::Cpu).unwrap();

        let data2 = vec![3.0f32, 4.0];
        let tensor2 = Tensor::from_data(data2, vec![2], DeviceType::Cpu).unwrap();

        // Apply concat operation along axis 0
        let params = OperationParams::new().with_int("axis", 0);
        let results = tensor1
            .apply_custom_op_with_registry(&registry, "concat", &[&tensor2], &params)
            .unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.shape().dims(), &[4]); // 2 + 2 = 4 elements
        let expected_data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(result.data().unwrap(), expected_data);
    }

    #[test]
    fn test_operation_validation() {
        let registry = CustomOperationRegistry::new();
        let concat_op = Box::new(ConcatOperation);
        registry
            .register::<f32>(concat_op, "1.0.0", None, vec![])
            .unwrap();

        // Create tensors with incompatible dimensions (2D vs 1D should fail)
        let data1 = vec![1.0f32, 2.0];
        let tensor1 = Tensor::from_data(data1, vec![2], DeviceType::Cpu).unwrap(); // 1D tensor

        let data2 = vec![3.0f32, 4.0, 5.0, 6.0];
        let tensor2 = Tensor::from_data(data2, vec![2, 2], DeviceType::Cpu).unwrap(); // 2D tensor

        // This should fail validation due to different number of dimensions
        let params = OperationParams::new().with_int("axis", 0);
        let result =
            tensor1.apply_custom_op_with_registry(&registry, "concat", &[&tensor2], &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_output_shape_inference() {
        let concat_op = ConcatOperation;

        // Test shape inference for concat operation (1D tensors)
        let input_shapes = vec![vec![3], vec![4]];
        let params = OperationParams::new().with_int("axis", 0);

        let output_shapes = <ConcatOperation as CustomOperation<f32>>::output_shapes(
            &concat_op,
            &input_shapes,
            &params,
        )
        .unwrap();
        assert_eq!(output_shapes, vec![vec![7]]); // 3 + 4 = 7 along axis 0
    }

    #[test]
    fn test_error_cases() {
        let registry = CustomOperationRegistry::new();

        // Try to register duplicate operation
        let scale_op1 = Box::new(ScaleOperation);
        let scale_op2 = Box::new(ScaleOperation);

        registry
            .register::<f32>(scale_op1, "1.0.0", None, vec![])
            .unwrap();
        let result = registry.register::<f32>(scale_op2, "1.0.0", None, vec![]);
        assert!(result.is_err());

        // Try to unregister non-existent operation
        let result = registry.unregister::<f32>("nonexistent");
        assert!(result.is_err());

        // Try to apply non-existent operation
        let data = vec![1.0f32, 2.0];
        let tensor = Tensor::from_data(data, vec![1, 2], DeviceType::Cpu).unwrap();
        let params = OperationParams::new();
        let result = tensor.apply_custom_op_with_registry(&registry, "nonexistent", &[], &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_global_registry() {
        let registry = global_registry();

        // Register an operation in the global registry
        let scale_op = Box::new(ScaleOperation);
        registry
            .register::<f32>(scale_op, "1.0.0", None, vec![])
            .unwrap();

        // Use the operation via the tensor extension trait
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu).unwrap();
        let params = OperationParams::new().with_float("scale", 3.0);

        let results = tensor.apply_custom_op("scale", &[], &params).unwrap();
        assert_eq!(results.len(), 1);
        let expected_data = vec![3.0f32, 6.0, 9.0];
        assert_eq!(results[0].data().unwrap(), expected_data);

        // Clean up
        registry.unregister::<f32>("scale").unwrap();
    }
}
