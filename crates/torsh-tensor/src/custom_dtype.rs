//! Custom Data Type Registration System
//!
//! This module provides a flexible system for registering custom data types with the tensor framework.
//! It allows users to define their own numeric types and use them with tensor operations,
//! as long as they implement the required traits.
//!
//! # Features
//!
//! - **Type registration**: Register custom types with metadata
//! - **Operation registration**: Define custom operations for types
//! - **Type validation**: Ensure types meet requirements for tensor operations
//! - **Type conversion**: Define conversions between custom and built-in types
//! - **Trait-based extensibility**: Use Rust's trait system for type safety

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

/// Metadata for a registered custom data type
#[derive(Clone)]
pub struct CustomTypeMetadata {
    /// Human-readable name of the type
    pub name: String,
    /// Type ID for runtime type checking
    pub type_id: TypeId,
    /// Size of the type in bytes
    pub size_bytes: usize,
    /// Alignment requirement in bytes
    pub alignment: usize,
    /// Whether the type supports SIMD operations
    pub simd_capable: bool,
    /// Whether the type is a floating-point type
    pub is_float: bool,
    /// Whether the type is a complex number type
    pub is_complex: bool,
    /// Whether the type is signed
    pub is_signed: bool,
}

impl CustomTypeMetadata {
    /// Create metadata for a custom type
    pub fn new<T: TensorElement + 'static>(
        name: String,
        simd_capable: bool,
        is_float: bool,
        is_complex: bool,
        is_signed: bool,
    ) -> Self {
        Self {
            name,
            type_id: TypeId::of::<T>(),
            size_bytes: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
            simd_capable,
            is_float,
            is_complex,
            is_signed,
        }
    }

    /// Check if this type matches a given TypeId
    pub fn matches<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }
}

impl fmt::Debug for CustomTypeMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomTypeMetadata")
            .field("name", &self.name)
            .field("size_bytes", &self.size_bytes)
            .field("alignment", &self.alignment)
            .field("simd_capable", &self.simd_capable)
            .field("is_float", &self.is_float)
            .field("is_complex", &self.is_complex)
            .field("is_signed", &self.is_signed)
            .finish()
    }
}

/// Custom operation that can be registered for a data type
pub trait CustomOperation: Send + Sync {
    /// Name of the operation
    fn name(&self) -> &str;

    /// Execute the operation (boxed for dynamic dispatch)
    fn execute(&self, inputs: &[Box<dyn Any>]) -> Result<Box<dyn Any>>;

    /// Get the expected number of inputs
    fn num_inputs(&self) -> usize;

    /// Get a description of the operation
    fn description(&self) -> &str;
}

/// Registry for custom data types and their operations
pub struct CustomTypeRegistry {
    /// Registered type metadata
    types: RwLock<HashMap<TypeId, CustomTypeMetadata>>,
    /// Registered operations per type
    operations: RwLock<HashMap<TypeId, HashMap<String, Arc<dyn CustomOperation>>>>,
    /// Type conversion functions
    conversions: RwLock<HashMap<(TypeId, TypeId), Arc<dyn Any + Send + Sync>>>,
}

impl CustomTypeRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            types: RwLock::new(HashMap::new()),
            operations: RwLock::new(HashMap::new()),
            conversions: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom data type
    pub fn register_type(&self, metadata: CustomTypeMetadata) -> Result<()> {
        let mut types = self.types.write().unwrap();

        if types.contains_key(&metadata.type_id) {
            return Err(TorshError::InvalidArgument(format!(
                "Type {} is already registered",
                metadata.name
            )));
        }

        types.insert(metadata.type_id, metadata);
        Ok(())
    }

    /// Register a type with automatic metadata creation
    pub fn register_type_auto<T: TensorElement + 'static>(
        &self,
        name: String,
        simd_capable: bool,
        is_float: bool,
        is_complex: bool,
        is_signed: bool,
    ) -> Result<()> {
        let metadata =
            CustomTypeMetadata::new::<T>(name, simd_capable, is_float, is_complex, is_signed);
        self.register_type(metadata)
    }

    /// Get metadata for a registered type
    pub fn get_metadata<T: 'static>(&self) -> Option<CustomTypeMetadata> {
        let types = self.types.read().unwrap();
        types.get(&TypeId::of::<T>()).cloned()
    }

    /// Check if a type is registered
    pub fn is_registered<T: 'static>(&self) -> bool {
        let types = self.types.read().unwrap();
        types.contains_key(&TypeId::of::<T>())
    }

    /// Get all registered type names
    pub fn registered_types(&self) -> Vec<String> {
        let types = self.types.read().unwrap();
        types.values().map(|meta| meta.name.clone()).collect()
    }

    /// Register a custom operation for a type
    pub fn register_operation<T: 'static>(
        &self,
        operation: Arc<dyn CustomOperation>,
    ) -> Result<()> {
        let type_id = TypeId::of::<T>();

        // Ensure type is registered
        {
            let types = self.types.read().unwrap();
            if !types.contains_key(&type_id) {
                return Err(TorshError::InvalidArgument(
                    "Type must be registered before adding operations".to_string(),
                ));
            }
        }

        let mut ops = self.operations.write().unwrap();
        let type_ops = ops.entry(type_id).or_insert_with(HashMap::new);

        if type_ops.contains_key(operation.name()) {
            return Err(TorshError::InvalidArgument(format!(
                "Operation {} is already registered for this type",
                operation.name()
            )));
        }

        type_ops.insert(operation.name().to_string(), operation);
        Ok(())
    }

    /// Get a registered operation for a type
    pub fn get_operation<T: 'static>(&self, op_name: &str) -> Option<Arc<dyn CustomOperation>> {
        let ops = self.operations.read().unwrap();
        let type_id = TypeId::of::<T>();

        ops.get(&type_id)
            .and_then(|type_ops| type_ops.get(op_name))
            .cloned()
    }

    /// List all operations registered for a type
    pub fn list_operations<T: 'static>(&self) -> Vec<String> {
        let ops = self.operations.read().unwrap();
        let type_id = TypeId::of::<T>();

        ops.get(&type_id)
            .map(|type_ops| type_ops.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Register a type conversion function
    pub fn register_conversion<From: 'static, To: 'static, F>(&self, conversion_fn: F) -> Result<()>
    where
        F: Fn(From) -> To + Send + Sync + 'static,
    {
        let from_id = TypeId::of::<From>();
        let to_id = TypeId::of::<To>();

        let mut conversions = self.conversions.write().unwrap();

        if conversions.contains_key(&(from_id, to_id)) {
            return Err(TorshError::InvalidArgument(
                "Conversion already registered".to_string(),
            ));
        }

        conversions.insert((from_id, to_id), Arc::new(conversion_fn));
        Ok(())
    }

    /// Check if a conversion exists
    pub fn has_conversion<From: 'static, To: 'static>(&self) -> bool {
        let conversions = self.conversions.read().unwrap();
        conversions.contains_key(&(TypeId::of::<From>(), TypeId::of::<To>()))
    }

    /// Clear all registrations
    pub fn clear(&self) {
        self.types.write().unwrap().clear();
        self.operations.write().unwrap().clear();
        self.conversions.write().unwrap().clear();
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        let types = self.types.read().unwrap();
        let operations = self.operations.read().unwrap();
        let conversions = self.conversions.read().unwrap();

        let total_operations: usize = operations.values().map(|ops| ops.len()).sum();

        RegistryStats {
            num_types: types.len(),
            num_operations: total_operations,
            num_conversions: conversions.len(),
        }
    }
}

impl Default for CustomTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the custom type registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Number of registered types
    pub num_types: usize,
    /// Total number of registered operations
    pub num_operations: usize,
    /// Number of registered type conversions
    pub num_conversions: usize,
}

/// Get a reference to the global custom type registry
///
/// Uses lazy_static-style initialization with a RwLock
pub fn global_registry() -> &'static CustomTypeRegistry {
    use std::sync::OnceLock;
    static GLOBAL_REGISTRY: OnceLock<CustomTypeRegistry> = OnceLock::new();
    GLOBAL_REGISTRY.get_or_init(CustomTypeRegistry::new)
}

/// Builder for registering custom types with a fluent API
pub struct CustomTypeBuilder<T: TensorElement + 'static> {
    name: String,
    simd_capable: bool,
    is_float: bool,
    is_complex: bool,
    is_signed: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TensorElement + 'static> CustomTypeBuilder<T> {
    /// Create a new builder for a custom type
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            simd_capable: false,
            is_float: false,
            is_complex: false,
            is_signed: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Mark the type as SIMD-capable
    pub fn simd_capable(mut self, capable: bool) -> Self {
        self.simd_capable = capable;
        self
    }

    /// Mark the type as floating-point
    pub fn is_float(mut self, is_float: bool) -> Self {
        self.is_float = is_float;
        self
    }

    /// Mark the type as complex number
    pub fn is_complex(mut self, is_complex: bool) -> Self {
        self.is_complex = is_complex;
        self
    }

    /// Mark the type as signed
    pub fn is_signed(mut self, is_signed: bool) -> Self {
        self.is_signed = is_signed;
        self
    }

    /// Register the type with the global registry
    pub fn register(self) -> Result<()> {
        global_registry().register_type_auto::<T>(
            self.name,
            self.simd_capable,
            self.is_float,
            self.is_complex,
            self.is_signed,
        )
    }

    /// Register the type with a specific registry
    pub fn register_with(self, registry: &CustomTypeRegistry) -> Result<()> {
        registry.register_type_auto::<T>(
            self.name,
            self.simd_capable,
            self.is_float,
            self.is_complex,
            self.is_signed,
        )
    }
}

/// Helper macro for defining custom operations
#[macro_export]
macro_rules! define_custom_operation {
    ($name:ident, $input_count:expr, $desc:expr, $execute_fn:expr) => {
        struct $name;

        impl CustomOperation for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn num_inputs(&self) -> usize {
                $input_count
            }

            fn description(&self) -> &str {
                $desc
            }

            fn execute(&self, inputs: &[Box<dyn Any>]) -> Result<Box<dyn Any>> {
                $execute_fn(inputs)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Define a simple custom type for testing
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MyCustomFloat(f64);

    impl TensorElement for MyCustomFloat {
        fn dtype() -> torsh_core::DType {
            torsh_core::DType::F64
        }

        fn zero() -> Self {
            MyCustomFloat(0.0)
        }

        fn one() -> Self {
            MyCustomFloat(1.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }

        fn is_one(&self) -> bool {
            self.0 == 1.0
        }

        fn from_f64(value: f64) -> Option<Self> {
            Some(MyCustomFloat(value))
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0)
        }
    }

    #[test]
    fn test_type_registration() {
        let registry = CustomTypeRegistry::new();

        let result = CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .is_float(true)
            .is_signed(true)
            .simd_capable(false)
            .register_with(&registry);

        assert!(result.is_ok());
        assert!(registry.is_registered::<MyCustomFloat>());

        let metadata = registry.get_metadata::<MyCustomFloat>().unwrap();
        assert_eq!(metadata.name, "MyCustomFloat");
        assert!(metadata.is_float);
        assert!(metadata.is_signed);
        assert!(!metadata.simd_capable);
        assert_eq!(metadata.size_bytes, std::mem::size_of::<MyCustomFloat>());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = CustomTypeRegistry::new();

        CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .register_with(&registry)
            .unwrap();

        let result =
            CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat").register_with(&registry);

        assert!(result.is_err());
    }

    #[test]
    fn test_registered_types_list() {
        let registry = CustomTypeRegistry::new();

        CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .register_with(&registry)
            .unwrap();

        let types = registry.registered_types();
        assert_eq!(types.len(), 1);
        assert!(types.contains(&"MyCustomFloat".to_string()));
    }

    #[test]
    fn test_registry_stats() {
        let registry = CustomTypeRegistry::new();

        CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .register_with(&registry)
            .unwrap();

        let stats = registry.stats();
        assert_eq!(stats.num_types, 1);
        assert_eq!(stats.num_operations, 0);
        assert_eq!(stats.num_conversions, 0);
    }

    #[test]
    fn test_conversion_registration() {
        let registry = CustomTypeRegistry::new();

        CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .register_with(&registry)
            .unwrap();

        CustomTypeBuilder::<f64>::new("f64")
            .is_float(true)
            .register_with(&registry)
            .unwrap();

        let result =
            registry.register_conversion::<MyCustomFloat, f64, _>(|val: MyCustomFloat| val.0);
        assert!(result.is_ok());

        assert!(registry.has_conversion::<MyCustomFloat, f64>());
        assert!(!registry.has_conversion::<f64, MyCustomFloat>());
    }

    #[test]
    fn test_registry_clear() {
        let registry = CustomTypeRegistry::new();

        CustomTypeBuilder::<MyCustomFloat>::new("MyCustomFloat")
            .register_with(&registry)
            .unwrap();

        assert!(registry.is_registered::<MyCustomFloat>());

        registry.clear();

        assert!(!registry.is_registered::<MyCustomFloat>());
        let stats = registry.stats();
        assert_eq!(stats.num_types, 0);
    }

    #[test]
    fn test_metadata_matches() {
        let metadata = CustomTypeMetadata::new::<MyCustomFloat>(
            "MyCustomFloat".to_string(),
            false,
            true,
            false,
            true,
        );

        assert!(metadata.matches::<MyCustomFloat>());
        assert!(!metadata.matches::<f64>());
    }
}
