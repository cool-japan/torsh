//! Example custom operations for demonstrating custom data type support

use crate::{
    custom_types::{ExtendedCustomOperation, ExtendedShapeInfo},
    TorshResult,
};
use std::any::TypeId;
use std::collections::HashMap;
use torsh_core::{
    dtype::{CustomInt16, DType, ExtendedDType},
    error::TorshError,
};
use torsh_tensor::Tensor;

/// Example custom operation that works with CustomInt16 data type
pub struct CustomInt16AddOperation;

/// Custom multiplication operation for CustomInt16 tensors
pub struct CustomInt16MulOperation;

/// Custom subtraction operation for CustomInt16 tensors  
pub struct CustomInt16SubOperation;

impl ExtendedCustomOperation for CustomInt16AddOperation {
    fn execute_extended(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Add requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // Validate tensors are compatible
        if a.shape() != b.shape() {
            return Err(TorshError::InvalidArgument(
                "Input tensors must have the same shape".to_string(),
            ));
        }

        // Implement actual CustomInt16 addition with custom semantics
        // This would ideally access the tensor's raw data and perform element-wise operations
        // For now, we simulate the custom addition behavior

        // Extract the underlying data (this is a simplified simulation)
        // In a real implementation, we would:
        // 1. Access the tensor's storage buffer
        // 2. Interpret it as CustomInt16 elements
        // 3. Perform element-wise addition using CustomInt16::add_element
        // 4. Create a new tensor with the result

        // For demonstration, we perform the addition using the custom semantics:
        // - Values are added with saturation
        // - Metadata uses max operation

        // Since we can't directly access CustomInt16 tensor data without more infrastructure,
        // we delegate to regular tensor addition but document the intended behavior
        let result = a.add_op(b)?;

        // Note: In a complete implementation, this would perform:
        // for each pair of CustomInt16 elements (elem_a, elem_b):
        //   result_value = elem_a.value.saturating_add(elem_b.value)
        //   result_metadata = elem_a.metadata.max(elem_b.metadata)
        //   result_element = CustomInt16 { value: result_value, metadata: result_metadata }

        Ok(result)
    }

    fn name(&self) -> &str {
        "custom_int16_add"
    }

    fn infer_shape_extended(
        &self,
        input_shapes: &[ExtendedShapeInfo],
    ) -> TorshResult<ExtendedShapeInfo> {
        if input_shapes.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Add requires exactly 2 inputs".to_string(),
            ));
        }

        // Both inputs should have CustomInt16 type
        for shape_info in input_shapes {
            match &shape_info.dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Add requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Add requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        // Shapes must be broadcastable
        if input_shapes[0].shape.dims() != input_shapes[1].shape.dims() {
            return Err(TorshError::InvalidArgument(
                "Input shapes must be identical for CustomInt16Add".to_string(),
            ));
        }

        // Output has the same shape and type as inputs
        Ok(input_shapes[0].clone())
    }

    fn validate_types_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<()> {
        if input_types.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Add requires exactly 2 inputs".to_string(),
            ));
        }

        for dtype in input_types {
            match dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Add requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Add requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn infer_type_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<ExtendedDType> {
        self.validate_types_extended(input_types)?;
        Ok(input_types[0].clone())
    }

    fn supports_custom_type(&self, type_id: TypeId) -> bool {
        type_id == TypeId::of::<CustomInt16>()
    }

    fn supported_custom_types(&self) -> Vec<TypeId> {
        vec![TypeId::of::<CustomInt16>()]
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "Element-wise addition for CustomInt16 tensors".to_string(),
        );
        metadata.insert(
            "custom_semantics".to_string(),
            "Combines metadata using max operation".to_string(),
        );
        Some(metadata)
    }
}

impl ExtendedCustomOperation for CustomInt16MulOperation {
    fn execute_extended(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Mul requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // Validate tensors are compatible
        if a.shape() != b.shape() {
            return Err(TorshError::InvalidArgument(
                "Input tensors must have the same shape".to_string(),
            ));
        }

        // Implement CustomInt16 multiplication with custom semantics:
        // - Values are multiplied with saturation
        // - Metadata is added (saturating)
        let result = a.mul_op(b)?;

        // Note: In a complete implementation, this would perform:
        // for each pair of CustomInt16 elements (elem_a, elem_b):
        //   result_value = elem_a.value.saturating_mul(elem_b.value)
        //   result_metadata = elem_a.metadata.saturating_add(elem_b.metadata)
        //   result_element = CustomInt16 { value: result_value, metadata: result_metadata }

        Ok(result)
    }

    fn name(&self) -> &str {
        "custom_int16_mul"
    }

    fn infer_shape_extended(
        &self,
        input_shapes: &[ExtendedShapeInfo],
    ) -> TorshResult<ExtendedShapeInfo> {
        if input_shapes.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Mul requires exactly 2 inputs".to_string(),
            ));
        }

        // Both inputs should have CustomInt16 type and same shape
        for shape_info in input_shapes {
            match &shape_info.dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Mul requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Mul requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        if input_shapes[0].shape.dims() != input_shapes[1].shape.dims() {
            return Err(TorshError::InvalidArgument(
                "Input shapes must be identical for CustomInt16Mul".to_string(),
            ));
        }

        Ok(input_shapes[0].clone())
    }

    fn validate_types_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<()> {
        if input_types.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Mul requires exactly 2 inputs".to_string(),
            ));
        }

        for dtype in input_types {
            match dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Mul requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Mul requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn infer_type_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<ExtendedDType> {
        self.validate_types_extended(input_types)?;
        Ok(input_types[0].clone())
    }

    fn supports_custom_type(&self, type_id: TypeId) -> bool {
        type_id == TypeId::of::<CustomInt16>()
    }

    fn supported_custom_types(&self) -> Vec<TypeId> {
        vec![TypeId::of::<CustomInt16>()]
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "Element-wise multiplication for CustomInt16 tensors".to_string(),
        );
        metadata.insert(
            "custom_semantics".to_string(),
            "Values multiplied with saturation, metadata added".to_string(),
        );
        Some(metadata)
    }
}

impl ExtendedCustomOperation for CustomInt16SubOperation {
    fn execute_extended(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Sub requires exactly 2 inputs".to_string(),
            ));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // Validate tensors are compatible
        if a.shape() != b.shape() {
            return Err(TorshError::InvalidArgument(
                "Input tensors must have the same shape".to_string(),
            ));
        }

        // Implement CustomInt16 subtraction with custom semantics:
        // - Values are subtracted with saturation
        // - Metadata uses difference (min 0)
        let result = a.sub(b)?;

        // Note: In a complete implementation, this would perform:
        // for each pair of CustomInt16 elements (elem_a, elem_b):
        //   result_value = elem_a.value.saturating_sub(elem_b.value)
        //   result_metadata = elem_a.metadata.saturating_sub(elem_b.metadata)
        //   result_element = CustomInt16 { value: result_value, metadata: result_metadata }

        Ok(result)
    }

    fn name(&self) -> &str {
        "custom_int16_sub"
    }

    fn infer_shape_extended(
        &self,
        input_shapes: &[ExtendedShapeInfo],
    ) -> TorshResult<ExtendedShapeInfo> {
        if input_shapes.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Sub requires exactly 2 inputs".to_string(),
            ));
        }

        // Both inputs should have CustomInt16 type and same shape
        for shape_info in input_shapes {
            match &shape_info.dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Sub requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Sub requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        if input_shapes[0].shape.dims() != input_shapes[1].shape.dims() {
            return Err(TorshError::InvalidArgument(
                "Input shapes must be identical for CustomInt16Sub".to_string(),
            ));
        }

        Ok(input_shapes[0].clone())
    }

    fn validate_types_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<()> {
        if input_types.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CustomInt16Sub requires exactly 2 inputs".to_string(),
            ));
        }

        for dtype in input_types {
            match dtype {
                ExtendedDType::Custom(type_id) => {
                    if *type_id != TypeId::of::<CustomInt16>() {
                        return Err(TorshError::InvalidArgument(
                            "CustomInt16Sub requires CustomInt16 inputs".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "CustomInt16Sub requires custom CustomInt16 inputs".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn infer_type_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<ExtendedDType> {
        self.validate_types_extended(input_types)?;
        Ok(input_types[0].clone())
    }

    fn supports_custom_type(&self, type_id: TypeId) -> bool {
        type_id == TypeId::of::<CustomInt16>()
    }

    fn supported_custom_types(&self) -> Vec<TypeId> {
        vec![TypeId::of::<CustomInt16>()]
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "Element-wise subtraction for CustomInt16 tensors".to_string(),
        );
        metadata.insert(
            "custom_semantics".to_string(),
            "Values subtracted with saturation, metadata subtracted".to_string(),
        );
        Some(metadata)
    }
}

/// Example custom operation that converts between standard and custom types
pub struct TypeConversionOperation {
    from_type: ExtendedDType,
    to_type: ExtendedDType,
}

impl TypeConversionOperation {
    pub fn new(from_type: ExtendedDType, to_type: ExtendedDType) -> Self {
        Self { from_type, to_type }
    }

    /// Create a conversion from standard int16 to CustomInt16
    pub fn standard_to_custom_int16() -> TorshResult<Self> {
        Ok(Self {
            from_type: ExtendedDType::Standard(DType::I16),
            to_type: ExtendedDType::custom::<CustomInt16>().ok_or_else(|| {
                TorshError::InvalidArgument("CustomInt16 should be registered".to_string())
            })?,
        })
    }

    /// Create a conversion from CustomInt16 to standard int16
    pub fn custom_int16_to_standard() -> TorshResult<Self> {
        Ok(Self {
            from_type: ExtendedDType::custom::<CustomInt16>().ok_or_else(|| {
                TorshError::InvalidArgument("CustomInt16 should be registered".to_string())
            })?,
            to_type: ExtendedDType::Standard(DType::I16),
        })
    }
}

impl ExtendedCustomOperation for TypeConversionOperation {
    fn execute_extended(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "TypeConversion requires exactly 1 input".to_string(),
            ));
        }

        let input = &inputs[0];

        // For demonstration purposes, this is a simplified implementation
        // In practice, this would perform actual type conversion
        match (&self.from_type, &self.to_type) {
            (ExtendedDType::Standard(DType::I16), ExtendedDType::Custom(_)) => {
                // Convert standard i16 to CustomInt16
                // For now, just return the input (would need tensor crate support)
                Ok(input.clone())
            }
            (ExtendedDType::Custom(_), ExtendedDType::Standard(DType::I16)) => {
                // Convert CustomInt16 to standard i16
                // For now, just return the input (would need tensor crate support)
                Ok(input.clone())
            }
            _ => Err(TorshError::InvalidArgument(
                "Unsupported type conversion".to_string(),
            )),
        }
    }

    fn name(&self) -> &str {
        match (&self.from_type, &self.to_type) {
            (ExtendedDType::Standard(DType::I16), ExtendedDType::Custom(_)) => {
                "standard_to_custom_int16"
            }
            (ExtendedDType::Custom(_), ExtendedDType::Standard(DType::I16)) => {
                "custom_int16_to_standard"
            }
            _ => "type_conversion",
        }
    }

    fn infer_shape_extended(
        &self,
        input_shapes: &[ExtendedShapeInfo],
    ) -> TorshResult<ExtendedShapeInfo> {
        if input_shapes.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "TypeConversion requires exactly 1 input".to_string(),
            ));
        }

        // Shape is preserved, only type changes
        Ok(ExtendedShapeInfo::new(
            input_shapes[0].shape.clone(),
            self.to_type.clone(),
        ))
    }

    fn validate_types_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<()> {
        if input_types.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "TypeConversion requires exactly 1 input".to_string(),
            ));
        }

        // Check if input type matches expected source type
        if std::mem::discriminant(&input_types[0]) != std::mem::discriminant(&self.from_type) {
            return Err(TorshError::InvalidArgument(
                "Input type does not match expected source type".to_string(),
            ));
        }

        Ok(())
    }

    fn infer_type_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<ExtendedDType> {
        self.validate_types_extended(input_types)?;
        Ok(self.to_type.clone())
    }

    fn supports_custom_type(&self, type_id: TypeId) -> bool {
        match (&self.from_type, &self.to_type) {
            (ExtendedDType::Custom(from_id), _) => *from_id == type_id,
            (_, ExtendedDType::Custom(to_id)) => *to_id == type_id,
            _ => false,
        }
    }

    fn supported_custom_types(&self) -> Vec<TypeId> {
        let mut types = Vec::new();

        if let ExtendedDType::Custom(type_id) = &self.from_type {
            types.push(*type_id);
        }

        if let ExtendedDType::Custom(type_id) = &self.to_type {
            if !types.contains(type_id) {
                types.push(*type_id);
            }
        }

        types
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "Type conversion between standard and custom types".to_string(),
        );
        metadata.insert("from_type".to_string(), self.from_type.name());
        metadata.insert("to_type".to_string(), self.to_type.name());
        Some(metadata)
    }
}

/// Example operation that works with multiple custom types
pub struct CustomTypeUnifyOperation;

impl ExtendedCustomOperation for CustomTypeUnifyOperation {
    fn execute_extended(&self, inputs: Vec<Tensor>) -> TorshResult<Tensor> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "CustomTypeUnify requires at least 1 input".to_string(),
            ));
        }

        // For demonstration: simply return the first input
        // In practice, this would perform custom unification logic
        Ok(inputs[0].clone())
    }

    fn name(&self) -> &str {
        "custom_type_unify"
    }

    fn infer_shape_extended(
        &self,
        input_shapes: &[ExtendedShapeInfo],
    ) -> TorshResult<ExtendedShapeInfo> {
        if input_shapes.is_empty() {
            return Err(TorshError::InvalidArgument(
                "CustomTypeUnify requires at least 1 input".to_string(),
            ));
        }

        // For demonstration: return shape and type of first input
        // In practice, this would implement custom unification rules
        Ok(input_shapes[0].clone())
    }

    fn validate_types_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<()> {
        if input_types.is_empty() {
            return Err(TorshError::InvalidArgument(
                "CustomTypeUnify requires at least 1 input".to_string(),
            ));
        }

        // Check that all inputs are custom types
        for dtype in input_types {
            if !dtype.is_custom() {
                return Err(TorshError::InvalidArgument(
                    "CustomTypeUnify requires all inputs to be custom types".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn infer_type_extended(&self, input_types: &[ExtendedDType]) -> TorshResult<ExtendedDType> {
        self.validate_types_extended(input_types)?;

        // For demonstration: return first input type
        // In practice, this would implement custom type unification rules
        Ok(input_types[0].clone())
    }

    fn supports_custom_type(&self, _type_id: TypeId) -> bool {
        true // Supports any custom type
    }

    fn supported_custom_types(&self) -> Vec<TypeId> {
        // Return all registered custom types
        use torsh_core::dtype::CustomDTypeRegistry;
        CustomDTypeRegistry::list_types()
            .into_iter()
            .map(|info| info.type_id)
            .collect()
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            "Unifies multiple custom type tensors into a single result".to_string(),
        );
        metadata.insert(
            "flexibility".to_string(),
            "Supports any registered custom type".to_string(),
        );
        Some(metadata)
    }
}

/// Helper function to register all example custom operations
pub fn register_example_operations() -> TorshResult<()> {
    use crate::custom_types::register_extended_operation;

    // Register the CustomInt16 type first (ignore if already registered)
    use crate::custom_types::CustomTypeUtils;
    let _ = CustomTypeUtils::register_custom_type::<CustomInt16>();

    // Register custom operations (ignore if already registered)
    let _ = register_extended_operation(CustomInt16AddOperation);
    let _ = register_extended_operation(CustomInt16MulOperation);
    let _ = register_extended_operation(CustomInt16SubOperation);
    if let Ok(op) = TypeConversionOperation::standard_to_custom_int16() {
        let _ = register_extended_operation(op);
    }
    if let Ok(op) = TypeConversionOperation::custom_int16_to_standard() {
        let _ = register_extended_operation(op);
    }
    let _ = register_extended_operation(CustomTypeUnifyOperation);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_types::{global_extended_registry, ExtendedShapeInfo};
    use crate::CustomTypeUtils;
    use std::any::TypeId;
    use torsh_core::{dtype::CustomInt16, shape::Shape};

    #[test]
    fn test_custom_int16_add_operation() {
        let op = CustomInt16AddOperation;

        // Test name
        assert_eq!(op.name(), "custom_int16_add");

        // Test custom type support
        assert!(op.supports_custom_type(TypeId::of::<CustomInt16>()));
        assert!(!op.supports_custom_type(TypeId::of::<i32>()));

        // Test supported types
        let supported = op.supported_custom_types();
        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&TypeId::of::<CustomInt16>()));

        // Test metadata
        let metadata = op.metadata().unwrap();
        assert!(metadata.contains_key("description"));
        assert!(metadata.contains_key("custom_semantics"));
    }

    #[test]
    fn test_type_conversion_operation() {
        // Register the CustomInt16 type first
        let _ = CustomTypeUtils::register_custom_type::<CustomInt16>();

        let op = TypeConversionOperation::standard_to_custom_int16().unwrap();

        // Test name
        assert_eq!(op.name(), "standard_to_custom_int16");

        // Test custom type support
        assert!(op.supports_custom_type(TypeId::of::<CustomInt16>()));

        // Test metadata
        let metadata = op.metadata().unwrap();
        assert!(metadata.contains_key("description"));
        assert!(metadata.contains_key("from_type"));
        assert!(metadata.contains_key("to_type"));
    }

    #[test]
    fn test_custom_type_unify_operation() {
        let op = CustomTypeUnifyOperation;

        // Test name
        assert_eq!(op.name(), "custom_type_unify");

        // Test that it supports any custom type
        assert!(op.supports_custom_type(TypeId::of::<CustomInt16>()));
        assert!(op.supports_custom_type(TypeId::of::<i64>())); // Any type ID
    }

    #[test]
    fn test_shape_inference() {
        // Register the CustomInt16 type first
        let _ = CustomTypeUtils::register_custom_type::<CustomInt16>();

        let op = CustomInt16AddOperation;

        // Create test shapes with custom types
        let custom_dtype =
            ExtendedDType::custom::<CustomInt16>().expect("CustomInt16 should be available");

        let shape1 = ExtendedShapeInfo::new(Shape::new(vec![2, 3]), custom_dtype.clone());
        let shape2 = ExtendedShapeInfo::new(Shape::new(vec![2, 3]), custom_dtype.clone());

        let result = op.infer_shape_extended(&[shape1, shape2]).unwrap();
        assert_eq!(result.shape.dims(), &[2, 3]);
        assert!(result.dtype.is_custom());
    }

    #[test]
    fn test_type_validation() {
        // Register the CustomInt16 type first
        let _ = CustomTypeUtils::register_custom_type::<CustomInt16>();

        let op = CustomInt16AddOperation;

        let custom_dtype =
            ExtendedDType::custom::<CustomInt16>().expect("CustomInt16 should be available");
        let standard_dtype = ExtendedDType::Standard(DType::I16);

        // Valid: two custom types
        assert!(op
            .validate_types_extended(&[custom_dtype.clone(), custom_dtype.clone()])
            .is_ok());

        // Invalid: mixed types
        assert!(op
            .validate_types_extended(&[custom_dtype, standard_dtype])
            .is_err());

        // Invalid: wrong number of inputs
        assert!(op.validate_types_extended(&[]).is_err());
    }

    #[test]
    fn test_custom_int16_mul_operation() {
        let op = CustomInt16MulOperation;

        // Test name
        assert_eq!(op.name(), "custom_int16_mul");

        // Test custom type support
        assert!(op.supports_custom_type(TypeId::of::<CustomInt16>()));
        assert!(!op.supports_custom_type(TypeId::of::<i32>()));

        // Test supported types
        let supported = op.supported_custom_types();
        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&TypeId::of::<CustomInt16>()));

        // Test metadata
        let metadata = op.metadata().unwrap();
        assert!(metadata.contains_key("description"));
        assert!(metadata.contains_key("custom_semantics"));
        assert!(metadata["description"].contains("multiplication"));
    }

    #[test]
    fn test_custom_int16_sub_operation() {
        let op = CustomInt16SubOperation;

        // Test name
        assert_eq!(op.name(), "custom_int16_sub");

        // Test custom type support
        assert!(op.supports_custom_type(TypeId::of::<CustomInt16>()));
        assert!(!op.supports_custom_type(TypeId::of::<i32>()));

        // Test supported types
        let supported = op.supported_custom_types();
        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&TypeId::of::<CustomInt16>()));

        // Test metadata
        let metadata = op.metadata().unwrap();
        assert!(metadata.contains_key("description"));
        assert!(metadata.contains_key("custom_semantics"));
        assert!(metadata["description"].contains("subtraction"));
    }

    #[test]
    fn test_register_example_operations() {
        let result = register_example_operations();
        assert!(result.is_ok());

        // Check that operations are registered
        let registry = global_extended_registry();
        assert!(registry.is_registered("custom_int16_add"));
        assert!(registry.is_registered("custom_int16_mul"));
        assert!(registry.is_registered("custom_int16_sub"));
        assert!(registry.is_registered("standard_to_custom_int16"));
        assert!(registry.is_registered("custom_int16_to_standard"));
        assert!(registry.is_registered("custom_type_unify"));

        // Check operations for CustomInt16 type
        let ops = registry.get_operations_for_type(TypeId::of::<CustomInt16>());
        assert!(!ops.is_empty());
        assert!(ops.contains(&"custom_int16_add".to_string()));
        assert!(ops.contains(&"custom_int16_mul".to_string()));
        assert!(ops.contains(&"custom_int16_sub".to_string()));
    }
}
