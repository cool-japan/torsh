use crate::error::{FfiError, FfiResult};
use once_cell::sync::Lazy;
use torsh_core::{dtype::TypePromotion, DType};

/// Comprehensive type mapping system for cross-framework compatibility
#[derive(Debug, Clone)]
pub struct TypeMapper {
    /// Maps between different framework dtype representations
    framework_mappings: std::collections::HashMap<String, FrameworkTypeInfo>,
}

#[derive(Debug, Clone)]
pub struct FrameworkTypeInfo {
    pub torsh_dtype: DType,
    pub numpy_dtype: String,
    pub pytorch_dtype: String,
    pub element_size: usize,
    pub is_floating: bool,
    pub is_signed: bool,
}

impl TypeMapper {
    /// Create a new type mapper with predefined mappings
    pub fn new() -> Self {
        let mut framework_mappings = std::collections::HashMap::new();

        // Float32 mappings
        framework_mappings.insert(
            "f32".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::F32,
                numpy_dtype: "float32".to_string(),
                pytorch_dtype: "torch.float32".to_string(),
                element_size: 4,
                is_floating: true,
                is_signed: true,
            },
        );

        // Float64 mappings
        framework_mappings.insert(
            "f64".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::F64,
                numpy_dtype: "float64".to_string(),
                pytorch_dtype: "torch.float64".to_string(),
                element_size: 8,
                is_floating: true,
                is_signed: true,
            },
        );

        // Int32 mappings
        framework_mappings.insert(
            "i32".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::I32,
                numpy_dtype: "int32".to_string(),
                pytorch_dtype: "torch.int32".to_string(),
                element_size: 4,
                is_floating: false,
                is_signed: true,
            },
        );

        // Int64 mappings
        framework_mappings.insert(
            "i64".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::I64,
                numpy_dtype: "int64".to_string(),
                pytorch_dtype: "torch.int64".to_string(),
                element_size: 8,
                is_floating: false,
                is_signed: true,
            },
        );

        // Boolean mappings
        framework_mappings.insert(
            "bool".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::Bool,
                numpy_dtype: "bool".to_string(),
                pytorch_dtype: "torch.bool".to_string(),
                element_size: 1,
                is_floating: false,
                is_signed: false,
            },
        );

        // Additional integer types
        framework_mappings.insert(
            "u8".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::U8,
                numpy_dtype: "uint8".to_string(),
                pytorch_dtype: "torch.uint8".to_string(),
                element_size: 1,
                is_floating: false,
                is_signed: false,
            },
        );

        framework_mappings.insert(
            "i8".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::I8,
                numpy_dtype: "int8".to_string(),
                pytorch_dtype: "torch.int8".to_string(),
                element_size: 1,
                is_floating: false,
                is_signed: true,
            },
        );

        framework_mappings.insert(
            "i16".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::I16,
                numpy_dtype: "int16".to_string(),
                pytorch_dtype: "torch.int16".to_string(),
                element_size: 2,
                is_floating: false,
                is_signed: true,
            },
        );

        // Half precision float
        framework_mappings.insert(
            "f16".to_string(),
            FrameworkTypeInfo {
                torsh_dtype: DType::F16,
                numpy_dtype: "float16".to_string(),
                pytorch_dtype: "torch.float16".to_string(),
                element_size: 2,
                is_floating: true,
                is_signed: true,
            },
        );

        Self { framework_mappings }
    }

    /// Convert from NumPy dtype string to ToRSh DType
    pub fn numpy_to_torsh(&self, numpy_dtype: &str) -> FfiResult<DType> {
        for info in self.framework_mappings.values() {
            if info.numpy_dtype == numpy_dtype {
                return Ok(info.torsh_dtype);
            }
        }

        // Additional NumPy aliases
        match numpy_dtype {
            "float" | "double" => Ok(DType::F64),
            "single" => Ok(DType::F32),
            "int" | "long" => Ok(DType::I64),
            "intc" => Ok(DType::I32),
            "byte" => Ok(DType::I8),
            "ubyte" => Ok(DType::U8),
            "short" => Ok(DType::I16),
            "half" => Ok(DType::F16),
            "bool_" | "boolean" => Ok(DType::Bool),
            _ => Err(FfiError::DTypeMismatch {
                expected: "f32, f64, i32, i64, i16, i8, u8, f16, bool".to_string(),
                actual: numpy_dtype.to_string(),
            }),
        }
    }

    /// Convert from PyTorch dtype string to ToRSh DType
    pub fn pytorch_to_torsh(&self, pytorch_dtype: &str) -> FfiResult<DType> {
        for info in self.framework_mappings.values() {
            if info.pytorch_dtype == pytorch_dtype {
                return Ok(info.torsh_dtype);
            }
        }

        // Additional PyTorch aliases
        match pytorch_dtype {
            "torch.float" => Ok(DType::F32),
            "torch.double" => Ok(DType::F64),
            "torch.long" => Ok(DType::I64),
            "torch.int" => Ok(DType::I32),
            "torch.short" => Ok(DType::I16),
            "torch.char" => Ok(DType::I8),
            "torch.byte" => Ok(DType::U8),
            "torch.half" => Ok(DType::F16),
            _ => Err(FfiError::DTypeMismatch {
                expected: "torch.float32, torch.float64, torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.float16, torch.bool".to_string(),
                actual: pytorch_dtype.to_string(),
            }),
        }
    }

    /// Convert from ToRSh DType to NumPy dtype string
    pub fn torsh_to_numpy(&self, dtype: DType) -> String {
        match dtype {
            DType::F32 => "float32".to_string(),
            DType::F64 => "float64".to_string(),
            DType::I32 => "int32".to_string(),
            DType::I64 => "int64".to_string(),
            DType::I16 => "int16".to_string(),
            DType::I8 => "int8".to_string(),
            DType::U8 => "uint8".to_string(),
            DType::F16 => "float16".to_string(),
            DType::Bool => "bool".to_string(),
            _ => "float32".to_string(), // fallback
        }
    }

    /// Convert from ToRSh DType to PyTorch dtype string
    pub fn torsh_to_pytorch(&self, dtype: DType) -> String {
        match dtype {
            DType::F32 => "torch.float32".to_string(),
            DType::F64 => "torch.float64".to_string(),
            DType::I32 => "torch.int32".to_string(),
            DType::I64 => "torch.int64".to_string(),
            DType::I16 => "torch.int16".to_string(),
            DType::I8 => "torch.int8".to_string(),
            DType::U8 => "torch.uint8".to_string(),
            DType::F16 => "torch.float16".to_string(),
            DType::Bool => "torch.bool".to_string(),
            _ => "torch.float32".to_string(), // fallback
        }
    }

    /// Get element size for a given dtype
    pub fn element_size(&self, dtype: DType) -> usize {
        dtype.size() // Use the built-in size method from DType
    }

    /// Check if two dtypes are compatible for operations
    pub fn are_compatible(&self, dtype1: DType, dtype2: DType) -> bool {
        // Simple type compatibility logic - most types can work together
        matches!(
            (dtype1, dtype2),
            (DType::F32, DType::F64)
                | (DType::F64, DType::F32)
                | (DType::F32, DType::F32)
                | (DType::F64, DType::F64)
                | (DType::I32, DType::F32)
                | (DType::F32, DType::I32)
                | (DType::I32, DType::F64)
                | (DType::F64, DType::I32)
                | (DType::I32, DType::I32)
                | (DType::I64, DType::I64)
                | (DType::U32, DType::U32)
                | (DType::U64, DType::U64)
        )
    }

    /// Promote dtypes to a common type for operations
    pub fn promote_dtypes(&self, dtype1: DType, dtype2: DType) -> DType {
        // Simple promotion: prefer floating point, then larger types
        match (dtype1, dtype2) {
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            (DType::I64, _) | (_, DType::I64) => DType::I64,
            (DType::U64, _) | (_, DType::U64) => DType::U64,
            (DType::I32, _) | (_, DType::I32) => DType::I32,
            (DType::U32, _) | (_, DType::U32) => DType::U32,
            _ => dtype1, // Default to first type
        }
    }

    /// Get detailed type information
    pub fn get_type_info(&self, dtype: DType) -> FrameworkTypeInfo {
        FrameworkTypeInfo {
            torsh_dtype: dtype,
            numpy_dtype: self.torsh_to_numpy(dtype),
            pytorch_dtype: self.torsh_to_pytorch(dtype),
            element_size: dtype.size(),
            is_floating: dtype.is_float(),
            is_signed: match dtype {
                DType::U8 => false,
                DType::Bool => false,
                _ => true,
            },
        }
    }
}

impl Default for TypeMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Global type mapper instance
pub static TYPE_MAPPER: Lazy<TypeMapper> = Lazy::new(TypeMapper::default);
