//! Data type handling for Python bindings

use crate::error::PyResult;
use pyo3::prelude::*;
use torsh_core::dtype::DType;

/// Python wrapper for ToRSh data types
#[pyclass(name = "dtype")]
#[derive(Clone, Debug)]
pub struct PyDType {
    pub(crate) dtype: DType,
}

#[pymethods]
impl PyDType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let dtype = match name {
            "float32" | "f32" => DType::F32,
            "float64" | "f64" => DType::F64,
            "int8" | "i8" => DType::I8,
            "int16" | "i16" => DType::I16,
            "int32" | "i32" => DType::I32,
            "int64" | "i64" => DType::I64,
            "uint8" | "u8" => DType::U8,
            "uint16" | "u16" => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "uint16/u16 data type is not supported in ToRSh",
                ))
            }
            "uint32" | "u32" => DType::U32,
            "uint64" | "u64" => DType::U64,
            "bool" => DType::Bool,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown dtype: {}",
                    name
                )))
            }
        };
        Ok(Self { dtype })
    }

    fn __str__(&self) -> String {
        match self.dtype {
            DType::F32 => "torch.float32".to_string(),
            DType::F64 => "torch.float64".to_string(),
            DType::I8 => "torch.int8".to_string(),
            DType::I16 => "torch.int16".to_string(),
            DType::I32 => "torch.int32".to_string(),
            DType::I64 => "torch.int64".to_string(),
            DType::U8 => "torch.uint8".to_string(),
            DType::U32 => "torch.uint32".to_string(),
            DType::U64 => "torch.uint64".to_string(),
            DType::Bool => "torch.bool".to_string(),
            DType::F16 => "torch.float16".to_string(),
            DType::BF16 => "torch.bfloat16".to_string(),
            DType::C64 => "torch.complex64".to_string(),
            DType::C128 => "torch.complex128".to_string(),
            DType::QInt8 => "torch.qint8".to_string(),
            _ => format!("torch.{:?}", self.dtype).to_lowercase(),
        }
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __eq__(&self, other: &PyDType) -> bool {
        self.dtype == other.dtype
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.dtype.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn name(&self) -> String {
        match self.dtype {
            DType::F32 => "float32".to_string(),
            DType::F64 => "float64".to_string(),
            DType::I8 => "int8".to_string(),
            DType::I16 => "int16".to_string(),
            DType::I32 => "int32".to_string(),
            DType::I64 => "int64".to_string(),
            DType::U8 => "uint8".to_string(),
            DType::U32 => "uint32".to_string(),
            DType::U64 => "uint64".to_string(),
            DType::Bool => "bool".to_string(),
            DType::F16 => "float16".to_string(),
            DType::BF16 => "bfloat16".to_string(),
            DType::C64 => "complex64".to_string(),
            DType::C128 => "complex128".to_string(),
            DType::QInt8 => "qint8".to_string(),
            _ => format!("{:?}", self.dtype).to_lowercase(),
        }
    }

    #[getter]
    fn itemsize(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Bool => 1,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::C64 => 8,
            DType::C128 => 16,
            DType::QInt8 => 1,
            _ => 4, // Default to 4 bytes for unknown types
        }
    }

    #[getter]
    fn is_floating_point(&self) -> bool {
        matches!(
            self.dtype,
            DType::F16 | DType::F32 | DType::F64 | DType::BF16
        )
    }

    #[getter]
    fn is_signed(&self) -> bool {
        matches!(
            self.dtype,
            DType::F16
                | DType::F32
                | DType::F64
                | DType::BF16
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::QInt8
        )
    }

    /// Check if this is a complex dtype.
    ///
    /// # Returns
    ///
    /// True if dtype is complex (complex64, complex128), False otherwise
    ///
    /// # Examples
    ///
    /// ```python
    /// float32 = torsh.PyDType("float32")
    /// print(float32.is_complex)  # False
    /// ```
    #[getter]
    fn is_complex(&self) -> bool {
        matches!(self.dtype, DType::C64 | DType::C128)
    }

    /// Check if this is an integer dtype.
    ///
    /// # Returns
    ///
    /// True if dtype is integer (int8, int16, int32, int64, uint8, etc.), False otherwise
    ///
    /// # Examples
    ///
    /// ```python
    /// int32 = torsh.PyDType("int32")
    /// print(int32.is_integer)  # True
    ///
    /// float32 = torsh.PyDType("float32")
    /// print(float32.is_integer)  # False
    /// ```
    #[getter]
    fn is_integer(&self) -> bool {
        matches!(
            self.dtype,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U32 | DType::U64
        )
    }

    /// Get the NumPy-compatible dtype string.
    ///
    /// # Returns
    ///
    /// NumPy dtype string (e.g., 'float32', 'int64')
    ///
    /// # Examples
    ///
    /// ```python
    /// dtype = torsh.PyDType("float32")
    /// print(dtype.numpy_dtype)  # 'float32'
    /// ```
    #[getter]
    fn numpy_dtype(&self) -> String {
        match self.dtype {
            DType::F32 => "float32".to_string(),
            DType::F64 => "float64".to_string(),
            DType::F16 => "float16".to_string(),
            DType::I8 => "int8".to_string(),
            DType::I16 => "int16".to_string(),
            DType::I32 => "int32".to_string(),
            DType::I64 => "int64".to_string(),
            DType::U8 => "uint8".to_string(),
            DType::U32 => "uint32".to_string(),
            DType::U64 => "uint64".to_string(),
            DType::Bool => "bool".to_string(),
            DType::C64 => "complex64".to_string(),
            DType::C128 => "complex128".to_string(),
            _ => format!("{:?}", self.dtype).to_lowercase(),
        }
    }

    /// Check if this dtype can be safely cast to another dtype.
    ///
    /// # Arguments
    ///
    /// * `other` - Target dtype to check casting compatibility
    ///
    /// # Returns
    ///
    /// True if safe cast is possible, False otherwise
    ///
    /// # Examples
    ///
    /// ```python
    /// int32 = torsh.PyDType("int32")
    /// int64 = torsh.PyDType("int64")
    /// float32 = torsh.PyDType("float32")
    ///
    /// print(int32.can_cast(int64))    # True (widening)
    /// print(int64.can_cast(int32))    # False (narrowing)
    /// print(int32.can_cast(float32))  # True (int to float)
    /// ```
    fn can_cast(&self, other: &PyDType) -> bool {
        // Same type is always safe
        if self.dtype == other.dtype {
            return true;
        }

        // Casting rules based on type promotion
        match (self.dtype, other.dtype) {
            // Integer widening is safe
            (DType::I8, DType::I16 | DType::I32 | DType::I64) => true,
            (DType::I16, DType::I32 | DType::I64) => true,
            (DType::I32, DType::I64) => true,

            // Unsigned integer widening is safe
            (DType::U8, DType::U32 | DType::U64) => true,
            (DType::U32, DType::U64) => true,

            // Integer to float is generally safe (may lose precision for large integers)
            (DType::I8 | DType::I16 | DType::I32, DType::F32 | DType::F64) => true,
            (DType::I64, DType::F64) => true,
            (DType::U8 | DType::U32, DType::F32 | DType::F64) => true,

            // Float widening is safe
            (DType::F16, DType::F32 | DType::F64) => true,
            (DType::F32, DType::F64) => true,

            // Bool can be cast to any numeric type
            (DType::Bool, DType::I8 | DType::I16 | DType::I32 | DType::I64) => true,
            (DType::Bool, DType::U8 | DType::U32 | DType::U64) => true,
            (DType::Bool, DType::F16 | DType::F32 | DType::F64) => true,

            // Float to complex
            (DType::F32, DType::C64 | DType::C128) => true,
            (DType::F64, DType::C128) => true,

            // Complex widening
            (DType::C64, DType::C128) => true,

            // Everything else is not safe
            _ => false,
        }
    }
}

impl From<DType> for PyDType {
    fn from(dtype: DType) -> Self {
        Self { dtype }
    }
}

impl From<PyDType> for DType {
    fn from(py_dtype: PyDType) -> Self {
        py_dtype.dtype
    }
}

impl std::fmt::Display for PyDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

/// Register dtype constants and utility functions with the module.
///
/// This function adds:
/// - Dtype constants (float32, int64, etc.)
/// - PyTorch-style aliases (float, double, long, etc.)
/// - Utility functions for dtype operations
pub fn register_dtype_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;

    // Create dtype constants similar to PyTorch
    m.add("float32", PyDType { dtype: DType::F32 })?;
    m.add("float64", PyDType { dtype: DType::F64 })?;
    m.add("int8", PyDType { dtype: DType::I8 })?;
    m.add("int16", PyDType { dtype: DType::I16 })?;
    m.add("int32", PyDType { dtype: DType::I32 })?;
    m.add("int64", PyDType { dtype: DType::I64 })?;
    m.add("uint8", PyDType { dtype: DType::U8 })?;
    m.add("uint32", PyDType { dtype: DType::U32 })?;
    m.add("uint64", PyDType { dtype: DType::U64 })?;
    m.add("bool", PyDType { dtype: DType::Bool })?;

    // PyTorch-style aliases
    m.add("float", PyDType { dtype: DType::F32 })?;
    m.add("double", PyDType { dtype: DType::F64 })?;
    m.add("long", PyDType { dtype: DType::I64 })?;
    m.add("int", PyDType { dtype: DType::I32 })?;
    m.add("short", PyDType { dtype: DType::I16 })?;
    m.add("char", PyDType { dtype: DType::I8 })?;
    m.add("byte", PyDType { dtype: DType::U8 })?;

    // Utility functions
    /// Promote two dtypes to a common dtype for operations.
    ///
    /// # Arguments
    ///
    /// * `dtype1` - First dtype
    /// * `dtype2` - Second dtype
    ///
    /// # Returns
    ///
    /// Promoted dtype that can safely represent both inputs
    ///
    /// # Examples
    ///
    /// ```python
    /// result = torsh.promote_types(torsh.int32, torsh.float32)
    /// print(result)  # float32
    ///
    /// result = torsh.promote_types(torsh.int32, torsh.int64)
    /// print(result)  # int64
    /// ```
    #[pyfunction]
    fn promote_types(dtype1: &PyDType, dtype2: &PyDType) -> PyDType {
        use DType::*;

        // If same type, return it
        if dtype1.dtype == dtype2.dtype {
            return dtype1.clone();
        }

        // Type promotion rules (similar to NumPy/PyTorch)
        let promoted = match (dtype1.dtype, dtype2.dtype) {
            // Bool promotes to anything else
            (Bool, other) | (other, Bool) => other,

            // Complex types take precedence
            (C128, _) | (_, C128) => C128,
            (C64, _) | (_, C64) => C64,

            // Float promotion
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,

            // Integer promotion - signed takes precedence, larger size wins
            (I64, I8 | I16 | I32 | U8 | U32 | U64) | (I8 | I16 | I32 | U8 | U32 | U64, I64) => I64,
            (I32, I8 | I16 | U8) | (I8 | I16 | U8, I32) => I32,
            (I16, I8 | U8) | (I8 | U8, I16) => I16,

            // Unsigned integer promotion
            (U64, U8 | U32) | (U8 | U32, U64) => U64,
            (U32, U8) | (U8, U32) => U32,

            // Default to the larger type
            (a, b) => {
                let size_a = dtype1.itemsize();
                let size_b = dtype2.itemsize();
                if size_a >= size_b {
                    a
                } else {
                    b
                }
            }
        };

        PyDType { dtype: promoted }
    }

    /// Get the result dtype for a binary operation between two dtypes.
    ///
    /// # Arguments
    ///
    /// * `dtype1` - First operand dtype
    /// * `dtype2` - Second operand dtype
    ///
    /// # Returns
    ///
    /// Result dtype for the operation
    ///
    /// # Examples
    ///
    /// ```python
    /// result = torsh.result_type(torsh.int32, torsh.float32)
    /// print(result)  # float32
    /// ```
    #[pyfunction]
    fn result_type(dtype1: &PyDType, dtype2: &PyDType) -> PyDType {
        // For now, result_type is the same as promote_types
        // In the future, this could have different rules for specific operations
        promote_types(dtype1, dtype2)
    }

    /// Check if two dtypes are compatible for operations.
    ///
    /// # Arguments
    ///
    /// * `dtype1` - First dtype
    /// * `dtype2` - Second dtype
    ///
    /// # Returns
    ///
    /// True if dtypes can be used together in operations
    ///
    /// # Examples
    ///
    /// ```python
    /// print(torsh.can_operate(torsh.int32, torsh.float32))  # True
    /// print(torsh.can_operate(torsh.bool, torsh.int32))     # True
    /// ```
    #[pyfunction]
    fn can_operate(_dtype1: &PyDType, _dtype2: &PyDType) -> bool {
        // Most dtypes can operate together (via promotion)
        // Only complex and non-numeric types might be incompatible
        true
    }

    m.add_function(wrap_pyfunction!(promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    m.add_function(wrap_pyfunction!(can_operate, m)?)?;

    Ok(())
}
