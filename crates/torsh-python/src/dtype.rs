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

/// Register dtype constants with the module
pub fn register_dtype_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    Ok(())
}
