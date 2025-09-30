//! Interoperability traits and utilities for ToRSh
//!
//! This module provides conversion traits and utilities for interoperating
//! with other tensor libraries and data formats, including NumPy arrays,
//! ndarray, Apache Arrow, and standard Rust types.

use crate::{DType, Result, Shape, TorshError};
use std::collections::HashMap;

/// Trait for converting from external tensor formats to ToRSh types
pub trait FromExternal<T> {
    /// Convert from an external type to a ToRSh type
    fn from_external(value: T) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for converting ToRSh types to external tensor formats
pub trait ToExternal<T> {
    /// Convert a ToRSh type to an external type
    fn to_external(&self) -> Result<T>;
}

/// Trait for zero-copy conversion from external types when possible
pub trait FromExternalZeroCopy<T> {
    /// Attempt zero-copy conversion, falling back to copy if necessary
    fn from_external_zero_copy(value: T) -> Result<Self>
    where
        Self: Sized;

    /// Check if zero-copy conversion is possible for the given value
    fn can_zero_copy(value: &T) -> bool;
}

/// Trait for zero-copy conversion to external types when possible
pub trait ToExternalZeroCopy<T> {
    /// Attempt zero-copy conversion, falling back to copy if necessary
    fn to_external_zero_copy(&self) -> Result<T>;

    /// Check if zero-copy conversion is possible
    fn can_zero_copy(&self) -> bool;
}

/// NumPy-compatible array metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumpyArrayInfo {
    /// Array shape
    pub shape: Vec<usize>,
    /// Array strides in bytes
    pub strides: Vec<isize>,
    /// Data type
    pub dtype: DType,
    /// Whether the array is C-contiguous
    pub c_contiguous: bool,
    /// Whether the array is Fortran-contiguous
    pub f_contiguous: bool,
    /// Total size in bytes
    pub nbytes: usize,
}

impl NumpyArrayInfo {
    /// Create new NumPy array info
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let strides = Self::compute_c_strides(&shape, dtype.size());
        let nbytes = shape.iter().product::<usize>() * dtype.size();

        Self {
            c_contiguous: true,
            f_contiguous: shape.len() <= 1,
            shape,
            strides,
            dtype,
            nbytes,
        }
    }

    /// Create NumPy array info with custom strides
    pub fn with_strides(shape: Vec<usize>, strides: Vec<isize>, dtype: DType) -> Self {
        let nbytes = shape.iter().product::<usize>() * dtype.size();
        let c_strides = Self::compute_c_strides(&shape, dtype.size());
        let f_strides = Self::compute_f_strides(&shape, dtype.size());

        Self {
            shape,
            strides: strides.clone(),
            dtype,
            c_contiguous: strides == c_strides,
            f_contiguous: strides == f_strides,
            nbytes,
        }
    }

    /// Compute C-contiguous strides
    fn compute_c_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
        let mut strides = vec![0; shape.len()];
        if !shape.is_empty() {
            let mut stride = itemsize as isize;
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride *= shape[i] as isize;
            }
        }
        strides
    }

    /// Compute Fortran-contiguous strides
    fn compute_f_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
        let mut strides = vec![0; shape.len()];
        if !shape.is_empty() {
            let mut stride = itemsize as isize;
            for i in 0..shape.len() {
                strides[i] = stride;
                stride *= shape[i] as isize;
            }
        }
        strides
    }
}

/// ONNX tensor type information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxTensorInfo {
    /// Element type
    pub elem_type: OnnxDataType,
    /// Shape (None for unknown dimensions)
    pub shape: Vec<Option<usize>>,
    /// Optional name
    pub name: Option<String>,
}

/// ONNX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxDataType {
    /// Undefined
    Undefined = 0,
    /// 32-bit floating point
    Float = 1,
    /// 8-bit unsigned integer
    Uint8 = 2,
    /// 8-bit signed integer
    Int8 = 3,
    /// 16-bit unsigned integer
    Uint16 = 4,
    /// 16-bit signed integer
    Int16 = 5,
    /// 32-bit signed integer
    Int32 = 6,
    /// 64-bit signed integer
    Int64 = 7,
    /// String
    String = 8,
    /// Boolean
    Bool = 9,
    /// 16-bit floating point
    Float16 = 10,
    /// 64-bit floating point
    Double = 11,
    /// 32-bit unsigned integer
    Uint32 = 12,
    /// 64-bit unsigned integer
    Uint64 = 13,
    /// Complex 64-bit
    Complex64 = 14,
    /// Complex 128-bit
    Complex128 = 15,
    /// Brain floating point 16-bit
    Bfloat16 = 16,
}

impl From<DType> for OnnxDataType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => OnnxDataType::Float,
            DType::F64 => OnnxDataType::Double,
            DType::F16 => OnnxDataType::Float16,
            DType::BF16 => OnnxDataType::Bfloat16,
            DType::I8 => OnnxDataType::Int8,
            DType::U8 => OnnxDataType::Uint8,
            DType::I16 => OnnxDataType::Int16,
            DType::I32 => OnnxDataType::Int32,
            DType::I64 => OnnxDataType::Int64,
            DType::U32 => OnnxDataType::Uint32,
            DType::U64 => OnnxDataType::Uint64,
            DType::Bool => OnnxDataType::Bool,
            DType::C64 => OnnxDataType::Complex64,
            DType::C128 => OnnxDataType::Complex128,
            DType::QInt8 => OnnxDataType::Int8, // Quantized types map to base types
            DType::QUInt8 => OnnxDataType::Uint8,
        }
    }
}

impl TryFrom<OnnxDataType> for DType {
    type Error = TorshError;

    fn try_from(onnx_type: OnnxDataType) -> Result<Self> {
        match onnx_type {
            OnnxDataType::Float => Ok(DType::F32),
            OnnxDataType::Double => Ok(DType::F64),
            OnnxDataType::Float16 => Ok(DType::F16),
            OnnxDataType::Bfloat16 => Ok(DType::BF16),
            OnnxDataType::Int8 => Ok(DType::I8),
            OnnxDataType::Uint8 => Ok(DType::U8),
            OnnxDataType::Int16 => Ok(DType::I16),
            OnnxDataType::Int32 => Ok(DType::I32),
            OnnxDataType::Int64 => Ok(DType::I64),
            OnnxDataType::Bool => Ok(DType::Bool),
            OnnxDataType::Complex64 => Ok(DType::C64),
            OnnxDataType::Complex128 => Ok(DType::C128),
            _ => Err(TorshError::UnsupportedOperation {
                op: "ONNX data type conversion".to_string(),
                dtype: format!("{onnx_type:?}"),
            }),
        }
    }
}

/// Apache Arrow type information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowTypeInfo {
    /// Arrow data type
    pub data_type: ArrowDataType,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Simplified Arrow data types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrowDataType {
    /// Boolean
    Boolean,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 16-bit floating point
    Float16,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Fixed-size list
    FixedSizeList(Box<ArrowDataType>, usize),
}

impl From<DType> for ArrowDataType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::Bool => ArrowDataType::Boolean,
            DType::I8 | DType::QInt8 => ArrowDataType::Int8,
            DType::U8 | DType::QUInt8 => ArrowDataType::UInt8,
            DType::I16 => ArrowDataType::Int16,
            DType::I32 => ArrowDataType::Int32,
            DType::I64 => ArrowDataType::Int64,
            DType::U32 => ArrowDataType::UInt32,
            DType::U64 => ArrowDataType::UInt64,
            DType::F16 => ArrowDataType::Float16,
            DType::F32 => ArrowDataType::Float32,
            DType::F64 => ArrowDataType::Float64,
            DType::BF16 => ArrowDataType::Float32, // Best approximation
            DType::C64 => ArrowDataType::FixedSizeList(Box::new(ArrowDataType::Float32), 2),
            DType::C128 => ArrowDataType::FixedSizeList(Box::new(ArrowDataType::Float64), 2),
        }
    }
}

/// Conversion utilities
pub struct ConversionUtils;

impl ConversionUtils {
    /// Convert ToRSh shape to NumPy shape
    pub fn torsh_shape_to_numpy(shape: &Shape) -> Vec<usize> {
        shape.dims().to_vec()
    }

    /// Convert NumPy shape to ToRSh shape
    pub fn numpy_shape_to_torsh(shape: Vec<usize>) -> Result<Shape> {
        Ok(Shape::new(shape))
    }

    /// Check if two arrays are memory layout compatible
    pub fn is_layout_compatible(
        shape1: &[usize],
        strides1: &[isize],
        shape2: &[usize],
        strides2: &[isize],
    ) -> bool {
        if shape1.len() != shape2.len() || shape1 != shape2 {
            return false;
        }

        strides1 == strides2
    }

    /// Compute memory layout efficiency score (0.0 to 1.0)
    pub fn layout_efficiency_score(shape: &[usize], strides: &[isize], itemsize: usize) -> f64 {
        if shape.is_empty() {
            return 1.0;
        }

        // Check if it's C-contiguous
        let c_strides = NumpyArrayInfo::compute_c_strides(shape, itemsize);
        if strides == c_strides {
            return 1.0;
        }

        // Check if it's Fortran-contiguous
        let f_strides = NumpyArrayInfo::compute_f_strides(shape, itemsize);
        if strides == f_strides {
            return 0.9;
        }

        // Compute efficiency based on stride patterns
        let total_elements: usize = shape.iter().product();
        let expected_size = total_elements * itemsize;
        let actual_span = Self::compute_memory_span(shape, strides, itemsize);

        if actual_span == 0 {
            return 0.0;
        }

        (expected_size as f64 / actual_span as f64).min(1.0)
    }

    /// Compute the span of memory used by an array
    fn compute_memory_span(shape: &[usize], strides: &[isize], itemsize: usize) -> usize {
        if shape.is_empty() {
            return 0;
        }

        let mut min_offset = 0isize;
        let mut max_offset = 0isize;

        for (&dim, &stride) in shape.iter().zip(strides.iter()) {
            if dim > 1 {
                let offset = stride * (dim as isize - 1);
                min_offset = min_offset.min(offset);
                max_offset = max_offset.max(offset);
            }
        }

        (max_offset - min_offset) as usize + itemsize
    }
}

/// Documentation utilities for the interop module
pub struct InteropDocs;

impl InteropDocs {
    /// Generate documentation for supported conversions
    pub fn supported_conversions() -> String {
        let conversions = vec![
            ("NumPy", "ndarray", "Zero-copy when C-contiguous"),
            ("ndarray", "Array", "Zero-copy when contiguous"),
            ("ONNX", "TensorProto", "Schema mapping"),
            ("Arrow", "Array", "Type mapping with metadata"),
            ("Rust", "Vec<T>", "Direct conversion"),
        ];

        let mut doc = String::from("Supported Tensor Format Conversions:\n");
        doc.push_str("=========================================\n\n");

        for (from, to, notes) in conversions {
            doc.push_str(&format!("• {from} ↔ {to}: {notes}\n"));
        }

        doc
    }

    /// Generate examples for common conversion patterns
    pub fn conversion_examples() -> String {
        r#"
Conversion Examples:
==================

// NumPy-style array info
let numpy_info = NumpyArrayInfo::new(vec![2, 3, 4], DType::F32);
assert!(numpy_info.c_contiguous);

// ONNX type conversion
let onnx_type = OnnxDataType::from(DType::F32);
let back_to_dtype = DType::try_from(onnx_type).unwrap();

// Arrow type conversion
let arrow_type = ArrowDataType::from(DType::C64);
match arrow_type {
    ArrowDataType::FixedSizeList(inner, size) => {
        assert_eq!(size, 2); // Real and imaginary parts
    }
    _ => panic!("Unexpected type"),
}

// Layout efficiency checking
let shape = vec![1000, 1000];
let c_strides = vec![4000, 4]; // C-contiguous for f32
let efficiency = ConversionUtils::layout_efficiency_score(&shape, &c_strides, 4);
assert_eq!(efficiency, 1.0);
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numpy_array_info() {
        let info = NumpyArrayInfo::new(vec![2, 3, 4], DType::F32);
        assert_eq!(info.shape, vec![2, 3, 4]);
        assert_eq!(info.strides, vec![48, 16, 4]); // C-contiguous strides for f32
        assert!(info.c_contiguous);
        assert!(!info.f_contiguous);
        assert_eq!(info.nbytes, 96); // 2*3*4*4 bytes
    }

    #[test]
    fn test_onnx_dtype_conversion() {
        // Test round-trip conversion
        let dtypes = vec![
            DType::F32,
            DType::F64,
            DType::I8,
            DType::U8,
            DType::I32,
            DType::Bool,
            DType::C64,
        ];

        for dtype in dtypes {
            let onnx_type = OnnxDataType::from(dtype);
            let back_to_dtype = DType::try_from(onnx_type).unwrap();
            assert_eq!(dtype, back_to_dtype);
        }
    }

    #[test]
    fn test_arrow_dtype_conversion() {
        assert_eq!(ArrowDataType::from(DType::F32), ArrowDataType::Float32);
        assert_eq!(ArrowDataType::from(DType::Bool), ArrowDataType::Boolean);

        // Test complex types
        match ArrowDataType::from(DType::C64) {
            ArrowDataType::FixedSizeList(inner, size) => {
                assert_eq!(*inner, ArrowDataType::Float32);
                assert_eq!(size, 2);
            }
            _ => panic!("Expected FixedSizeList for C64"),
        }
    }

    #[test]
    fn test_layout_efficiency() {
        let shape = vec![10, 10];
        let itemsize = 4;

        // C-contiguous (perfect efficiency)
        let c_strides = vec![40, 4];
        let efficiency = ConversionUtils::layout_efficiency_score(&shape, &c_strides, itemsize);
        assert_eq!(efficiency, 1.0);

        // F-contiguous (very good efficiency)
        let f_strides = vec![4, 40];
        let efficiency = ConversionUtils::layout_efficiency_score(&shape, &f_strides, itemsize);
        assert_eq!(efficiency, 0.9);
    }

    #[test]
    fn test_conversion_utils() {
        let shape = Shape::new(vec![2, 3, 4]);
        let numpy_shape = ConversionUtils::torsh_shape_to_numpy(&shape);
        assert_eq!(numpy_shape, vec![2, 3, 4]);

        let back_to_shape = ConversionUtils::numpy_shape_to_torsh(numpy_shape).unwrap();
        assert_eq!(shape.dims(), back_to_shape.dims());
    }

    #[test]
    fn test_layout_compatibility() {
        let shape1 = vec![2, 3];
        let strides1 = vec![12, 4];
        let shape2 = vec![2, 3];
        let strides2 = vec![12, 4];

        assert!(ConversionUtils::is_layout_compatible(
            &shape1, &strides1, &shape2, &strides2
        ));

        let strides3 = vec![4, 8]; // Different strides
        assert!(!ConversionUtils::is_layout_compatible(
            &shape1, &strides1, &shape2, &strides3
        ));
    }
}
