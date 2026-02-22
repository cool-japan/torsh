//! Unified type system for FFI bindings
//!
//! This module provides a centralized type system to reduce code duplication
//! across different language bindings and ensure consistent type handling.
//!
//! # Goals
//!
//! - **Consistency**: Single source of truth for type conversions
//! - **Maintainability**: Reduce duplication across language bindings
//! - **Type Safety**: Strong typing with compile-time checks
//! - **Performance**: Zero-cost abstractions where possible
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │         Language-Specific Types                 │
//! │  Python │ Java │ C# │ Go │ Swift │ WASM │ ...   │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//! ┌───────────────────▼─────────────────────────────┐
//! │         Unified Type System (this module)       │
//! │  • Common traits  • Type conversions            │
//! │  • Error handling • Memory management           │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//! ┌───────────────────▼─────────────────────────────┐
//! │              ToRSh Core Types                   │
//! │  Tensor │ DType │ Device │ Shape │ Storage      │
//! └─────────────────────────────────────────────────┘
//! ```

use crate::error::{FfiError, FfiResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unified data type enumeration for cross-language compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnifiedDType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit unsigned integer
    UInt16,
    /// Boolean (1 byte)
    Bool,
    /// 32-bit complex number
    Complex32,
    /// 64-bit complex number
    Complex64,
}

impl UnifiedDType {
    /// Get size in bytes for this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
            Self::Int16 => 2,
            Self::UInt16 => 2,
            Self::Bool => 1,
            Self::Complex32 => 8,
            Self::Complex64 => 16,
        }
    }

    /// Check if this is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Int32 | Self::Int64 | Self::Int16 | Self::UInt8 | Self::UInt16
        )
    }

    /// Check if this is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }

    /// Get the corresponding floating point type (for promotion)
    pub fn to_floating_point(&self) -> Self {
        match self {
            Self::Float32 | Self::Int32 | Self::Int16 | Self::UInt8 | Self::UInt16 | Self::Bool => {
                Self::Float32
            }
            Self::Float64 | Self::Int64 => Self::Float64,
            Self::Complex32 => Self::Complex32,
            Self::Complex64 => Self::Complex64,
        }
    }
}

impl fmt::Display for UnifiedDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "float32"),
            Self::Float64 => write!(f, "float64"),
            Self::Int32 => write!(f, "int32"),
            Self::Int64 => write!(f, "int64"),
            Self::UInt8 => write!(f, "uint8"),
            Self::Int16 => write!(f, "int16"),
            Self::UInt16 => write!(f, "uint16"),
            Self::Bool => write!(f, "bool"),
            Self::Complex32 => write!(f, "complex32"),
            Self::Complex64 => write!(f, "complex64"),
        }
    }
}

/// Unified device enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnifiedDevice {
    /// CPU device
    Cpu,
    /// CUDA GPU device with index
    Cuda(usize),
    /// Metal GPU device (Apple Silicon)
    Metal(usize),
    /// Vulkan device
    Vulkan(usize),
    /// WebGPU device (for WASM)
    WebGpu(usize),
}

impl UnifiedDevice {
    /// Check if device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Check if device is GPU (any type)
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Get device index (0 for CPU)
    pub fn index(&self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Cuda(i) | Self::Metal(i) | Self::Vulkan(i) | Self::WebGpu(i) => *i,
        }
    }
}

impl fmt::Display for UnifiedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(i) => write!(f, "cuda:{}", i),
            Self::Metal(i) => write!(f, "metal:{}", i),
            Self::Vulkan(i) => write!(f, "vulkan:{}", i),
            Self::WebGpu(i) => write!(f, "webgpu:{}", i),
        }
    }
}

impl Default for UnifiedDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Unified shape representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnifiedShape {
    dims: Vec<usize>,
}

impl UnifiedShape {
    /// Create new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get dimensions as slice
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Check if shape is scalar (0 dimensions)
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Check if shapes are broadcastable
    pub fn is_broadcastable_with(&self, other: &Self) -> bool {
        let max_ndim = self.ndim().max(other.ndim());

        for i in 0..max_ndim {
            let dim1 = self.dims.get(self.ndim().saturating_sub(i + 1));
            let dim2 = other.dims.get(other.ndim().saturating_sub(i + 1));

            match (dim1, dim2) {
                (Some(&d1), Some(&d2)) if d1 != d2 && d1 != 1 && d2 != 1 => return false,
                _ => {}
            }
        }

        true
    }

    /// Compute broadcast shape with another shape
    pub fn broadcast_shape(&self, other: &Self) -> FfiResult<Self> {
        if !self.is_broadcastable_with(other) {
            return Err(FfiError::ShapeMismatch {
                expected: self.dims.clone(),
                actual: other.dims.clone(),
            });
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = self.dims.get(self.ndim().saturating_sub(i + 1)).copied();
            let dim2 = other.dims.get(other.ndim().saturating_sub(i + 1)).copied();

            let result_dim = match (dim1, dim2) {
                (Some(d1), Some(d2)) => d1.max(d2),
                (Some(d), None) | (None, Some(d)) => d,
                (None, None) => unreachable!(),
            };

            result_dims.push(result_dim);
        }

        result_dims.reverse();
        Ok(Self::new(result_dims))
    }
}

impl From<Vec<usize>> for UnifiedShape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for UnifiedShape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl fmt::Display for UnifiedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// Unified tensor metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedTensorMetadata {
    pub dtype: UnifiedDType,
    pub shape: UnifiedShape,
    pub device: UnifiedDevice,
    pub requires_grad: bool,
    pub is_leaf: bool,
}

impl UnifiedTensorMetadata {
    /// Create new metadata
    pub fn new(
        dtype: UnifiedDType,
        shape: UnifiedShape,
        device: UnifiedDevice,
        requires_grad: bool,
    ) -> Self {
        Self {
            dtype,
            shape,
            device,
            requires_grad,
            is_leaf: true,
        }
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Check compatibility with another metadata
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.shape == other.shape && self.device == other.device
    }
}

/// Type conversion trait for language bindings
pub trait TypeConverter {
    /// Target language type
    type LanguageType;

    /// Convert from unified dtype to language-specific type
    fn from_unified_dtype(dtype: UnifiedDType) -> Self::LanguageType;

    /// Convert from language-specific type to unified dtype
    fn to_unified_dtype(lang_type: Self::LanguageType) -> FfiResult<UnifiedDType>;

    /// Convert from unified device to language-specific device
    fn from_unified_device(device: UnifiedDevice) -> Self::LanguageType;

    /// Convert from language-specific device to unified device
    fn to_unified_device(lang_device: Self::LanguageType) -> FfiResult<UnifiedDevice>;
}

/// Memory layout for tensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Row-major (C-style) contiguous layout
    RowMajor,
    /// Column-major (Fortran-style) contiguous layout
    ColumnMajor,
    /// Strided (non-contiguous) layout
    Strided,
}

impl MemoryLayout {
    /// Check if layout is contiguous
    pub fn is_contiguous(&self) -> bool {
        matches!(self, Self::RowMajor | Self::ColumnMajor)
    }
}

/// Strides information for tensor indexing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Strides {
    strides: Vec<isize>,
}

impl Strides {
    /// Create strides from shape for row-major layout
    pub fn from_shape_row_major(shape: &UnifiedShape) -> Self {
        let dims = shape.dims();
        if dims.is_empty() {
            return Self {
                strides: Vec::new(),
            };
        }

        let mut strides = vec![1isize; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1] as isize;
        }

        Self { strides }
    }

    /// Create strides from shape for column-major layout
    pub fn from_shape_column_major(shape: &UnifiedShape) -> Self {
        let dims = shape.dims();
        if dims.is_empty() {
            return Self {
                strides: Vec::new(),
            };
        }

        let mut strides = vec![1isize; dims.len()];
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }

        Self { strides }
    }

    /// Get strides as slice
    pub fn as_slice(&self) -> &[isize] {
        &self.strides
    }

    /// Compute linear index from multi-dimensional index
    pub fn compute_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.strides.len(),
            "Index dimensions must match strides"
        );

        let mut offset = 0isize;
        for (idx, stride) in indices.iter().zip(self.strides.iter()) {
            offset += *idx as isize * stride;
        }

        offset as usize
    }
}

/// Common tensor operations trait
pub trait TensorOps {
    /// Get tensor metadata
    fn metadata(&self) -> &UnifiedTensorMetadata;

    /// Get tensor shape
    fn shape(&self) -> &UnifiedShape {
        &self.metadata().shape
    }

    /// Get tensor dtype
    fn dtype(&self) -> UnifiedDType {
        self.metadata().dtype
    }

    /// Get tensor device
    fn device(&self) -> UnifiedDevice {
        self.metadata().device
    }

    /// Get number of elements
    fn numel(&self) -> usize {
        self.metadata().numel()
    }

    /// Check if requires gradient
    fn requires_grad(&self) -> bool {
        self.metadata().requires_grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_dtype_size() {
        assert_eq!(UnifiedDType::Float32.size_bytes(), 4);
        assert_eq!(UnifiedDType::Float64.size_bytes(), 8);
        assert_eq!(UnifiedDType::Int32.size_bytes(), 4);
        assert_eq!(UnifiedDType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_unified_dtype_properties() {
        assert!(UnifiedDType::Float32.is_floating_point());
        assert!(UnifiedDType::Int32.is_integer());
        assert!(UnifiedDType::Complex64.is_complex());
        assert!(!UnifiedDType::Float32.is_integer());
    }

    #[test]
    fn test_unified_device() {
        let cpu = UnifiedDevice::Cpu;
        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.index(), 0);

        let cuda = UnifiedDevice::Cuda(1);
        assert!(!cuda.is_cpu());
        assert!(cuda.is_gpu());
        assert_eq!(cuda.index(), 1);
    }

    #[test]
    fn test_unified_shape() {
        let shape = UnifiedShape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert!(!shape.is_scalar());

        let scalar = UnifiedShape::new(vec![]);
        assert!(scalar.is_scalar());
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = UnifiedShape::new(vec![2, 3]);
        let shape2 = UnifiedShape::new(vec![1, 3]);

        assert!(shape1.is_broadcastable_with(&shape2));

        let broadcast = shape1.broadcast_shape(&shape2).unwrap();
        assert_eq!(broadcast.dims(), &[2, 3]);
    }

    #[test]
    fn test_strides_row_major() {
        let shape = UnifiedShape::new(vec![2, 3, 4]);
        let strides = Strides::from_shape_row_major(&shape);

        assert_eq!(strides.as_slice(), &[12, 4, 1]);

        let offset = strides.compute_offset(&[1, 2, 3]);
        assert_eq!(offset, 1 * 12 + 2 * 4 + 3 * 1);
    }

    #[test]
    fn test_strides_column_major() {
        let shape = UnifiedShape::new(vec![2, 3, 4]);
        let strides = Strides::from_shape_column_major(&shape);

        assert_eq!(strides.as_slice(), &[1, 2, 6]);

        let offset = strides.compute_offset(&[1, 2, 3]);
        assert_eq!(offset, 1 * 1 + 2 * 2 + 3 * 6);
    }

    #[test]
    fn test_tensor_metadata() {
        let metadata = UnifiedTensorMetadata::new(
            UnifiedDType::Float32,
            UnifiedShape::new(vec![2, 3]),
            UnifiedDevice::Cpu,
            true,
        );

        assert_eq!(metadata.numel(), 6);
        assert_eq!(metadata.size_bytes(), 24); // 6 * 4 bytes
        assert!(metadata.requires_grad);
    }
}
