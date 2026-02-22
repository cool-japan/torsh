//! Unified type conversion utilities
//!
//! This module provides centralized conversion utilities to reduce code duplication
//! across different language bindings.
//!
//! # Features
//!
//! - **Consistent Conversions**: Single source of truth for all type conversions
//! - **Error Handling**: Comprehensive error reporting for failed conversions
//! - **Performance**: Zero-copy conversions where possible
//! - **Type Safety**: Compile-time type checking with runtime validation

use crate::error::{FfiError, FfiResult};
use crate::type_system::{UnifiedDType, UnifiedDevice, UnifiedShape};
use torsh_core::DType;

/// Convert ToRSh DType to unified dtype
pub fn torsh_dtype_to_unified(dtype: DType) -> UnifiedDType {
    match dtype {
        DType::F32 => UnifiedDType::Float32,
        DType::F64 => UnifiedDType::Float64,
        DType::I8 | DType::I16 => UnifiedDType::Int16,
        DType::I32 | DType::QInt8 | DType::QUInt8 => UnifiedDType::Int32,
        DType::I64 => UnifiedDType::Int64,
        DType::U8 => UnifiedDType::UInt8,
        DType::U32 | DType::U64 | DType::QInt32 => UnifiedDType::UInt16, // Map to closest supported type
        DType::Bool => UnifiedDType::Bool,
        DType::F16 | DType::BF16 => UnifiedDType::Float32, // Map half precision to float32
        DType::C64 => UnifiedDType::Complex64,
        DType::C128 => UnifiedDType::Complex64, // Map C128 to C64
    }
}

/// Convert unified dtype to ToRSh DType
pub fn unified_to_torsh_dtype(dtype: UnifiedDType) -> FfiResult<DType> {
    match dtype {
        UnifiedDType::Float32 => Ok(DType::F32),
        UnifiedDType::Float64 => Ok(DType::F64),
        UnifiedDType::Int32 => Ok(DType::I32),
        UnifiedDType::Int64 => Ok(DType::I64),
        UnifiedDType::UInt8 => Ok(DType::U8),
        UnifiedDType::Int16 => Ok(DType::I16),
        UnifiedDType::Bool => Ok(DType::Bool),
        UnifiedDType::UInt16 | UnifiedDType::Complex32 | UnifiedDType::Complex64 => {
            Err(FfiError::UnsupportedOperation {
                operation: format!("Type not yet supported in ToRSh: {}", dtype),
            })
        }
    }
}

/// Convert shape slice to unified shape
pub fn shape_to_unified(shape: &[usize]) -> UnifiedShape {
    UnifiedShape::from(shape)
}

/// Convert unified shape to vec
pub fn unified_to_shape_vec(shape: &UnifiedShape) -> Vec<usize> {
    shape.dims().to_vec()
}

/// String parsing utilities for cross-language compatibility
pub mod string_parsing {
    use super::*;

    /// Parse dtype from string (e.g., "float32", "int64")
    pub fn parse_dtype(s: &str) -> FfiResult<UnifiedDType> {
        match s.to_lowercase().as_str() {
            "float32" | "f32" => Ok(UnifiedDType::Float32),
            "float64" | "f64" | "double" => Ok(UnifiedDType::Float64),
            "int32" | "i32" | "int" => Ok(UnifiedDType::Int32),
            "int64" | "i64" | "long" => Ok(UnifiedDType::Int64),
            "uint8" | "u8" | "byte" => Ok(UnifiedDType::UInt8),
            "int16" | "i16" | "short" => Ok(UnifiedDType::Int16),
            "uint16" | "u16" => Ok(UnifiedDType::UInt16),
            "bool" | "boolean" => Ok(UnifiedDType::Bool),
            "complex32" | "c32" => Ok(UnifiedDType::Complex32),
            "complex64" | "c64" => Ok(UnifiedDType::Complex64),
            _ => Err(FfiError::InvalidConversion {
                message: format!("Unknown dtype string: {}", s),
            }),
        }
    }

    /// Parse device from string (e.g., "cpu", "cuda:0", "metal:1")
    pub fn parse_device(s: &str) -> FfiResult<UnifiedDevice> {
        let s = s.to_lowercase();
        if s == "cpu" {
            return Ok(UnifiedDevice::Cpu);
        }

        if let Some(suffix) = s.strip_prefix("cuda:") {
            let index = suffix
                .parse::<usize>()
                .map_err(|_| FfiError::InvalidConversion {
                    message: format!("Invalid CUDA device index: {}", suffix),
                })?;
            return Ok(UnifiedDevice::Cuda(index));
        }

        if let Some(suffix) = s.strip_prefix("metal:") {
            let index = suffix
                .parse::<usize>()
                .map_err(|_| FfiError::InvalidConversion {
                    message: format!("Invalid Metal device index: {}", suffix),
                })?;
            return Ok(UnifiedDevice::Metal(index));
        }

        if let Some(suffix) = s.strip_prefix("vulkan:") {
            let index = suffix
                .parse::<usize>()
                .map_err(|_| FfiError::InvalidConversion {
                    message: format!("Invalid Vulkan device index: {}", suffix),
                })?;
            return Ok(UnifiedDevice::Vulkan(index));
        }

        if let Some(suffix) = s.strip_prefix("webgpu:") {
            let index = suffix
                .parse::<usize>()
                .map_err(|_| FfiError::InvalidConversion {
                    message: format!("Invalid WebGPU device index: {}", suffix),
                })?;
            return Ok(UnifiedDevice::WebGpu(index));
        }

        Err(FfiError::InvalidConversion {
            message: format!("Unknown device string: {}", s),
        })
    }

    /// Parse shape from string (e.g., "[2, 3, 4]" or "2,3,4")
    pub fn parse_shape(s: &str) -> FfiResult<UnifiedShape> {
        let s = s.trim().trim_matches(|c| c == '[' || c == ']');
        let dims: Result<Vec<usize>, _> = s
            .split(',')
            .map(|part| part.trim().parse::<usize>())
            .collect();

        dims.map(UnifiedShape::from)
            .map_err(|_| FfiError::InvalidConversion {
                message: format!("Invalid shape string: {}", s),
            })
    }
}

/// Numeric conversion utilities
pub mod numeric {
    use super::*;

    /// Convert f32 slice to f64 vec
    pub fn f32_to_f64(data: &[f32]) -> Vec<f64> {
        data.iter().map(|&x| x as f64).collect()
    }

    /// Convert f64 slice to f32 vec
    pub fn f64_to_f32(data: &[f64]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }

    /// Convert i32 slice to f32 vec
    pub fn i32_to_f32(data: &[i32]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }

    /// Convert i64 slice to f64 vec
    pub fn i64_to_f64(data: &[i64]) -> Vec<f64> {
        data.iter().map(|&x| x as f64).collect()
    }

    /// Convert u8 slice to f32 vec (for image data)
    pub fn u8_to_f32_normalized(data: &[u8]) -> Vec<f32> {
        data.iter().map(|&x| x as f32 / 255.0).collect()
    }

    /// Convert f32 slice to u8 vec (for image data)
    pub fn f32_to_u8_normalized(data: &[f32]) -> Vec<u8> {
        data.iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }

    /// Generic numeric type casting with overflow checking
    pub fn safe_numeric_cast<T, U>(value: T) -> FfiResult<U>
    where
        T: scirs2_core::numeric::ToPrimitive + std::fmt::Display,
        U: scirs2_core::numeric::FromPrimitive + std::fmt::Display,
    {
        U::from_f64(value.to_f64().ok_or_else(|| FfiError::InvalidConversion {
            message: format!("Failed to convert {} to f64", value),
        })?)
        .ok_or_else(|| FfiError::InvalidConversion {
            message: format!("Numeric overflow or precision loss converting {}", value),
        })
    }
}

/// Buffer conversion utilities
pub mod buffer {
    #![allow(unused_imports)]
    use super::*;

    /// Convert vec to raw pointer and length (for C FFI)
    pub fn vec_to_raw_parts<T>(mut vec: Vec<T>) -> (*mut T, usize, usize) {
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        std::mem::forget(vec); // Prevent deallocation
        (ptr, len, cap)
    }

    /// Reconstruct vec from raw pointer and length (for C FFI cleanup)
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - `ptr` must have been allocated with `Vec::into_raw_parts` or equivalent
    /// - `len` and `cap` must match the original allocation
    /// - The pointer must not be used after calling this function
    pub unsafe fn vec_from_raw_parts<T>(ptr: *mut T, len: usize, cap: usize) -> Vec<T> {
        Vec::from_raw_parts(ptr, len, cap)
    }

    /// Create a boxed slice from a pointer and length
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for reads of `len * std::mem::size_of::<T>()` bytes
    /// - `ptr` must be properly aligned
    /// - The data pointed to must remain valid for the lifetime of the returned slice
    pub unsafe fn slice_from_raw_parts<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Handle management utilities for C API
pub mod handle {
    use parking_lot::RwLock;
    use std::sync::Arc;

    /// Convert a boxed value to an opaque handle
    pub fn box_to_handle<T>(value: T) -> *mut T {
        Box::into_raw(Box::new(value))
    }

    /// Convert an opaque handle back to a box
    ///
    /// # Safety
    ///
    /// - `handle` must have been created with `box_to_handle`
    /// - `handle` must not be used after calling this function
    pub unsafe fn handle_to_box<T>(handle: *mut T) -> Box<T> {
        Box::from_raw(handle)
    }

    /// Borrow from handle without taking ownership
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid pointer to `T`
    /// - The pointer must remain valid for the returned reference lifetime
    pub unsafe fn handle_ref<'a, T>(handle: *const T) -> Option<&'a T> {
        handle.as_ref()
    }

    /// Mutable borrow from handle without taking ownership
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid pointer to `T`
    /// - No other references to the same data must exist
    /// - The pointer must remain valid for the returned reference lifetime
    pub unsafe fn handle_mut<'a, T>(handle: *mut T) -> Option<&'a mut T> {
        handle.as_mut()
    }

    /// Convert Arc to raw pointer (for thread-safe shared ownership)
    pub fn arc_to_handle<T>(arc: Arc<T>) -> *const T {
        Arc::into_raw(arc)
    }

    /// Convert raw pointer back to Arc
    ///
    /// # Safety
    ///
    /// - `ptr` must have been created with `arc_to_handle`
    pub unsafe fn handle_to_arc<T>(ptr: *const T) -> Arc<T> {
        Arc::from_raw(ptr)
    }

    /// Clone Arc from handle (increment reference count)
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid Arc pointer
    pub unsafe fn clone_arc_handle<T>(ptr: *const T) -> Arc<T> {
        let arc = Arc::from_raw(ptr);
        let cloned = Arc::clone(&arc);
        std::mem::forget(arc); // Don't decrement count
        cloned
    }

    /// Convert Arc<RwLock<T>> to handle for thread-safe mutable access
    pub fn arc_rwlock_to_handle<T>(value: Arc<RwLock<T>>) -> *const RwLock<T> {
        Arc::into_raw(value)
    }

    /// Convert handle back to Arc<RwLock<T>>
    ///
    /// # Safety
    ///
    /// - `ptr` must have been created with `arc_rwlock_to_handle`
    pub unsafe fn handle_to_arc_rwlock<T>(ptr: *const RwLock<T>) -> Arc<RwLock<T>> {
        Arc::from_raw(ptr)
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate shape is non-empty and has valid dimensions
    pub fn validate_shape(shape: &[usize]) -> FfiResult<()> {
        if shape.is_empty() {
            return Err(FfiError::InvalidParameter {
                parameter: "shape".to_string(),
                value: "empty".to_string(),
            });
        }

        for (i, &dim) in shape.iter().enumerate() {
            if dim == 0 {
                return Err(FfiError::InvalidParameter {
                    parameter: format!("shape[{}]", i),
                    value: "0".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate data length matches shape
    pub fn validate_data_shape_match(data_len: usize, shape: &[usize]) -> FfiResult<()> {
        let expected_len: usize = shape.iter().product();
        if data_len != expected_len {
            return Err(FfiError::ShapeMismatch {
                expected: vec![expected_len],
                actual: vec![data_len],
            });
        }
        Ok(())
    }

    /// Validate pointer is not null
    pub fn validate_ptr<T>(ptr: *const T, name: &str) -> FfiResult<()> {
        if ptr.is_null() {
            return Err(FfiError::InvalidParameter {
                parameter: name.to_string(),
                value: "null pointer".to_string(),
            });
        }
        Ok(())
    }

    /// Validate numeric parameter is finite
    pub fn validate_finite_f32(value: f32, name: &str) -> FfiResult<()> {
        if !value.is_finite() {
            return Err(FfiError::InvalidParameter {
                parameter: name.to_string(),
                value: format!("{}", value),
            });
        }
        Ok(())
    }

    /// Validate numeric parameter is positive
    pub fn validate_positive_f32(value: f32, name: &str) -> FfiResult<()> {
        if value <= 0.0 {
            return Err(FfiError::InvalidParameter {
                parameter: name.to_string(),
                value: format!("{} (must be positive)", value),
            });
        }
        Ok(())
    }

    /// Validate numeric parameter is in range
    pub fn validate_range_f32(value: f32, min: f32, max: f32, name: &str) -> FfiResult<()> {
        if value < min || value > max {
            return Err(FfiError::InvalidParameter {
                parameter: name.to_string(),
                value: format!("{} (must be in range [{}, {}])", value, min, max),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_dtype_conversions() {
        let torsh_dtype = DType::F32;
        let unified = torsh_dtype_to_unified(torsh_dtype);
        assert_eq!(unified, UnifiedDType::Float32);

        let back = unified_to_torsh_dtype(unified).unwrap();
        assert_eq!(back, torsh_dtype);
    }

    #[test]
    fn test_string_parsing() {
        let dtype = string_parsing::parse_dtype("float32").unwrap();
        assert_eq!(dtype, UnifiedDType::Float32);

        let device = string_parsing::parse_device("cuda:0").unwrap();
        assert_eq!(device, UnifiedDevice::Cuda(0));

        let shape = string_parsing::parse_shape("[2, 3, 4]").unwrap();
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_numeric_conversions() {
        let f32_data = vec![1.0f32, 2.0, 3.0];
        let f64_data = numeric::f32_to_f64(&f32_data);
        assert_eq!(f64_data, vec![1.0f64, 2.0, 3.0]);

        let back = numeric::f64_to_f32(&f64_data);
        assert_eq!(back, f32_data);
    }

    #[test]
    fn test_validation() {
        assert!(validation::validate_shape(&[2, 3, 4]).is_ok());
        assert!(validation::validate_shape(&[]).is_err());
        assert!(validation::validate_shape(&[2, 0, 4]).is_err());

        assert!(validation::validate_data_shape_match(24, &[2, 3, 4]).is_ok());
        assert!(validation::validate_data_shape_match(20, &[2, 3, 4]).is_err());

        assert!(validation::validate_positive_f32(1.0, "lr").is_ok());
        assert!(validation::validate_positive_f32(0.0, "lr").is_err());
        assert!(validation::validate_positive_f32(-1.0, "lr").is_err());
    }

    #[test]
    fn test_handle_management() {
        let value = vec![1, 2, 3, 4];
        let handle = handle::box_to_handle(value.clone());
        assert!(!handle.is_null());

        unsafe {
            let borrowed = handle::handle_ref(handle).unwrap();
            assert_eq!(borrowed, &value);

            let boxed = handle::handle_to_box(handle);
            assert_eq!(*boxed, value);
        }
    }

    #[test]
    fn test_arc_handle() {
        let value = Arc::new(vec![1, 2, 3]);
        let handle = handle::arc_to_handle(value.clone());

        unsafe {
            let cloned = handle::clone_arc_handle(handle);
            assert_eq!(*cloned, vec![1, 2, 3]);

            let _restored = handle::handle_to_arc(handle);
        }
    }
}
