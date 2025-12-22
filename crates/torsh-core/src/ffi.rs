//! FFI-safe type wrappers for C/C++ integration
//!
//! This module provides C-compatible wrappers around ToRSh core types,
//! enabling integration with C and C++ code.

use crate::device::DeviceType;
use crate::dtype::DType;
use crate::error::{Result, TorshError};
use crate::shape::Shape;
use std::ffi::CString;
use std::os::raw::{c_char, c_uchar};
use std::ptr;
use std::slice;

/// FFI-safe representation of DType
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorshDType {
    /// Internal type identifier
    type_id: u8,
    /// Size in bytes
    size_bytes: u8,
}

impl TorshDType {
    /// Create FFI-safe DType from internal DType
    pub fn from_dtype(dtype: DType) -> Self {
        let (type_id, size_bytes) = match dtype {
            DType::U8 => (0, 1),
            DType::I8 => (1, 1),
            DType::I16 => (2, 2),
            DType::I32 => (3, 4),
            DType::I64 => (4, 8),
            DType::U32 => (14, 4),
            DType::U64 => (15, 8),
            DType::F16 => (5, 2),
            DType::F32 => (6, 4),
            DType::F64 => (7, 8),
            DType::Bool => (8, 1),
            DType::BF16 => (9, 2),
            DType::C64 => (10, 8),
            DType::C128 => (11, 16),
            DType::QInt8 => (12, 1),
            DType::QUInt8 => (13, 1),
            DType::QInt32 => (16, 4),
        };
        TorshDType {
            type_id,
            size_bytes,
        }
    }

    /// Convert back to internal DType
    pub fn to_dtype(self) -> Result<DType> {
        match self.type_id {
            0 => Ok(DType::U8),
            1 => Ok(DType::I8),
            2 => Ok(DType::I16),
            3 => Ok(DType::I32),
            4 => Ok(DType::I64),
            5 => Ok(DType::F16),
            6 => Ok(DType::F32),
            7 => Ok(DType::F64),
            8 => Ok(DType::Bool),
            9 => Ok(DType::BF16),
            10 => Ok(DType::C64),
            11 => Ok(DType::C128),
            12 => Ok(DType::QInt8),
            13 => Ok(DType::QUInt8),
            14 => Ok(DType::U32),
            15 => Ok(DType::U64),
            16 => Ok(DType::QInt32),
            _ => Err(TorshError::InvalidArgument(format!(
                "Invalid FFI dtype ID: {}",
                self.type_id
            ))),
        }
    }
}

/// FFI-safe representation of DeviceType
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorshDevice {
    /// Device type: 0=CPU, 1=CUDA, 2=Metal, 3=WebGPU
    device_type: u8,
    /// Device index (0 for CPU)
    device_index: u32,
}

impl TorshDevice {
    /// Create FFI-safe device from internal DeviceType
    pub fn from_device_type(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => TorshDevice {
                device_type: 0,
                device_index: 0,
            },
            DeviceType::Cuda(idx) => TorshDevice {
                device_type: 1,
                device_index: idx as u32,
            },
            DeviceType::Metal(idx) => TorshDevice {
                device_type: 2,
                device_index: idx as u32,
            },
            DeviceType::Wgpu(idx) => TorshDevice {
                device_type: 3,
                device_index: idx as u32,
            },
        }
    }

    /// Convert back to internal DeviceType
    pub fn to_device_type(self) -> Result<DeviceType> {
        match self.device_type {
            0 => Ok(DeviceType::Cpu),
            1 => Ok(DeviceType::Cuda(self.device_index as usize)),
            2 => Ok(DeviceType::Metal(self.device_index as usize)),
            3 => Ok(DeviceType::Wgpu(self.device_index as usize)),
            _ => Err(TorshError::InvalidArgument(format!(
                "Invalid FFI device type: {}",
                self.device_type
            ))),
        }
    }
}

/// FFI-safe representation of Shape
#[repr(C)]
pub struct TorshShape {
    /// Pointer to dimensions array
    dims: *mut usize,
    /// Number of dimensions
    ndim: usize,
    /// Capacity of dims array (for memory management)
    capacity: usize,
}

impl TorshShape {
    /// Create FFI-safe shape from internal Shape
    pub fn from_shape(shape: &Shape) -> Self {
        let dims_vec = shape.dims().to_vec();
        let ndim = dims_vec.len();
        let capacity = ndim;

        let mut dims_boxed = dims_vec.into_boxed_slice();
        let dims = dims_boxed.as_mut_ptr();

        // Prevent deallocation by forgetting the box
        std::mem::forget(dims_boxed);

        TorshShape {
            dims,
            ndim,
            capacity,
        }
    }

    /// Convert back to internal Shape
    ///
    /// # Safety
    /// The dims pointer must be valid and contain ndim elements
    pub unsafe fn to_shape(&self) -> Result<Shape> {
        if self.dims.is_null() || self.ndim == 0 {
            return Ok(Shape::new(vec![]));
        }

        let dims_slice = slice::from_raw_parts(self.dims, self.ndim);
        Ok(Shape::new(dims_slice.to_vec()))
    }

    /// Get dimensions as a slice
    ///
    /// # Safety
    /// The dims pointer must be valid and contain ndim elements
    pub unsafe fn dims_slice(&self) -> &[usize] {
        if self.dims.is_null() {
            &[]
        } else {
            slice::from_raw_parts(self.dims, self.ndim)
        }
    }
}

impl Drop for TorshShape {
    fn drop(&mut self) {
        if !self.dims.is_null() && self.capacity > 0 {
            unsafe {
                let _ = Vec::from_raw_parts(self.dims, self.ndim, self.capacity);
            }
        }
    }
}

/// FFI-safe error handling
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorshErrorCode {
    Success = 0,
    InvalidArgument = 1,
    ShapeError = 2,
    DimensionMismatch = 3,
    DeviceError = 4,
    AllocationError = 5,
    SynchronizationError = 6,
    UnknownError = 7,
}

impl From<&TorshError> for TorshErrorCode {
    fn from(error: &TorshError) -> Self {
        match error {
            TorshError::InvalidArgument(_) => TorshErrorCode::InvalidArgument,
            TorshError::Shape(_) => TorshErrorCode::ShapeError,
            TorshError::ShapeMismatch { .. } => TorshErrorCode::ShapeError,
            TorshError::BroadcastError { .. } => TorshErrorCode::ShapeError,
            TorshError::DeviceError(_) => TorshErrorCode::DeviceError,
            TorshError::AllocationError(_) => TorshErrorCode::AllocationError,
            TorshError::SynchronizationError(_) => TorshErrorCode::SynchronizationError,
            _ => TorshErrorCode::UnknownError,
        }
    }
}

// C-compatible function exports
extern "C" {
    // These would be implemented in a separate C header
}

/// C-compatible functions for external use
#[no_mangle]
pub extern "C" fn torsh_dtype_create(type_id: u8) -> TorshDType {
    // Create from known type IDs
    TorshDType {
        type_id,
        size_bytes: match type_id {
            0 | 1 | 8 | 12 | 13 => 1, // U8, I8, Bool, QInt8, QUInt8
            2 | 5 | 9 => 2,           // I16, F16, BF16
            3 | 6 => 4,               // I32, F32
            4 | 7 | 10 => 8,          // I64, F64, C64
            11 => 16,                 // C128
            _ => 0,                   // Invalid
        },
    }
}

#[no_mangle]
pub extern "C" fn torsh_dtype_size(dtype: TorshDType) -> u8 {
    dtype.size_bytes
}

#[no_mangle]
pub extern "C" fn torsh_dtype_is_float(dtype: TorshDType) -> c_uchar {
    match dtype.type_id {
        5 | 6 | 7 | 9 => 1, // F16, F32, F64, BF16
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn torsh_dtype_is_integer(dtype: TorshDType) -> c_uchar {
    match dtype.type_id {
        0..=4 => 1, // U8, I8, I16, I32, I64
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn torsh_device_create(device_type: u8, device_index: u32) -> TorshDevice {
    TorshDevice {
        device_type,
        device_index,
    }
}

#[no_mangle]
pub extern "C" fn torsh_device_is_cpu(device: TorshDevice) -> c_uchar {
    if device.device_type == 0 {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn torsh_device_is_gpu(device: TorshDevice) -> c_uchar {
    if device.device_type > 0 {
        1
    } else {
        0
    }
}

/// Create a new TorshShape from raw dimensions
///
/// # Safety
/// - `dims` must be a valid pointer to `ndim` elements
/// - `dims` must point to readable memory for `ndim * size_of::<usize>()` bytes
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_create(dims: *const usize, ndim: usize) -> *mut TorshShape {
    if dims.is_null() || ndim == 0 {
        let shape = TorshShape {
            dims: ptr::null_mut(),
            ndim: 0,
            capacity: 0,
        };
        return Box::into_raw(Box::new(shape));
    }

    let dims_slice = slice::from_raw_parts(dims, ndim);
    let dims_vec = dims_slice.to_vec();
    let capacity = ndim;

    let mut dims_boxed = dims_vec.into_boxed_slice();
    let dims_ptr = dims_boxed.as_mut_ptr();
    std::mem::forget(dims_boxed);

    let shape = TorshShape {
        dims: dims_ptr,
        ndim,
        capacity,
    };
    Box::into_raw(Box::new(shape))
}

/// Destroy a TorshShape and free its memory
///
/// # Safety
/// - `shape` must be a valid pointer returned by `torsh_shape_create`
/// - `shape` must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_destroy(shape: *mut TorshShape) {
    if !shape.is_null() {
        let _ = Box::from_raw(shape);
    }
}

/// Get the number of dimensions in a TorshShape
///
/// # Safety
/// - `shape` must be a valid pointer to a TorshShape
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_ndim(shape: *const TorshShape) -> usize {
    if shape.is_null() {
        return 0;
    }
    (*shape).ndim
}

/// Get the size of a specific dimension in a TorshShape
///
/// # Safety
/// - `shape` must be a valid pointer to a TorshShape
/// - `dim` should be less than the number of dimensions
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_size(shape: *const TorshShape, dim: usize) -> usize {
    if shape.is_null() {
        return 0;
    }

    let shape_ref = &*shape;
    if dim >= shape_ref.ndim {
        return 0;
    }

    let dims_slice = slice::from_raw_parts(shape_ref.dims, shape_ref.ndim);
    dims_slice[dim]
}

/// Get the total number of elements in a TorshShape
///
/// # Safety
/// - `shape` must be a valid pointer to a TorshShape
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_numel(shape: *const TorshShape) -> usize {
    if shape.is_null() {
        return 0;
    }

    let shape_ref = &*shape;
    if shape_ref.dims.is_null() || shape_ref.ndim == 0 {
        return 0;
    }

    let dims_slice = slice::from_raw_parts(shape_ref.dims, shape_ref.ndim);
    dims_slice.iter().product()
}

/// Check if two TorshShapes are broadcast compatible
///
/// # Safety
/// - `shape1` and `shape2` must be valid pointers to TorshShape objects
#[no_mangle]
pub unsafe extern "C" fn torsh_shape_broadcast_compatible(
    shape1: *const TorshShape,
    shape2: *const TorshShape,
) -> c_uchar {
    if shape1.is_null() || shape2.is_null() {
        return 0;
    }

    let shape1_ref = &*shape1;
    let shape2_ref = &*shape2;

    if let (Ok(s1), Ok(s2)) = (shape1_ref.to_shape(), shape2_ref.to_shape()) {
        if s1.is_broadcastable_with(&s2) {
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// Get last error message as C string
///
/// Note: This is a simplified error handling mechanism.
/// In a real implementation, you'd want thread-local error storage.
static mut LAST_ERROR: Option<CString> = None;

#[no_mangle]
pub extern "C" fn torsh_get_last_error() -> *const c_char {
    unsafe {
        match &raw const LAST_ERROR {
            ptr if !ptr.is_null() => {
                if let Some(error) = &(*ptr) {
                    error.as_ptr()
                } else {
                    ptr::null()
                }
            }
            _ => ptr::null(),
        }
    }
}

/// Set error message (internal use)
#[allow(dead_code)]
pub(crate) fn set_last_error(message: &str) {
    unsafe {
        LAST_ERROR = CString::new(message).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_ffi_conversion() {
        let original = DType::F32;
        let ffi_dtype = TorshDType::from_dtype(original);
        let converted = ffi_dtype.to_dtype().unwrap();
        assert_eq!(original, converted);
    }

    #[test]
    fn test_device_ffi_conversion() {
        let original = DeviceType::Cuda(1);
        let ffi_device = TorshDevice::from_device_type(original);
        let converted = ffi_device.to_device_type().unwrap();
        assert_eq!(original, converted);
    }

    #[test]
    fn test_shape_ffi_conversion() {
        let original = Shape::new(vec![2, 3, 4]);
        let ffi_shape = TorshShape::from_shape(&original);

        unsafe {
            let converted = ffi_shape.to_shape().unwrap();
            assert_eq!(original.dims(), converted.dims());
        }
    }

    #[test]
    fn test_c_api_dtype() {
        let dtype = torsh_dtype_create(6); // F32
        assert_eq!(torsh_dtype_size(dtype), 4);
        assert_eq!(torsh_dtype_is_float(dtype), 1);
        assert_eq!(torsh_dtype_is_integer(dtype), 0);
    }

    #[test]
    fn test_c_api_device() {
        let device = torsh_device_create(1, 2); // CUDA:2
        assert_eq!(torsh_device_is_cpu(device), 0);
        assert_eq!(torsh_device_is_gpu(device), 1);
    }

    #[test]
    fn test_c_api_shape() {
        let dims = [2, 3, 4];
        let shape = unsafe { torsh_shape_create(dims.as_ptr(), dims.len()) };

        unsafe {
            assert_eq!(torsh_shape_ndim(shape), 3);
            assert_eq!(torsh_shape_size(shape, 0), 2);
            assert_eq!(torsh_shape_size(shape, 1), 3);
            assert_eq!(torsh_shape_size(shape, 2), 4);
            assert_eq!(torsh_shape_numel(shape), 24);

            torsh_shape_destroy(shape);
        }
    }

    #[test]
    fn test_broadcast_compatibility_ffi() {
        let dims1 = [2, 1, 4];
        let dims2 = [1, 3, 1];

        unsafe {
            let shape1 = torsh_shape_create(dims1.as_ptr(), dims1.len());
            let shape2 = torsh_shape_create(dims2.as_ptr(), dims2.len());

            assert_eq!(torsh_shape_broadcast_compatible(shape1, shape2), 1);

            torsh_shape_destroy(shape1);
            torsh_shape_destroy(shape2);
        }
    }
}
