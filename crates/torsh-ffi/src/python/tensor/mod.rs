//! Python tensor wrapper module
//!
//! This module provides comprehensive Python FFI support for ToRSh tensors,
//! including memory management, type mapping, device management, and tensor operations.
//!
//! The module is organized into focused submodules:
//! - `memory`: Memory pool for efficient tensor allocation
//! - `types`: Cross-framework type mapping and compatibility
//! - `device`: Device management and properties
//! - `storage`: Reference-counted tensor storage
//! - `tensor`: Main PyTensor implementation

pub mod device;
pub mod memory;
pub mod storage;
pub mod tensor;
pub mod types;

// Re-export core types for backward compatibility
pub use device::{DeviceProperties, DeviceType};
pub use memory::{MemoryPool, MemoryPoolStats, MEMORY_POOL};
pub use storage::TensorStorage;
pub use tensor::PyTensor;
pub use types::{FrameworkTypeInfo, TypeMapper, TYPE_MAPPER};

// Re-export for Python module registration
pub use tensor::PyTensor as Tensor;

use crate::error::FfiResult;
use pyo3::prelude::*;

/// Initialize the tensor module for Python
pub fn init_tensor_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}

/// Utility functions for tensor creation and management

/// Create a new tensor from raw data with validation
pub fn create_tensor_validated(
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: torsh_core::DType,
    requires_grad: bool,
) -> FfiResult<PyTensor> {
    // Validate shape consistency
    let expected_size: usize = shape.iter().product();
    if data.len() != expected_size {
        return Err(crate::error::FfiError::ShapeMismatch {
            expected: vec![expected_size],
            actual: vec![data.len()],
        });
    }

    Ok(PyTensor::from_raw(data, shape, dtype, requires_grad))
}

/// Get memory pool statistics
pub fn get_memory_stats() -> FfiResult<MemoryPoolStats> {
    MEMORY_POOL.stats()
}

/// Clear memory pool
pub fn clear_memory_pool() -> FfiResult<()> {
    MEMORY_POOL.clear()
}

/// Convert between framework type names
pub fn convert_dtype_name(
    dtype_name: &str,
    from_framework: &str,
    to_framework: &str,
) -> FfiResult<String> {
    let torsh_dtype = match from_framework {
        "numpy" => TYPE_MAPPER.numpy_to_torsh(dtype_name)?,
        "pytorch" => TYPE_MAPPER.pytorch_to_torsh(dtype_name)?,
        "torsh" => match dtype_name {
            "f32" => torsh_core::DType::F32,
            "f64" => torsh_core::DType::F64,
            "i32" => torsh_core::DType::I32,
            "i64" => torsh_core::DType::I64,
            "i16" => torsh_core::DType::I16,
            "i8" => torsh_core::DType::I8,
            "u8" => torsh_core::DType::U8,
            "f16" => torsh_core::DType::F16,
            "bool" => torsh_core::DType::Bool,
            _ => {
                return Err(crate::error::FfiError::DTypeMismatch {
                    expected: "f32, f64, i32, i64, i16, i8, u8, f16, bool".to_string(),
                    actual: dtype_name.to_string(),
                })
            }
        },
        _ => {
            return Err(crate::error::FfiError::DTypeMismatch {
                expected: "numpy, pytorch, torsh".to_string(),
                actual: from_framework.to_string(),
            })
        }
    };

    let result = match to_framework {
        "numpy" => TYPE_MAPPER.torsh_to_numpy(torsh_dtype),
        "pytorch" => TYPE_MAPPER.torsh_to_pytorch(torsh_dtype),
        "torsh" => format!("{:?}", torsh_dtype).to_lowercase(),
        _ => {
            return Err(crate::error::FfiError::DTypeMismatch {
                expected: "numpy, pytorch, torsh".to_string(),
                actual: to_framework.to_string(),
            })
        }
    };

    Ok(result)
}

/// Check device availability
pub fn check_device_availability(device_str: &str) -> FfiResult<bool> {
    let device = DeviceType::from_string(device_str)?;
    Ok(device.is_available())
}

/// Get device properties as a formatted string
pub fn get_device_info(device_str: &str) -> FfiResult<String> {
    let device = DeviceType::from_string(device_str)?;
    let props = device.properties();
    Ok(format!(
        "Device: {}\nTotal Memory: {} GB\nAvailable Memory: {} GB\nCompute Capability: {}\nMulti-processor Count: {}\nIntegrated: {}",
        props.name,
        props.memory_total / (1024 * 1024 * 1024),
        props.memory_available / (1024 * 1024 * 1024),
        props.compute_capability,
        props.multi_processor_count,
        props.is_integrated
    ))
}

/// Tensor creation convenience functions

/// Create a zeros tensor
pub fn zeros(shape: Vec<usize>, dtype: Option<torsh_core::DType>) -> FfiResult<PyTensor> {
    let tensor_dtype = dtype.unwrap_or(torsh_core::DType::F32);
    let size: usize = shape.iter().product();
    let data = vec![0.0; size];
    create_tensor_validated(data, shape, tensor_dtype, false)
}

/// Create a ones tensor
pub fn ones(shape: Vec<usize>, dtype: Option<torsh_core::DType>) -> FfiResult<PyTensor> {
    let tensor_dtype = dtype.unwrap_or(torsh_core::DType::F32);
    let size: usize = shape.iter().product();
    let data = vec![1.0; size];
    create_tensor_validated(data, shape, tensor_dtype, false)
}

/// Create a random tensor with normal distribution
pub fn randn(shape: Vec<usize>, dtype: Option<torsh_core::DType>) -> FfiResult<PyTensor> {
    let tensor_dtype = dtype.unwrap_or(torsh_core::DType::F32);
    let size: usize = shape.iter().product();

    let data: Vec<f32> = (0..size)
        .map(|_| {
            // Simple box-muller transform for normal distribution
            use std::f32::consts::PI;
            static mut SPARE: Option<f32> = None;
            static mut HAS_SPARE: bool = false;

            unsafe {
                if HAS_SPARE {
                    HAS_SPARE = false;
                    SPARE.unwrap()
                } else {
                    HAS_SPARE = true;
                    let u = fastrand::f32();
                    let v = fastrand::f32();
                    let mag = 0.1 * (-2.0 * u.ln()).sqrt();
                    SPARE = Some(mag * (2.0 * PI * v).sin());
                    mag * (2.0 * PI * v).cos()
                }
            }
        })
        .collect();

    create_tensor_validated(data, shape, tensor_dtype, false)
}

/// Create an identity matrix
pub fn eye(size: usize, dtype: Option<torsh_core::DType>) -> FfiResult<PyTensor> {
    let tensor_dtype = dtype.unwrap_or(torsh_core::DType::F32);
    let mut data = vec![0.0; size * size];

    for i in 0..size {
        data[i * size + i] = 1.0;
    }

    create_tensor_validated(data, vec![size, size], tensor_dtype, false)
}

/// Advanced tensor operations

/// Perform matrix multiplication between tensors
pub fn matmul(a: &PyTensor, b: &PyTensor) -> FfiResult<PyTensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(crate::error::FfiError::ShapeMismatch {
            expected: vec![2],
            actual: vec![a.shape.len(), b.shape.len()],
        });
    }

    let [m, k] = [a.shape[0], a.shape[1]];
    let [k2, n] = [b.shape[0], b.shape[1]];

    if k != k2 {
        return Err(crate::error::FfiError::ShapeMismatch {
            expected: vec![k],
            actual: vec![k2],
        });
    }

    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.data[i * k + l] * b.data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    create_tensor_validated(
        result,
        vec![m, n],
        a.dtype,
        a.requires_grad || b.requires_grad,
    )
}

/// Broadcasting utilities

/// Check if two shapes are broadcastable
pub fn are_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let max_len = shape1.len().max(shape2.len());

    for i in 0..max_len {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Compute broadcast shape
pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> FfiResult<Vec<usize>> {
    if !are_broadcastable(shape1, shape2) {
        return Err(crate::error::FfiError::ShapeMismatch {
            expected: shape1.to_vec(),
            actual: shape2.to_vec(),
        });
    }

    let max_len = shape1.len().max(shape2.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        result.push(dim1.max(dim2));
    }

    result.reverse();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DType;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = create_tensor_validated(data, shape, DType::F32, false);
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, DType::F32);
        assert!(!tensor.requires_grad);
    }

    #[test]
    fn test_zeros_creation() {
        let tensor = zeros(vec![3, 3], Some(DType::F32));
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_creation() {
        let tensor = ones(vec![2, 3], Some(DType::F32));
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape, vec![2, 3]);
        assert!(tensor.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_eye_creation() {
        let tensor = eye(3, Some(DType::F32));
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape, vec![3, 3]);

        // Check diagonal elements are 1.0
        for i in 0..3 {
            assert_eq!(tensor.data[i * 3 + i], 1.0);
        }

        // Check off-diagonal elements are 0.0
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert_eq!(tensor.data[i * 3 + j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcasting() {
        assert!(are_broadcastable(&[3, 4], &[4]));
        assert!(are_broadcastable(&[2, 1, 4], &[3, 4]));
        assert!(!are_broadcastable(&[3, 4], &[3, 5]));

        let result = broadcast_shape(&[3, 4], &[4]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_dtype_conversion() {
        let result = convert_dtype_name("float32", "numpy", "pytorch");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "torch.float32");

        let result = convert_dtype_name("torch.int64", "pytorch", "numpy");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "int64");
    }

    #[test]
    fn test_device_operations() {
        assert!(check_device_availability("cpu").unwrap());

        let info = get_device_info("cpu");
        assert!(info.is_ok());
        assert!(info.unwrap().contains("Device: CPU"));
    }

    #[test]
    fn test_memory_pool() {
        let stats = get_memory_stats();
        assert!(stats.is_ok());

        let clear_result = clear_memory_pool();
        assert!(clear_result.is_ok());
    }
}
