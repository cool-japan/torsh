//! Data conversion utilities for Python-Rust interop

use crate::error::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyList;
use torsh_core::{device::DeviceType, dtype::DType};

/// Convert Python list to `Vec<f32>`
pub fn python_list_to_vec(list: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
    let mut data = Vec::new();
    let len = list.len();

    for i in 0..len {
        let item = list.get_item(i)?;
        if let Ok(val) = item.extract::<f32>() {
            data.push(val);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot convert item at index {} to f32",
                i
            )));
        }
    }

    Ok(data)
}

/// Convert device string to DeviceType
pub fn parse_device_string(device_str: &str) -> PyResult<DeviceType> {
    match device_str {
        "cpu" => Ok(DeviceType::Cpu),
        "cuda" | "cuda:0" => Ok(DeviceType::Cuda(0)),
        "metal" | "metal:0" => Ok(DeviceType::Metal(0)),
        s if s.starts_with("cuda:") => {
            let id: usize = s[5..].parse().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid CUDA device ID: {}",
                    &s[5..]
                ))
            })?;
            Ok(DeviceType::Cuda(id))
        }
        s if s.starts_with("metal:") => {
            let id: usize = s[6..].parse().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid Metal device ID: {}",
                    &s[6..]
                ))
            })?;
            Ok(DeviceType::Metal(id))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown device: {}",
            device_str
        ))),
    }
}

/// Convert dtype string to DType
pub fn parse_dtype_string(dtype_str: &str) -> PyResult<DType> {
    match dtype_str {
        "float32" | "f32" => Ok(DType::F32),
        "float64" | "f64" => Ok(DType::F64),
        "int8" | "i8" => Ok(DType::I8),
        "int16" | "i16" => Ok(DType::I16),
        "int32" | "i32" => Ok(DType::I32),
        "int64" | "i64" => Ok(DType::I64),
        "uint8" | "u8" => Ok(DType::U8),
        "uint32" | "u32" => Ok(DType::U32),
        "uint64" | "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "float16" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown dtype: {}",
            dtype_str
        ))),
    }
}

/// Convert Python objects to shape vector
pub fn extract_shape(shape_obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(shape_list) = shape_obj.extract::<Vec<i32>>() {
        Ok(shape_list.into_iter().map(|s| s as usize).collect())
    } else if let Ok(shape_tuple) = shape_obj.extract::<(i32,)>() {
        Ok(vec![shape_tuple.0 as usize])
    } else if let Ok(single_dim) = shape_obj.extract::<i32>() {
        Ok(vec![single_dim as usize])
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Shape must be an integer, tuple, or list of integers",
        ))
    }
}
