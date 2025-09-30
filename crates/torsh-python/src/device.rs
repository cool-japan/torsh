//! Device handling for Python bindings

use crate::error::PyResult;
use pyo3::prelude::*;
use torsh_core::device::DeviceType;

/// Python wrapper for ToRSh devices
#[pyclass(name = "device")]
#[derive(Clone, Debug)]
pub struct PyDevice {
    pub(crate) device: DeviceType,
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(device: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device_type = if let Ok(s) = device.extract::<String>() {
            match s.as_str() {
                "cpu" => DeviceType::Cpu,
                "cuda" | "cuda:0" => DeviceType::Cuda(0),
                "metal" | "metal:0" => DeviceType::Metal(0),
                s if s.starts_with("cuda:") => {
                    let id: usize = s[5..].parse().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid CUDA device ID: {}",
                            &s[5..]
                        ))
                    })?;
                    DeviceType::Cuda(id)
                }
                s if s.starts_with("metal:") => {
                    let id: usize = s[6..].parse().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid Metal device ID: {}",
                            &s[6..]
                        ))
                    })?;
                    DeviceType::Metal(id)
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown device: {}",
                        s
                    )))
                }
            }
        } else if let Ok(i) = device.extract::<i32>() {
            // Accept integer for CUDA device ID
            if i < 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Device ID must be non-negative",
                ));
            }
            DeviceType::Cuda(i as usize)
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Device must be a string or integer",
            ));
        };

        Ok(Self {
            device: device_type,
        })
    }

    fn __str__(&self) -> String {
        match self.device {
            DeviceType::Cpu => "cpu".to_string(),
            DeviceType::Cuda(id) => format!("cuda:{}", id),
            DeviceType::Metal(id) => format!("metal:{}", id),
            DeviceType::Wgpu(id) => format!("wgpu:{}", id),
        }
    }

    fn __repr__(&self) -> String {
        match self.index() {
            Some(idx) => format!("device(type='{}', index={})", self.type_(), idx),
            None => format!("device(type='{}')", self.type_()),
        }
    }

    fn __eq__(&self, other: &PyDevice) -> bool {
        self.device == other.device
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.device.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn type_(&self) -> String {
        match self.device {
            DeviceType::Cpu => "cpu".to_string(),
            DeviceType::Cuda(_) => "cuda".to_string(),
            DeviceType::Metal(_) => "metal".to_string(),
            DeviceType::Wgpu(_) => "wgpu".to_string(),
        }
    }

    #[getter]
    fn index(&self) -> Option<u32> {
        match self.device {
            DeviceType::Cpu => None,
            DeviceType::Cuda(id) => Some(id as u32),
            DeviceType::Metal(id) => Some(id as u32),
            DeviceType::Wgpu(id) => Some(id as u32),
        }
    }
}

impl From<DeviceType> for PyDevice {
    fn from(device: DeviceType) -> Self {
        Self { device }
    }
}

impl From<PyDevice> for DeviceType {
    fn from(py_device: PyDevice) -> Self {
        py_device.device
    }
}

impl std::fmt::Display for PyDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

/// Helper function to parse device from Python arguments
pub fn parse_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<DeviceType> {
    match device {
        Some(d) => Ok(PyDevice::new(d)?.device),
        None => Ok(DeviceType::Cpu), // Default to CPU
    }
}

/// Register device constants and functions
pub fn register_device_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;

    // Create device constants
    m.add(
        "cpu",
        PyDevice {
            device: DeviceType::Cpu,
        },
    )?;

    // Add device utility functions
    #[pyfunction]
    fn device_count() -> u32 {
        // For now, return 1 (would need proper device discovery)
        1
    }

    #[pyfunction]
    fn is_available() -> bool {
        true
    }

    #[pyfunction]
    fn cuda_is_available() -> bool {
        // Would need proper CUDA detection
        false
    }

    #[pyfunction]
    fn mps_is_available() -> bool {
        // Metal Performance Shaders availability
        false
    }

    #[pyfunction]
    fn get_device_name(device: Option<PyDevice>) -> String {
        match device {
            Some(d) => d.__str__(),
            None => "cpu".to_string(),
        }
    }

    m.add_function(wrap_pyfunction!(device_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(mps_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_name, m)?)?;

    Ok(())
}
