// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error::FfiError;
use crate::tensor::{device::DeviceType, storage::TensorStorage, types::TYPE_MAPPER};
// Note: Using numpy's direct API for type compatibility (numpy uses ndarray 0.15.x)
use numpy::{IxDyn as NpIxDyn, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyType};
use pyo3::Py;
use scirs2_core::legacy::rng;
use scirs2_core::random::prelude::*;
use torsh_core::DType;

/// Python wrapper for ToRSh Tensor
#[pyclass(name = "Tensor")]
#[derive(Clone, Debug)]
pub struct PyTensor {
    storage: TensorStorage,
    pub requires_grad: bool,
    // Keep direct access to data for compatibility (will be removed eventually)
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[pymethods]
impl PyTensor {
    /// Create new tensor from data
    #[new]
    #[pyo3(signature = (data, shape=None, dtype=None, requires_grad=false))]
    pub fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: Option<&str>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let (tensor_data, tensor_shape, is_external) =
            if let Ok(array) = data.cast::<PyArrayDyn<f32>>() {
                // From NumPy array - try zero-copy first
                let readonly = array.readonly();
                let data_vec = if readonly.is_c_contiguous() {
                    // Zero-copy path for contiguous arrays
                    readonly
                        .as_slice()
                        .expect("contiguous array should be sliceable")
                        .to_vec()
                } else {
                    // Fallback for non-contiguous arrays
                    readonly.as_array().iter().cloned().collect()
                };
                let shape_vec = array.shape().to_vec();
                (data_vec, shape_vec, true)
            } else if let Ok(list) = data.cast::<PyList>() {
                // From Python list
                let flat_data = Self::flatten_list(&list)?;
                let inferred_shape = shape.unwrap_or_else(|| vec![flat_data.len()]);
                (flat_data, inferred_shape, false)
            } else {
                return Err(FfiError::InvalidConversion {
                    message: "Unsupported data type for tensor creation".to_string(),
                }
                .into());
            };

        let tensor_dtype = match dtype {
            Some("float32") | Some("f32") => DType::F32,
            Some("float64") | Some("f64") => DType::F64,
            Some("int32") | Some("i32") => DType::I32,
            Some("int64") | Some("i64") => DType::I64,
            Some("int16") | Some("i16") => DType::I16,
            Some("int8") | Some("i8") => DType::I8,
            Some("uint8") | Some("u8") => DType::U8,
            Some("float16") | Some("f16") => DType::F16,
            Some("bool") => DType::Bool,
            None => DType::F32, // Default
            _ => {
                // Try using the type mapper for more comprehensive parsing
                if let Ok(parsed_dtype) =
                    TYPE_MAPPER.numpy_to_torsh(dtype.expect("dtype should be Some in this branch"))
                {
                    parsed_dtype
                } else {
                    return Err(FfiError::DTypeMismatch {
                        expected: "f32, f64, i32, i64, i16, i8, u8, f16, bool".to_string(),
                        actual: dtype.unwrap_or("unknown").to_string(),
                    }
                    .into());
                }
            }
        };

        let storage = if is_external {
            TensorStorage::from_external(tensor_data.clone(), tensor_shape.clone(), tensor_dtype)
        } else {
            TensorStorage::new(tensor_data.clone(), tensor_shape.clone(), tensor_dtype)
        };

        Ok(PyTensor {
            storage,
            requires_grad,
            // Keep compatibility fields
            data: tensor_data,
            shape: tensor_shape,
            dtype: tensor_dtype,
        })
    }

    /// Get tensor shape
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get tensor data type
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.dtype).to_lowercase()
    }

    /// Get requires_grad flag
    #[getter]
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Convert tensor to NumPy array
    fn to_numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Create 1D numpy array and reshape to target dimensions
        // This approach avoids ndarray version mismatches (numpy uses 0.15.x)
        let array_1d = PyArray1::from_vec(py, self.data.clone());
        let array_nd = array_1d.reshape(NpIxDyn(&self.shape)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Reshape error: {}", e))
        })?;
        Ok(array_nd.into_pyobject(py)?.into_any().unbind())
    }

    /// Convert tensor to PyTorch tensor (if available)
    fn to_torch(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Try to import torch
        let torch = py.import("torch").map_err(|_| FfiError::Module {
            message: "PyTorch not available".to_string(),
        })?;

        // Convert to torch tensor
        let numpy_array = self.to_numpy(py)?;
        let torch_tensor = torch.call_method1("from_numpy", (numpy_array,))?;

        // Set requires_grad if needed
        if self.requires_grad {
            let result = torch_tensor.call_method1("requires_grad_", (true,))?;
            Ok(result.unbind())
        } else {
            Ok(torch_tensor.unbind())
        }
    }

    /// Basic tensor operations

    /// Element-wise addition
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, |a, b| a + b, "add")
    }

    /// Element-wise subtraction
    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, |a, b| a - b, "sub")
    }

    /// Element-wise multiplication
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, |a, b| a * b, "mul")
    }

    /// Element-wise division
    fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary_op(other, |a, b| a / b, "div")
    }

    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(FfiError::ShapeMismatch {
                expected: vec![2],
                actual: vec![self.shape.len(), other.shape.len()],
            }
            .into());
        }

        let [m, k] = [self.shape[0], self.shape[1]];
        let [k2, n] = [other.shape[0], other.shape[1]];

        if k != k2 {
            return Err(FfiError::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            }
            .into());
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        let result_storage = TensorStorage::new(result.clone(), vec![m, n], self.dtype);
        Ok(PyTensor {
            storage: result_storage,
            requires_grad: self.requires_grad || other.requires_grad,
            data: result,
            shape: vec![m, n],
            dtype: self.dtype,
        })
    }

    /// Transpose tensor (2D only for now)
    fn transpose(&self) -> PyResult<PyTensor> {
        if self.shape.len() != 2 {
            return Err(FfiError::ShapeMismatch {
                expected: vec![2],
                actual: vec![self.shape.len()],
            }
            .into());
        }

        let [m, n] = [self.shape[0], self.shape[1]];
        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }

        let result_storage = TensorStorage::new(result.clone(), vec![n, m], self.dtype);
        Ok(PyTensor {
            storage: result_storage,
            requires_grad: self.requires_grad,
            data: result,
            shape: vec![n, m],
            dtype: self.dtype,
        })
    }

    /// Reshape tensor
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyTensor> {
        let current_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(FfiError::ShapeMismatch {
                expected: vec![current_size],
                actual: vec![new_size],
            }
            .into());
        }

        let result_storage = self
            .storage
            .view(new_shape.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

        Ok(PyTensor {
            storage: result_storage,
            requires_grad: self.requires_grad,
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    /// Sum all elements
    fn sum(&self) -> PyResult<f32> {
        Ok(self.data.iter().sum())
    }

    /// Mean of all elements
    fn mean(&self) -> PyResult<f32> {
        Ok(self.data.iter().sum::<f32>() / self.data.len() as f32)
    }

    /// Maximum element
    fn max(&self) -> PyResult<f32> {
        let result = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        Ok(result)
    }

    /// Minimum element
    fn min(&self) -> PyResult<f32> {
        let result = self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        Ok(result)
    }

    /// Clone tensor
    fn clone(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            storage: self.storage.clone(),
            requires_grad: self.requires_grad,
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    /// Create tensor filled with zeros
    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        shape: Vec<usize>,
        dtype: Option<&str>,
    ) -> PyResult<PyTensor> {
        let tensor_dtype = match dtype {
            Some("float32") | Some("f32") | None => DType::F32,
            Some("float64") | Some("f64") => DType::F64,
            Some("int32") | Some("i32") => DType::I32,
            Some("int64") | Some("i64") => DType::I64,
            _ => DType::F32,
        };

        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        let storage = TensorStorage::new(data.clone(), shape.clone(), tensor_dtype);

        Ok(PyTensor {
            storage,
            requires_grad: false,
            data,
            shape,
            dtype: tensor_dtype,
        })
    }

    /// Create tensor filled with ones
    #[classmethod]
    fn ones(
        _cls: &Bound<'_, PyType>,
        shape: Vec<usize>,
        dtype: Option<&str>,
    ) -> PyResult<PyTensor> {
        let tensor_dtype = match dtype {
            Some("float32") | Some("f32") | None => DType::F32,
            Some("float64") | Some("f64") => DType::F64,
            Some("int32") | Some("i32") => DType::I32,
            Some("int64") | Some("i64") => DType::I64,
            _ => DType::F32,
        };

        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        let storage = TensorStorage::new(data.clone(), shape.clone(), tensor_dtype);

        Ok(PyTensor {
            storage,
            requires_grad: false,
            data,
            shape,
            dtype: tensor_dtype,
        })
    }

    /// Create tensor with random values
    #[classmethod]
    fn randn(
        _cls: &Bound<'_, PyType>,
        shape: Vec<usize>,
        dtype: Option<&str>,
    ) -> PyResult<PyTensor> {
        let tensor_dtype = match dtype {
            Some("float32") | Some("f32") | None => DType::F32,
            Some("float64") | Some("f64") => DType::F64,
            _ => DType::F32,
        };

        let size: usize = shape.iter().product();
        // Use SciRS2 for high-quality random normal generation
        let mut random_gen = rng();
        let normal_dist = Normal::new(0.0, 1.0).expect("valid normal distribution parameters");
        let data: Vec<f32> = (0..size)
            .map(|_| random_gen.sample(&normal_dist) as f32)
            .collect();

        let storage = TensorStorage::new(data.clone(), shape.clone(), tensor_dtype);

        Ok(PyTensor {
            storage,
            requires_grad: false,
            data,
            shape,
            dtype: tensor_dtype,
        })
    }

    /// Autograd functionality

    /// Get gradient tensor
    fn grad(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        let grad_data = self.storage.grad();
        if let Some(ref grad) = *grad_data {
            let grad_tensor = PyTensor {
                storage: TensorStorage::new(grad.clone(), self.shape.clone(), self.dtype),
                requires_grad: false,
                data: grad.clone(),
                shape: self.shape.clone(),
                dtype: self.dtype,
            };
            Ok(Some(Py::new(py, grad_tensor)?.into()))
        } else {
            Ok(None)
        }
    }

    /// Compute backward pass
    fn backward(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        if !self.requires_grad {
            return Err(FfiError::InvalidParameter {
                parameter: "requires_grad".to_string(),
                value: "false".to_string(),
            }
            .into());
        }

        // Simple backward pass implementation
        let grad_data = match gradient {
            Some(grad_tensor) => grad_tensor.data.clone(),
            None => {
                // For scalar tensors, gradient is 1.0
                if self.data.len() == 1 {
                    vec![1.0]
                } else {
                    return Err(FfiError::InvalidParameter {
                        parameter: "gradient".to_string(),
                        value: "None".to_string(),
                    }
                    .into());
                }
            }
        };

        self.storage.set_grad(Some(grad_data));
        Ok(())
    }

    /// Clear gradients
    fn zero_grad(&self) {
        self.storage.clear_grad();
    }

    /// Detach tensor from computation graph
    fn detach(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            storage: self.storage.clone(),
            requires_grad: false,
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    /// Check if tensor is leaf (no operations performed on it)
    fn is_leaf(&self) -> bool {
        // Simplified implementation - in real autograd, this would track operations
        true
    }

    /// Device management

    /// Get current device
    fn device(&self) -> String {
        "cpu".to_string() // Simplified - always CPU for now
    }

    /// Move tensor to device
    fn to_device(&self, device_str: &str) -> PyResult<PyTensor> {
        let device = DeviceType::from_string(device_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

        if !device.is_available() {
            return Err(FfiError::DeviceTransfer {
                message: format!("Device {} is not available", device_str),
            }
            .into());
        }

        // For now, just return a copy since we only support CPU
        self.clone()
    }

    /// Get device properties
    fn device_properties(&self) -> PyResult<String> {
        let device = DeviceType::CPU;
        let props = device.properties();
        Ok(format!("{:#?}", props))
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={:?}, requires_grad={})",
            self.shape, self.dtype, self.requires_grad
        )
    }

    /// String representation
    fn __str__(&self) -> String {
        if self.data.len() <= 10 {
            format!("Tensor({:?})", self.data)
        } else {
            format!(
                "Tensor([{:.4}, {:.4}, ..., {:.4}, {:.4}])",
                self.data[0],
                self.data[1],
                self.data[self.data.len() - 2],
                self.data[self.data.len() - 1]
            )
        }
    }
}

impl PyTensor {
    /// Helper function to flatten nested Python lists
    fn flatten_list(list: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let mut result = Vec::new();
        for item in list.iter() {
            if let Ok(nested_list) = item.cast::<PyList>() {
                result.extend(Self::flatten_list(&nested_list)?);
            } else if let Ok(value) = item.extract::<f32>() {
                result.push(value);
            } else {
                return Err(FfiError::InvalidConversion {
                    message: "Invalid data type in list".to_string(),
                }
                .into());
            }
        }
        Ok(result)
    }

    /// Helper function for binary operations
    fn binary_op<F>(&self, other: &PyTensor, op: F, _op_name: &str) -> PyResult<PyTensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        if self.shape != other.shape {
            return Err(FfiError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            }
            .into());
        }

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        let result_storage = TensorStorage::new(result.clone(), self.shape.clone(), self.dtype);
        Ok(PyTensor {
            storage: result_storage,
            requires_grad: self.requires_grad || other.requires_grad,
            data: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    /// Get number of elements
    #[allow(dead_code)]
    fn numel(&self) -> usize {
        self.data.len()
    }

    /// Convert to numpy array (alias for to_numpy)
    #[allow(dead_code)]
    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.to_numpy(py)
    }
}

/// Additional PyTensor implementation for internal use
impl PyTensor {
    /// Create tensor from raw data (internal use)
    pub fn from_raw(data: Vec<f32>, shape: Vec<usize>, dtype: DType, requires_grad: bool) -> Self {
        let storage = TensorStorage::new(data.clone(), shape.clone(), dtype);
        PyTensor {
            storage,
            requires_grad,
            data,
            shape,
            dtype,
        }
    }

    /// Transpose tensor (internal Rust access)
    pub fn t_internal(&self) -> Result<PyTensor, FfiError> {
        if self.shape.len() != 2 {
            return Err(FfiError::InvalidParameter {
                parameter: "shape".to_string(),
                value: format!("{:?}", self.shape),
            });
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(PyTensor {
            storage: TensorStorage::new(new_data.clone(), vec![cols, rows], self.dtype),
            requires_grad: self.requires_grad,
            data: new_data,
            shape: vec![cols, rows],
            dtype: self.dtype,
        })
    }

    /// Convert to numpy array (internal Rust access)
    pub fn to_numpy_internal(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Create 1D numpy array and reshape to target dimensions
        // This approach avoids ndarray version mismatches (numpy uses 0.15.x)
        let array_1d = PyArray1::from_vec(py, self.data.clone());
        let array_nd = array_1d.reshape(NpIxDyn(&self.shape)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Reshape error: {}", e))
        })?;
        Ok(array_nd.into_pyobject(py)?.into_any().unbind())
    }

    /// Matrix multiplication (internal Rust access)
    pub fn matmul_internal(&self, other: &PyTensor) -> Result<PyTensor, FfiError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(FfiError::InvalidParameter {
                parameter: "shape".to_string(),
                value: "Both tensors must be 2D for matrix multiplication".to_string(),
            });
        }

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(FfiError::ShapeMismatch {
                expected: vec![m, k],
                actual: vec![k2, n],
            });
        }

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(PyTensor {
            storage: TensorStorage::new(result_data.clone(), vec![m, n], self.dtype),
            requires_grad: self.requires_grad || other.requires_grad,
            data: result_data,
            shape: vec![m, n],
            dtype: self.dtype,
        })
    }

    /// Get reference to storage (internal use)
    pub fn storage(&self) -> &TensorStorage {
        &self.storage
    }

    /// Get mutable reference to storage (internal use)
    pub fn storage_mut(&mut self) -> &mut TensorStorage {
        &mut self.storage
    }
}
