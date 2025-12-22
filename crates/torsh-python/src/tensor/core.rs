//! Core tensor implementation - PyTensor struct and fundamental operations

use crate::{device::PyDevice, dtype::PyDType, error::PyResult, py_result};
// ✅ SCIRS2 Policy: Use scirs2_core::ndarray instead of direct ndarray
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use scirs2_core::ndarray::{Array, IxDyn};
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Python wrapper for ToRSh Tensor (simplified version)
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) tensor: Tensor<f32>, // For now, default to f32
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(
        data: &Bound<'_, PyAny>,
        _dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        // Convert Python data to Rust tensor
        let tensor = if let Ok(arr) = data.clone().cast_into::<PyArray1<f32>>() {
            // 1D NumPy array
            let data = arr.to_vec()?;
            let shape = vec![data.len()];
            py_result!(Tensor::from_data(data, shape, device))?
        } else if let Ok(arr) = data.clone().cast_into::<PyArray2<f32>>() {
            // 2D NumPy array
            let data = arr.to_vec()?;
            let shape = arr.shape().to_vec();
            py_result!(Tensor::from_data(data, shape, device))?
        } else if let Ok(arr) = data.clone().cast_into::<PyArrayDyn<f32>>() {
            // N-D NumPy array
            let data = arr.to_vec()?;
            let shape = arr.shape().to_vec();
            py_result!(Tensor::from_data(data, shape, device))?
        } else if let Ok(list) = data.clone().cast_into::<PyList>() {
            // Python list - simplified version
            Self::from_py_list(&list, device)?
        } else if let Ok(scalar) = data.extract::<f32>() {
            // Scalar value
            py_result!(Tensor::from_data(vec![scalar], vec![], device))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported data type for tensor creation",
            ));
        };

        let tensor = tensor.requires_grad_(requires_grad);

        Ok(Self { tensor })
    }

    // Basic properties
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    #[getter]
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType::from(self.tensor.dtype())
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from(self.tensor.device())
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.tensor.requires_grad()
    }

    // String representation
    fn __repr__(&self) -> String {
        let binding = self.tensor.shape();
        let shape = binding.dims();
        let device_str = match self.tensor.device() {
            DeviceType::Cpu => String::new(),
            dev => format!(", device='{}'", PyDevice::from(dev)),
        };
        let grad_str = if self.tensor.requires_grad() {
            ", requires_grad=True"
        } else {
            ""
        };

        format!(
            "tensor({:?}, shape={}{}{}, dtype={})",
            // For now, just show shape info instead of actual data
            shape,
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            device_str,
            grad_str,
            PyDType::from(self.tensor.dtype())
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Convert tensor to NumPy array with proper shape preservation
    fn numpy(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let data = py_result!(self.tensor.to_vec())?;
            let binding = self.tensor.shape();
            let shape = binding.dims();

            // Convert shape from usize to PyArrayDyn compatible format
            let shape_vec: Vec<usize> = shape.iter().copied().collect();

            match shape.len() {
                0 => {
                    // Scalar
                    Ok(data[0].into_pyobject(py)?.into_any().unbind())
                }
                1 => {
                    // 1D array
                    let array = data.to_pyarray(py);
                    Ok(array.into_pyobject(py)?.into_any().unbind())
                }
                2 => {
                    // 2D array - properly reshape
                    // Create ndarray and convert to PyArrayDyn
                    let ndarray = Array::from_vec(data)
                        .into_dyn()
                        .to_shape(IxDyn(&shape_vec))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Shape error: {}",
                                e
                            ))
                        })?
                        .into_owned();
                    let array = ndarray.to_pyarray(py);
                    Ok(array.into_pyobject(py)?.into_any().unbind())
                }
                _ => {
                    // N-D array - properly reshape for arbitrary dimensions
                    let ndarray = Array::from_vec(data)
                        .into_dyn()
                        .to_shape(IxDyn(&shape_vec))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Shape error: {}",
                                e
                            ))
                        })?
                        .into_owned();
                    let array = ndarray.to_pyarray(py);
                    Ok(array.into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Extract single scalar value from tensor
    fn item(&self) -> PyResult<f32> {
        if self.tensor.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only one element tensors can be converted to Python scalars",
            ));
        }
        let data = py_result!(self.tensor.to_vec())?;
        Ok(data[0])
    }

    /// Convert tensor to nested Python lists
    fn tolist(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let data = py_result!(self.tensor.to_vec())?;
            let binding = self.tensor.shape();
            let shape = binding.dims();

            if shape.is_empty() {
                // Scalar
                Ok(data[0].into_pyobject(py)?.into_any().unbind())
            } else {
                // Create nested lists based on tensor shape
                let nested_list = self.create_nested_list(py, &data, shape, 0, &mut 0)?;
                Ok(nested_list)
            }
        })
    }

    /// Create a copy of tensor on specified device (NumPy-compatible method)
    fn copy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Get tensor stride information (NumPy-compatible)
    fn stride(&self) -> Vec<usize> {
        // For now, return a simple stride calculation
        let binding = self.tensor.shape();
        let shape = binding.dims();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get number of bytes per element (NumPy-compatible)
    fn itemsize(&self) -> usize {
        std::mem::size_of::<f32>()
    }

    /// Get total number of bytes (NumPy-compatible)
    fn nbytes(&self) -> usize {
        self.tensor.numel() * self.itemsize()
    }

    /// Check if tensor data is C-contiguous (NumPy-compatible)
    fn is_c_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    // ===============================
    // Mathematical Operations
    // ===============================

    /// Add operation (tensor + other)
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.add(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Subtract operation (tensor - other)
    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.sub(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Multiply operation (tensor * other)
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.mul(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Divide operation (tensor / other)
    fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.div(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Power operation (tensor ** exponent)
    fn pow(&self, exponent: f32) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.pow(exponent))?;
        Ok(PyTensor { tensor: result })
    }

    /// Scalar addition
    fn add_scalar(&self, scalar: f32) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.add_scalar(scalar))?;
        Ok(PyTensor { tensor: result })
    }

    /// Scalar multiplication
    fn mul_scalar(&self, scalar: f32) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.mul_scalar(scalar))?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Tensor Manipulation Operations
    // ===============================

    /// Reshape tensor to new shape
    fn reshape(&self, shape: Vec<i64>) -> PyResult<PyTensor> {
        let i32_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let result = py_result!(self.tensor.reshape(&i32_shape))?;
        Ok(PyTensor { tensor: result })
    }

    /// Transpose tensor (swap dimensions)
    fn transpose(&self, dim0: i64, dim1: i64) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.transpose(dim0 as i32, dim1 as i32))?;
        Ok(PyTensor { tensor: result })
    }

    /// Transpose 2D tensor
    fn t(&self) -> PyResult<PyTensor> {
        if self.tensor.ndim() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "t() can only be called on 2D tensors",
            ));
        }
        self.transpose(0, 1)
    }

    /// Squeeze tensor (remove dimensions of size 1)
    fn squeeze(&self, dim: Option<i64>) -> PyResult<PyTensor> {
        let dim_to_squeeze = dim.map(|d| d as i32).unwrap_or(0i32);
        let result = py_result!(self.tensor.squeeze(dim_to_squeeze))?;
        Ok(PyTensor { tensor: result })
    }

    /// Unsqueeze tensor (add dimension of size 1)
    fn unsqueeze(&self, dim: i64) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.unsqueeze(dim as i32))?;
        Ok(PyTensor { tensor: result })
    }

    /// Flatten tensor
    fn flatten(&self, _start_dim: Option<i64>, _end_dim: Option<i64>) -> PyResult<PyTensor> {
        // For now, use basic flatten - may need different implementation
        let result = py_result!(self.tensor.flatten())?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Reduction Operations
    // ===============================

    /// Sum along specified dimensions
    fn sum(&self, _dim: Option<Vec<i64>>, _keepdim: Option<bool>) -> PyResult<PyTensor> {
        // For now, use basic sum - may need different implementation
        let result = py_result!(self.tensor.sum())?;
        Ok(PyTensor { tensor: result })
    }

    /// Mean along specified dimensions
    fn mean(&self, dim: Option<Vec<i64>>, keepdim: Option<bool>) -> PyResult<PyTensor> {
        let keepdim = keepdim.unwrap_or(false);
        let result = if let Some(dims) = dim {
            let usize_dims: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
            py_result!(self.tensor.mean(Some(&usize_dims), keepdim))?
        } else {
            py_result!(self.tensor.mean(None, keepdim))?
        };
        Ok(PyTensor { tensor: result })
    }

    /// Maximum along specified dimensions
    fn max(&self, dim: Option<i64>, keepdim: Option<bool>) -> PyResult<PyTensor> {
        let dim_opt = dim.map(|d| d as usize);
        let keepdim = keepdim.unwrap_or(false);
        let result = py_result!(self.tensor.max(dim_opt, keepdim))?;
        Ok(PyTensor { tensor: result })
    }

    /// Minimum along specified dimensions
    fn min(&self, _dim: Option<i64>, _keepdim: Option<bool>) -> PyResult<PyTensor> {
        // For now, use basic min - may need different implementation
        let result = py_result!(self.tensor.min())?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Linear Algebra Operations
    // ===============================

    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.matmul(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Dot product (1D tensors)
    fn dot(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.dot(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Activation Functions
    // ===============================

    /// ReLU activation function
    fn relu(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.relu())?;
        Ok(PyTensor { tensor: result })
    }

    /// Sigmoid activation function
    fn sigmoid(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.sigmoid())?;
        Ok(PyTensor { tensor: result })
    }

    /// Tanh activation function
    fn tanh(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.tanh())?;
        Ok(PyTensor { tensor: result })
    }

    /// Softmax along specified dimension
    fn softmax(&self, dim: i64) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.softmax(dim as i32))?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Trigonometric Functions
    // ===============================

    /// Sine function
    fn sin(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.sin())?;
        Ok(PyTensor { tensor: result })
    }

    /// Cosine function
    fn cos(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.cos())?;
        Ok(PyTensor { tensor: result })
    }

    /// Exponential function
    fn exp(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.exp())?;
        Ok(PyTensor { tensor: result })
    }

    /// Natural logarithm
    fn log(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.log())?;
        Ok(PyTensor { tensor: result })
    }

    /// Square root
    fn sqrt(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.sqrt())?;
        Ok(PyTensor { tensor: result })
    }

    /// Absolute value
    fn abs(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.abs())?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Comparison Operations
    // ===============================

    /// Element-wise equality (returns f32 tensor with 0.0/1.0 values)
    fn eq(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.eq(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise inequality (returns f32 tensor with 0.0/1.0 values)
    fn ne(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.ne(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise less than (returns f32 tensor with 0.0/1.0 values)
    fn lt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.lt(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise greater than (returns f32 tensor with 0.0/1.0 values)
    fn gt(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.gt(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise less than or equal (returns f32 tensor with 0.0/1.0 values)
    fn le(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.le(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise greater than or equal (returns f32 tensor with 0.0/1.0 values)
    fn ge(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.ge(&other.tensor))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Element-wise maximum
    fn maximum(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.maximum(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Element-wise minimum
    fn minimum(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.minimum(&other.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    // Scalar comparison operations

    /// Scalar equality (returns f32 tensor with 0.0/1.0 values)
    fn eq_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.eq_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Scalar inequality (returns f32 tensor with 0.0/1.0 values)
    fn ne_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.ne_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Scalar less than (returns f32 tensor with 0.0/1.0 values)
    fn lt_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.lt_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Scalar greater than (returns f32 tensor with 0.0/1.0 values)
    fn gt_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.gt_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Scalar less than or equal (returns f32 tensor with 0.0/1.0 values)
    fn le_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.le_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    /// Scalar greater than or equal (returns f32 tensor with 0.0/1.0 values)
    fn ge_scalar(&self, value: f32) -> PyResult<PyTensor> {
        let bool_result = py_result!(self.tensor.ge_scalar(value))?;
        let float_tensor = py_result!(Self::bool_to_float_tensor(bool_result))?;
        Ok(PyTensor {
            tensor: float_tensor,
        })
    }

    // ===============================
    // Utility Methods
    // ===============================

    /// Create a copy of the tensor
    fn clone_tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Create a detached copy (no gradients)
    fn detach(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.tensor.detach(),
        })
    }

    /// Move tensor to specified device
    fn to_device(&self, device: PyDevice) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.clone().to(device.device))?;
        Ok(PyTensor { tensor: result })
    }

    /// Check if tensor is contiguous in memory
    fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    /// Make tensor contiguous in memory
    fn contiguous(&self) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.contiguous())?;
        Ok(PyTensor { tensor: result })
    }

    // ===============================
    // Additional PyTorch-Compatible Operations
    // ===============================

    /// Clamp tensor values to specified range
    fn clamp(&self, min: Option<f32>, max: Option<f32>) -> PyResult<PyTensor> {
        let result = if let (Some(min_val), Some(max_val)) = (min, max) {
            py_result!(self.tensor.clamp(min_val, max_val))?
        } else if let Some(min_val) = min {
            // Use clamp with a very large max value for min-only clamping
            py_result!(self.tensor.clamp(min_val, f32::MAX))?
        } else if let Some(max_val) = max {
            // Use clamp with a very small min value for max-only clamping
            py_result!(self.tensor.clamp(f32::MIN, max_val))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "At least one of min or max must be specified",
            ));
        };
        Ok(PyTensor { tensor: result })
    }

    /// Fill tensor with specified value (in-place operation)
    fn fill_(&mut self, value: f32) -> PyResult<PyTensor> {
        py_result!(self.tensor.fill_(value))?;
        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Zero out tensor (in-place operation)
    fn zero_(&mut self) -> PyResult<PyTensor> {
        py_result!(self.tensor.zero_())?;
        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Apply uniform random initialization
    fn uniform_(&mut self, from: Option<f32>, to: Option<f32>) -> PyResult<PyTensor> {
        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::{thread_rng, Distribution, Uniform};

        let from = from.unwrap_or(0.0);
        let to = to.unwrap_or(1.0);

        let mut rng = thread_rng();
        let dist = Uniform::new(from, to).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid uniform distribution parameters: {}",
                e
            ))
        })?;

        let mut data = py_result!(self.tensor.data())?;
        for val in data.iter_mut() {
            *val = dist.sample(&mut rng);
        }

        let shape = self.tensor.shape().dims().to_vec();
        let device = self.tensor.device();
        self.tensor = py_result!(torsh_tensor::Tensor::from_data(data, shape, device))?
            .requires_grad_(self.tensor.requires_grad());

        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Apply normal random initialization
    fn normal_(&mut self, mean: Option<f32>, std: Option<f32>) -> PyResult<PyTensor> {
        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::{thread_rng, Distribution, Normal};

        let mean = mean.unwrap_or(0.0);
        let std = std.unwrap_or(1.0);

        let normal = Normal::new(mean as f64, std as f64).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid normal distribution parameters: {}",
                e
            ))
        })?;

        let mut rng = thread_rng();
        let mut data = py_result!(self.tensor.data())?;
        for val in data.iter_mut() {
            *val = normal.sample(&mut rng) as f32;
        }

        let shape = self.tensor.shape().dims().to_vec();
        let device = self.tensor.device();
        self.tensor = py_result!(torsh_tensor::Tensor::from_data(data, shape, device))?
            .requires_grad_(self.tensor.requires_grad());

        Ok(PyTensor {
            tensor: self.tensor.clone(),
        })
    }

    /// Repeat tensor along specified dimensions
    fn repeat(&self, repeats: Vec<usize>) -> PyResult<PyTensor> {
        let result = py_result!(self.tensor.repeat(&repeats))?;
        Ok(PyTensor { tensor: result })
    }

    /// Expand tensor to specified shape (broadcasting)
    fn expand(&self, size: Vec<i64>) -> PyResult<PyTensor> {
        let size: Vec<usize> = size.iter().map(|&x| x as usize).collect();
        let result = py_result!(self.tensor.expand(&size))?;
        Ok(PyTensor { tensor: result })
    }

    /// Expand tensor to match another tensor's shape
    fn expand_as(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let other_shape: Vec<usize> = other.tensor.shape().dims().iter().map(|&x| x).collect();
        let result = py_result!(self.tensor.expand(&other_shape))?;
        Ok(PyTensor { tensor: result })
    }

    /// Select elements from tensor along specified dimension
    fn index_select(&self, dim: i64, index: &PyTensor) -> PyResult<PyTensor> {
        let index_i32 = py_result!(index.tensor.to_i32_simd())?;
        let index_i64 = py_result!(index_i32.to_i64_simd())?;
        let result = py_result!(self.tensor.index_select(dim as i32, &index_i64))?;
        Ok(PyTensor { tensor: result })
    }

    /// Gather elements from tensor
    fn gather(&self, dim: i64, index: &PyTensor) -> PyResult<PyTensor> {
        let index_i32 = py_result!(index.tensor.to_i32_simd())?;
        let index_i64 = py_result!(index_i32.to_i64_simd())?;
        let result = py_result!(self.tensor.gather(dim as usize, &index_i64))?;
        Ok(PyTensor { tensor: result })
    }

    /// Scatter elements into tensor
    fn scatter(&self, dim: i64, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
        let index_i32 = py_result!(index.tensor.to_i32_simd())?;
        let index_i64 = py_result!(index_i32.to_i64_simd())?;
        let result = py_result!(self.tensor.scatter(dim as usize, &index_i64, &src.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// Masked fill operation
    fn masked_fill(&self, mask: &PyTensor, value: f32) -> PyResult<PyTensor> {
        // ✅ Proper masked fill implementation
        let mut data = py_result!(self.tensor.data())?;
        let mask_data = py_result!(mask.tensor.data())?;

        if data.len() != mask_data.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Mask must have the same number of elements as tensor",
            ));
        }

        for (val, &mask_val) in data.iter_mut().zip(mask_data.iter()) {
            if mask_val != 0.0 {
                *val = value;
            }
        }

        let shape = self.tensor.shape().dims().to_vec();
        let device = self.tensor.device();
        let result = py_result!(torsh_tensor::Tensor::from_data(data, shape, device))?
            .requires_grad_(self.tensor.requires_grad());

        Ok(PyTensor { tensor: result })
    }

    /// Masked select operation
    fn masked_select(&self, mask: &PyTensor) -> PyResult<PyTensor> {
        // Convert f32 mask to bool by treating non-zero as true
        let mask_data = py_result!(mask.tensor.data())?;
        let bool_data: Vec<bool> = mask_data.iter().map(|&x| x != 0.0).collect();
        let mask_bool = py_result!(torsh_tensor::Tensor::from_data(
            bool_data,
            mask.tensor.shape().dims().to_vec(),
            mask.tensor.device()
        ))?;
        let result = py_result!(self.tensor.masked_select(&mask_bool))?;
        Ok(PyTensor { tensor: result })
    }

    /// Concatenate tensors along specified dimension
    fn cat(&self, tensors: Vec<PyTensor>, dim: i64) -> PyResult<PyTensor> {
        let tensor_refs: Vec<&torsh_tensor::Tensor<f32>> =
            tensors.iter().map(|t| &t.tensor).collect();
        let result = py_result!(torsh_tensor::Tensor::cat(&tensor_refs, dim as i32))?;
        Ok(PyTensor { tensor: result })
    }

    /// Stack tensors along new dimension
    fn stack(&self, tensors: Vec<PyTensor>, dim: i64) -> PyResult<PyTensor> {
        let tensor_refs: Vec<&torsh_tensor::Tensor<f32>> =
            tensors.iter().map(|t| &t.tensor).collect();
        // Use cat as a simplified stacking implementation
        let result = py_result!(torsh_tensor::Tensor::cat(&tensor_refs, dim as i32))?;
        Ok(PyTensor { tensor: result })
    }

    /// Split tensor into chunks
    fn chunk(&self, chunks: usize, dim: i64) -> PyResult<Vec<PyTensor>> {
        // ✅ Proper chunk implementation
        let shape = self.tensor.shape().dims().to_vec();
        let dim_usize = dim as usize;

        if dim_usize >= shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        let dim_size = shape[dim_usize];
        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division

        let mut result = Vec::new();
        for i in 0..chunks {
            let start = i * chunk_size;
            if start >= dim_size {
                break;
            }
            let end = std::cmp::min(start + chunk_size, dim_size);

            // Use narrow to extract chunk
            let chunk = py_result!(self.tensor.narrow(dim as i32, start as i64, end - start))?;
            result.push(PyTensor { tensor: chunk });
        }

        Ok(result)
    }

    /// Split tensor at specified sizes
    fn split(&self, split_sizes: Vec<usize>, dim: i64) -> PyResult<Vec<PyTensor>> {
        // ✅ Proper split implementation
        let shape = self.tensor.shape().dims().to_vec();
        let dim_usize = dim as usize;

        if dim_usize >= shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        let total_size: usize = split_sizes.iter().sum();
        if total_size != shape[dim_usize] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Split sizes sum to {} but dimension {} has size {}",
                total_size, dim, shape[dim_usize]
            )));
        }

        let mut result = Vec::new();
        let mut start = 0;

        for &size in &split_sizes {
            let chunk = py_result!(self.tensor.narrow(dim as i32, start as i64, size))?;
            result.push(PyTensor { tensor: chunk });
            start += size;
        }

        Ok(result)
    }

    /// Permute tensor dimensions
    fn permute(&self, dims: Vec<i64>) -> PyResult<PyTensor> {
        let dims: Vec<i32> = dims.iter().map(|&x| x as i32).collect();
        let result = py_result!(self.tensor.permute(&dims))?;
        Ok(PyTensor { tensor: result })
    }

    /// Get diagonal elements
    fn diag(&self, diagonal: Option<i64>) -> PyResult<PyTensor> {
        // ✅ Proper diagonal extraction implementation

        let offset = diagonal.unwrap_or(0);
        let shape = self.tensor.shape().dims().to_vec();

        if shape.len() == 1 {
            // 1D tensor -> create diagonal matrix
            let n = shape[0];
            let size = n + offset.abs() as usize;
            let mut data = vec![0.0; size * size];
            let input_data = py_result!(self.tensor.data())?;

            for (i, &val) in input_data.iter().enumerate() {
                if offset >= 0 {
                    let row = i;
                    let col = i + offset as usize;
                    data[row * size + col] = val;
                } else {
                    let row = i + (-offset) as usize;
                    let col = i;
                    data[row * size + col] = val;
                }
            }

            let result = py_result!(torsh_tensor::Tensor::from_data(
                data,
                vec![size, size],
                self.tensor.device()
            ))?;
            Ok(PyTensor { tensor: result })
        } else if shape.len() == 2 {
            // 2D tensor -> extract diagonal
            let rows = shape[0];
            let cols = shape[1];
            let data = py_result!(self.tensor.data())?;

            let mut diag_data = Vec::new();

            if offset >= 0 {
                let offset_u = offset as usize;
                for i in 0..std::cmp::min(rows, cols.saturating_sub(offset_u)) {
                    diag_data.push(data[i * cols + i + offset_u]);
                }
            } else {
                let offset_u = (-offset) as usize;
                for i in 0..std::cmp::min(rows.saturating_sub(offset_u), cols) {
                    diag_data.push(data[(i + offset_u) * cols + i]);
                }
            }

            let diag_len = diag_data.len();
            let result = py_result!(torsh_tensor::Tensor::from_data(
                diag_data,
                vec![diag_len],
                self.tensor.device()
            ))?;
            Ok(PyTensor { tensor: result })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "diag() only supports 1D or 2D tensors",
            ))
        }
    }

    /// Trace of matrix
    fn trace(&self) -> PyResult<PyTensor> {
        // ✅ Proper trace computation
        let shape = self.tensor.shape().dims().to_vec();

        if shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "trace() requires a 2D tensor",
            ));
        }

        let rows = shape[0];
        let cols = shape[1];
        let min_dim = std::cmp::min(rows, cols);

        let data = py_result!(self.tensor.data())?;
        let mut trace_sum = 0.0;

        for i in 0..min_dim {
            trace_sum += data[i * cols + i];
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            vec![trace_sum],
            vec![],
            self.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Norm calculation
    fn norm(
        &self,
        p: Option<f32>,
        _dim: Option<Vec<i64>>,
        _keepdim: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _p = p.unwrap_or(2.0);
        // For now, use simple L2 norm regardless of parameters
        // TODO: Implement full norm_lp functionality when ops module is exposed
        let result = py_result!(self.tensor.norm())?;
        Ok(PyTensor { tensor: result })
    }

    /// Standard deviation
    fn std(
        &self,
        dim: Option<Vec<i64>>,
        keepdim: Option<bool>,
        unbiased: Option<bool>,
    ) -> PyResult<PyTensor> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let stat_mode = if unbiased {
            torsh_tensor::stats::StatMode::Sample
        } else {
            torsh_tensor::stats::StatMode::Population
        };
        let result = if let Some(dims) = dim {
            let usize_dims: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
            py_result!(self.tensor.std(Some(&usize_dims), keepdim, stat_mode))?
        } else {
            py_result!(self.tensor.std(None, keepdim, stat_mode))?
        };
        Ok(PyTensor { tensor: result })
    }

    /// Variance calculation
    fn var(
        &self,
        dim: Option<Vec<i64>>,
        keepdim: Option<bool>,
        unbiased: Option<bool>,
    ) -> PyResult<PyTensor> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let stat_mode = if unbiased {
            torsh_tensor::stats::StatMode::Sample
        } else {
            torsh_tensor::stats::StatMode::Population
        };
        let result = if let Some(dims) = dim {
            let usize_dims: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
            py_result!(self.tensor.var(Some(&usize_dims), keepdim, stat_mode))?
        } else {
            py_result!(self.tensor.var(None, keepdim, stat_mode))?
        };
        Ok(PyTensor { tensor: result })
    }

    /// Argmax operation
    fn argmax(&self, dim: Option<i64>, _keepdim: Option<bool>) -> PyResult<PyTensor> {
        let result = if let Some(d) = dim {
            py_result!(self.tensor.argmax(Some(d as i32)))?
        } else {
            py_result!(self.tensor.argmax(None))?
        };
        let result_f32 = py_result!(result.to_f32_simd())?;
        Ok(PyTensor { tensor: result_f32 })
    }

    /// Argmin operation
    fn argmin(&self, dim: Option<i64>, _keepdim: Option<bool>) -> PyResult<PyTensor> {
        let result = if let Some(d) = dim {
            py_result!(self.tensor.argmin(Some(d as i32)))?
        } else {
            py_result!(self.tensor.argmin(None))?
        };
        let result_f32 = py_result!(result.to_f32_simd())?;
        Ok(PyTensor { tensor: result_f32 })
    }
}

impl PyTensor {
    /// Convert from Python list to tensor (simplified)
    fn from_py_list(list: &Bound<'_, PyList>, device: DeviceType) -> PyResult<Tensor<f32>> {
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

        py_result!(Tensor::from_data(data, vec![len], device))
    }

    /// Helper method to create nested Python lists from tensor data
    fn create_nested_list(
        &self,
        py: Python<'_>,
        data: &[f32],
        shape: &[usize],
        dim: usize,
        index: &mut usize,
    ) -> PyResult<Py<PyAny>> {
        if dim == shape.len() - 1 {
            // Leaf dimension: create list of values
            let mut items = Vec::new();
            for _ in 0..shape[dim] {
                items.push(data[*index].into_pyobject(py)?.into_any().unbind());
                *index += 1;
            }
            Ok(items.into_pyobject(py)?.into_any().unbind())
        } else {
            // Intermediate dimension: create list of nested lists
            let mut items = Vec::new();
            for _ in 0..shape[dim] {
                let nested = self.create_nested_list(py, data, shape, dim + 1, index)?;
                items.push(nested);
            }
            Ok(items.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Convert boolean tensor to float tensor (0.0 for false, 1.0 for true)
    fn bool_to_float_tensor(
        bool_tensor: torsh_tensor::Tensor<bool>,
    ) -> torsh_core::error::Result<torsh_tensor::Tensor<f32>> {
        let bool_data = bool_tensor.data()?;
        let float_data: Vec<f32> = bool_data
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let shape = bool_tensor.shape().dims().to_vec();
        let device = bool_tensor.device();

        torsh_tensor::Tensor::from_data(float_data, shape, device)
    }
}
