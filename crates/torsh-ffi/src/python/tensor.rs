//! Python tensor wrapper

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use std::sync::Arc;
use torsh_core::DType;
use crate::error::{FfiError, FfiResult};

/// Python wrapper for ToRSh Tensor
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    // For now, we'll use a simplified tensor representation
    // In a full implementation, this would wrap torsh_tensor::Tensor
    pub data: Vec<f32>,
    shape: Vec<usize>,
    dtype: DType,
    requires_grad: bool,
}

#[pymethods]
impl PyTensor {
    /// Create new tensor from data
    #[new]
    #[pyo3(signature = (data, shape=None, dtype=None, requires_grad=false))]
    fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: Option<&str>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let (tensor_data, tensor_shape) = if let Ok(array) = data.downcast::<PyArrayDyn<f32>>() {
            // From NumPy array
            let data_vec = array.readonly().as_array().iter().cloned().collect();
            let shape_vec = array.shape().to_vec();
            (data_vec, shape_vec)
        } else if let Ok(list) = data.downcast::<PyList>() {
            // From Python list
            let flat_data = Self::flatten_list(&list)?;
            let inferred_shape = shape.unwrap_or_else(|| vec![flat_data.len()]);
            (flat_data, inferred_shape)
        } else {
            return Err(FfiError::InvalidConversion {
                message: "Unsupported data type for tensor creation".to_string(),
            }.into());
        };
        
        let tensor_dtype = match dtype {
            Some("float32") | Some("f32") => DType::F32,
            Some("float64") | Some("f64") => DType::F64,
            Some("int32") | Some("i32") => DType::I32,
            Some("int64") | Some("i64") => DType::I64,
            None => DType::F32, // Default
            _ => return Err(FfiError::DTypeMismatch {
                expected: "f32, f64, i32, i64".to_string(),
                actual: dtype.unwrap_or("unknown").to_string(),
            }.into()),
        };
        
        Ok(PyTensor {
            data: tensor_data,
            shape: tensor_shape,
            dtype: tensor_dtype,
            requires_grad,
        })
    }
    
    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
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
    
    /// Set requires_grad flag
    #[setter]
    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
    
    /// Get number of dimensions
    fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get total number of elements
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Convert to NumPy array
    fn numpy(&self, py: Python) -> PyResult<PyObject> {
        let array = self.data.clone().into_pyarray(py);
        let reshaped = array.reshape(self.shape.clone())?;
        Ok(reshaped.into_py(py))
    }
    
    /// Convert to Python list
    fn tolist(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let flat_list = PyList::new(py, &self.data);
            if self.shape.len() == 1 {
                Ok(flat_list.into_py(py))
            } else {
                Self::reshape_to_nested_list(py, &self.data, &self.shape, 0)
            }
        })
    }
    
    /// Reshape tensor
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let total_elements: usize = shape.iter().product();
        if total_elements != self.numel() {
            return Err(FfiError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: vec![total_elements],
            }.into());
        }
        
        Ok(PyTensor {
            data: self.data.clone(),
            shape,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
        })
    }
    
    /// Transpose tensor
    fn t(&self) -> PyResult<Self> {
        if self.shape.len() != 2 {
            return Err(FfiError::InvalidConversion {
                message: "Transpose only supported for 2D tensors".to_string(),
            }.into());
        }
        
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        
        Ok(PyTensor {
            data: transposed_data,
            shape: vec![cols, rows],
            dtype: self.dtype,
            requires_grad: self.requires_grad,
        })
    }
    
    /// Element-wise addition
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.shape != other.shape {
            return Err(FfiError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            }.into());
        }
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(PyTensor {
            data: result_data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }
    
    /// Element-wise multiplication
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.shape != other.shape {
            return Err(FfiError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            }.into());
        }
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        Ok(PyTensor {
            data: result_data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }
    
    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(FfiError::InvalidConversion {
                message: "Matrix multiplication requires 2D tensors".to_string(),
            }.into());
        }
        
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k != k2 {
            return Err(FfiError::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            }.into());
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
            data: result_data,
            shape: vec![m, n],
            dtype: self.dtype,
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }
    
    /// Sum along all dimensions
    fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
    
    /// Mean along all dimensions
    fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, requires_grad={})",
            self.shape, 
            self.dtype(),
            self.requires_grad
        )
    }
    
    /// String representation
    fn __str__(&self) -> String {
        // For small tensors, show the data
        if self.numel() <= 100 {
            format!("tensor({:?})", self.data)
        } else {
            format!("tensor(shape={:?}, dtype={})", self.shape, self.dtype())
        }
    }
}

impl PyTensor {
    /// Flatten nested Python list to Vec<f32>
    fn flatten_list(list: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let mut result = Vec::new();
        for item in list.iter() {
            if let Ok(sublist) = item.downcast::<PyList>() {
                result.extend(Self::flatten_list(&sublist)?);
            } else if let Ok(val) = item.extract::<f32>() {
                result.push(val);
            } else if let Ok(val) = item.extract::<i32>() {
                result.push(val as f32);
            } else {
                return Err(FfiError::InvalidConversion {
                    message: "List contains non-numeric values".to_string(),
                }.into());
            }
        }
        Ok(result)
    }
    
    /// Reshape flat data to nested Python list
    fn reshape_to_nested_list(
        py: Python,
        data: &[f32],
        shape: &[usize],
        offset: usize,
    ) -> PyResult<PyObject> {
        if shape.len() == 1 {
            let slice = &data[offset..offset + shape[0]];
            Ok(PyList::new(py, slice).into_py(py))
        } else {
            let mut result = Vec::new();
            let stride = shape[1..].iter().product::<usize>();
            
            for i in 0..shape[0] {
                let sublist = Self::reshape_to_nested_list(
                    py, 
                    data, 
                    &shape[1..], 
                    offset + i * stride
                )?;
                result.push(sublist);
            }
            
            Ok(PyList::new(py, result).into_py(py))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    
    #[test]
    fn test_tensor_creation() {
        Python::with_gil(|py| {
            let data = PyList::new_bound(py, vec![1.0, 2.0, 3.0, 4.0]);
            let tensor = PyTensor::new(
                data.as_ref(), 
                Some(vec![2, 2]), 
                Some("f32"), 
                false
            ).unwrap();
            
            assert_eq!(tensor.shape(), vec![2, 2]);
            assert_eq!(tensor.numel(), 4);
            assert_eq!(tensor.dtype(), "f32");
        });
    }
    
    #[test]
    fn test_tensor_operations() {
        Python::with_gil(|py| {
            let data1 = PyList::new_bound(py, vec![1.0, 2.0]);
            let data2 = PyList::new_bound(py, vec![3.0, 4.0]);
            
            let tensor1 = PyTensor::new(data1.as_ref(), None, None, false).unwrap();
            let tensor2 = PyTensor::new(data2.as_ref(), None, None, false).unwrap();
            
            let result = tensor1.__add__(&tensor2).unwrap();
            assert_eq!(result.data, vec![4.0, 6.0]);
            
            let result = tensor1.__mul__(&tensor2).unwrap();
            assert_eq!(result.data, vec![3.0, 8.0]);
        });
    }
}