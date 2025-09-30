//! Linear (fully connected) neural network layer

use super::module::PyModule;
use crate::{device::PyDevice, error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Linear (fully connected) layer
#[pyclass(name = "Linear", extends = PyModule)]
pub struct PyLinear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    training: bool,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        use torsh_core::device::DeviceType;

        let has_bias = bias.unwrap_or(true);

        // Initialize weight with Xavier/Glorot uniform initialization
        let weight_shape = vec![out_features, in_features];
        let weight = py_result!(torsh_tensor::creation::randn(&weight_shape))?.requires_grad_(true);

        // Initialize bias if needed
        let bias = if has_bias {
            let bias_shape = vec![out_features];
            Some(py_result!(torsh_tensor::creation::zeros(&bias_shape))?.requires_grad_(true))
        } else {
            None
        };

        Ok((
            Self {
                weight,
                bias,
                in_features,
                out_features,
                has_bias,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through the linear layer
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Compute input @ weight.T
        let result = py_result!(input.tensor.matmul(&self.weight))?;

        // Add bias if present
        let result = if let Some(ref bias) = self.bias {
            py_result!(result.add(bias))?
        } else {
            result
        };

        Ok(PyTensor { tensor: result })
    }

    /// Get all parameters (weight and bias if present)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();

        // Add weight parameter
        params.push(PyTensor {
            tensor: self.weight.clone(),
        });

        // Add bias parameter if present
        if let Some(ref bias) = self.bias {
            params.push(PyTensor {
                tensor: bias.clone(),
            });
        }

        Ok(params)
    }

    /// Get named parameters
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut named_params = HashMap::new();

        // Add weight parameter
        named_params.insert(
            "weight".to_string(),
            PyTensor {
                tensor: self.weight.clone(),
            },
        );

        // Add bias parameter if present
        if let Some(ref bias) = self.bias {
            named_params.insert(
                "bias".to_string(),
                PyTensor {
                    tensor: bias.clone(),
                },
            );
        }

        Ok(named_params)
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) {
        self.training = mode.unwrap_or(true);
        // Linear layers don't have different behavior in train/eval mode
        // but we track the state for consistency
    }

    /// Set evaluation mode
    fn eval(&mut self) {
        self.training = false;
    }

    /// Move layer to specified device
    fn to(&mut self, device: PyDevice) -> PyResult<()> {
        // Move weight to device
        self.weight = py_result!(self.weight.clone().to(device.device))?;

        // Move bias to device if present
        if let Some(ref bias) = self.bias {
            self.bias = Some(py_result!(bias.clone().to(device.device))?);
        }

        Ok(())
    }

    /// Zero gradients of all parameters
    fn zero_grad(&mut self) {
        // Zero gradients for weight and bias
        let _ = self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            let _ = bias.zero_grad();
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features, self.out_features, self.has_bias
        )
    }

    /// Get input features
    #[getter]
    fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[getter]
    fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if bias is enabled
    #[getter]
    fn bias(&self) -> bool {
        self.has_bias
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }

    /// Get weight tensor
    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.weight.clone(),
        })
    }

    /// Load state dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, PyTensor>) -> PyResult<()> {
        // Load weight
        if let Some(weight_tensor) = state_dict.get("weight") {
            self.weight = weight_tensor.tensor.clone();
        }

        // Load bias if present
        if self.has_bias {
            if let Some(bias_tensor) = state_dict.get("bias") {
                self.bias = Some(bias_tensor.tensor.clone());
            }
        }

        Ok(())
    }
}
