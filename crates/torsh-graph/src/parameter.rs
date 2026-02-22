//! Minimal Parameter implementation for torsh-graph
//! This is a temporary replacement for torsh_nn::Parameter to avoid compilation issues

use torsh_tensor::Tensor;

/// Minimal Parameter wrapper for torsh-graph
#[derive(Clone, Debug)]
pub struct Parameter {
    tensor: Tensor,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    /// Get the underlying tensor data (clone)
    pub fn clone_data(&self) -> Tensor {
        self.tensor.clone()
    }

    /// Get a reference to the tensor
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}
