//! Linear (fully connected) layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Linear (fully connected) layer
pub struct Linear {
    base: ModuleBase,
    in_features: usize,
    out_features: usize,
    use_bias: bool,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut base = ModuleBase::new();

        // Initialize weight with shape [in_features, out_features] for direct matmul
        // This way input[batch, in_features] @ weight[in_features, out_features] = output[batch, out_features]
        let weight = crate::init::xavier_uniform(&[in_features, out_features])
            .expect("Failed to initialize linear layer weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_features]).expect("Failed to create bias tensor");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            in_features,
            out_features,
            use_bias: bias,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified linear transformation using basic tensor operations
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Compute input @ weight
        let output = input.matmul(&weight)?;

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            Ok(output.add(&bias)?)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl core::fmt::Debug for Linear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("use_bias", &self.use_bias)
            .finish()
    }
}

/// Flatten layer to reshape tensor to 1D (except batch dimension)
pub struct Flatten {
    base: ModuleBase,
    start_dim: usize,
    end_dim: Option<usize>,
}

impl Flatten {
    /// Create a new flatten layer
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            start_dim: 1,
            end_dim: None,
        }
    }

    /// Create a flatten layer with custom dimensions
    pub fn with_dims(start_dim: usize, end_dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            start_dim,
            end_dim,
        }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let dims = shape.dims();

        if dims.is_empty() {
            return Ok(input.clone());
        }

        let start = self.start_dim.min(dims.len());
        let end = self.end_dim.unwrap_or(dims.len()).min(dims.len());

        if start >= end {
            return Ok(input.clone());
        }

        // Calculate new shape
        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        new_shape.extend_from_slice(&dims[..start]);

        // Flatten dimensions from start_dim to end_dim
        let flattened_size: usize = dims[start..end].iter().product();
        new_shape.push(flattened_size);

        // Keep dimensions after end_dim
        if end < dims.len() {
            new_shape.extend_from_slice(&dims[end..]);
        }

        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        input.reshape(&new_shape_i32)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }
}

impl core::fmt::Debug for Flatten {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Flatten")
            .field("start_dim", &self.start_dim)
            .field("end_dim", &self.end_dim)
            .finish()
    }
}
