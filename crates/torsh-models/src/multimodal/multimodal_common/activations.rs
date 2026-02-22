//! Common activation functions for multimodal models

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// QuickGELU activation function used in CLIP
pub struct QuickGELU;

impl QuickGELU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // QuickGELU: x * sigmoid(1.702 * x)
        let scaled = x.mul_scalar(1.702)?;
        let sigmoid = scaled.sigmoid()?;
        x.mul(&sigmoid)
    }
}

impl Module for QuickGELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new() // No parameters
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        true // Stateless
    }

    fn train(&mut self) {
        // No-op for stateless activation
    }

    fn eval(&mut self) {
        // No-op for stateless activation
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(()) // No parameters to move
    }
}

/// Swish/SiLU activation function
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SiLU: x * sigmoid(x)
        let sigmoid = x.sigmoid()?;
        x.mul(&sigmoid)
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        true
    }

    fn train(&mut self) {
        // No-op for stateless activation
    }

    fn eval(&mut self) {
        // No-op for stateless activation
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }
}

/// Factory function to create activation functions by name
pub fn create_activation(activation_name: &str) -> Result<Box<dyn Module>> {
    match activation_name.to_lowercase().as_str() {
        "quick_gelu" | "quickgelu" => Ok(Box::new(QuickGELU::new())),
        "silu" | "swish" => Ok(Box::new(SiLU::new())),
        "gelu" => {
            use torsh_nn::prelude::GELU;
            Ok(Box::new(GELU::new(false)))
        }
        "relu" => {
            use torsh_nn::prelude::ReLU;
            Ok(Box::new(ReLU::new()))
        }
        _ => Err(TorshError::Other(format!(
            "Unsupported activation function: {}",
            activation_name
        ))),
    }
}
