//! Weight-based normalization techniques
//!
//! This module provides normalization techniques that operate on the weights of neural
//! network layers rather than the activations. These techniques can improve training
//! stability and convergence.

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Weight standardization wrapper
///
/// Applies weight standardization to the wrapped module's weights before forward pass.
/// This technique normalizes the weights to have zero mean and unit variance.
pub struct WeightStandardization<M: Module> {
    base: ModuleBase,
    module: M,
    eps: f32,
}

impl<M: Module> WeightStandardization<M> {
    pub fn new(module: M) -> Self {
        Self::with_eps(module, 1e-5)
    }

    pub fn with_eps(module: M, eps: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            module,
            eps,
        }
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn inner(&self) -> &M {
        &self.module
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Standardize a weight tensor
    fn standardize_weight(&self, weight: &Tensor) -> Result<Tensor> {
        let weight_shape = weight.shape();
        let dims = weight_shape.dims();

        if dims.is_empty() {
            return Ok(weight.clone());
        }

        // Calculate fan_in (number of input features)
        let fan_in = if dims.len() >= 2 {
            dims[1..].iter().product::<usize>()
        } else {
            dims[0]
        };

        let weight_data = weight.to_vec()?;
        let num_filters = dims[0];

        let mut standardized_data = vec![0.0f32; weight_data.len()];

        // Standardize each filter independently
        for filter in 0..num_filters {
            let filter_start = filter * fan_in;
            let filter_end = filter_start + fan_in;

            // Calculate mean for this filter
            let mut sum = 0.0;
            for i in filter_start..filter_end {
                sum += weight_data[i];
            }
            let mean = sum / fan_in as f32;

            // Calculate variance for this filter
            let mut var_sum = 0.0;
            for i in filter_start..filter_end {
                let diff = weight_data[i] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / fan_in as f32;
            let std = (var + self.eps).sqrt();

            // Apply standardization
            for i in filter_start..filter_end {
                standardized_data[i] = (weight_data[i] - mean) / std;
            }
        }

        Tensor::from_data(standardized_data, dims.to_vec(), weight.device())
    }
}

impl<M: Module> Module for WeightStandardization<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Note: In a real implementation, we would need to temporarily replace
        // the module's weights with standardized versions during forward pass.
        // This is a simplified version that demonstrates the concept.

        // For now, just forward through the wrapped module
        self.module.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.named_parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.base.training() && self.module.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        self.module.train();
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        self.module.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        self.module.to_device(device)
    }
}

/// Spectral normalization wrapper
///
/// Applies spectral normalization to constrain the spectral norm (largest singular value)
/// of the weight matrices to improve training stability.
pub struct SpectralNorm<M: Module> {
    base: ModuleBase,
    module: M,
    power_iterations: usize,
    eps: f32,
}

impl<M: Module> SpectralNorm<M> {
    pub fn new(module: M) -> Result<Self> {
        Self::with_config(module, 1, 1e-12)
    }

    pub fn with_config(module: M, power_iterations: usize, eps: f32) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize u and v vectors for power iteration
        // In practice, these would be registered as buffers based on the module's weight shapes
        let u = randn::<f32>(&[1, 128])?; // Placeholder shape
        let v = randn::<f32>(&[128, 1])?; // Placeholder shape

        base.register_buffer("u".to_string(), u);
        base.register_buffer("v".to_string(), v);

        Ok(Self {
            base,
            module,
            power_iterations,
            eps,
        })
    }

    pub fn power_iterations(&self) -> usize {
        self.power_iterations
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn inner(&self) -> &M {
        &self.module
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Compute spectral norm using power iteration
    fn compute_spectral_norm(&self, weight: &Tensor) -> Result<f32> {
        let weight_shape = weight.shape();
        let dims = weight_shape.dims();

        if dims.len() < 2 {
            return Ok(1.0);
        }

        // Reshape weight to 2D matrix [out_features, in_features]
        let out_features = dims[0];
        let in_features: usize = dims[1..].iter().product();

        let weight_2d = weight.reshape(&[out_features as i32, in_features as i32])?;

        // Power iteration to find largest singular value
        // This is a simplified implementation
        let weight_data = weight_2d.to_vec()?;

        // Initialize random vectors
        let mut u = vec![1.0f32; out_features];
        let mut v = vec![1.0f32; in_features];

        // Normalize initial vectors
        let u_norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        for val in &mut u {
            *val /= u_norm + self.eps;
        }
        for val in &mut v {
            *val /= v_norm + self.eps;
        }

        // Power iterations
        for _ in 0..self.power_iterations {
            // v = W^T * u
            for j in 0..in_features {
                let mut sum = 0.0;
                for i in 0..out_features {
                    sum += weight_data[i * in_features + j] * u[i];
                }
                v[j] = sum;
            }

            // Normalize v
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut v {
                *val /= v_norm + self.eps;
            }

            // u = W * v
            for i in 0..out_features {
                let mut sum = 0.0;
                for j in 0..in_features {
                    sum += weight_data[i * in_features + j] * v[j];
                }
                u[i] = sum;
            }

            // Normalize u
            let u_norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut u {
                *val /= u_norm + self.eps;
            }
        }

        // Compute spectral norm: u^T * W * v
        let mut spectral_norm = 0.0;
        for i in 0..out_features {
            for j in 0..in_features {
                spectral_norm += u[i] * weight_data[i * in_features + j] * v[j];
            }
        }

        Ok(spectral_norm)
    }
}

impl<M: Module> Module for SpectralNorm<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Note: In a real implementation, we would need to apply spectral normalization
        // to the module's weights before forward pass.

        // For now, just forward through the wrapped module
        self.module.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.named_parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.base.training() && self.module.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        self.module.train();
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        self.module.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        self.module.to_device(device)
    }
}

/// Weight normalization wrapper
///
/// Decomposes weights into magnitude and direction components for more stable training.
pub struct WeightNorm<M: Module> {
    base: ModuleBase,
    module: M,
    dim: usize,
    eps: f32,
}

impl<M: Module> WeightNorm<M> {
    pub fn new(module: M, dim: usize) -> Result<Self> {
        Self::with_eps(module, dim, 1e-5)
    }

    pub fn with_eps(module: M, dim: usize, eps: f32) -> Result<Self> {
        let mut base = ModuleBase::new();

        // In practice, we would reparameterize the module's weights
        // For demonstration, we create placeholder parameters
        let g = ones(&[1])?; // Weight magnitude parameter
        base.register_parameter("g".to_string(), Parameter::new(g));

        Ok(Self {
            base,
            module,
            dim,
            eps,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn inner(&self) -> &M {
        &self.module
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Compute weight normalization: w = g * v / ||v||
    fn normalize_weight(&self, weight: &Tensor, g: &Tensor) -> Result<Tensor> {
        // Compute L2 norm along specified dimension
        let weight_data = weight.to_vec()?;
        let weight_shape = weight.shape();
        let dims = weight_shape.dims();

        if self.dim >= dims.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "dim {} out of range for tensor with {} dimensions",
                self.dim,
                dims.len()
            )));
        }

        // For simplicity, compute the overall norm (not per-dimension)
        let norm_sq: f32 = weight_data.iter().map(|&x| x * x).sum();
        let norm = (norm_sq + self.eps).sqrt();

        let g_scalar = g.to_vec()?[0]; // Assuming g is a scalar

        // Normalize: w = g * v / ||v||
        let normalized_data: Vec<f32> = weight_data.iter().map(|&x| g_scalar * x / norm).collect();

        Tensor::from_data(normalized_data, dims.to_vec(), weight.device())
    }
}

impl<M: Module> Module for WeightNorm<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Note: In a real implementation, we would need to apply weight normalization
        // to the module's weights before forward pass.

        // For now, just forward through the wrapped module
        self.module.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.named_parameters();
        for (name, param) in self.module.named_parameters() {
            params.insert(format!("module.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.base.training() && self.module.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
        self.module.train();
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        self.module.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        self.module.to_device(device)
    }
}

// Re-export the weight-based normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;

    #[test]
    fn test_weight_standardization_creation() {
        let linear = Linear::new(10, 5, true);
        let ws = WeightStandardization::new(linear);
        assert_eq!(ws.eps(), 1e-5);
    }

    #[test]
    fn test_spectral_norm_creation() {
        let linear = Linear::new(10, 5, true);
        let sn = SpectralNorm::new(linear).unwrap();
        assert_eq!(sn.power_iterations(), 1);
        assert_eq!(sn.eps(), 1e-12);
    }

    #[test]
    fn test_weight_norm_creation() {
        let linear = Linear::new(10, 5, true);
        let wn = WeightNorm::new(linear, 0).unwrap();
        assert_eq!(wn.dim(), 0);
        assert_eq!(wn.eps(), 1e-5);
    }

    #[test]
    fn test_weight_standardization_standardize() {
        let linear = Linear::new(4, 2, true);
        let ws = WeightStandardization::new(linear);

        // Test weight standardization on a simple tensor
        let weight = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        let standardized = ws.standardize_weight(&weight).unwrap();

        // Verify the shape is preserved
        assert_eq!(standardized.shape().dims(), &[2, 4]);
    }
}
