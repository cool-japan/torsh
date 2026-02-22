//! Instance normalization layers
//!
//! Instance normalization normalizes each sample independently across spatial dimensions.
//! This is particularly useful for style transfer and generative models where batch
//! statistics may not be meaningful.

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

use super::common::{utils, NormalizationConfig};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// 1D instance normalization layer
pub struct InstanceNorm1d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm1d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    fn compute_instance_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size * channels];
        let mut vars = vec![0.0f32; batch_size * channels];

        // For 1D instance norm, each sample's each channel is normalized independently
        for batch in 0..batch_size {
            for c in 0..channels {
                let idx = batch * channels + c;
                means[idx] = input_data[idx];
                vars[idx] = 0.0; // No variance for single element
            }
        }

        let mean_tensor = Tensor::from_data(means, vec![batch_size, channels], input.device())?;
        let var_tensor = Tensor::from_data(vars, vec![batch_size, channels], input.device())?;

        Ok((mean_tensor, var_tensor))
    }
}

impl Module for InstanceNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm1d expects 2D input (N, C), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute instance statistics
        let (mean, var) = self.compute_instance_stats(input)?;

        // Get learnable parameters
        let weight = if self.config.affine {
            self.base.parameters.get("weight")
        } else {
            None
        };

        let bias = if self.config.affine {
            self.base.parameters.get("bias")
        } else {
            None
        };

        // Apply normalization
        let weight_tensor = weight.as_ref().map(|p| p.tensor().read().clone());
        let bias_tensor = bias.as_ref().map(|p| p.tensor().read().clone());

        utils::apply_normalization(
            input,
            &mean,
            &var,
            weight_tensor.as_ref(),
            bias_tensor.as_ref(),
            self.config.eps,
        )
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// 2D instance normalization layer
pub struct InstanceNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    fn compute_instance_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size * channels];
        let mut vars = vec![0.0f32; batch_size * channels];

        let spatial_size = (height * width) as f32;

        // Compute mean and variance for each instance-channel pair
        for batch in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;

                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w;
                        let val = input_data[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let mean = sum / spatial_size;
                let var = (sum_sq / spatial_size) - (mean * mean);

                let stat_idx = batch * channels + c;
                means[stat_idx] = mean;
                vars[stat_idx] = var;
            }
        }

        let mean_tensor = Tensor::from_data(means, vec![batch_size, channels], input.device())?;
        let var_tensor = Tensor::from_data(vars, vec![batch_size, channels], input.device())?;

        Ok((mean_tensor, var_tensor))
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm2d expects 4D input (N, C, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute instance statistics
        let (mean, var) = self.compute_instance_stats(input)?;

        // Expand dimensions for broadcasting
        let mean_expanded = mean.unsqueeze(2)?.unsqueeze(3)?;
        let var_expanded = var.unsqueeze(2)?.unsqueeze(3)?;

        // Get learnable parameters
        let weight = if self.config.affine {
            self.base.parameters.get("weight")
        } else {
            None
        };

        let bias = if self.config.affine {
            self.base.parameters.get("bias")
        } else {
            None
        };

        // Apply normalization
        let weight_tensor = weight.as_ref().map(|p| p.tensor().read().clone());
        let bias_tensor = bias.as_ref().map(|p| p.tensor().read().clone());

        utils::apply_normalization(
            input,
            &mean_expanded,
            &var_expanded,
            weight_tensor.as_ref(),
            bias_tensor.as_ref(),
            self.config.eps,
        )
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// 3D instance normalization layer
pub struct InstanceNorm3d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm3d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    fn compute_instance_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let depth = dims[2];
        let height = dims[3];
        let width = dims[4];

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size * channels];
        let mut vars = vec![0.0f32; batch_size * channels];

        let spatial_size = (depth * height * width) as f32;

        // Compute mean and variance for each instance-channel pair
        for batch in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;

                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = batch * (channels * depth * height * width)
                                + c * (depth * height * width)
                                + d * (height * width)
                                + h * width
                                + w;
                            let val = input_data[idx];
                            sum += val;
                            sum_sq += val * val;
                        }
                    }
                }

                let mean = sum / spatial_size;
                let var = (sum_sq / spatial_size) - (mean * mean);

                let stat_idx = batch * channels + c;
                means[stat_idx] = mean;
                vars[stat_idx] = var;
            }
        }

        let mean_tensor = Tensor::from_data(means, vec![batch_size, channels], input.device())?;
        let var_tensor = Tensor::from_data(vars, vec![batch_size, channels], input.device())?;

        Ok((mean_tensor, var_tensor))
    }
}

impl Module for InstanceNorm3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 5 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm3d expects 5D input (N, C, D, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute instance statistics
        let (mean, var) = self.compute_instance_stats(input)?;

        // Expand dimensions for broadcasting
        let mean_expanded = mean.unsqueeze(2)?.unsqueeze(3)?.unsqueeze(4)?;
        let var_expanded = var.unsqueeze(2)?.unsqueeze(3)?.unsqueeze(4)?;

        // Get learnable parameters
        let weight = if self.config.affine {
            self.base.parameters.get("weight")
        } else {
            None
        };

        let bias = if self.config.affine {
            self.base.parameters.get("bias")
        } else {
            None
        };

        // Apply normalization
        let weight_tensor = weight.as_ref().map(|p| p.tensor().read().clone());
        let bias_tensor = bias.as_ref().map(|p| p.tensor().read().clone());

        utils::apply_normalization(
            input,
            &mean_expanded,
            &var_expanded,
            weight_tensor.as_ref(),
            bias_tensor.as_ref(),
            self.config.eps,
        )
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

// Re-export the instance normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm_2d_creation() {
        let instance_norm = InstanceNorm2d::new(64).unwrap();
        assert_eq!(instance_norm.num_features(), 64);
        assert_eq!(instance_norm.eps(), 1e-5);
    }

    #[test]
    fn test_instance_norm_2d_shape_validation() {
        let instance_norm = InstanceNorm2d::new(3).unwrap();

        // Valid input
        let input = zeros(&[2, 3, 32, 32]).unwrap();
        assert!(instance_norm.forward(&input).is_ok());

        // Invalid dimensions
        let input_3d = zeros(&[2, 3, 32]).unwrap();
        assert!(instance_norm.forward(&input_3d).is_err());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 4, 32, 32]).unwrap();
        assert!(instance_norm.forward(&input_wrong_channels).is_err());
    }
}
