//! Layer and Group normalization implementations
//!
//! This module provides normalization techniques that operate on different dimensions:
//! - Layer normalization: normalizes across the feature dimension
//! - Group normalization: normalizes across grouped features

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

/// Layer normalization
pub struct LayerNorm {
    base: ModuleBase,
    normalized_shape: Vec<usize>,
    config: NormalizationConfig,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Result<Self> {
        Self::with_config(normalized_shape, NormalizationConfig::default())
    }

    pub fn with_config(normalized_shape: Vec<usize>, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&normalized_shape)?;
            let bias = zeros(&normalized_shape)?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            normalized_shape,
            config,
        })
    }

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    fn compute_layer_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        // Determine which dimensions to normalize over
        let normalized_dims = self.normalized_shape.len();
        let input_dims = dims.len();

        if input_dims < normalized_dims {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Input has {} dims but normalized_shape has {} dims",
                input_dims, normalized_dims
            )));
        }

        // Check that the last dimensions match normalized_shape
        let start_idx = input_dims - normalized_dims;
        for (i, &norm_dim) in self.normalized_shape.iter().enumerate() {
            if dims[start_idx + i] != norm_dim {
                return Err(torsh_core::error::TorshError::InvalidShape(format!(
                    "Expected dimension {} to be {}, got {}",
                    start_idx + i,
                    norm_dim,
                    dims[start_idx + i]
                )));
            }
        }

        // Calculate the number of elements to normalize over
        let norm_elements: usize = self.normalized_shape.iter().product();
        let batch_size: usize = dims[..start_idx].iter().product();

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size];
        let mut vars = vec![0.0f32; batch_size];

        // Compute mean and variance for each batch element
        for batch in 0..batch_size {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            let batch_start = batch * norm_elements;
            for i in 0..norm_elements {
                let val = input_data[batch_start + i];
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / norm_elements as f32;
            let var = (sum_sq / norm_elements as f32) - (mean * mean);

            means[batch] = mean;
            vars[batch] = var;
        }

        // Reshape to match input batch dimensions
        let mut batch_shape = dims[..start_idx].to_vec();
        for _ in 0..normalized_dims {
            batch_shape.push(1);
        }

        let mean_tensor = Tensor::from_data(means, dims[..start_idx].to_vec(), input.device())?
            .reshape(&batch_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?;
        let var_tensor = Tensor::from_data(vars, dims[..start_idx].to_vec(), input.device())?
            .reshape(&batch_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?;

        Ok((mean_tensor, var_tensor))
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Compute layer statistics
        let (mean, var) = self.compute_layer_stats(input)?;

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

/// Group normalization
pub struct GroupNorm {
    base: ModuleBase,
    num_groups: usize,
    num_channels: usize,
    config: NormalizationConfig,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Result<Self> {
        Self::with_config(num_groups, num_channels, NormalizationConfig::default())
    }

    pub fn with_config(
        num_groups: usize,
        num_channels: usize,
        config: NormalizationConfig,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            )));
        }

        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_channels])?;
            let bias = zeros(&[num_channels])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_groups,
            num_channels,
            config,
        })
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    fn compute_group_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "GroupNorm expects at least 2D input, got {:?}",
                dims
            )));
        }

        let batch_size = dims[0];
        let channels = dims[1];

        if channels != self.num_channels {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} channels, got {}",
                self.num_channels, channels
            )));
        }

        let channels_per_group = self.num_channels / self.num_groups;
        let spatial_size: usize = if dims.len() > 2 {
            dims[2..].iter().product()
        } else {
            1
        };

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size * self.num_groups];
        let mut vars = vec![0.0f32; batch_size * self.num_groups];

        let group_elements = channels_per_group * spatial_size;

        // Compute statistics for each group
        for batch in 0..batch_size {
            for group in 0..self.num_groups {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;

                let group_start_channel = group * channels_per_group;
                let group_end_channel = group_start_channel + channels_per_group;

                for c in group_start_channel..group_end_channel {
                    for spatial in 0..spatial_size {
                        let idx = batch * (channels * spatial_size) + c * spatial_size + spatial;
                        let val = input_data[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let mean = sum / group_elements as f32;
                let var = (sum_sq / group_elements as f32) - (mean * mean);

                let stat_idx = batch * self.num_groups + group;
                means[stat_idx] = mean;
                vars[stat_idx] = var;
            }
        }

        // Expand statistics to match channel dimension
        let mut expanded_means = vec![0.0f32; batch_size * channels];
        let mut expanded_vars = vec![0.0f32; batch_size * channels];

        for batch in 0..batch_size {
            for c in 0..channels {
                let group = c / channels_per_group;
                let stat_idx = batch * self.num_groups + group;
                let channel_idx = batch * channels + c;

                expanded_means[channel_idx] = means[stat_idx];
                expanded_vars[channel_idx] = vars[stat_idx];
            }
        }

        // Create result shape for broadcasting
        let mut result_shape = vec![batch_size, channels];
        for _ in 2..dims.len() {
            result_shape.push(1);
        }

        let mean_tensor =
            Tensor::from_data(expanded_means, vec![batch_size, channels], input.device())?
                .reshape(&result_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?;
        let var_tensor =
            Tensor::from_data(expanded_vars, vec![batch_size, channels], input.device())?
                .reshape(&result_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?;

        Ok((mean_tensor, var_tensor))
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Compute group statistics
        let (mean, var) = self.compute_group_stats(input)?;

        // Get learnable parameters
        let weight = if self.config.affine {
            if let Some(w) = self.base.parameters.get("weight") {
                // Reshape weight to match input dimensions for broadcasting
                let input_shape = input.shape();
                let dims = input_shape.dims();
                let mut weight_shape = vec![1, self.num_channels];
                for _ in 2..dims.len() {
                    weight_shape.push(1);
                }
                Some(
                    w.tensor()
                        .read()
                        .reshape(&weight_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?,
                )
            } else {
                None
            }
        } else {
            None
        };

        let bias = if self.config.affine {
            if let Some(b) = self.base.parameters.get("bias") {
                // Reshape bias to match input dimensions for broadcasting
                let input_shape = input.shape();
                let dims = input_shape.dims();
                let mut bias_shape = vec![1, self.num_channels];
                for _ in 2..dims.len() {
                    bias_shape.push(1);
                }
                Some(
                    b.tensor()
                        .read()
                        .reshape(&bias_shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Apply normalization
        utils::apply_normalization(
            input,
            &mean,
            &var,
            weight.as_ref(),
            bias.as_ref(),
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

// Re-export the layer and group normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::new(vec![128]).unwrap();
        assert_eq!(layer_norm.normalized_shape(), &[128]);
        assert_eq!(layer_norm.eps(), 1e-5);

        let layer_norm_2d = LayerNorm::new(vec![64, 64]).unwrap();
        assert_eq!(layer_norm_2d.normalized_shape(), &[64, 64]);
    }

    #[test]
    fn test_group_norm_creation() {
        let group_norm = GroupNorm::new(8, 32).unwrap();
        assert_eq!(group_norm.num_groups(), 8);
        assert_eq!(group_norm.num_channels(), 32);
        assert_eq!(group_norm.eps(), 1e-5);

        // Test invalid configuration
        assert!(GroupNorm::new(8, 30).is_err()); // 30 not divisible by 8
    }

    #[test]
    fn test_group_norm_shape_validation() {
        let group_norm = GroupNorm::new(4, 8).unwrap();

        // Valid input
        let input = zeros(&[2, 8, 16, 16]).unwrap();
        assert!(group_norm.forward(&input).is_ok());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 16, 16, 16]).unwrap();
        assert!(group_norm.forward(&input_wrong_channels).is_err());
    }
}
