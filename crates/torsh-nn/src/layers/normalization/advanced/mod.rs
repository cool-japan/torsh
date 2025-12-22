//! Advanced normalization techniques
//!
//! This module provides sophisticated normalization methods that combine or adapt
//! multiple normalization strategies for improved performance.

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

/// Switchable normalization that learns to combine different normalization techniques
///
/// This layer learns weights to combine batch normalization, instance normalization,
/// and layer normalization adaptively for each channel.
pub struct SwitchableNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    #[allow(dead_code)]
    using_movavg: bool,
}

impl SwitchableNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize switchable weights for combining different normalizations
        let switch_weight = ones(&[3, num_features])?; // 3 normalization types
        base.register_parameter("switch_weight".to_string(), Parameter::new(switch_weight));

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        // Initialize running statistics for batch norm component
        if config.track_running_stats {
            let running_mean = zeros(&[num_features])?;
            let running_var = ones(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);
            base.register_buffer("running_var".to_string(), running_var);
            base.register_buffer("num_batches_tracked".to_string(), zeros(&[1])?);
        }

        let using_movavg = config.track_running_stats;

        Ok(Self {
            base,
            num_features,
            config,
            using_movavg,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    /// Compute batch normalization statistics
    fn compute_batch_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        utils::compute_channel_mean(input)
            .and_then(|mean| utils::compute_channel_variance(input, &mean).map(|var| (mean, var)))
    }

    /// Compute instance normalization statistics
    fn compute_instance_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
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

        let mean_tensor =
            Tensor::from_data(means, vec![batch_size, channels, 1, 1], input.device())?;
        let var_tensor = Tensor::from_data(vars, vec![batch_size, channels, 1, 1], input.device())?;

        Ok((mean_tensor, var_tensor))
    }

    /// Compute layer normalization statistics
    fn compute_layer_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let dims = input_shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let input_data = input.to_vec()?;
        let mut means = vec![0.0f32; batch_size];
        let mut vars = vec![0.0f32; batch_size];

        let layer_size = (channels * height * width) as f32;

        // Compute mean and variance for each sample across all channels and spatial dims
        for batch in 0..batch_size {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            let batch_start = batch * (channels * height * width);
            for i in 0..(channels * height * width) {
                let val = input_data[batch_start + i];
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / layer_size;
            let var = (sum_sq / layer_size) - (mean * mean);

            means[batch] = mean;
            vars[batch] = var;
        }

        let mean_tensor = Tensor::from_data(means, vec![batch_size, 1, 1, 1], input.device())?;
        let var_tensor = Tensor::from_data(vars, vec![batch_size, 1, 1, 1], input.device())?;

        Ok((mean_tensor, var_tensor))
    }

    /// Apply switchable normalization
    fn apply_switchable_norm(&self, input: &Tensor) -> Result<Tensor> {
        // Compute statistics for all three normalization types
        let (bn_mean, bn_var) = self.compute_batch_norm_stats(input)?;
        let (in_mean, in_var) = self.compute_instance_norm_stats(input)?;
        let (ln_mean, ln_var) = self.compute_layer_norm_stats(input)?;

        // Get switch weights and apply softmax to normalize
        let switch_weight = self.base.parameters.get("switch_weight").ok_or_else(|| {
            torsh_core::error::TorshError::InvalidOperation(
                "Switch weight parameter not found".to_string(),
            )
        })?;

        let switch_data = switch_weight.tensor().read().to_vec()?;
        let mut normalized_weights = vec![0.0f32; switch_data.len()];

        // Apply softmax for each channel (3 weights per channel)
        for c in 0..self.num_features {
            let mut max_val = switch_data[c];
            for norm_type in 1..3 {
                let idx = norm_type * self.num_features + c;
                if switch_data[idx] > max_val {
                    max_val = switch_data[idx];
                }
            }

            let mut sum = 0.0;
            for norm_type in 0..3 {
                let idx = norm_type * self.num_features + c;
                let exp_val = (switch_data[idx] - max_val).exp();
                normalized_weights[idx] = exp_val;
                sum += exp_val;
            }

            for norm_type in 0..3 {
                let idx = norm_type * self.num_features + c;
                normalized_weights[idx] /= sum;
            }
        }

        // Expand means and variances for broadcasting
        let bn_mean_expanded = bn_mean.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let bn_var_expanded = bn_var.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;

        // Combine statistics using learned weights
        let input_shape = input.shape();
        let dims = input_shape.dims();
        let mut combined_mean_data = vec![0.0f32; dims.iter().product()];
        let mut combined_var_data = vec![0.0f32; dims.iter().product()];

        let bn_mean_data = bn_mean_expanded.to_vec()?;
        let bn_var_data = bn_var_expanded.to_vec()?;
        let in_mean_data = in_mean.to_vec()?;
        let in_var_data = in_var.to_vec()?;
        let ln_mean_data = ln_mean.to_vec()?;
        let ln_var_data = ln_var.to_vec()?;

        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        for batch in 0..batch_size {
            for c in 0..channels {
                let bn_weight = normalized_weights[c];
                let in_weight = normalized_weights[self.num_features + c];
                let ln_weight = normalized_weights[2 * self.num_features + c];

                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w;

                        // Combine means
                        let bn_idx = c;
                        let in_idx = batch * channels + c;
                        let ln_idx = batch;

                        combined_mean_data[idx] = bn_weight * bn_mean_data[bn_idx]
                            + in_weight * in_mean_data[in_idx]
                            + ln_weight * ln_mean_data[ln_idx];

                        // Combine variances
                        combined_var_data[idx] = bn_weight * bn_var_data[bn_idx]
                            + in_weight * in_var_data[in_idx]
                            + ln_weight * ln_var_data[ln_idx];
                    }
                }
            }
        }

        let combined_mean = Tensor::from_data(combined_mean_data, dims.to_vec(), input.device())?;
        let combined_var = Tensor::from_data(combined_var_data, dims.to_vec(), input.device())?;

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

        // Apply final normalization
        let weight_tensor = weight.as_ref().map(|p| p.tensor().read().clone());
        let bias_tensor = bias.as_ref().map(|p| p.tensor().read().clone());

        utils::apply_normalization(
            input,
            &combined_mean,
            &combined_var,
            weight_tensor.as_ref(),
            bias_tensor.as_ref(),
            self.config.eps,
        )
    }
}

impl Module for SwitchableNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "SwitchableNorm2d expects 4D input (N, C, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        self.apply_switchable_norm(input)
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

// Re-export the advanced normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switchable_norm_creation() {
        let switchable_norm = SwitchableNorm2d::new(64).unwrap();
        assert_eq!(switchable_norm.num_features(), 64);
        assert_eq!(switchable_norm.eps(), 1e-5);
    }

    #[test]
    fn test_switchable_norm_shape_validation() {
        let switchable_norm = SwitchableNorm2d::new(3).unwrap();

        // Valid input
        let input = zeros(&[2, 3, 32, 32]).unwrap();
        assert!(switchable_norm.forward(&input).is_ok());

        // Invalid dimensions
        let input_3d = zeros(&[2, 3, 32]).unwrap();
        assert!(switchable_norm.forward(&input_3d).is_err());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 4, 32, 32]).unwrap();
        assert!(switchable_norm.forward(&input_wrong_channels).is_err());
    }
}
