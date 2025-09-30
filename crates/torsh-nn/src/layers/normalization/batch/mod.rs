//! Batch normalization layers and variants
//!
//! This module implements various batch normalization techniques:
//! - Standard batch normalization (1D, 2D, 3D)
//! - Synchronized batch normalization for distributed training
//! - Virtual batch normalization for stable training
//! - Batch renormalization for improved stability

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

use super::common::{utils, NormalizationConfig, NormalizationStats};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// 1D batch normalization layer
pub struct BatchNorm1d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<NormalizationStats>,
}

impl BatchNorm1d {
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

        // Initialize statistics if tracking
        let stats = if config.track_running_stats {
            let running_mean = zeros(&[num_features])?;
            let running_var = ones(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);
            base.register_buffer("running_var".to_string(), running_var);
            base.register_buffer("num_batches_tracked".to_string(), zeros(&[1])?);
            Some(NormalizationStats::new(num_features, true)?)
        } else {
            None
        };

        Ok(Self {
            base,
            num_features,
            config,
            stats,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    pub fn momentum(&self) -> f32 {
        self.config.momentum
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "BatchNorm1d expects 2D input (N, C), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute batch statistics
        let batch_mean = utils::compute_channel_mean(input)?;
        let batch_var = utils::compute_channel_variance(input, &batch_mean)?;

        // Use running statistics during inference
        let (mean, var) = if self.training() {
            (&batch_mean, &batch_var)
        } else if let Some(ref stats) = self.stats {
            if let (Some(ref running_mean), Some(ref running_var)) =
                (&stats.running_mean, &stats.running_var)
            {
                (running_mean, running_var)
            } else {
                (&batch_mean, &batch_var)
            }
        } else {
            (&batch_mean, &batch_var)
        };

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
            mean,
            var,
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

/// 2D batch normalization layer
pub struct BatchNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<NormalizationStats>,
}

impl std::fmt::Debug for BatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("num_features", &self.num_features)
            .field("training", &self.base.training())
            .finish()
    }
}

impl BatchNorm2d {
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

        // Initialize statistics if tracking
        let stats = if config.track_running_stats {
            let running_mean = zeros(&[num_features])?;
            let running_var = ones(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);
            base.register_buffer("running_var".to_string(), running_var);
            base.register_buffer("num_batches_tracked".to_string(), zeros(&[1])?);
            Some(NormalizationStats::new(num_features, true)?)
        } else {
            None
        };

        Ok(Self {
            base,
            num_features,
            config,
            stats,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    pub fn momentum(&self) -> f32 {
        self.config.momentum
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "BatchNorm2d expects 4D input (N, C, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute batch statistics
        let batch_mean = utils::compute_channel_mean(input)?;
        let batch_var = utils::compute_channel_variance(input, &batch_mean)?;

        // Use running statistics during inference
        let (mean, var) = if self.training() {
            (&batch_mean, &batch_var)
        } else if let Some(ref stats) = self.stats {
            if let (Some(ref running_mean), Some(ref running_var)) =
                (&stats.running_mean, &stats.running_var)
            {
                (running_mean, running_var)
            } else {
                (&batch_mean, &batch_var)
            }
        } else {
            (&batch_mean, &batch_var)
        };

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
            mean,
            var,
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

/// 3D batch normalization layer
pub struct BatchNorm3d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<NormalizationStats>,
}

impl BatchNorm3d {
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

        // Initialize statistics if tracking
        let stats = if config.track_running_stats {
            let running_mean = zeros(&[num_features])?;
            let running_var = ones(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);
            base.register_buffer("running_var".to_string(), running_var);
            base.register_buffer("num_batches_tracked".to_string(), zeros(&[1])?);
            Some(NormalizationStats::new(num_features, true)?)
        } else {
            None
        };

        Ok(Self {
            base,
            num_features,
            config,
            stats,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    pub fn momentum(&self) -> f32 {
        self.config.momentum
    }
}

impl Module for BatchNorm3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 5 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "BatchNorm3d expects 5D input (N, C, D, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        // Compute batch statistics
        let batch_mean = utils::compute_channel_mean(input)?;
        let batch_var = utils::compute_channel_variance(input, &batch_mean)?;

        // Use running statistics during inference
        let (mean, var) = if self.training() {
            (&batch_mean, &batch_var)
        } else if let Some(ref stats) = self.stats {
            if let (Some(ref running_mean), Some(ref running_var)) =
                (&stats.running_mean, &stats.running_var)
            {
                (running_mean, running_var)
            } else {
                (&batch_mean, &batch_var)
            }
        } else {
            (&batch_mean, &batch_var)
        };

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
            mean,
            var,
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

// Forward declarations for specialized batch norm variants (to be implemented)
pub struct SyncBatchNorm2d;
pub struct VirtualBatchNorm2d;
pub struct BatchRenorm2d;

// Re-export the batch normalization components (already defined in this module)
