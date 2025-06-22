//! Normalization layers

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

/// 2D batch normalization layer
pub struct BatchNorm2d {
    base: ModuleBase,
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        let mut base = ModuleBase::new();
        
        // Initialize parameters
        let weight = ones(&[num_features]);
        let bias = zeros(&[num_features]);
        let running_mean = zeros(&[num_features]);
        let running_var = ones(&[num_features]);
        
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        base.register_parameter("bias".to_string(), Parameter::new(bias));
        base.register_buffer("running_mean".to_string(), running_mean);
        base.register_buffer("running_var".to_string(), running_var);
        base.register_buffer("num_batches_tracked".to_string(), zeros(&[1]));

        Self {
            base,
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }

    pub fn with_config(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
    ) -> Self {
        let mut bn = Self::new(num_features);
        bn.eps = eps;
        bn.momentum = momentum;
        bn.affine = affine;
        bn.track_running_stats = track_running_stats;
        bn
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Batch normalization for 2D inputs
        // Input shape: [batch_size, num_features, height, width]
        
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let bias = self.base.parameters["bias"].tensor().read().clone();
        
        // Simplified batch norm - real implementation would handle training/eval modes
        let normalized = input.clone(); // Placeholder
        
        if self.affine {
            let weight_expanded = weight.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
            let bias_expanded = bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
            let scaled = normalized.mul(&weight_expanded)?;
            scaled.add(&bias_expanded)
        } else {
            Ok(normalized)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Layer normalization
pub struct LayerNorm {
    base: ModuleBase,
    normalized_shape: Vec<usize>,
    eps: f32,
    elementwise_affine: bool,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let mut base = ModuleBase::new();
        
        let num_elements: usize = normalized_shape.iter().product();
        let weight = ones(&[num_elements]);
        let bias = zeros(&[num_elements]);
        
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        base.register_parameter("bias".to_string(), Parameter::new(bias));

        Self {
            base,
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    pub fn with_config(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
    ) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.eps = eps;
        ln.elementwise_affine = elementwise_affine;
        ln
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Layer normalization
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let bias = self.base.parameters["bias"].tensor().read().clone();
        
        // Simplified layer norm - real implementation would normalize over specified dimensions
        let normalized = input.clone(); // Placeholder
        
        if self.elementwise_affine {
            let scaled = normalized.mul(&weight)?;
            scaled.add(&bias)
        } else {
            Ok(normalized)
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for BatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .finish()
    }
}

impl std::fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm")
            .field("normalized_shape", &self.normalized_shape)
            .field("eps", &self.eps)
            .finish()
    }
}