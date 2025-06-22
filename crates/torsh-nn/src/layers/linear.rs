//! Linear (fully connected) layers

use crate::{Module, ModuleBase, Parameter};
use parking_lot::RwLock;
use scirs2::neural::{activations as sci_act, layers as sci_layers};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_autograd::prelude::*;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
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
        let weight = crate::init::xavier_uniform(&[in_features, out_features]);
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_features]);
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

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("use_bias", &self.use_bias)
            .finish()
    }
}