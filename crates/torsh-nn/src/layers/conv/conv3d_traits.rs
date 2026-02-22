//! # Conv3d - Trait Implementations
//!
//! This module contains trait implementations for `Conv3d`.
//!
//! ## Implemented Traits
//!
//! - `Module`
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Module, Parameter};
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

use super::types::Conv3d;

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let mut output = self.conv3d_direct(input, &weight)?;
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .unsqueeze(4)?;
            output = output.add_op(&reshaped_bias)?;
        }
        Ok(output)
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

impl std::fmt::Debug for Conv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv3d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("groups", &self.groups)
            .finish()
    }
}
