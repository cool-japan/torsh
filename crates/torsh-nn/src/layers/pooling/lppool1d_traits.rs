//! # LPPool1d - Trait Implementations
//!
//! This module contains trait implementations for `LPPool1d`.
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
use torsh_tensor::{creation::*, Tensor};

use super::types::LPPool1d;

impl Module for LPPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);
        let output_length = if self.ceil_mode {
            ((input_shape[2] - self.kernel_size) as f32 / stride as f32).ceil() as usize + 1
        } else {
            (input_shape[2] - self.kernel_size) / stride + 1
        };
        let output_shape = [input_shape[0], input_shape[1], output_length];
        let output = zeros(&output_shape)?;
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

impl std::fmt::Debug for LPPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LPPool1d")
            .field("norm_type", &self.norm_type)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .finish()
    }
}
