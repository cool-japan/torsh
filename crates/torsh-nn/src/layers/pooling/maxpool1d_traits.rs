//! # MaxPool1d - Trait Implementations
//!
//! This module contains trait implementations for `MaxPool1d`.
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

use super::types::MaxPool1d;

impl Module for MaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let stride = self.stride.unwrap_or(self.kernel_size);
        if input_shape.len() != 2 && input_shape.len() != 3 {
            return Err(torsh_core::TorshError::InvalidArgument(
                "MaxPool1d expects 2D [N, L] or 3D [N, C, L] input".to_string(),
            ));
        }
        let length_dim = input_shape.len() - 1;
        let input_length = input_shape[length_dim];
        let output_length = if self.ceil_mode {
            ((input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) as f32
                / stride as f32)
                .ceil() as usize
                + 1
        } else {
            (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / stride
                + 1
        };
        let mut output_shape = input_shape.to_vec();
        output_shape[length_dim] = output_length;
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

impl std::fmt::Debug for MaxPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool1d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("ceil_mode", &self.ceil_mode)
            .finish()
    }
}
