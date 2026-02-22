//! # Conv2d - Trait Implementations
//!
//! This module contains trait implementations for `Conv2d`.
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

use super::types::Conv2d;

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let binding = input.shape();
        let input_shape = binding.dims();
        if input_shape.len() < 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Conv2d expects 4D input (batch_size, channels, height, width), got {}D: {:?}",
                input_shape.len(),
                input_shape
            )));
        }
        let output_height =
            (input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let output_width =
            (input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;
        let _output_shape = [
            input_shape[0],
            self.out_channels,
            output_height,
            output_width,
        ];
        let mut output = self.conv2d_im2col(input, &weight)?;
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
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

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("groups", &self.groups)
            .finish()
    }
}
