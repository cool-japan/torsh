//! # AdaptiveAvgPool2d - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveAvgPool2d`.
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

use super::types::AdaptiveAvgPool2d;

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        if input_shape.len() < 4 {
            return Err(
                torsh_core::error::TorshError::InvalidArgument(
                    format!(
                        "AdaptiveAvgPool2d expects 4D input (batch_size, channels, height, width), got {}D: {:?}",
                        input_shape.len(), input_shape
                    ),
                ),
            );
        }
        let output_height = self.output_size.0.unwrap_or(input_shape[2]);
        let output_width = self.output_size.1.unwrap_or(input_shape[3]);
        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];
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

impl std::fmt::Debug for AdaptiveAvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveAvgPool2d")
            .field("output_size", &self.output_size)
            .finish()
    }
}
