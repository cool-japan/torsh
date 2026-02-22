//! # DepthwiseSeparableConv - Trait Implementations
//!
//! This module contains trait implementations for `DepthwiseSeparableConv`.
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

use super::types::DepthwiseSeparableConv;

impl Module for DepthwiseSeparableConv {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let depthwise_out = self.depthwise.forward(input)?;
        let output = self.pointwise.forward(&depthwise_out)?;
        Ok(output)
    }
    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.depthwise.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }
        for (name, param) in self.pointwise.parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }
        params
    }
    fn training(&self) -> bool {
        self.depthwise.training()
    }
    fn train(&mut self) {
        self.depthwise.train();
        self.pointwise.train();
    }
    fn eval(&mut self) {
        self.depthwise.eval();
        self.pointwise.eval();
    }
    fn set_training(&mut self, training: bool) {
        self.depthwise.set_training(training);
        self.pointwise.set_training(training);
    }
    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.depthwise.to_device(device)?;
        self.pointwise.to_device(device)?;
        Ok(())
    }
    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.depthwise.named_parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }
        for (name, param) in self.pointwise.named_parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }
        params
    }
}

impl std::fmt::Debug for DepthwiseSeparableConv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthwiseSeparableConv")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}
