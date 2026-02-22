//! # AdaptiveMaxPool3d - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveMaxPool3d`.
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

use super::types::AdaptiveMaxPool3d;

impl Module for AdaptiveMaxPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_indices(input)?;
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

impl std::fmt::Debug for AdaptiveMaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMaxPool3d")
            .field("output_size", &self.output_size)
            .field("return_indices", &self.return_indices)
            .finish()
    }
}
