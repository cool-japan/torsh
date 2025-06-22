//! Regularization layers

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Dropout layer for regularization
pub struct Dropout {
    base: ModuleBase,
    p: f32,
    inplace: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace: false,
        }
    }

    pub fn with_inplace(p: f32, inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace,
        }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() {
            // During evaluation, just return the input
            return Ok(input.clone());
        }

        if self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return Ok(zeros(input.shape().dims()));
        }

        // Generate random mask for dropout
        // In a real implementation, this would use proper random number generation
        let keep_prob = 1.0 - self.p;
        let mask = full(input.shape().dims(), keep_prob); // Simplified - should be random

        let dropped = input.mul(&mask)?;
        // Scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / keep_prob;
        dropped.mul(&full(input.shape().dims(), scale))
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

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("inplace", &self.inplace)
            .finish()
    }
}
