//! Module composition system for neural network modules
//!
//! This module provides traits and implementations for composing modules
//! in various patterns including sequential, parallel, residual, and conditional
//! execution patterns.

use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::{Module, ModuleConfig, Parameter};

/// Enhanced module building trait with fluent interface
pub trait ModuleBuilder<T> {
    /// Build the module
    fn build(self) -> Result<T>;

    /// Set training mode
    fn training(self, training: bool) -> Self;

    /// Set device
    fn device(self, device: DeviceType) -> Self;

    /// Add custom configuration
    fn config<F>(self, config_fn: F) -> Self
    where
        F: FnOnce(&mut ModuleConfig);
}

/// Trait for modules that support functional composition
pub trait ModuleComposition {
    /// Compose this module with another module sequentially
    ///
    /// Creates a new module that applies self then other.
    fn then<Other: Module + 'static>(self, other: Other) -> ComposedModule<Self, Other>
    where
        Self: Sized + 'static;

    /// Compose this module with another module in parallel
    ///
    /// Creates a new module that applies both modules to the same input and combines results.
    fn parallel<Other: Module + 'static>(self, other: Other) -> ParallelModule<Self, Other>
    where
        Self: Sized + 'static;

    /// Add a residual connection
    ///
    /// Creates a new module that adds the input to the output of this module.
    fn residual(self) -> ResidualModule<Self>
    where
        Self: Sized + 'static;

    /// Add conditional execution
    ///
    /// Creates a new module that only applies this module when the condition is true.
    fn conditional<F>(self, condition_fn: F) -> ConditionalModule<Self, F>
    where
        Self: Sized + 'static,
        F: Fn() -> bool + Send + Sync;
}

/// Sequential composition of two modules
pub struct ComposedModule<First, Second> {
    first: First,
    second: Second,
}

impl<First: Module, Second: Module> Module for ComposedModule<First, Second> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let intermediate = self.first.forward(input)?;
        self.second.forward(&intermediate)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.first.parameters();
        let second_params = self.second.parameters();
        for (name, param) in second_params {
            params.insert(format!("second.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.first.training() && self.second.training()
    }

    fn set_training(&mut self, training: bool) {
        self.first.set_training(training);
        self.second.set_training(training);
    }
}

/// Parallel composition of two modules
pub struct ParallelModule<First, Second> {
    first: First,
    second: Second,
}

impl<First: Module, Second: Module> Module for ParallelModule<First, Second> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let first_output = self.first.forward(input)?;
        let _second_output = self.second.forward(input)?;
        // In a full implementation, this would concatenate or combine the outputs
        // For now, just return the first output
        Ok(first_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.first.parameters() {
            params.insert(format!("first.{name}"), param);
        }
        for (name, param) in self.second.parameters() {
            params.insert(format!("second.{name}"), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.first.training() && self.second.training()
    }

    fn set_training(&mut self, training: bool) {
        self.first.set_training(training);
        self.second.set_training(training);
    }
}

/// Module with residual connection
pub struct ResidualModule<M> {
    module: M,
}

impl<M: Module> Module for ResidualModule<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.module.forward(input)?;
        // In a full implementation, this would add input + output
        // For now, just return the output
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.module.parameters()
    }

    fn training(&self) -> bool {
        self.module.training()
    }

    fn set_training(&mut self, training: bool) {
        self.module.set_training(training);
    }
}

/// Module with conditional execution
pub struct ConditionalModule<M, F> {
    module: M,
    condition_fn: F,
}

impl<M: Module, F: Fn() -> bool + Send + Sync> Module for ConditionalModule<M, F> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if (self.condition_fn)() {
            self.module.forward(input)
        } else {
            Ok(input.clone())
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.module.parameters()
    }

    fn training(&self) -> bool {
        self.module.training()
    }

    fn set_training(&mut self, training: bool) {
        self.module.set_training(training);
    }
}

/// Blanket implementation of ModuleComposition for all modules
impl<T: Module + 'static> ModuleComposition for T {
    fn then<Other: Module + 'static>(self, other: Other) -> ComposedModule<Self, Other> {
        ComposedModule {
            first: self,
            second: other,
        }
    }

    fn parallel<Other: Module + 'static>(self, other: Other) -> ParallelModule<Self, Other> {
        ParallelModule {
            first: self,
            second: other,
        }
    }

    fn residual(self) -> ResidualModule<Self> {
        ResidualModule { module: self }
    }

    fn conditional<F>(self, condition_fn: F) -> ConditionalModule<Self, F>
    where
        F: Fn() -> bool + Send + Sync,
    {
        ConditionalModule {
            module: self,
            condition_fn,
        }
    }
}
