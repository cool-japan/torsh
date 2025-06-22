//! Neural network modules for ToRSh
//!
//! This crate provides PyTorch-compatible neural network layers and modules,
//! built on top of scirs2-neural for optimized implementations.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod container;
pub mod functional;
pub mod init;
pub mod layers;
pub mod modules;
pub mod parameter;
#[cfg(feature = "serialize")]
pub mod serialization;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Re-export scirs2 neural functionality
use scirs2::neural as sci_nn;

/// Base trait for all neural network modules
pub trait Module: Send + Sync {
    /// Forward pass through the module
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all parameters in the module
    fn parameters(&self) -> HashMap<String, Parameter>;

    /// Get named parameters
    fn named_parameters(&self) -> HashMap<String, Parameter>;

    /// Check if in training mode
    fn training(&self) -> bool;

    /// Set training mode
    fn train(&mut self);

    /// Set evaluation mode
    fn eval(&mut self);

    /// Move module to device
    fn to_device(&mut self, device: DeviceType) -> Result<()>;

    /// Get all buffers (non-trainable parameters)
    fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    /// Get named buffers
    fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    /// Get all submodules
    fn children(&self) -> Vec<&dyn Module> {
        Vec::new()
    }

    /// Zero all gradients
    fn zero_grad(&mut self) {
        for param in self.parameters().values() {
            // This would zero gradients when autograd is fully implemented
        }
    }

    /// Get string representation
    fn extra_repr(&self) -> String {
        String::new()
    }
}

/// Extension trait for applying functions to modules (separate to maintain dyn compatibility)
pub trait ModuleApply {
    /// Apply a function to all submodules recursively
    fn apply<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&mut dyn Module) -> Result<()>;
}

/// Blanket implementation for all modules
impl<T: Module> ModuleApply for T {
    fn apply<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&mut dyn Module) -> Result<()>,
    {
        f(self)
    }
}

/// Parameter wrapper for trainable tensors
#[derive(Clone)]
pub struct Parameter {
    data: Arc<RwLock<Tensor>>,
    requires_grad: bool,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(tensor: Tensor) -> Self {
        Self {
            data: Arc::new(RwLock::new(tensor)),
            requires_grad: true,
        }
    }

    /// Create a parameter that doesn't require gradients
    pub fn new_no_grad(tensor: Tensor) -> Self {
        Self {
            data: Arc::new(RwLock::new(tensor)),
            requires_grad: false,
        }
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> Arc<RwLock<Tensor>> {
        self.data.clone()
    }

    /// Set whether this parameter requires gradients
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        // Note: torsh_tensor doesn't support requires_grad yet
        // This will be implemented when autograd is available
        self
    }
}

/// Base module implementation helper
pub struct ModuleBase {
    training: bool,
    device: DeviceType,
    pub parameters: HashMap<String, Parameter>,
    buffers: HashMap<String, Arc<RwLock<Tensor>>>,
    modules: HashMap<String, Box<dyn Module>>,
}

impl Default for ModuleBase {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleBase {
    pub fn new() -> Self {
        Self {
            training: true,
            device: DeviceType::Cpu,
            parameters: HashMap::new(),
            buffers: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    /// Check if in training mode
    pub fn training(&self) -> bool {
        self.training
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in self.modules.values_mut() {
            if training {
                module.train();
            } else {
                module.eval();
            }
        }
    }

    /// Get named parameters
    pub fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters.clone()
    }

    /// Move to device
    pub fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.device = device;
        // In a full implementation, would move all parameters to device
        for module in self.modules.values_mut() {
            module.to_device(device)?;
        }
        Ok(())
    }

    /// Register a parameter
    pub fn register_parameter(&mut self, name: String, param: Parameter) {
        self.parameters.insert(name, param);
    }

    /// Register a buffer
    pub fn register_buffer(&mut self, name: String, tensor: Tensor) {
        self.buffers.insert(name, Arc::new(RwLock::new(tensor)));
    }

    /// Register a submodule
    pub fn register_module(&mut self, name: String, module: Box<dyn Module>) {
        self.modules.insert(name, module);
    }

    /// Get all parameters including submodules
    pub fn all_parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params: Vec<_> = self.parameters.values().map(|p| p.tensor()).collect();

        for module in self.modules.values() {
            let module_params = module.parameters();
            for param in module_params.values() {
                params.push(param.tensor());
            }
        }

        params
    }

    /// Get all named parameters including submodules
    pub fn all_named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        let mut params = HashMap::new();

        for (name, param) in &self.parameters {
            params.insert(name.clone(), param.tensor());
        }

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param.tensor());
            }
        }

        params
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::container::*;
    pub use crate::init;
    pub use crate::modules::*;
    pub use crate::{Module, ModuleBase, Parameter};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter() {
        let tensor = torsh_tensor::creation::ones(&[3, 4]);
        let param = Parameter::new(tensor);
        assert!(param.requires_grad);
    }
}
