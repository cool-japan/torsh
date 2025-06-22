//! Neural network modules for ToRSh
//!
//! This crate provides PyTorch-compatible neural network layers and modules,
//! built on top of scirs2-neural for optimized implementations.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod modules;
pub mod init;
pub mod functional;
pub mod container;
pub mod parameter;
#[cfg(feature = "serialize")]
pub mod serialization;

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
use torsh_core::device::DeviceType;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

// Re-export scirs2 neural functionality
use scirs2::neural as sci_nn;

/// Base trait for all neural network modules
pub trait Module: Send + Sync {
    /// Forward pass through the module
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    
    /// Get all parameters in the module
    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>>;
    
    /// Get named parameters
    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>>;
    
    /// Get all buffers (non-trainable parameters)
    fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }
    
    /// Get named buffers
    fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }
    
    /// Set training mode
    fn train(&mut self, mode: bool);
    
    /// Check if in training mode
    fn training(&self) -> bool;
    
    /// Move module to device
    fn to(&mut self, device: DeviceType) -> Result<()>;
    
    /// Get all submodules
    fn children(&self) -> Vec<&dyn Module> {
        Vec::new()
    }
    
    /// Zero all gradients
    fn zero_grad(&mut self) {
        for param in self.parameters() {
            param.write().zero_grad();
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
    parameters: HashMap<String, Parameter>,
    buffers: HashMap<String, Arc<RwLock<Tensor>>>,
    modules: HashMap<String, Box<dyn Module>>,
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
        let mut params: Vec<_> = self.parameters.values()
            .map(|p| p.tensor())
            .collect();
        
        for module in self.modules.values() {
            params.extend(module.parameters());
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
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }
        
        params
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{Module, Parameter, ModuleBase};
    pub use crate::modules::*;
    pub use crate::container::*;
    pub use crate::init;
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