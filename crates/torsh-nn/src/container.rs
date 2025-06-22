//! Container modules for organizing layers

use crate::{Module, ModuleBase};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Sequential container
pub struct Sequential {
    base: ModuleBase,
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create a new sequential container
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            modules: Vec::new(),
        }
    }

    /// Add a module to the sequential container
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Add a function as a module
    pub fn add_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    {
        self.modules.push(Box::new(FunctionModule::new(f)));
        self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();

        for module in &self.modules {
            output = module.forward(&output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params = Vec::new();

        for module in &self.modules {
            params.extend(module.parameters());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
        for module in &mut self.modules {
            module.train(mode);
        }
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for module in &mut self.modules {
            module.to(device)?;
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.iter().map(|m| m.as_ref()).collect()
    }
}

/// ModuleList container
pub struct ModuleList {
    base: ModuleBase,
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    /// Create a new module list
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            modules: Vec::new(),
        }
    }

    /// Add a module to the list
    pub fn append<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    /// Get the number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the list is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a module by index
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// Get a mutable module by index (placeholder - not implemented due to lifetime issues)
    /// TODO: Implement proper mutable access when needed
    pub fn get_mut(&mut self, _index: usize) -> Option<()> {
        // Placeholder implementation to avoid lifetime issues
        None
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleList {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TorshError::Other(
            "ModuleList is a container and does not define a forward method".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params = Vec::new();

        for module in &self.modules {
            params.extend(module.parameters());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
        for module in &mut self.modules {
            module.train(mode);
        }
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for module in &mut self.modules {
            module.to(device)?;
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.iter().map(|m| m.as_ref()).collect()
    }
}

/// ModuleDict container
pub struct ModuleDict {
    base: ModuleBase,
    modules: HashMap<String, Box<dyn Module>>,
}

impl ModuleDict {
    /// Create a new module dictionary
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            modules: HashMap::new(),
        }
    }

    /// Insert a module
    pub fn insert<M: Module + 'static>(&mut self, key: String, module: M) {
        self.modules.insert(key, Box::new(module));
    }

    /// Get a module by key
    pub fn get(&self, key: &str) -> Option<&dyn Module> {
        self.modules.get(key).map(|m| m.as_ref())
    }

    /// Get a mutable module by key (placeholder - not implemented due to lifetime issues)
    /// TODO: Implement proper mutable access when needed  
    pub fn get_mut(&mut self, _key: &str) -> Option<()> {
        // Placeholder implementation to avoid lifetime issues
        None
    }

    /// Remove a module
    pub fn remove(&mut self, key: &str) -> Option<Box<dyn Module>> {
        self.modules.remove(key)
    }

    /// Get the number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.modules.keys()
    }
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleDict {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TorshError::Other(
            "ModuleDict is a container and does not define a forward method".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params = Vec::new();

        for module in self.modules.values() {
            params.extend(module.parameters());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        let mut params = HashMap::new();

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }

        params
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
        for module in self.modules.values_mut() {
            module.train(mode);
        }
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        for module in self.modules.values_mut() {
            module.to(device)?;
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.values().map(|m| m.as_ref()).collect()
    }
}

/// Wrapper for functions to be used as modules
struct FunctionModule<F> {
    base: ModuleBase,
    function: F,
}

impl<F> FunctionModule<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    fn new(function: F) -> Self {
        Self {
            base: ModuleBase::new(),
            function,
        }
    }
}

impl<F> Module for FunctionModule<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        (self.function)(input)
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        Vec::new()
    }

    fn named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        HashMap::new()
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn training(&self) -> bool {
        self.base.training
    }

    fn to(&mut self, device: DeviceType) -> Result<()> {
        self.base.device = device;
        Ok(())
    }
}
