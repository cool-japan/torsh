//! Basic container modules for organizing layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Sequential container
pub struct Sequential {
    base: ModuleBase,
    modules: Vec<Box<dyn Module>>,
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("modules_count", &self.modules.len())
            .field("training", &self.base.training())
            .finish()
    }
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
    #[allow(clippy::should_implement_trait)]
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

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        for module in &mut self.modules {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        for module in &mut self.modules {
            module.eval();
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        for module in &mut self.modules {
            module.to_device(device)?;
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

impl std::fmt::Debug for ModuleList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleList")
            .field("modules_count", &self.modules.len())
            .field("training", &self.base.training())
            .finish()
    }
}

impl ModuleList {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            modules: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn extend<I>(&mut self, modules: I)
    where
        I: IntoIterator<Item = Box<dyn Module>>,
    {
        self.modules.extend(modules);
    }

    pub fn get(&self, _index: usize) -> Option<&dyn Module> {
        self.modules.get(_index).map(|m| m.as_ref())
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut (dyn Module + '_)> {
        if let Some(m) = self.modules.get_mut(index) {
            Some(&mut **m)
        } else {
            None
        }
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleList {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // ModuleList doesn't define forward pass - each module should be called individually
        Err(TorshError::InvalidArgument(
            "ModuleList doesn't define forward pass".to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", i, name), param);
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        for module in &mut self.modules {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        for module in &mut self.modules {
            module.eval();
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        for module in &mut self.modules {
            module.to_device(device)?;
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

impl std::fmt::Debug for ModuleDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleDict")
            .field("modules_count", &self.modules.len())
            .field("training", &self.base.training())
            .finish()
    }
}

impl ModuleDict {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            modules: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn insert<M: Module + 'static>(&mut self, key: String, module: M) {
        self.modules.insert(key, Box::new(module));
    }

    pub fn get(&self, key: &str) -> Option<&dyn Module> {
        self.modules.get(key).map(|m| m.as_ref())
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut (dyn Module + '_)> {
        if let Some(m) = self.modules.get_mut(key) {
            Some(&mut **m)
        } else {
            None
        }
    }

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
        // ModuleDict doesn't define forward pass - each module should be called individually
        Err(TorshError::InvalidArgument(
            "ModuleDict doesn't define forward pass".to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        for module in self.modules.values_mut() {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        for module in self.modules.values_mut() {
            module.eval();
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        for module in self.modules.values_mut() {
            module.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        for module in self.modules.values_mut() {
            module.to_device(device)?;
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.values().map(|m| m.as_ref()).collect()
    }
}

/// Function module wrapper
pub struct FunctionModule<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    base: ModuleBase,
    func: F,
}

impl<F> FunctionModule<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            base: ModuleBase::new(),
            func,
        }
    }
}

impl<F> Module for FunctionModule<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        (self.func)(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}
