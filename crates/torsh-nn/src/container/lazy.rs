//! Lazy container modules that defer initialization until first use

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

// Use parking_lot::Mutex for both std and no_std
use parking_lot::Mutex;

/// Lazy Sequential container that defers module creation until first forward pass
pub struct LazySequential {
    base: ModuleBase,
    /// Factory functions that create modules based on input shape
    module_factories: Vec<Box<dyn Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync>>,
    /// Initialized modules (None until first forward pass)
    modules: Mutex<Option<Vec<Box<dyn Module>>>>,
    /// Track if initialization has occurred
    initialized: Mutex<bool>,
}

impl LazySequential {
    /// Create a new lazy sequential container
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            module_factories: Vec::new(),
            modules: Mutex::new(None),
            initialized: Mutex::new(false),
        }
    }

    /// Add a module factory function that creates a module based on input shape
    pub fn add_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync + 'static,
    {
        self.module_factories.push(Box::new(factory));
        self
    }

    /// Add a pre-built module wrapped in a factory
    pub fn add_module<M: Module + 'static>(mut self, module: M) -> Self {
        let module = Arc::new(Mutex::new(Some(module)));
        self.module_factories.push(Box::new(move |_shape| {
            if let Some(m) = module.lock().take() {
                Ok(Box::new(m) as Box<dyn Module>)
            } else {
                Err(TorshError::Other("Module already used".to_string()))
            }
        }));
        self
    }

    /// Initialize all modules based on input shape
    fn initialize(&self, input_shape: &[usize]) -> Result<()> {
        let mut modules_guard = self.modules.lock();
        if modules_guard.is_some() {
            return Ok(()); // Already initialized
        }

        let mut modules = Vec::new();
        let mut current_shape = input_shape.to_vec();

        for factory in &self.module_factories {
            let module = factory(&current_shape)?;

            // Create a dummy input to infer the output shape
            let dummy_input = torsh_tensor::creation::zeros(&current_shape)?;
            let dummy_output = module.forward(&dummy_input)?;
            current_shape = dummy_output.shape().dims().to_vec();

            modules.push(module);
        }

        *modules_guard = Some(modules);
        *self.initialized.lock() = true;
        Ok(())
    }

    /// Check if the container is initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock()
    }

    /// Get the number of module factories
    pub fn len(&self) -> usize {
        self.module_factories.len()
    }

    /// Check if the container is empty
    pub fn is_empty(&self) -> bool {
        self.module_factories.is_empty()
    }
}

impl Default for LazySequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LazySequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Initialize if not already done
        if !self.is_initialized() {
            self.initialize(input.shape().dims())?;
        }

        let modules_guard = self.modules.lock();
        let modules = modules_guard.as_ref().unwrap();

        let mut output = input.clone();
        for module in modules {
            output = module.forward(&output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (i, module) in modules.iter().enumerate() {
                for (name, param) in module.parameters() {
                    params.insert(format!("{}.{}", i, name), param);
                }
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (i, module) in modules.iter().enumerate() {
                for (name, param) in module.named_parameters() {
                    params.insert(format!("{}.{}", i, name), param);
                }
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.train();
            }
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.eval();
            }
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.set_training(training);
            }
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.to_device(device)?;
            }
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        // Return empty vector for now to avoid lifetime issues
        Vec::new()
    }
}

/// Lazy ModuleList that supports deferred module creation
pub struct LazyModuleList {
    base: ModuleBase,
    /// Factory functions for creating modules
    module_factories: Vec<Box<dyn Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync>>,
    /// Initialized modules
    modules: Mutex<Option<Vec<Box<dyn Module>>>>,
    /// Track initialization state
    initialized: Mutex<bool>,
}

impl LazyModuleList {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            module_factories: Vec::new(),
            modules: Mutex::new(None),
            initialized: Mutex::new(false),
        }
    }

    /// Add a module factory
    pub fn push_factory<F>(&mut self, factory: F)
    where
        F: Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync + 'static,
    {
        self.module_factories.push(Box::new(factory));
    }

    /// Add a pre-built module wrapped in a factory
    pub fn push_module<M: Module + 'static>(&mut self, module: M) {
        let module = Arc::new(Mutex::new(Some(module)));
        self.module_factories.push(Box::new(move |_shape| {
            if let Some(m) = module.lock().take() {
                Ok(Box::new(m) as Box<dyn Module>)
            } else {
                Err(TorshError::Other("Module already used".to_string()))
            }
        }));
    }

    /// Initialize all modules with given input shape
    pub fn initialize_lazy(&self, input_shape: &[usize]) -> Result<()> {
        let mut modules_guard = self.modules.lock();
        if modules_guard.is_some() {
            return Ok(());
        }

        let mut modules = Vec::new();
        for factory in &self.module_factories {
            modules.push(factory(input_shape)?);
        }

        *modules_guard = Some(modules);
        *self.initialized.lock() = true;
        Ok(())
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock()
    }

    /// Get number of factories/modules
    pub fn len(&self) -> usize {
        self.module_factories.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.module_factories.is_empty()
    }

    /// Get a module by index (only after initialization)
    pub fn get(&self, _index: usize) -> Option<&dyn Module> {
        // Return None for now to avoid lifetime issues
        None
    }

    /// Apply a function to a specific module (only after initialization)
    pub fn apply_to_module<F, R>(&self, index: usize, f: F) -> Option<R>
    where
        F: FnOnce(&dyn Module) -> R,
    {
        if let Some(modules) = self.modules.lock().as_ref() {
            modules.get(index).map(|m| f(m.as_ref()))
        } else {
            None
        }
    }
}

impl Default for LazyModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LazyModuleList {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TorshError::InvalidArgument(
            "LazyModuleList doesn't define forward pass - initialize and use modules individually"
                .to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (i, module) in modules.iter().enumerate() {
                for (name, param) in module.parameters() {
                    params.insert(format!("{}.{}", i, name), param);
                }
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (i, module) in modules.iter().enumerate() {
                for (name, param) in module.named_parameters() {
                    params.insert(format!("{}.{}", i, name), param);
                }
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.train();
            }
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.eval();
            }
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.set_training(training);
            }
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules {
                module.to_device(device)?;
            }
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        // Return empty vector for now to avoid lifetime issues
        Vec::new()
    }
}

/// Lazy ModuleDict that supports deferred module creation
pub struct LazyModuleDict {
    base: ModuleBase,
    /// Factory functions for creating modules
    module_factories:
        HashMap<String, Box<dyn Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync>>,
    /// Initialized modules
    modules: Mutex<Option<HashMap<String, Box<dyn Module>>>>,
    /// Track initialization state
    initialized: Mutex<bool>,
}

impl LazyModuleDict {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            module_factories: HashMap::new(),
            modules: Mutex::new(None),
            initialized: Mutex::new(false),
        }
    }

    /// Insert a module factory
    pub fn insert_factory<F>(&mut self, key: String, factory: F)
    where
        F: Fn(&[usize]) -> Result<Box<dyn Module>> + Send + Sync + 'static,
    {
        self.module_factories.insert(key, Box::new(factory));
    }

    /// Insert a pre-built module wrapped in a factory
    pub fn insert_module<M: Module + 'static>(&mut self, key: String, module: M) {
        let module = Arc::new(Mutex::new(Some(module)));
        self.module_factories.insert(
            key,
            Box::new(move |_shape| {
                if let Some(m) = module.lock().take() {
                    Ok(Box::new(m) as Box<dyn Module>)
                } else {
                    Err(TorshError::Other("Module already used".to_string()))
                }
            }),
        );
    }

    /// Initialize all modules with given input shape
    pub fn initialize_lazy(&self, input_shape: &[usize]) -> Result<()> {
        let mut modules_guard = self.modules.lock();
        if modules_guard.is_some() {
            return Ok(());
        }

        let mut modules = HashMap::new();
        for (key, factory) in &self.module_factories {
            modules.insert(key.clone(), factory(input_shape)?);
        }

        *modules_guard = Some(modules);
        *self.initialized.lock() = true;
        Ok(())
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock()
    }

    /// Get number of factories/modules
    pub fn len(&self) -> usize {
        self.module_factories.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.module_factories.is_empty()
    }

    /// Get a module by key (only after initialization)
    pub fn get(&self, _key: &str) -> Option<&dyn Module> {
        // Return None for now to avoid lifetime issues
        None
    }

    /// Apply a function to a specific module (only after initialization)
    pub fn apply_to_module<F, R>(&self, key: &str, f: F) -> Option<R>
    where
        F: FnOnce(&dyn Module) -> R,
    {
        if let Some(modules) = self.modules.lock().as_ref() {
            modules.get(key).map(|m| f(m.as_ref()))
        } else {
            None
        }
    }

    /// Get keys from factories
    pub fn factory_keys(&self) -> impl Iterator<Item = &String> {
        self.module_factories.keys()
    }

    /// Get keys from initialized modules (empty if not initialized)
    pub fn module_keys(&self) -> Vec<String> {
        if let Some(modules) = self.modules.lock().as_ref() {
            modules.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for LazyModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LazyModuleDict {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TorshError::InvalidArgument(
            "LazyModuleDict doesn't define forward pass - initialize and use modules individually"
                .to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (module_name, module) in modules {
                for (param_name, param) in module.parameters() {
                    params.insert(format!("{}.{}", module_name, param_name), param);
                }
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(modules) = self.modules.lock().as_ref() {
            for (module_name, module) in modules {
                for (param_name, param) in module.named_parameters() {
                    params.insert(format!("{}.{}", module_name, param_name), param);
                }
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules.values_mut() {
                module.train();
            }
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules.values_mut() {
                module.eval();
            }
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules.values_mut() {
                module.set_training(training);
            }
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        if let Some(modules) = self.modules.lock().as_mut() {
            for module in modules.values_mut() {
                module.to_device(device)?;
            }
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        // Return empty vector for now to avoid lifetime issues
        Vec::new()
    }
}
