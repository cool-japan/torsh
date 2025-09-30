//! Module base infrastructure for neural network implementations
//!
//! This module provides the ModuleBase helper struct that serves as a foundation
//! for implementing neural network modules with integrated parameter management,
//! hook system, and training state.

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::{HookCallback, HookHandle, HookRegistry, HookType, Parameter};

/// Base module implementation helper
pub struct ModuleBase {
    training: bool,
    device: DeviceType,
    pub parameters: HashMap<String, Parameter>,
    buffers: HashMap<String, Arc<RwLock<Tensor>>>,
    modules: HashMap<String, Box<dyn crate::Module>>,
    hook_registry: HookRegistry,
}

impl core::fmt::Debug for ModuleBase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModuleBase")
            .field("training", &self.training)
            .field("device", &self.device)
            .field("parameters_count", &self.parameters.len())
            .field("buffers_count", &self.buffers.len())
            .field("modules_count", &self.modules.len())
            .field("hook_registry", &self.hook_registry)
            .finish()
    }
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
            hook_registry: HookRegistry::new(),
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
            module.set_training(training);
        }
    }

    /// Apply function to all parameters in this module
    pub fn apply_to_parameters<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut Parameter) -> Result<()>,
    {
        use crate::ModuleApply;
        for param in self.parameters.values_mut() {
            f(param)?;
        }
        for module in self.modules.values_mut() {
            module.apply_to_parameters(&f)?;
        }
        Ok(())
    }

    /// Apply function to all modules recursively
    pub fn apply_to_modules<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut dyn crate::Module) -> Result<()>,
    {
        use crate::ModuleApply;
        for module in self.modules.values_mut() {
            f(module.as_mut())?;
            module.apply_to_modules(&f)?;
        }
        Ok(())
    }

    /// Get children modules as references
    pub fn children(&self) -> Vec<&dyn crate::Module> {
        self.modules.values().map(|m| m.as_ref()).collect()
    }

    /// Get named children modules
    pub fn named_children(&self) -> Vec<(String, &dyn crate::Module)> {
        self.modules
            .iter()
            .map(|(name, module)| (name.clone(), module.as_ref()))
            .collect()
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
    pub fn register_module(&mut self, name: String, module: Box<dyn crate::Module>) {
        self.modules.insert(name, module);
    }

    /// Get all parameters including submodules (legacy method)
    pub fn all_parameter_tensors(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params: Vec<_> = self.parameters.values().map(|p| p.tensor()).collect();

        for module in self.modules.values() {
            let module_params = module.parameters();
            for param in module_params.values() {
                params.push(param.tensor());
            }
        }

        params
    }

    /// Get all parameters with module hierarchical names
    pub fn get_all_named_parameters(&self) -> HashMap<String, Parameter> {
        let mut all_params = HashMap::new();

        // Add own parameters
        for (name, param) in &self.parameters {
            all_params.insert(name.clone(), param.clone());
        }

        // Add child parameters with prefixes
        for (module_name, module) in &self.modules {
            for (param_name, param) in module.all_named_parameters() {
                let full_name = if param_name.is_empty() {
                    module_name.clone()
                } else {
                    format!("{module_name}.{param_name}")
                };
                all_params.insert(full_name, param);
            }
        }

        all_params
    }

    /// Get all named parameters including submodules
    pub fn all_named_parameters(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        let mut params = HashMap::new();

        for (name, param) in &self.parameters {
            params.insert(name.clone(), param.tensor());
        }

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{module_name}.{param_name}"), param.tensor());
            }
        }

        params
    }

    /// Register a hook for this module
    pub fn register_hook(&mut self, hook_type: HookType, callback: HookCallback) -> HookHandle {
        self.hook_registry.register_hook(hook_type, callback)
    }

    /// Remove a hook by handle
    pub fn remove_hook(&mut self, hook_type: HookType, handle: HookHandle) -> bool {
        self.hook_registry.remove_hook(hook_type, handle)
    }

    /// Execute hooks of a specific type
    pub fn execute_hooks(
        &self,
        hook_type: HookType,
        module: &dyn crate::Module,
        input: &Tensor,
        output: Option<&Tensor>,
    ) -> Result<()> {
        self.hook_registry
            .execute_hooks(hook_type, module, input, output)
    }

    /// Check if any hooks are registered
    pub fn has_hooks(&self, hook_type: HookType) -> bool {
        self.hook_registry.has_hooks(hook_type)
    }

    /// Get hook count for a specific type
    pub fn hook_count(&self, hook_type: HookType) -> usize {
        self.hook_registry.hook_count(hook_type)
    }

    /// Clear all hooks of a specific type
    pub fn clear_hooks(&mut self, hook_type: HookType) {
        self.hook_registry.clear_hooks(hook_type)
    }

    /// Clear all hooks
    pub fn clear_all_hooks(&mut self) {
        self.hook_registry.clear_all_hooks()
    }
}
