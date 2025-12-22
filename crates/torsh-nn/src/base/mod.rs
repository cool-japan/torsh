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

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_module_base_creation() {
        let base = ModuleBase::new();
        assert!(base.training());
        assert_eq!(base.device, DeviceType::Cpu);
        assert_eq!(base.parameters.len(), 0);
        assert_eq!(base.buffers.len(), 0);
        assert_eq!(base.modules.len(), 0);
    }

    #[test]
    fn test_module_base_default() {
        let base = ModuleBase::default();
        assert!(base.training());
        assert_eq!(base.device, DeviceType::Cpu);
    }

    #[test]
    fn test_training_mode() {
        let mut base = ModuleBase::new();
        assert!(base.training());

        base.set_training(false);
        assert!(!base.training());

        base.set_training(true);
        assert!(base.training());
    }

    #[test]
    fn test_register_parameter() {
        let mut base = ModuleBase::new();
        let tensor = zeros(&[3, 4]).unwrap();
        let param = Parameter::new(tensor);

        base.register_parameter("weight".to_string(), param);
        assert_eq!(base.parameters.len(), 1);
        assert!(base.parameters.contains_key("weight"));
    }

    #[test]
    fn test_register_multiple_parameters() {
        let mut base = ModuleBase::new();

        let weight = Parameter::new(zeros(&[10, 5]).unwrap());
        let bias = Parameter::new(zeros(&[5]).unwrap());

        base.register_parameter("weight".to_string(), weight);
        base.register_parameter("bias".to_string(), bias);

        assert_eq!(base.parameters.len(), 2);
        assert!(base.parameters.contains_key("weight"));
        assert!(base.parameters.contains_key("bias"));
    }

    #[test]
    fn test_register_buffer() {
        let mut base = ModuleBase::new();
        let tensor = zeros(&[10]).unwrap();

        base.register_buffer("running_mean".to_string(), tensor);
        assert_eq!(base.buffers.len(), 1);
        assert!(base.buffers.contains_key("running_mean"));
    }

    #[test]
    fn test_named_parameters() {
        let mut base = ModuleBase::new();
        let param = Parameter::new(zeros(&[3, 4]).unwrap());
        base.register_parameter("weight".to_string(), param);

        let named_params = base.named_parameters();
        assert_eq!(named_params.len(), 1);
        assert!(named_params.contains_key("weight"));
    }

    #[test]
    fn test_children_empty() {
        let base = ModuleBase::new();
        let children = base.children();
        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_named_children_empty() {
        let base = ModuleBase::new();
        let named_children = base.named_children();
        assert_eq!(named_children.len(), 0);
    }

    #[test]
    fn test_to_device_cpu() -> Result<()> {
        let mut base = ModuleBase::new();
        base.to_device(DeviceType::Cpu)?;
        assert_eq!(base.device, DeviceType::Cpu);
        Ok(())
    }

    #[test]
    fn test_all_parameter_tensors() {
        let mut base = ModuleBase::new();
        let param1 = Parameter::new(zeros(&[2, 3]).unwrap());
        let param2 = Parameter::new(zeros(&[4]).unwrap());

        base.register_parameter("weight".to_string(), param1);
        base.register_parameter("bias".to_string(), param2);

        let all_params = base.all_parameter_tensors();
        assert_eq!(all_params.len(), 2);
    }

    #[test]
    fn test_all_named_parameters() {
        let mut base = ModuleBase::new();
        let param = Parameter::new(zeros(&[3, 4]).unwrap());
        base.register_parameter("weight".to_string(), param);

        let all_named = base.all_named_parameters();
        assert_eq!(all_named.len(), 1);
    }

    #[test]
    fn test_hook_registration() {
        use crate::HookType;

        let mut base = ModuleBase::new();
        let callback: HookCallback = Box::new(|_module, _input, _output| Ok(()));

        let handle = base.register_hook(HookType::PreForward, callback);
        assert!(base.has_hooks(HookType::PreForward));
        assert_eq!(base.hook_count(HookType::PreForward), 1);

        let removed = base.remove_hook(HookType::PreForward, handle);
        assert!(removed);
        assert!(!base.has_hooks(HookType::PreForward));
    }

    #[test]
    fn test_hook_multiple_registration() {
        use crate::HookType;

        let mut base = ModuleBase::new();
        let callback1: HookCallback = Box::new(|_m, _i, _o| Ok(()));
        let callback2: HookCallback = Box::new(|_m, _i, _o| Ok(()));

        base.register_hook(HookType::PreForward, callback1);
        base.register_hook(HookType::PreForward, callback2);

        assert_eq!(base.hook_count(HookType::PreForward), 2);
    }

    #[test]
    fn test_clear_hooks() {
        use crate::HookType;

        let mut base = ModuleBase::new();
        let callback1: HookCallback = Box::new(|_m, _i, _o| Ok(()));
        let callback2: HookCallback = Box::new(|_m, _i, _o| Ok(()));

        base.register_hook(HookType::PreForward, callback1);
        base.register_hook(HookType::PreBackward, callback2);

        assert!(base.has_hooks(HookType::PreForward));
        assert!(base.has_hooks(HookType::PreBackward));

        base.clear_hooks(HookType::PreForward);
        assert!(!base.has_hooks(HookType::PreForward));
        assert!(base.has_hooks(HookType::PreBackward));
    }

    #[test]
    fn test_clear_all_hooks() {
        use crate::HookType;

        let mut base = ModuleBase::new();
        let callback1: HookCallback = Box::new(|_m, _i, _o| Ok(()));
        let callback2: HookCallback = Box::new(|_m, _i, _o| Ok(()));

        base.register_hook(HookType::PreForward, callback1);
        base.register_hook(HookType::PreBackward, callback2);

        assert!(base.has_hooks(HookType::PreForward));
        assert!(base.has_hooks(HookType::PreBackward));

        base.clear_all_hooks();
        assert!(!base.has_hooks(HookType::PreForward));
        assert!(!base.has_hooks(HookType::PreBackward));
    }

    #[test]
    fn test_hook_count_zero() {
        use crate::HookType;

        let base = ModuleBase::new();
        assert_eq!(base.hook_count(HookType::PreForward), 0);
        assert_eq!(base.hook_count(HookType::PreBackward), 0);
    }

    #[test]
    fn test_debug_format() {
        let mut base = ModuleBase::new();
        base.register_parameter(
            "weight".to_string(),
            Parameter::new(zeros(&[2, 3]).unwrap()),
        );

        let debug_str = format!("{:?}", base);
        assert!(debug_str.contains("ModuleBase"));
        assert!(debug_str.contains("training"));
        assert!(debug_str.contains("parameters_count"));
    }

    #[test]
    fn test_parameter_replacement() {
        let mut base = ModuleBase::new();

        // Register initial parameter
        let param1 = Parameter::new(zeros(&[2, 3]).unwrap());
        base.register_parameter("weight".to_string(), param1);
        assert_eq!(base.parameters.len(), 1);

        // Replace with new parameter (same name)
        let param2 = Parameter::new(zeros(&[4, 5]).unwrap());
        base.register_parameter("weight".to_string(), param2);
        assert_eq!(base.parameters.len(), 1); // Still just one parameter

        // Verify new shape
        let weight_arc = base.parameters["weight"].tensor();
        let weight = weight_arc.read();
        assert_eq!(weight.shape().dims(), &[4, 5]);
    }

    #[test]
    fn test_buffer_replacement() {
        let mut base = ModuleBase::new();

        // Register initial buffer
        base.register_buffer("running_mean".to_string(), zeros(&[10]).unwrap());
        assert_eq!(base.buffers.len(), 1);

        // Replace with new buffer
        base.register_buffer("running_mean".to_string(), zeros(&[20]).unwrap());
        assert_eq!(base.buffers.len(), 1); // Still just one buffer

        // Verify new shape
        let buffer = base.buffers["running_mean"].read();
        assert_eq!(buffer.shape().dims(), &[20]);
    }

    #[test]
    fn test_empty_base_all_named_parameters() {
        let base = ModuleBase::new();
        let all_named = base.all_named_parameters();
        assert_eq!(all_named.len(), 0);
    }

    #[test]
    fn test_empty_base_get_all_named_parameters() {
        let base = ModuleBase::new();
        let all_named = base.get_all_named_parameters();
        assert_eq!(all_named.len(), 0);
    }
}
