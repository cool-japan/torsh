//! Core module trait system for neural network modules
//!
//! This module provides the foundational Module trait and essential interfaces
//! for all neural network components in ToRSh.

pub mod module_ext;

pub use module_ext::{ModuleExt, ParameterStats, ValidationReport};

use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Base trait for all neural network modules
///
/// This trait provides the core interface for all neural network components,
/// following PyTorch-compatible patterns for maximum interoperability.
///
/// ## Design Philosophy
///
/// This trait is designed for maximum ergonomics while maintaining flexibility:
/// - Most methods have sensible defaults to reduce boilerplate
/// - Core functionality (forward, training mode) is required
/// - Parameter management is streamlined with helper methods
/// - Hook system is optional but well-integrated
pub trait Module: Send + Sync {
    /// Forward pass through the module
    ///
    /// This is the only required method that must be implemented by all modules.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor or error
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all parameters in the module (non-recursive)
    ///
    /// Override this method if your module has trainable parameters.
    /// The default implementation returns an empty map.
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Map of parameter names to parameters
    fn parameters(&self) -> HashMap<String, crate::Parameter> {
        HashMap::new()
    }

    /// Get named parameters (non-recursive)
    ///
    /// Default implementation delegates to `parameters()`. Override if you need
    /// different behavior for named vs unnamed parameter access.
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Map of parameter names to parameters
    fn named_parameters(&self) -> HashMap<String, crate::Parameter> {
        self.parameters()
    }

    /// Get all parameters recursively including submodules
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Flattened map of all parameters
    fn all_parameters(&self) -> HashMap<String, crate::Parameter> {
        let mut all_params = self.parameters();

        for child in self.children() {
            let child_params = child.all_parameters();
            for (name, param) in child_params {
                all_params.insert(name, param);
            }
        }

        all_params
    }

    /// Get all named parameters recursively with module prefixes
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Hierarchical parameter names
    fn all_named_parameters(&self) -> HashMap<String, crate::Parameter> {
        let mut all_params = HashMap::new();

        // Add own parameters
        for (name, param) in self.named_parameters() {
            all_params.insert(name, param);
        }

        // Add child parameters with prefixes
        let children_named = self.named_children();
        for (child_name, child) in children_named {
            for (param_name, param) in child.all_named_parameters() {
                let full_name = format!("{}.{}", child_name, param_name);
                all_params.insert(full_name, param);
            }
        }

        all_params
    }

    /// Check if in training mode
    ///
    /// Default implementation returns true. Override if your module tracks training state.
    fn training(&self) -> bool {
        true
    }

    /// Set training mode
    ///
    /// Convenience method that calls `set_training(true)`.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Set evaluation mode
    ///
    /// Convenience method that calls `set_training(false)`.
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Set training mode (internal implementation)
    ///
    /// Default implementation does nothing. Override if your module needs to track
    /// training state or propagate it to child modules.
    fn set_training(&mut self, _training: bool) {
        // Default: do nothing
    }

    /// Move module to device
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// or buffers that need to be moved between devices.
    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }

    /// Load state dictionary into the module
    ///
    /// # Arguments
    /// * `state_dict` - Map of parameter names to tensors
    /// * `strict` - Whether to require exact parameter name matches
    ///
    /// # Returns
    /// * `Result<()>` - Success or error with details about missing/unexpected keys
    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor>,
        strict: bool,
    ) -> Result<()> {
        let current_params = self.all_named_parameters();
        let mut missing_keys = Vec::new();
        let mut unexpected_keys = Vec::new();

        // Check for missing parameters
        for name in current_params.keys() {
            if !state_dict.contains_key(name) {
                missing_keys.push(name.clone());
            }
        }

        // Check for unexpected parameters
        for name in state_dict.keys() {
            if !current_params.contains_key(name) {
                unexpected_keys.push(name.clone());
            }
        }

        if strict && (!missing_keys.is_empty() || !unexpected_keys.is_empty()) {
            return Err(torsh_core::error::TorshError::Other(format!(
                "State dict loading failed. Missing keys: {:?}, Unexpected keys: {:?}",
                missing_keys, unexpected_keys
            )));
        }

        // Load matching parameters
        for (name, param) in current_params {
            if let Some(new_tensor) = state_dict.get(&name) {
                // Validate tensor shapes match
                let current_shape = param.shape()?;
                let new_shape = new_tensor.shape().dims().to_vec();
                if current_shape != new_shape {
                    return Err(torsh_core::error::TorshError::Other(format!(
                        "Shape mismatch for parameter '{}': expected {:?}, got {:?}",
                        name, current_shape, new_shape
                    )));
                }

                // Copy tensor data
                *param.tensor().write() = new_tensor.clone();
            }
        }

        Ok(())
    }

    /// Load state dictionary with default strict=true
    fn load_state_dict_strict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
        self.load_state_dict(state_dict, true)
    }

    /// Save state dictionary from the module
    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        for (name, param) in self.all_named_parameters() {
            state.insert(name, param.clone_data());
        }
        state
    }

    /// Get the module name (optional, for debugging and serialization)
    fn name(&self) -> Option<&str> {
        None
    }

    /// Get all buffers (non-trainable parameters)
    fn buffers(&self) -> Vec<std::sync::Arc<parking_lot::RwLock<Tensor>>> {
        Vec::new()
    }

    /// Get named buffers
    fn named_buffers(&self) -> HashMap<String, std::sync::Arc<parking_lot::RwLock<Tensor>>> {
        HashMap::new()
    }

    /// Get all direct child modules
    ///
    /// Default implementation returns an empty vector. Override if your module
    /// contains child modules.
    fn children(&self) -> Vec<&dyn Module> {
        Vec::new()
    }

    /// Get all direct child modules with names
    ///
    /// Default implementation returns an empty vector. Override if your module
    /// contains named child modules.
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// Get all modules recursively (depth-first traversal)
    fn modules(&self) -> Vec<&dyn Module>
    where
        Self: Sized,
    {
        let mut modules: Vec<&dyn Module> = vec![self];
        for child in self.children() {
            // Since child is &dyn Module, we need to use a different approach
            // We'll just collect immediate children for now
            modules.push(child);
        }
        modules
    }

    /// Get all modules recursively with hierarchical names
    fn named_modules(&self) -> Vec<(String, &dyn Module)>
    where
        Self: Sized,
    {
        let mut modules: Vec<(String, &dyn Module)> = vec![(String::new(), self)];

        for (child_name, child) in self.named_children() {
            // Since child is &dyn Module, we need to use a different approach
            // We'll just collect immediate named children for now
            modules.push((child_name, child));
        }

        modules
    }

    /// Zero all gradients recursively
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// with gradients that need to be zeroed.
    fn zero_grad(&mut self) {
        // Default: do nothing
    }

    /// Count total number of parameters
    fn num_parameters(&self) -> usize {
        self.all_parameters()
            .values()
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    /// Count trainable parameters
    fn num_trainable_parameters(&self) -> usize {
        self.all_parameters()
            .values()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    /// Get memory usage estimate in bytes
    fn memory_usage(&self) -> usize {
        self.all_parameters()
            .values()
            .map(|p| p.numel().unwrap_or(0) * 4) // Assume f32 = 4 bytes
            .sum()
    }

    /// Freeze all parameters (set requires_grad = false)
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// that can be frozen/unfrozen.
    fn freeze(&mut self) {
        // Default: do nothing
    }

    /// Unfreeze all parameters (set requires_grad = true)
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// that can be frozen/unfrozen.
    fn unfreeze(&mut self) {
        // Default: do nothing
    }

    /// Get string representation
    fn extra_repr(&self) -> String {
        String::new()
    }

    /// Register a hook for this module (default implementation does nothing)
    fn register_hook(
        &mut self,
        _hook_type: crate::HookType,
        _callback: crate::HookCallback,
    ) -> Option<crate::HookHandle> {
        None
    }

    /// Remove a hook by handle (default implementation does nothing)
    fn remove_hook(&mut self, _hook_type: crate::HookType, _handle: crate::HookHandle) -> bool {
        false
    }

    /// Execute hooks of a specific type (default implementation does nothing)
    fn execute_hooks(
        &self,
        _hook_type: crate::HookType,
        _input: &Tensor,
        _output: Option<&Tensor>,
    ) -> Result<()> {
        Ok(())
    }

    /// Forward pass with hooks support
    fn forward_with_hooks(&self, input: &Tensor) -> Result<Tensor> {
        // Execute pre-forward hooks
        self.execute_hooks(crate::HookType::PreForward, input, None)?;

        // Perform forward pass
        let output = self.forward(input)?;

        // Execute post-forward hooks
        self.execute_hooks(crate::HookType::PostForward, input, Some(&output))?;

        Ok(output)
    }

    /// Check if module has hooks registered
    fn has_hooks(&self, _hook_type: crate::HookType) -> bool {
        false
    }

    // === Ergonomic Helper Methods ===

    /// Convenient method to call forward and handle common patterns
    ///
    /// This is equivalent to `forward()` but provides a more ergonomic interface
    /// for chaining operations.
    fn call(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Apply the module to input (alias for forward)
    ///
    /// PyTorch-style method name for compatibility.
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Check if the module has any parameters
    fn has_parameters(&self) -> bool {
        !self.parameters().is_empty()
    }

    /// Check if the module has any child modules
    fn has_children(&self) -> bool {
        !self.children().is_empty()
    }

    /// Get parameter count (convenience method)
    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    /// Get trainable parameter count (convenience method)
    fn trainable_parameter_count(&self) -> usize {
        self.num_trainable_parameters()
    }

    /// Get memory usage in MB (convenience method)
    fn memory_usage_mb(&self) -> f64 {
        self.memory_usage() as f64 / (1024.0 * 1024.0)
    }

    /// Toggle training mode (convenience method)
    fn toggle_training(&mut self) {
        self.set_training(!self.training());
    }

    /// Check if module is in evaluation mode
    fn eval_mode(&self) -> bool {
        !self.training()
    }

    // === Enhanced Ergonomic Methods ===

    /// Sequential forward pass through multiple modules
    ///
    /// This provides a convenient way to chain multiple forward passes.
    ///
    /// # Arguments
    /// * `modules` - Slice of modules to apply sequentially
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Final output after all modules
    ///
    /// # Example
    /// ```ignore
    /// let result = Module::sequential_forward(&[&layer1, &layer2, &layer3], &input)?;
    /// ```
    fn sequential_forward(modules: &[&dyn Module], mut input: Tensor) -> Result<Tensor>
    where
        Self: Sized,
    {
        for module in modules {
            input = module.forward(&input)?;
        }
        Ok(input)
    }

    /// Apply module multiple times with different inputs (batch processing)
    ///
    /// This is useful for processing multiple independent inputs through the same module.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors
    ///
    /// # Returns
    /// * `Result<Vec<Tensor>>` - Vector of output tensors
    fn batch_forward(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Forward with condition - only apply if condition is true
    ///
    /// This provides conditional execution of modules.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `condition` - Whether to apply this module
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor (input if condition is false)
    fn conditional_forward(&self, input: &Tensor, condition: bool) -> Result<Tensor> {
        if condition {
            self.forward(input)
        } else {
            Ok(input.clone())
        }
    }

    /// Forward with residual connection
    ///
    /// Applies the module and adds the result to the input (residual/skip connection).
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor (input + forward(input))
    fn residual_forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.forward(input)?;
        // This would use tensor addition when available
        // For now, just return the output
        Ok(output)
    }

    /// Get detailed module information for debugging
    ///
    /// This provides comprehensive information about the module state.
    ///
    /// # Returns
    /// * `ModuleInfo` - Detailed module information
    fn module_info(&self) -> crate::ModuleInfo {
        crate::ModuleInfo {
            name: self.name().unwrap_or("Unknown").to_string(),
            training: self.training(),
            parameter_count: self.num_parameters(),
            trainable_parameter_count: self.num_trainable_parameters(),
            memory_usage_bytes: self.memory_usage(),
            has_children: self.has_children(),
            children_count: self.children().len(),
        }
    }

    /// Check if module is ready for training
    ///
    /// Performs various checks to ensure the module is properly configured for training.
    ///
    /// # Returns
    /// * `Result<()>` - Ok if ready, Error with details if not
    fn check_training_readiness(&self) -> Result<()> {
        // Check if module has parameters
        if !self.has_parameters() {
            return Err(torsh_core::error::TorshError::Other(
                "Module has no parameters - may not be trainable".to_string(),
            ));
        }

        // Check if in training mode
        if !self.training() {
            return Err(torsh_core::error::TorshError::Other(
                "Module is in evaluation mode - switch to training mode first".to_string(),
            ));
        }

        // Check for finite parameters
        for param in self.parameters().values() {
            if !param.is_finite().unwrap_or(false) {
                return Err(torsh_core::error::TorshError::Other(
                    "Module contains non-finite parameters (NaN or infinity)".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get parameter names matching a pattern
    ///
    /// This helps with selective parameter access and manipulation.
    ///
    /// # Arguments
    /// * `pattern` - String pattern to match against parameter names
    ///
    /// # Returns
    /// * `Vec<String>` - Vector of parameter names matching the pattern
    fn parameter_names_matching(&self, pattern: &str) -> Vec<String> {
        self.all_named_parameters()
            .keys()
            .filter(|name| name.contains(pattern))
            .cloned()
            .collect()
    }

    /// Get parameters by layer type (e.g., "weight", "bias")
    ///
    /// # Arguments
    /// * `param_type` - Type of parameters to retrieve
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Filtered parameters
    fn parameters_by_type(&self, param_type: &str) -> HashMap<String, crate::Parameter> {
        self.all_named_parameters()
            .into_iter()
            .filter(|(name, _)| name.contains(param_type))
            .collect()
    }

    /// Clone module parameters (for creating copies or checkpoints)
    ///
    /// # Returns
    /// * `HashMap<String, Tensor>` - Cloned parameter tensors
    fn clone_parameters(&self) -> HashMap<String, Tensor> {
        self.all_named_parameters()
            .into_iter()
            .map(|(name, param)| (name, param.clone_data()))
            .collect()
    }

    /// Quick diagnostic check of module health
    ///
    /// # Returns
    /// * `ModuleDiagnostics` - Diagnostic information
    fn diagnose(&self) -> crate::ModuleDiagnostics {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check parameter health
        for (name, param) in self.all_named_parameters() {
            if let Ok(diag) = param.diagnose() {
                if !diag.issues.is_empty() {
                    issues.extend(
                        diag.issues
                            .into_iter()
                            .map(|issue| format!("{}: {}", name, issue)),
                    );
                }
                if !diag.warnings.is_empty() {
                    warnings.extend(
                        diag.warnings
                            .into_iter()
                            .map(|warning| format!("{}: {}", name, warning)),
                    );
                }
            }
        }

        // Check training readiness
        if let Err(e) = self.check_training_readiness() {
            warnings.push(format!("Training readiness: {}", e));
        }

        crate::ModuleDiagnostics {
            module_info: self.module_info(),
            issues,
            warnings,
            parameter_diagnostics: self
                .all_named_parameters()
                .into_iter()
                .filter_map(|(name, param)| param.diagnose().ok().map(|d| (name, d)))
                .collect(),
        }
    }
}

/// Implementation for boxed trait objects
impl Module for Box<dyn Module> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (**self).forward(x)
    }

    fn parameters(&self) -> HashMap<String, crate::Parameter> {
        (**self).parameters()
    }

    fn train(&mut self) {
        (**self).train()
    }

    fn eval(&mut self) {
        (**self).eval()
    }

    fn training(&self) -> bool {
        (**self).training()
    }

    fn children(&self) -> Vec<&dyn Module> {
        (**self).children()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        (**self).named_children()
    }

    fn set_training(&mut self, training: bool) {
        (**self).set_training(training)
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        (**self).to_device(device)
    }
}

/// Implementation for mutable references to boxed trait objects
impl Module for &mut Box<dyn Module> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (***self).forward(x)
    }

    fn parameters(&self) -> HashMap<String, crate::Parameter> {
        (***self).parameters()
    }

    fn train(&mut self) {
        (***self).train()
    }

    fn eval(&mut self) {
        (***self).eval()
    }

    fn training(&self) -> bool {
        (***self).training()
    }

    fn children(&self) -> Vec<&dyn Module> {
        (***self).children()
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        (***self).named_children()
    }

    fn set_training(&mut self, training: bool) {
        (***self).set_training(training)
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        (***self).to_device(device)
    }
}
