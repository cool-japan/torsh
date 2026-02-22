//! Module trait ergonomic extensions
//!
//! This module provides additional ergonomic helpers and utilities for the Module trait,
//! following Rust best practices for trait extension patterns.

use crate::Module;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Extension trait providing additional ergonomic methods for Module
///
/// This trait is automatically implemented for all types that implement Module,
/// providing additional convenience methods without requiring changes to existing code.
///
/// # Design Philosophy
///
/// This extension follows Rust's extension trait pattern to:
/// - Add functionality without breaking backward compatibility
/// - Keep the core Module trait focused on essential methods
/// - Provide advanced features for users who need them
/// - Enable fluent/builder-style APIs
pub trait ModuleExt: Module {
    // === Fluent API / Builder Pattern Methods ===

    /// Chain forward pass with a transformation function
    ///
    /// This enables functional-style chaining of operations.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `f` - Transformation function to apply to output
    ///
    /// # Returns
    /// * `Result<Tensor>` - Transformed output
    ///
    /// # Example
    /// ```ignore
    /// let output = layer.and_then(&input, |x| x.relu())?;
    /// ```
    fn and_then<F>(&self, input: &Tensor, f: F) -> Result<Tensor>
    where
        F: FnOnce(Tensor) -> Result<Tensor>,
    {
        let output = self.forward(input)?;
        f(output)
    }

    /// Apply module and map the output with a function
    ///
    /// Similar to `and_then` but for non-failable transformations.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `f` - Mapping function
    ///
    /// # Returns
    /// * `Result<Tensor>` - Mapped output
    fn map<F>(&self, input: &Tensor, f: F) -> Result<Tensor>
    where
        F: FnOnce(Tensor) -> Tensor,
    {
        let output = self.forward(input)?;
        Ok(f(output))
    }

    /// Forward pass with input transformation
    ///
    /// Apply a transformation to the input before forwarding.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `f` - Input transformation function
    ///
    /// # Returns
    /// * `Result<Tensor>` - Module output
    fn with_input<F>(&self, input: &Tensor, f: F) -> Result<Tensor>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        let transformed = f(input)?;
        self.forward(&transformed)
    }

    // === Inspection and Debugging Methods ===

    /// Get human-readable summary of the module
    ///
    /// # Returns
    /// * `String` - Formatted module summary
    fn summary(&self) -> String {
        let info = self.module_info();
        format!(
            "Module: {}\n\
             Training: {}\n\
             Parameters: {} ({} trainable)\n\
             Memory: {:.2} MB\n\
             Children: {}",
            info.name,
            info.training,
            info.parameter_count,
            info.trainable_parameter_count,
            info.memory_usage_bytes as f64 / (1024.0 * 1024.0),
            info.children_count
        )
    }

    /// Print module summary to stdout
    fn print_summary(&self) {
        println!("{}", self.summary());
    }

    /// Get parameter statistics
    ///
    /// # Returns
    /// * `ParameterStats` - Statistical information about parameters
    fn parameter_stats(&self) -> ParameterStats {
        let params = self.all_parameters();
        let mut total_params = 0;
        let mut trainable_params = 0;
        let mut frozen_params = 0;
        let mut total_memory = 0;

        for param in params.values() {
            let numel = param.numel().unwrap_or(0);
            total_params += numel;
            total_memory += numel * 4; // Assume f32

            if param.requires_grad() {
                trainable_params += numel;
            } else {
                frozen_params += numel;
            }
        }

        ParameterStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            frozen_parameters: frozen_params,
            total_memory_bytes: total_memory,
            parameter_count: params.len(),
        }
    }

    /// Check if module has NaN or Inf in parameters
    ///
    /// # Returns
    /// * `bool` - true if all parameters are finite
    fn has_finite_parameters(&self) -> bool {
        self.all_parameters()
            .values()
            .all(|p| p.is_finite().unwrap_or(false))
    }

    /// Get list of parameter names
    ///
    /// # Returns
    /// * `Vec<String>` - Sorted list of parameter names
    fn parameter_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.all_named_parameters().keys().cloned().collect();
        names.sort();
        names
    }

    /// Get parameter by name
    ///
    /// # Arguments
    /// * `name` - Parameter name
    ///
    /// # Returns
    /// * `Option<Parameter>` - Parameter if found
    fn get_parameter(&self, name: &str) -> Option<crate::Parameter> {
        self.all_named_parameters().get(name).cloned()
    }

    // === Training Utilities ===

    /// Freeze specific parameters by name pattern
    ///
    /// # Arguments
    /// * `pattern` - String pattern to match parameter names
    ///
    /// # Returns
    /// * `usize` - Number of parameters frozen
    ///
    /// # Note
    /// This method currently returns a count but doesn't actually freeze parameters
    /// because Parameter's requires_grad is immutable. This is a placeholder for
    /// future implementation when mutable parameter access is available.
    fn freeze_matching(&mut self, pattern: &str) -> usize {
        let mut count = 0;
        for (name, _param) in self.all_named_parameters() {
            if name.contains(pattern) {
                // TODO: Implement actual freezing when Parameter supports it
                count += 1;
            }
        }
        count
    }

    /// Unfreeze specific parameters by name pattern
    ///
    /// # Arguments
    /// * `pattern` - String pattern to match parameter names
    ///
    /// # Returns
    /// * `usize` - Number of parameters unfrozen
    ///
    /// # Note
    /// This method currently returns a count but doesn't actually unfreeze parameters
    /// because Parameter's requires_grad is immutable. This is a placeholder for
    /// future implementation when mutable parameter access is available.
    fn unfreeze_matching(&mut self, pattern: &str) -> usize {
        let mut count = 0;
        for (name, _param) in self.all_named_parameters() {
            if name.contains(pattern) {
                // TODO: Implement actual unfreezing when Parameter supports it
                count += 1;
            }
        }
        count
    }

    /// Get list of frozen parameters
    ///
    /// # Returns
    /// * `Vec<String>` - Names of frozen parameters
    fn frozen_parameters(&self) -> Vec<String> {
        self.all_named_parameters()
            .into_iter()
            .filter(|(_, p)| !p.requires_grad())
            .map(|(name, _)| name)
            .collect()
    }

    /// Get list of trainable parameters
    ///
    /// # Returns
    /// * `Vec<String>` - Names of trainable parameters
    fn trainable_parameters(&self) -> Vec<String> {
        self.all_named_parameters()
            .into_iter()
            .filter(|(_, p)| p.requires_grad())
            .map(|(name, _)| name)
            .collect()
    }

    // === Advanced Operations ===

    /// Clone module parameters into a new state dict
    ///
    /// # Returns
    /// * `HashMap<String, Tensor>` - Cloned state dictionary
    fn clone_state_dict(&self) -> HashMap<String, Tensor> {
        self.state_dict()
    }

    /// Apply a function to all parameters
    ///
    /// # Arguments
    /// * `f` - Function to apply to each parameter
    fn apply_to_parameters<F>(&self, mut f: F)
    where
        F: FnMut(&str, &crate::Parameter),
    {
        for (name, param) in self.all_named_parameters() {
            f(&name, &param);
        }
    }

    /// Count parameters by layer type
    ///
    /// # Returns
    /// * `HashMap<String, usize>` - Parameter count per layer type
    fn parameters_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for (name, param) in self.all_named_parameters() {
            // Extract layer type from name (first component)
            let layer_type = name.split('.').next().unwrap_or("unknown").to_string();

            let numel = param.numel().unwrap_or(0);
            *counts.entry(layer_type).or_insert(0) += numel;
        }

        counts
    }

    /// Validate module configuration
    ///
    /// Performs comprehensive validation of module state.
    ///
    /// # Returns
    /// * `Result<ValidationReport>` - Validation results
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        // Check for parameters
        if !self.has_parameters() {
            report.warnings.push("Module has no parameters".to_string());
        }

        // Check for finite parameters
        if !self.has_finite_parameters() {
            report
                .errors
                .push("Module has non-finite parameters (NaN or Inf)".to_string());
        }

        // Check memory usage
        let memory_mb = self.memory_usage_mb();
        if memory_mb > 1024.0 {
            report
                .warnings
                .push(format!("Large memory usage: {:.2} GB", memory_mb / 1024.0));
        }

        // Check parameter count
        let param_count = self.num_parameters();
        if param_count > 100_000_000 {
            report
                .warnings
                .push(format!("Very large model: {} parameters", param_count));
        }

        report.is_valid = report.errors.is_empty();
        Ok(report)
    }

    /// Get device of parameters (if consistent)
    ///
    /// # Returns
    /// * `Option<DeviceType>` - Device if all parameters are on same device
    ///
    /// # Note
    /// Currently returns None as Parameter doesn't expose device information.
    /// This is a placeholder for future implementation.
    fn device(&self) -> Option<DeviceType> {
        // TODO: Implement when Parameter exposes device information
        // For now, assume CPU as default
        if self.has_parameters() {
            Some(DeviceType::Cpu)
        } else {
            None
        }
    }

    /// Check if all parameters are on CPU
    ///
    /// # Returns
    /// * `bool` - true if all parameters on CPU
    fn is_cpu(&self) -> bool {
        self.device() == Some(DeviceType::Cpu)
    }

    /// Check if all parameters are on CUDA device
    ///
    /// # Returns
    /// * `bool` - true if all parameters on CUDA
    fn is_cuda(&self) -> bool {
        matches!(self.device(), Some(DeviceType::Cuda(_)))
    }
}

// Automatically implement ModuleExt for all types that implement Module
impl<T: Module + ?Sized> ModuleExt for T {}

// === Supporting Types ===

/// Parameter statistics for a module
#[derive(Debug, Clone)]
pub struct ParameterStats {
    /// Total number of parameter elements
    pub total_parameters: usize,
    /// Number of trainable parameter elements
    pub trainable_parameters: usize,
    /// Number of frozen parameter elements
    pub frozen_parameters: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Number of distinct parameters
    pub parameter_count: usize,
}

impl ParameterStats {
    /// Get memory usage in megabytes
    pub fn memory_mb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory usage in gigabytes
    pub fn memory_gb(&self) -> f64 {
        self.memory_mb() / 1024.0
    }

    /// Get percentage of parameters that are trainable
    pub fn trainable_percentage(&self) -> f64 {
        if self.total_parameters == 0 {
            0.0
        } else {
            (self.trainable_parameters as f64 / self.total_parameters as f64) * 100.0
        }
    }
}

/// Validation report for a module
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Whether the module is valid
    pub is_valid: bool,
    /// List of errors found
    pub errors: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Check if validation passed without errors
    pub fn passed(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }

    /// Get total number of issues (errors + warnings)
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut result = String::new();

        result.push_str(&format!(
            "Validation: {}\n",
            if self.is_valid { "PASSED" } else { "FAILED" }
        ));

        if !self.errors.is_empty() {
            result.push_str("\nErrors:\n");
            for error in &self.errors {
                result.push_str(&format!("  - {}\n", error));
            }
        }

        if !self.warnings.is_empty() {
            result.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                result.push_str(&format!("  - {}\n", warning));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {

    // Tests would go here - skipped for brevity
}
