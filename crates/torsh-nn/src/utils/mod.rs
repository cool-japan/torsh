//! Module utilities and helper functions
//!
//! This module provides utility traits and helper functions for working with
//! neural network modules, including module application patterns, parameter analysis,
//! and model introspection utilities.
//!
//! # Examples
//!
//! ## Parameter Analysis
//! ```rust,ignore
//! use torsh_nn::layers::linear::Linear;
//! use torsh_nn::utils::analysis;
//!
//! let model = Linear::new(784, 10, true);
//! let param_count = analysis::count_parameters(&model);
//! let trainable_count = analysis::count_trainable_parameters(&model);
//! println!("Total parameters: {}, Trainable: {}", param_count, trainable_count);
//! ```
//!
//! ## Model Introspection
//! ```rust,ignore
//! use torsh_nn::layers::linear::Linear;
//! use torsh_nn::utils::introspection;
//!
//! let model = Linear::new(784, 128, true);
//! introspection::print_parameter_summary(&model, "MyModel");
//! let issues = introspection::health_check(&model);
//! if !issues.is_empty() {
//!     println!("Model issues detected: {:?}", issues);
//! }
//! ```

use std::collections::HashMap;
use torsh_core::error::Result;

use crate::{Module, Parameter};

/// Extension trait for applying functions to modules (separate to maintain dyn compatibility)
pub trait ModuleApply {
    /// Apply a function to all submodules recursively
    fn apply<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&mut dyn Module) -> Result<()>;

    /// Apply function to all parameters recursively
    fn apply_to_parameters<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&mut Parameter) -> Result<()>;

    /// Apply function to all modules recursively
    fn apply_to_modules<F>(&mut self, f: &F) -> Result<()>
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

    fn apply_to_parameters<F>(&mut self, _f: &F) -> Result<()>
    where
        F: Fn(&mut Parameter) -> Result<()>,
    {
        // Default implementation does nothing - override in implementing types
        Ok(())
    }

    fn apply_to_modules<F>(&mut self, _f: &F) -> Result<()>
    where
        F: Fn(&mut dyn Module) -> Result<()>,
    {
        // Default implementation does nothing - override in implementing types
        Ok(())
    }
}

/// Utility functions for neural network analysis and debugging
pub mod analysis {
    use super::*;

    /// Count the total number of parameters in a module
    pub fn count_parameters(module: &dyn Module) -> usize {
        module
            .parameters()
            .values()
            .map(|param| param.tensor().read().shape().numel())
            .sum()
    }

    /// Count only trainable parameters (parameters with requires_grad = true)
    pub fn count_trainable_parameters(module: &dyn Module) -> usize {
        module
            .parameters()
            .values()
            .filter(|param| param.requires_grad())
            .map(|param| param.tensor().read().shape().numel())
            .sum()
    }

    /// Get detailed parameter statistics for a module
    pub fn parameter_statistics(module: &dyn Module) -> ModuleParameterStats {
        let parameters = module.parameters();
        let total_params = parameters
            .values()
            .map(|param| param.tensor().read().shape().numel())
            .sum();
        let trainable_params = parameters
            .values()
            .filter(|param| param.requires_grad())
            .map(|param| param.tensor().read().shape().numel())
            .sum();

        ModuleParameterStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            frozen_parameters: total_params - trainable_params,
            parameter_count_by_layer: parameters
                .iter()
                .map(|(name, param)| (name.clone(), param.tensor().read().shape().numel()))
                .collect(),
        }
    }

    /// Check if a module is in training mode
    pub fn is_training(module: &dyn Module) -> bool {
        module.training()
    }

    /// Get the names of all parameters in a module
    pub fn parameter_names(module: &dyn Module) -> Vec<String> {
        module.parameters().keys().cloned().collect()
    }

    /// Find parameters by name pattern (simple substring matching)
    pub fn find_parameters_by_pattern(
        module: &dyn Module,
        pattern: &str,
    ) -> HashMap<String, Parameter> {
        module
            .parameters()
            .into_iter()
            .filter(|(name, _)| name.contains(pattern))
            .collect()
    }
}

/// Parameter statistics structure for module analysis
#[derive(Debug, Clone)]
pub struct ModuleParameterStats {
    /// Total number of parameters (trainable + frozen)
    pub total_parameters: usize,
    /// Number of trainable parameters
    pub trainable_parameters: usize,
    /// Number of frozen parameters
    pub frozen_parameters: usize,
    /// Parameter count by layer name
    pub parameter_count_by_layer: HashMap<String, usize>,
}

impl ModuleParameterStats {
    /// Get the percentage of parameters that are trainable
    pub fn trainable_percentage(&self) -> f32 {
        if self.total_parameters == 0 {
            0.0
        } else {
            (self.trainable_parameters as f32 / self.total_parameters as f32) * 100.0
        }
    }

    /// Get memory usage estimate in bytes (assuming f32 parameters)
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_parameters * 4 // 4 bytes per f32
    }

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f32 {
        self.memory_usage_bytes() as f32 / (1024.0 * 1024.0)
    }
}

/// Utility functions for model introspection and debugging
pub mod introspection {
    use super::*;

    /// Print a summary of module parameters
    pub fn print_parameter_summary(module: &dyn Module, module_name: &str) {
        let stats = analysis::parameter_statistics(module);
        println!("=== {} Parameter Summary ===", module_name);
        println!("Total parameters: {}", stats.total_parameters);
        println!(
            "Trainable parameters: {} ({:.1}%)",
            stats.trainable_parameters,
            stats.trainable_percentage()
        );
        println!("Frozen parameters: {}", stats.frozen_parameters);
        println!("Memory usage: {:.2} MB", stats.memory_usage_mb());
        println!("Training mode: {}", analysis::is_training(module));

        if !stats.parameter_count_by_layer.is_empty() {
            println!("\nParameters by layer:");
            let mut layers: Vec<_> = stats.parameter_count_by_layer.iter().collect();
            layers.sort_by_key(|(name, _)| name.as_str());
            for (layer, count) in layers {
                println!("  {}: {}", layer, count);
            }
        }
        println!();
    }

    /// Check for common issues in module configuration
    pub fn health_check(module: &dyn Module) -> Vec<String> {
        let mut issues = Vec::new();
        let stats = analysis::parameter_statistics(module);

        // Check for modules with no parameters
        if stats.total_parameters == 0 {
            issues.push("Module has no parameters".to_string());
        }

        // Check for modules with all frozen parameters
        if stats.total_parameters > 0 && stats.trainable_parameters == 0 {
            issues.push("All parameters are frozen - module won't train".to_string());
        }

        // Check for very large models (>1GB)
        if stats.memory_usage_bytes() > 1024 * 1024 * 1024 {
            issues.push(format!(
                "Large model detected: {:.1} GB",
                stats.memory_usage_bytes() as f32 / (1024.0 * 1024.0 * 1024.0)
            ));
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;

    #[test]
    fn test_parameter_counting() {
        let linear = Linear::new(10, 5, true); // 10*5 + 5 = 55 parameters
        let count = analysis::count_parameters(&linear);
        assert_eq!(count, 55);
    }

    #[test]
    fn test_parameter_stats() {
        let linear = Linear::new(4, 2, true); // 4*2 + 2 = 10 parameters
        let stats = analysis::parameter_statistics(&linear);
        assert_eq!(stats.total_parameters, 10);
        assert_eq!(stats.trainable_parameters, 10);
        assert_eq!(stats.frozen_parameters, 0);
        assert_eq!(stats.trainable_percentage(), 100.0);
    }

    #[test]
    fn test_memory_calculation() {
        let stats = ModuleParameterStats {
            total_parameters: 1000,
            trainable_parameters: 800,
            frozen_parameters: 200,
            parameter_count_by_layer: HashMap::new(),
        };
        assert_eq!(stats.memory_usage_bytes(), 4000); // 1000 * 4 bytes
        assert_eq!(stats.memory_usage_mb(), 4000.0 / (1024.0 * 1024.0));
    }
}
