//! Parameter management utilities

use crate::Parameter;
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// ParameterList for managing a list of parameters
pub struct ParameterList {
    parameters: Vec<Parameter>,
}

impl ParameterList {
    /// Create a new parameter list
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Append a parameter
    pub fn append(&mut self, param: Parameter) {
        self.parameters.push(param);
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get a parameter by index
    pub fn get(&self, index: usize) -> Option<&Parameter> {
        self.parameters.get(index)
    }

    /// Get all parameters as tensors
    pub fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.parameters.iter().map(|p| p.tensor()).collect()
    }

    /// Move all parameters to a device
    pub fn to(&mut self, device: DeviceType) -> Result<(), torsh_core::error::TorshError> {
        for param in &mut self.parameters {
            let tensor_clone = param.tensor().read().clone();
            let moved_tensor = tensor_clone.to(device)?;
            *param.tensor().write() = moved_tensor;
        }
        Ok(())
    }
}

impl Default for ParameterList {
    fn default() -> Self {
        Self::new()
    }
}

/// ParameterDict for managing named parameters
pub struct ParameterDict {
    parameters: std::collections::HashMap<String, Parameter>,
}

impl ParameterDict {
    /// Create a new parameter dictionary
    pub fn new() -> Self {
        Self {
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Insert a parameter
    pub fn insert(&mut self, key: String, param: Parameter) {
        self.parameters.insert(key, param);
    }

    /// Get a parameter by key
    pub fn get(&self, key: &str) -> Option<&Parameter> {
        self.parameters.get(key)
    }

    /// Remove a parameter
    pub fn remove(&mut self, key: &str) -> Option<Parameter> {
        self.parameters.remove(key)
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.parameters.keys()
    }

    /// Get all parameters as tensors
    pub fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.parameters.values().map(|p| p.tensor()).collect()
    }

    /// Get named parameters
    pub fn named_parameters(&self) -> std::collections::HashMap<String, Arc<RwLock<Tensor>>> {
        self.parameters
            .iter()
            .map(|(k, v)| (k.clone(), v.tensor()))
            .collect()
    }

    /// Move all parameters to a device
    pub fn to(&mut self, device: DeviceType) -> Result<(), torsh_core::error::TorshError> {
        for param in self.parameters.values_mut() {
            let tensor_clone = param.tensor().read().clone();
            let moved_tensor = tensor_clone.to(device)?;
            *param.tensor().write() = moved_tensor;
        }
        Ok(())
    }
}

impl Default for ParameterDict {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for parameter manipulation
pub mod utils {
    use super::*;
    use torsh_autograd::grad_mode::clip::{clip_grad_norm, clip_grad_value};

    /// Count the total number of parameters
    pub fn count_parameters(parameters: &[Arc<RwLock<Tensor>>]) -> usize {
        parameters.iter().map(|p| p.read().numel()).sum()
    }

    /// Count trainable parameters
    pub fn count_trainable_parameters(parameters: &[Arc<RwLock<Tensor>>]) -> usize {
        parameters
            .iter()
            .filter(|p| p.read().requires_grad())
            .map(|p| p.read().numel())
            .sum()
    }

    /// Freeze parameters (disable gradients)
    pub fn freeze_parameters(parameters: &[Arc<RwLock<Tensor>>]) {
        for param in parameters {
            let mut tensor = param.write();
            *tensor = tensor.clone().requires_grad_(false);
        }
    }

    /// Unfreeze parameters (enable gradients)
    pub fn unfreeze_parameters(parameters: &[Arc<RwLock<Tensor>>]) {
        for param in parameters {
            let mut tensor = param.write();
            *tensor = tensor.clone().requires_grad_(true);
        }
    }

    /// Zero all gradients
    pub fn zero_grad(parameters: &[Arc<RwLock<Tensor>>]) {
        for param in parameters {
            param.write().zero_grad();
        }
    }

    /// Clip gradients by norm
    pub fn clip_grad_norm_<T: torsh_core::dtype::FloatElement>(
        parameters: &mut [Arc<RwLock<Tensor<T>>>],
        max_norm: f32,
        norm_type: f32,
    ) -> f32 {
        let mut tensors: Vec<_> = parameters.iter().map(|p| p.write().clone()).collect();

        clip_grad_norm(&mut tensors, max_norm, norm_type)
    }

    /// Clip gradients by value
    pub fn clip_grad_value_<T: torsh_core::dtype::FloatElement>(
        parameters: &mut [Arc<RwLock<Tensor<T>>>],
        clip_value: f32,
    ) {
        let mut tensors: Vec<_> = parameters.iter().map(|p| p.write().clone()).collect();

        clip_grad_value(&mut tensors, clip_value)
    }

    /// Get parameter statistics
    pub fn parameter_stats(parameters: &[Arc<RwLock<Tensor>>]) -> ParameterStats {
        let total_params = count_parameters(parameters);
        let trainable_params = count_trainable_parameters(parameters);

        let mut total_memory = 0;
        let mut param_groups = std::collections::HashMap::new();

        for param in parameters {
            let tensor = param.read();
            let memory = tensor.numel() * tensor.dtype().size();
            total_memory += memory;

            let shape_str = format!("{:?}", tensor.shape().dims());
            *param_groups.entry(shape_str).or_insert(0) += 1;
        }

        ParameterStats {
            total_params,
            trainable_params,
            non_trainable_params: total_params - trainable_params,
            total_memory_bytes: total_memory,
            param_groups,
        }
    }
}

/// Statistics about parameters
pub struct ParameterStats {
    pub total_params: usize,
    pub trainable_params: usize,
    pub non_trainable_params: usize,
    pub total_memory_bytes: usize,
    pub param_groups: std::collections::HashMap<String, usize>,
}

impl std::fmt::Display for ParameterStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parameter Statistics:")?;
        writeln!(f, "  Total parameters: {}", self.total_params)?;
        writeln!(f, "  Trainable parameters: {}", self.trainable_params)?;
        writeln!(
            f,
            "  Non-trainable parameters: {}",
            self.non_trainable_params
        )?;
        writeln!(
            f,
            "  Total memory: {:.2} MB",
            self.total_memory_bytes as f64 / 1_048_576.0
        )?;
        writeln!(f, "  Parameter groups:")?;
        for (shape, count) in &self.param_groups {
            writeln!(f, "    {}: {} tensors", shape, count)?;
        }
        Ok(())
    }
}
