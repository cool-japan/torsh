//! Parameter management extensions and utilities
//!
//! This module provides enhanced parameter management capabilities including:
//! - Parameter groups for differential learning rates
//! - Parameter constraints and regularization
//! - Advanced parameter inspection utilities
//! - Parameter transformation utilities

use super::Parameter;
use torsh_core::error::{Result, TorshError};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Parameter group for organizing parameters with shared hyperparameters
///
/// This is useful for implementing techniques like:
/// - Differential learning rates
/// - Layer-wise learning rate decay
/// - Parameter-specific weight decay
/// - Grouped parameter optimization
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Name of the parameter group
    pub name: String,
    /// Parameters in this group
    pub parameters: Vec<Parameter>,
    /// Learning rate multiplier for this group
    pub lr_multiplier: f32,
    /// Weight decay for this group
    pub weight_decay: f32,
    /// Whether to apply gradient clipping
    pub clip_gradients: bool,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl ParameterGroup {
    /// Create a new parameter group
    ///
    /// # Arguments
    /// * `name` - Group name
    /// * `parameters` - Parameters in this group
    ///
    /// # Returns
    /// * `ParameterGroup` - New parameter group with default settings
    pub fn new(name: String, parameters: Vec<Parameter>) -> Self {
        Self {
            name,
            parameters,
            lr_multiplier: 1.0,
            weight_decay: 0.0,
            clip_gradients: false,
            max_grad_norm: 1.0,
        }
    }

    /// Set learning rate multiplier (builder pattern)
    pub fn with_lr_multiplier(mut self, multiplier: f32) -> Self {
        self.lr_multiplier = multiplier;
        self
    }

    /// Set weight decay (builder pattern)
    pub fn with_weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Enable gradient clipping (builder pattern)
    pub fn with_gradient_clipping(mut self, max_norm: f32) -> Self {
        self.clip_gradients = true;
        self.max_grad_norm = max_norm;
        self
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.iter().map(|p| p.numel().unwrap_or(0)).sum()
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
}

/// Parameter constraint for enforcing parameter properties
///
/// Constraints can be applied after parameter updates to ensure
/// parameters stay within valid ranges or satisfy certain properties.
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Clamp parameters to a range
    ClampRange { min: f32, max: f32 },
    /// Ensure parameters are non-negative
    NonNegative,
    /// Normalize parameters (L2 norm = 1)
    UnitNorm,
    /// Ensure parameters are in [0, 1]
    Probability,
    /// Custom constraint function
    Custom { name: String },
}

impl ParameterConstraint {
    /// Apply constraint to a parameter
    ///
    /// # Arguments
    /// * `parameter` - Parameter to constrain
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn apply(&self, parameter: &Parameter) -> Result<()> {
        let tensor = parameter.tensor();
        let _data = tensor.write();

        match self {
            ParameterConstraint::ClampRange { min, max } => {
                // Would clamp values to [min, max]
                let _ = (min, max);
                // TODO: Implement when tensor supports clamp
                Ok(())
            }
            ParameterConstraint::NonNegative => {
                // Would set negative values to 0
                // TODO: Implement when tensor supports element-wise operations
                Ok(())
            }
            ParameterConstraint::UnitNorm => {
                // Would normalize to unit norm
                // TODO: Implement when tensor supports normalization
                Ok(())
            }
            ParameterConstraint::Probability => {
                // Would clamp to [0, 1] and normalize
                // TODO: Implement when tensor supports operations
                Ok(())
            }
            ParameterConstraint::Custom { name: _ } => {
                // Custom constraints would be implemented by users
                Ok(())
            }
        }
    }

    /// Get constraint name
    pub fn name(&self) -> &str {
        match self {
            ParameterConstraint::ClampRange { .. } => "ClampRange",
            ParameterConstraint::NonNegative => "NonNegative",
            ParameterConstraint::UnitNorm => "UnitNorm",
            ParameterConstraint::Probability => "Probability",
            ParameterConstraint::Custom { name } => name,
        }
    }
}

/// Parameter statistics and analysis
#[derive(Debug, Clone)]
pub struct ParameterAnalysis {
    /// Mean of parameter values
    pub mean: f32,
    /// Standard deviation of parameter values
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Number of elements
    pub numel: usize,
    /// Percentage of zero values
    pub sparsity: f32,
    /// Has NaN values
    pub has_nan: bool,
    /// Has Inf values
    pub has_inf: bool,
}

/// Extension trait for Parameter with advanced utilities
pub trait ParameterExt {
    /// Analyze parameter statistics
    ///
    /// # Returns
    /// * `Result<ParameterAnalysis>` - Statistical analysis of parameter
    fn analyze(&self) -> Result<ParameterAnalysis>;

    /// Check if parameter values are finite
    ///
    /// # Returns
    /// * `Result<bool>` - true if all values are finite (not NaN or Inf)
    fn is_finite(&self) -> Result<bool>;

    /// Get parameter L2 norm
    ///
    /// # Returns
    /// * `Result<f32>` - L2 norm of parameter
    fn norm(&self) -> Result<f32>;

    /// Get parameter L1 norm
    ///
    /// # Returns
    /// * `Result<f32>` - L1 norm of parameter
    fn l1_norm(&self) -> Result<f32>;

    /// Compute parameter gradient norm (when available)
    ///
    /// # Returns
    /// * `Result<f32>` - Gradient norm
    fn grad_norm(&self) -> Result<f32>;

    /// Check if parameter has gradient (when available)
    ///
    /// # Returns
    /// * `bool` - true if gradient is available
    fn has_grad(&self) -> bool;

    /// Get parameter as read-only data vector
    ///
    /// # Returns
    /// * `Result<Vec<f32>>` - Parameter data
    fn to_vec(&self) -> Result<Vec<f32>>;

    /// Get parameter dtype name
    ///
    /// # Returns
    /// * `&str` - Data type name
    fn dtype_name(&self) -> &str;

    /// Get memory usage in bytes
    ///
    /// # Returns
    /// * `usize` - Memory usage in bytes
    fn memory_bytes(&self) -> usize;

    /// Clone parameter with new requires_grad setting
    ///
    /// # Arguments
    /// * `requires_grad` - New requires_grad setting
    ///
    /// # Returns
    /// * `Parameter` - Cloned parameter
    fn clone_with_grad(&self, requires_grad: bool) -> Parameter;
}

impl ParameterExt for Parameter {
    fn analyze(&self) -> Result<ParameterAnalysis> {
        let tensor = self.tensor();
        let data_guard = tensor.read();
        let data = data_guard.to_vec()?;

        let numel = data.len();
        if numel == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot analyze empty parameter".to_string(),
            ));
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / numel as f32;

        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / numel as f32;
        let std = variance.sqrt();

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let zero_count = data.iter().filter(|&&x| x == 0.0).count();
        let sparsity = (zero_count as f32 / numel as f32) * 100.0;

        let has_nan = data.iter().any(|&x| x.is_nan());
        let has_inf = data.iter().any(|&x| x.is_infinite());

        Ok(ParameterAnalysis {
            mean,
            std,
            min,
            max,
            numel,
            sparsity,
            has_nan,
            has_inf,
        })
    }

    fn is_finite(&self) -> Result<bool> {
        let tensor = self.tensor();
        let data = tensor.read().to_vec()?;
        Ok(data.iter().all(|&x| x.is_finite()))
    }

    fn norm(&self) -> Result<f32> {
        let tensor = self.tensor();
        let data = tensor.read().to_vec()?;
        let sum_sq: f32 = data.iter().map(|&x| x * x).sum();
        Ok(sum_sq.sqrt())
    }

    fn l1_norm(&self) -> Result<f32> {
        let tensor = self.tensor();
        let data = tensor.read().to_vec()?;
        Ok(data.iter().map(|&x| x.abs()).sum())
    }

    fn grad_norm(&self) -> Result<f32> {
        // TODO: Implement when gradient support is available
        Ok(0.0)
    }

    fn has_grad(&self) -> bool {
        // TODO: Implement when gradient support is available
        false
    }

    fn to_vec(&self) -> Result<Vec<f32>> {
        let tensor = self.tensor();
        let data_guard = tensor.read();
        data_guard.to_vec()
    }

    fn dtype_name(&self) -> &str {
        "f32" // Currently all parameters are f32
    }

    fn memory_bytes(&self) -> usize {
        self.numel().unwrap_or(0) * 4 // f32 = 4 bytes
    }

    fn clone_with_grad(&self, requires_grad: bool) -> Parameter {
        let tensor = self.clone_data();
        if requires_grad {
            Parameter::new(tensor)
        } else {
            Parameter::new_no_grad(tensor)
        }
    }
}

/// Extension trait for ParameterCollection with additional utilities
///
/// This trait is implemented for the existing ParameterCollection type
/// to add advanced functionality without modifying the core implementation.
pub trait ParameterCollectionExt {
    /// Get total parameter count
    fn total_numel(&self) -> usize;

    /// Group parameters by name pattern
    fn group_by_patterns(
        &self,
        groups: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, ParameterGroup>;

    /// Filter parameters by property
    fn filter<F>(&self, predicate: F) -> HashMap<String, Parameter>
    where
        F: Fn(&str, &Parameter) -> bool;

    /// Get trainable parameters only
    fn trainable(&self) -> HashMap<String, Parameter>;

    /// Get frozen parameters only
    fn frozen(&self) -> HashMap<String, Parameter>;
}

impl ParameterCollectionExt for super::ParameterCollection {
    fn total_numel(&self) -> usize {
        // Access through public methods
        self.names()
            .iter()
            .filter_map(|name| self.get(name))
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    fn group_by_patterns(
        &self,
        groups: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, ParameterGroup> {
        let mut result = HashMap::new();

        for (group_name, patterns) in groups {
            let mut group_params = Vec::new();

            for param_name in self.names() {
                if patterns.iter().any(|pattern| param_name.contains(pattern)) {
                    if let Some(param) = self.get(param_name) {
                        group_params.push(param.clone());
                    }
                }
            }

            if !group_params.is_empty() {
                result.insert(
                    group_name.clone(),
                    ParameterGroup::new(group_name.clone(), group_params),
                );
            }
        }

        result
    }

    fn filter<F>(&self, predicate: F) -> HashMap<String, Parameter>
    where
        F: Fn(&str, &Parameter) -> bool,
    {
        let mut result = HashMap::new();

        for name in self.names() {
            if let Some(param) = self.get(name) {
                if predicate(name, param) {
                    result.insert(name.clone(), param.clone());
                }
            }
        }

        result
    }

    fn trainable(&self) -> HashMap<String, Parameter> {
        self.filter(|_, param| param.requires_grad())
    }

    fn frozen(&self) -> HashMap<String, Parameter> {
        self.filter(|_, param| !param.requires_grad())
    }
}

#[cfg(test)]
mod tests {

    // Tests would go here
}
