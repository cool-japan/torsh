//! Neural network pruning utilities
//!
//! This module provides utilities for pruning neural networks to reduce model size
//! and improve inference speed while maintaining accuracy.

use crate::Module;
use crate::Parameter;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Pruning strategies for neural networks
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// Magnitude-based pruning - removes weights with smallest absolute values
    MagnitudeBased,
    /// Structured pruning - removes entire channels/filters
    Structured,
    /// Gradual pruning - removes weights gradually over training
    Gradual {
        initial_sparsity: f32,
        final_sparsity: f32,
        begin_step: usize,
        end_step: usize,
    },
    /// Lottery ticket hypothesis - iterative magnitude pruning
    LotteryTicket,
}

/// Pruning scope - what to prune
#[derive(Debug, Clone)]
pub enum PruningScope {
    /// Prune all layers
    Global,
    /// Prune specific layers by name
    LayerSpecific(Vec<String>),
    /// Prune by layer type
    LayerType(String),
}

/// Pruning configuration
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Pruning strategy to use
    pub strategy: PruningStrategy,
    /// Scope of pruning
    pub scope: PruningScope,
    /// Target sparsity ratio (0.0 to 1.0)
    pub sparsity: f32,
    /// Whether to preserve structure
    pub structured: bool,
}

/// Pruning mask for a parameter
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// The mask tensor (1.0 for keep, 0.0 for prune)
    pub mask: Tensor<f32>,
    /// Original parameter name
    pub parameter_name: String,
    /// Current sparsity ratio
    pub sparsity: f32,
}

impl PruningMask {
    /// Create a new pruning mask
    pub fn new(mask: Tensor<f32>, parameter_name: String) -> Self {
        let total_elements = mask.numel();
        let mask_data = mask.data().unwrap_or_else(|_| vec![]);
        let zero_elements = mask_data.iter().filter(|&&x| x == 0.0).count();
        let sparsity = zero_elements as f32 / total_elements as f32;

        Self {
            mask,
            parameter_name,
            sparsity,
        }
    }

    /// Apply the mask to a parameter
    pub fn apply(&self, parameter: &Parameter) -> Result<Parameter, Box<dyn std::error::Error>> {
        let data = parameter.tensor().read().clone();
        let masked_data = data.mul_op(&self.mask)?;
        Ok(Parameter::new(masked_data))
    }

    /// Get the number of pruned parameters
    pub fn pruned_count(&self) -> usize {
        self.mask
            .data()
            .unwrap_or_else(|_| vec![])
            .iter()
            .filter(|&&x| x == 0.0)
            .count()
    }

    /// Get the total number of parameters
    pub fn total_count(&self) -> usize {
        self.mask.numel()
    }
}

/// Pruning utilities for neural networks
pub struct Pruner {
    config: PruningConfig,
    masks: HashMap<String, PruningMask>,
    current_step: usize,
}

impl Pruner {
    /// Create a new pruner with the given configuration
    pub fn new(config: PruningConfig) -> Self {
        Self {
            config,
            masks: HashMap::new(),
            current_step: 0,
        }
    }

    /// Prune a module according to the configuration
    pub fn prune_module<M: Module>(
        &mut self,
        module: &M,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let named_params = module.named_parameters();

        for (name, param) in named_params.iter() {
            if self.should_prune_parameter(name) {
                let mask = self.create_mask(param)?;
                self.masks.insert(name.to_string(), mask);
            }
        }

        Ok(())
    }

    /// Apply pruning masks to a module
    pub fn apply_masks<M: Module>(
        &self,
        _module: &mut M,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Note: This is a simplified implementation
        // In practice, you'd need to modify the module's parameters
        // This requires a mutable reference to the module's internal state
        Ok(())
    }

    /// Update pruning masks for gradual pruning
    pub fn update_masks(&mut self) {
        match &self.config.strategy {
            PruningStrategy::Gradual {
                initial_sparsity,
                final_sparsity,
                begin_step,
                end_step,
            } => {
                if self.current_step >= *begin_step && self.current_step <= *end_step {
                    let progress =
                        (self.current_step - begin_step) as f32 / (end_step - begin_step) as f32;
                    let current_sparsity =
                        initial_sparsity + progress * (final_sparsity - initial_sparsity);

                    // Update masks based on current sparsity
                    for (_, mask) in self.masks.iter_mut() {
                        if mask.sparsity < current_sparsity {
                            // Need to increase sparsity
                            // This would require recalculating the mask
                        }
                    }
                }
            }
            _ => {}
        }

        self.current_step += 1;
    }

    /// Get current sparsity statistics
    pub fn get_sparsity_stats(&self) -> HashMap<String, f32> {
        self.masks
            .iter()
            .map(|(name, mask)| (name.clone(), mask.sparsity))
            .collect()
    }

    /// Get total model sparsity
    pub fn get_total_sparsity(&self) -> f32 {
        if self.masks.is_empty() {
            return 0.0;
        }

        let total_pruned: usize = self.masks.values().map(|m| m.pruned_count()).sum();
        let total_params: usize = self.masks.values().map(|m| m.total_count()).sum();

        total_pruned as f32 / total_params as f32
    }

    /// Check if a parameter should be pruned
    fn should_prune_parameter(&self, param_name: &str) -> bool {
        match &self.config.scope {
            PruningScope::Global => true,
            PruningScope::LayerSpecific(names) => {
                names.iter().any(|name| param_name.contains(name))
            }
            PruningScope::LayerType(layer_type) => param_name.contains(layer_type),
        }
    }

    /// Create a pruning mask for a parameter
    fn create_mask(
        &self,
        parameter: &Parameter,
    ) -> Result<PruningMask, Box<dyn std::error::Error>> {
        let data = parameter.tensor().read().clone();
        let mask = match &self.config.strategy {
            PruningStrategy::MagnitudeBased => self.create_magnitude_mask(&data)?,
            PruningStrategy::Structured => self.create_structured_mask(&data)?,
            PruningStrategy::Gradual {
                initial_sparsity, ..
            } => self.create_magnitude_mask_with_sparsity(&data, *initial_sparsity)?,
            PruningStrategy::LotteryTicket => self.create_magnitude_mask(&data)?,
        };

        Ok(PruningMask::new(mask, "parameter".to_string()))
    }

    /// Create magnitude-based pruning mask
    fn create_magnitude_mask(
        &self,
        data: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        // Get absolute values
        let abs_values = data.abs()?;

        // Calculate threshold for desired sparsity
        let mut sorted_values: Vec<f32> = abs_values.data().unwrap_or_else(|_| vec![]);
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (sorted_values.len() as f32 * self.config.sparsity) as usize;
        let threshold = sorted_values[threshold_idx.min(sorted_values.len() - 1)];

        // Create mask
        let mask = abs_values.gt_scalar(threshold)?;
        // Convert bool mask to f32 mask
        let mask_data = mask.data()?;
        let f32_mask_data: Vec<f32> = mask_data
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let f32_mask =
            Tensor::from_data(f32_mask_data, mask.shape().dims().to_vec(), mask.device()).unwrap();
        Ok(f32_mask)
    }

    /// Create structured pruning mask (channels/filters)
    fn create_structured_mask(
        &self,
        data: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        // For structured pruning, we need to consider the tensor structure
        // This is a simplified implementation
        let shape = data.shape();
        let dims = shape.dims();

        if dims.len() == 4 {
            // Conv2d weights: [out_channels, in_channels, kernel_h, kernel_w]
            self.create_channel_mask(data)
        } else if dims.len() == 2 {
            // Linear weights: [out_features, in_features]
            self.create_magnitude_mask(data)
        } else {
            // Default to magnitude-based
            self.create_magnitude_mask(data)
        }
    }

    /// Create channel-wise pruning mask for convolutional layers
    fn create_channel_mask(
        &self,
        data: &Tensor<f32>,
    ) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        let binding = data.shape();
        let dims = binding.dims();
        let out_channels = dims[0];

        // Calculate L2 norm for each output channel
        let mut channel_norms = Vec::new();
        for i in 0..out_channels {
            let channel_data = data.slice(0, i, i + 1)?;
            // Calculate L2 norm: sqrt(sum(x^2))
            let channel_tensor = channel_data.to_tensor()?;
            let squared = channel_tensor.mul_op(&channel_tensor)?;
            let sum_squared = squared.sum()?;
            let norm = sum_squared.sqrt()?;
            channel_norms.push((i, norm.item()?));
        }

        // Sort by norm
        channel_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Determine channels to prune
        let channels_to_prune = (out_channels as f32 * self.config.sparsity) as usize;
        let pruned_channels: std::collections::HashSet<usize> = channel_norms
            .iter()
            .take(channels_to_prune)
            .map(|(idx, _)| *idx)
            .collect();

        // Create mask
        let mask_data: Vec<f32> = (0..data.numel())
            .map(|i| {
                // Calculate which channel this element belongs to
                let channel_idx = i / (data.numel() / out_channels);
                if pruned_channels.contains(&channel_idx) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();

        Ok(Tensor::from_data(mask_data, dims.to_vec(), data.device()).unwrap())
    }

    /// Create magnitude-based mask with specific sparsity
    fn create_magnitude_mask_with_sparsity(
        &self,
        data: &Tensor<f32>,
        sparsity: f32,
    ) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        let abs_values = data.abs()?;
        let mut sorted_values: Vec<f32> = abs_values.to_vec()?;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (sorted_values.len() as f32 * sparsity) as usize;
        let threshold = sorted_values[threshold_idx.min(sorted_values.len() - 1)];

        let mask = abs_values.gt_scalar(threshold)?;
        // Convert boolean mask to f32 mask
        let mask_data = mask.to_vec()?;
        let mask_f32: Vec<f32> = mask_data
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        Ok(Tensor::from_vec(mask_f32, mask.shape().dims())?)
    }
}

/// Convenience functions for common pruning operations
impl Pruner {
    /// Create a magnitude-based pruner
    pub fn magnitude_based(sparsity: f32) -> Self {
        Self::new(PruningConfig {
            strategy: PruningStrategy::MagnitudeBased,
            scope: PruningScope::Global,
            sparsity,
            structured: false,
        })
    }

    /// Create a structured pruner
    pub fn structured(sparsity: f32) -> Self {
        Self::new(PruningConfig {
            strategy: PruningStrategy::Structured,
            scope: PruningScope::Global,
            sparsity,
            structured: true,
        })
    }

    /// Create a gradual pruner
    pub fn gradual(
        initial_sparsity: f32,
        final_sparsity: f32,
        begin_step: usize,
        end_step: usize,
    ) -> Self {
        Self::new(PruningConfig {
            strategy: PruningStrategy::Gradual {
                initial_sparsity,
                final_sparsity,
                begin_step,
                end_step,
            },
            scope: PruningScope::Global,
            sparsity: final_sparsity,
            structured: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_pruning_mask_creation() {
        let mask_data = vec![1.0, 0.0, 1.0, 0.0];
        let mask_tensor = Tensor::from_data(mask_data, vec![2, 2], DeviceType::Cpu).unwrap();
        let mask = PruningMask::new(mask_tensor, "test_param".to_string());

        assert_eq!(mask.sparsity, 0.5);
        assert_eq!(mask.pruned_count(), 2);
        assert_eq!(mask.total_count(), 4);
    }

    #[test]
    fn test_pruner_creation() {
        let pruner = Pruner::magnitude_based(0.5);
        assert_eq!(pruner.config.sparsity, 0.5);
        assert!(matches!(
            pruner.config.strategy,
            PruningStrategy::MagnitudeBased
        ));
    }

    #[test]
    fn test_structured_pruner_creation() {
        let pruner = Pruner::structured(0.3);
        assert_eq!(pruner.config.sparsity, 0.3);
        assert!(matches!(
            pruner.config.strategy,
            PruningStrategy::Structured
        ));
        assert!(pruner.config.structured);
    }

    #[test]
    fn test_gradual_pruner_creation() {
        let pruner = Pruner::gradual(0.1, 0.9, 1000, 5000);
        assert_eq!(pruner.config.sparsity, 0.9);
        if let PruningStrategy::Gradual {
            initial_sparsity,
            final_sparsity,
            begin_step,
            end_step,
        } = pruner.config.strategy
        {
            assert_eq!(initial_sparsity, 0.1);
            assert_eq!(final_sparsity, 0.9);
            assert_eq!(begin_step, 1000);
            assert_eq!(end_step, 5000);
        } else {
            panic!("Expected gradual strategy");
        }
    }

    #[test]
    fn test_sparsity_stats_empty() {
        let pruner = Pruner::magnitude_based(0.5);
        let stats = pruner.get_sparsity_stats();
        assert!(stats.is_empty());
        assert_eq!(pruner.get_total_sparsity(), 0.0);
    }
}
