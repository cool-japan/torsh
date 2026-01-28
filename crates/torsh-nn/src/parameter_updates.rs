//! Optimized parameter update strategies for better training performance
//!
//! This module provides various optimizations for parameter updates including
//! vectorized operations, memory-efficient updates, and specialized routines
//! for different parameter types.

use crate::Parameter;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Configuration for parameter update optimizations
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// Whether to use vectorized operations
    pub use_vectorization: bool,
    /// Whether to use in-place updates when possible
    pub use_inplace_updates: bool,
    /// Whether to fuse operations for better cache efficiency
    pub use_operation_fusion: bool,
    /// Memory budget for batching operations (in bytes)
    pub memory_budget: usize,
    /// Whether to use async updates for non-critical parameters
    pub use_async_updates: bool,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            use_vectorization: true,
            use_inplace_updates: true,
            use_operation_fusion: true,
            memory_budget: 1024 * 1024 * 1024, // 1GB default
            use_async_updates: false,
        }
    }
}

/// Optimized parameter updater
pub struct ParameterUpdater {
    config: UpdateConfig,
    update_stats: UpdateStatistics,
}

impl ParameterUpdater {
    /// Create a new parameter updater with default configuration
    pub fn new() -> Self {
        Self {
            config: UpdateConfig::default(),
            update_stats: UpdateStatistics::new(),
        }
    }

    /// Create a parameter updater with custom configuration
    pub fn with_config(config: UpdateConfig) -> Self {
        Self {
            config,
            update_stats: UpdateStatistics::new(),
        }
    }

    /// Apply SGD update to parameters
    pub fn sgd_update(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        learning_rate: f32,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        if self.config.use_operation_fusion {
            self.fused_sgd_update(parameters, gradients, learning_rate)?;
        } else {
            self.standard_sgd_update(parameters, gradients, learning_rate)?;
        }

        self.update_stats.total_updates += 1;
        self.update_stats.total_time += start_time.elapsed();

        Ok(())
    }

    /// Apply Adam update to parameters
    pub fn adam_update(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        m_t: &mut HashMap<String, Tensor>, // First moment estimates
        v_t: &mut HashMap<String, Tensor>, // Second moment estimates
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        if self.config.use_operation_fusion {
            self.fused_adam_update(
                parameters,
                gradients,
                m_t,
                v_t,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                step,
            )?;
        } else {
            self.standard_adam_update(
                parameters,
                gradients,
                m_t,
                v_t,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                step,
            )?;
        }

        self.update_stats.total_updates += 1;
        self.update_stats.total_time += start_time.elapsed();

        Ok(())
    }

    /// Apply momentum update to parameters
    pub fn momentum_update(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        velocities: &mut HashMap<String, Tensor>,
        learning_rate: f32,
        momentum: f32,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let velocity = velocities.entry(name.clone()).or_insert_with(|| {
                    zeros_like(&param.tensor().read())
                        .expect("zeros_like should succeed for valid tensor")
                });

                if self.config.use_inplace_updates {
                    // v = momentum * v + learning_rate * grad
                    *velocity = velocity
                        .mul_op(&torsh_tensor::creation::tensor_scalar(momentum)?)?
                        .add_op(
                            &grad.mul_op(&torsh_tensor::creation::tensor_scalar(learning_rate)?)?,
                        )?;

                    // param = param - v
                    let tensor_guard = param.tensor();
                    let mut param_tensor = tensor_guard.write();
                    *param_tensor = param_tensor.sub(&velocity)?;
                } else {
                    // Standard update without in-place operations
                    let new_velocity = velocity
                        .mul_op(&torsh_tensor::creation::tensor_scalar(momentum)?)?
                        .add_op(
                            &grad.mul_op(&torsh_tensor::creation::tensor_scalar(learning_rate)?)?,
                        )?;

                    let binding = param.tensor();
                    let param_tensor = binding.write();
                    param_tensor.sub(&new_velocity)?;
                    *velocity = new_velocity;
                }
            }
        }

        self.update_stats.total_updates += 1;
        self.update_stats.total_time += start_time.elapsed();

        Ok(())
    }

    /// Apply RMSprop update to parameters
    pub fn rmsprop_update(
        &mut self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        squared_gradients: &mut HashMap<String, Tensor>,
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let sq_grad = squared_gradients.entry(name.clone()).or_insert_with(|| {
                    zeros_like(&param.tensor().read())
                        .expect("zeros_like should succeed for valid tensor")
                });

                // Update squared gradients: sq_grad = alpha * sq_grad + (1 - alpha) * grad^2
                let grad_squared = grad.mul_op(grad)?;
                let alpha_tensor = torsh_tensor::creation::tensor_scalar(alpha)?;
                let one_minus_alpha = torsh_tensor::creation::tensor_scalar(1.0 - alpha)?;

                *sq_grad = sq_grad
                    .mul_op(&alpha_tensor)?
                    .add_op(&grad_squared.mul_op(&one_minus_alpha)?)?;

                // Update parameters: param = param - lr * grad / (sqrt(sq_grad) + eps)
                let sqrt_sq_grad = sq_grad.sqrt()?;
                let denominator =
                    sqrt_sq_grad.add_op(&torsh_tensor::creation::tensor_scalar(epsilon)?)?;
                let update = grad
                    .div(&denominator)?
                    .mul_op(&torsh_tensor::creation::tensor_scalar(learning_rate)?)?;

                let binding = param.tensor();
                let param_tensor = binding.write();
                param_tensor.sub(&update)?;
            }
        }

        self.update_stats.total_updates += 1;
        self.update_stats.total_time += start_time.elapsed();

        Ok(())
    }

    /// Batch update multiple parameter groups for better cache efficiency
    pub fn batch_update<F>(
        &mut self,
        parameter_groups: &[HashMap<String, Parameter>],
        gradient_groups: &[HashMap<String, Tensor>],
        update_fn: F,
    ) -> Result<()>
    where
        F: Fn(&HashMap<String, Parameter>, &HashMap<String, Tensor>) -> Result<()>,
    {
        if parameter_groups.len() != gradient_groups.len() {
            return Err(TorshError::InvalidArgument(
                "Parameter and gradient groups must have the same length".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        // Sort by memory usage for better cache efficiency
        let mut groups: Vec<_> = parameter_groups
            .iter()
            .zip(gradient_groups.iter())
            .enumerate()
            .collect();

        groups.sort_by_key(|(_, (params, _))| {
            params
                .values()
                .map(|p| p.tensor().read().shape().numel())
                .sum::<usize>()
        });

        // Process groups in batches based on memory budget
        let mut current_memory = 0;
        let mut batch_start = 0;

        for (i, (_, (params, _grads))) in groups.iter().enumerate() {
            let group_memory: usize = params
                .values()
                .map(|p| p.tensor().read().shape().numel() * std::mem::size_of::<f32>())
                .sum();

            if current_memory + group_memory > self.config.memory_budget && i > batch_start {
                // Process current batch
                for j in batch_start..i {
                    let (_, (batch_params, batch_grads)) = &groups[j];
                    update_fn(batch_params, batch_grads)?;
                }
                batch_start = i;
                current_memory = group_memory;
            } else {
                current_memory += group_memory;
            }
        }

        // Process remaining batch
        for j in batch_start..groups.len() {
            let (_, (batch_params, batch_grads)) = &groups[j];
            update_fn(batch_params, batch_grads)?;
        }

        self.update_stats.total_updates += groups.len();
        self.update_stats.total_time += start_time.elapsed();

        Ok(())
    }

    /// Apply gradient clipping before updates
    pub fn clip_gradients(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        max_norm: f32,
    ) -> Result<f32> {
        // Calculate total gradient norm
        let mut total_norm_squared = 0.0f32;

        for grad in gradients.values() {
            let grad_norm_squared = grad.mul_op(grad)?.sum()?.item()?;
            total_norm_squared += grad_norm_squared;
        }

        let total_norm = total_norm_squared.sqrt();

        if total_norm > max_norm {
            let clip_ratio = max_norm / total_norm;

            for grad in gradients.values_mut() {
                *grad = grad.mul_op(&torsh_tensor::creation::tensor_scalar(clip_ratio)?)?;
            }
        }

        Ok(total_norm)
    }

    /// Get update statistics
    pub fn get_statistics(&self) -> &UpdateStatistics {
        &self.update_stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.update_stats = UpdateStatistics::new();
    }

    // Private helper methods

    fn standard_sgd_update(
        &self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        learning_rate: f32,
    ) -> Result<()> {
        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let update = grad.mul_op(&torsh_tensor::creation::tensor_scalar(learning_rate)?)?;
                let binding = param.tensor();
                let param_tensor = binding.write();
                param_tensor.sub(&update)?;
            }
        }
        Ok(())
    }

    fn fused_sgd_update(
        &self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        learning_rate: f32,
    ) -> Result<()> {
        // Group parameters by device and size for vectorized operations
        let lr_tensor = torsh_tensor::creation::tensor_scalar(learning_rate)?;

        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let binding = param.tensor();
                let param_tensor = binding.write();

                // Fused operation: param = param - lr * grad
                param_tensor.sub(&grad.mul_op(&lr_tensor)?)?;
            }
        }
        Ok(())
    }

    fn standard_adam_update(
        &self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        m_t: &mut HashMap<String, Tensor>,
        v_t: &mut HashMap<String, Tensor>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<()> {
        let beta1_tensor = torsh_tensor::creation::tensor_scalar(beta1)?;
        let beta2_tensor = torsh_tensor::creation::tensor_scalar(beta2)?;
        let one_minus_beta1 = torsh_tensor::creation::tensor_scalar(1.0 - beta1)?;
        let one_minus_beta2 = torsh_tensor::creation::tensor_scalar(1.0 - beta2)?;
        let _lr_tensor = torsh_tensor::creation::tensor_scalar(learning_rate)?;
        let eps_tensor = torsh_tensor::creation::tensor_scalar(epsilon)?;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let bias_correction2 = 1.0 - beta2.powi(step as i32);
        let corrected_lr = learning_rate * (bias_correction2.sqrt() / bias_correction1);
        let corrected_lr_tensor = torsh_tensor::creation::tensor_scalar(corrected_lr)?;

        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let m = m_t.entry(name.clone()).or_insert_with(|| {
                    zeros_like(&param.tensor().read())
                        .expect("zeros_like should succeed for valid tensor")
                });
                let v = v_t.entry(name.clone()).or_insert_with(|| {
                    zeros_like(&param.tensor().read())
                        .expect("zeros_like should succeed for valid tensor")
                });

                // Update biased first moment estimate: m_t = beta1 * m_t + (1 - beta1) * grad
                *m = m
                    .mul_op(&beta1_tensor)?
                    .add_op(&grad.mul_op(&one_minus_beta1)?)?;

                // Update biased second moment estimate: v_t = beta2 * v_t + (1 - beta2) * grad^2
                let grad_squared = grad.mul_op(grad)?;
                *v = v
                    .mul_op(&beta2_tensor)?
                    .add_op(&grad_squared.mul_op(&one_minus_beta2)?)?;

                // Update parameters: param = param - corrected_lr * m_t / (sqrt(v_t) + eps)
                let sqrt_v = v.sqrt()?;
                let denominator = sqrt_v.add_op(&eps_tensor)?;
                let update = m.div(&&denominator)?.mul_op(&corrected_lr_tensor)?;

                let binding = param.tensor();
                let param_tensor = binding.write();
                param_tensor.sub(&update)?;
            }
        }

        Ok(())
    }

    fn fused_adam_update(
        &self,
        parameters: &HashMap<String, Parameter>,
        gradients: &HashMap<String, Tensor>,
        m_t: &mut HashMap<String, Tensor>,
        v_t: &mut HashMap<String, Tensor>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<()> {
        // For now, use the standard Adam update
        // In a real implementation, this would use fused kernels
        self.standard_adam_update(
            parameters,
            gradients,
            m_t,
            v_t,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            step,
        )
    }
}

impl Default for ParameterUpdater {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics tracking for parameter updates
#[derive(Debug, Clone)]
pub struct UpdateStatistics {
    pub total_updates: usize,
    pub total_time: std::time::Duration,
    pub memory_peak: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl UpdateStatistics {
    pub fn new() -> Self {
        Self {
            total_updates: 0,
            total_time: std::time::Duration::default(),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get average update time
    pub fn average_update_time(&self) -> std::time::Duration {
        if self.total_updates > 0 {
            self.total_time / self.total_updates as u32
        } else {
            std::time::Duration::default()
        }
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f32 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses > 0 {
            self.cache_hits as f32 / total_accesses as f32
        } else {
            0.0
        }
    }

    /// Get updates per second
    pub fn updates_per_second(&self) -> f32 {
        if !self.total_time.is_zero() {
            self.total_updates as f32 / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }
}

impl Default for UpdateStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized optimizers for different layer types
pub struct LayerSpecificOptimizers;

impl LayerSpecificOptimizers {
    /// Optimized update for linear/dense layers
    pub fn update_linear_layer(
        weight: &Parameter,
        bias: Option<&Parameter>,
        weight_grad: &Tensor,
        bias_grad: Option<&Tensor>,
        learning_rate: f32,
    ) -> Result<()> {
        // Update weight
        let lr_tensor = torsh_tensor::creation::tensor_scalar(learning_rate)?;
        let weight_update = weight_grad.mul_op(&lr_tensor)?;
        let binding = weight.tensor();
        let weight_tensor = binding.write();
        weight_tensor.sub(&weight_update)?;

        // Update bias if present
        if let (Some(bias), Some(bias_grad)) = (bias, bias_grad) {
            let bias_update = bias_grad.mul_op(&lr_tensor)?;
            let binding = bias.tensor();
            let bias_tensor = binding.write();
            bias_tensor.sub(&bias_update)?;
        }

        Ok(())
    }

    /// Optimized update for convolutional layers
    pub fn update_conv_layer(
        weight: &Parameter,
        bias: Option<&Parameter>,
        weight_grad: &Tensor,
        bias_grad: Option<&Tensor>,
        learning_rate: f32,
    ) -> Result<()> {
        // For now, use the same logic as linear layers
        // In practice, this could use specialized convolution-aware updates
        Self::update_linear_layer(weight, bias, weight_grad, bias_grad, learning_rate)
    }

    /// Optimized update for normalization layers
    pub fn update_norm_layer(
        weight: &Parameter,
        bias: &Parameter,
        weight_grad: &Tensor,
        bias_grad: &Tensor,
        learning_rate: f32,
    ) -> Result<()> {
        let lr_tensor = torsh_tensor::creation::tensor_scalar(learning_rate)?;

        // Update weight (scale parameter)
        let weight_update = weight_grad.mul_op(&lr_tensor)?;
        let binding = weight.tensor();
        let weight_tensor = binding.write();
        weight_tensor.sub(&weight_update)?;

        // Update bias (shift parameter)
        let bias_update = bias_grad.mul_op(&lr_tensor)?;
        let binding = bias.tensor();
        let bias_tensor = binding.write();
        bias_tensor.sub(&bias_update)?;

        Ok(())
    }
}

/// Helper function to create a tensor with the same shape and device as the input
fn zeros_like(tensor: &Tensor) -> Result<Tensor> {
    torsh_tensor::creation::zeros(tensor.shape().dims())
}

/// Utility functions for parameter update optimizations
pub mod utils {
    use super::*;

    /// Calculate memory usage of parameter set
    pub fn calculate_memory_usage(parameters: &HashMap<String, Parameter>) -> usize {
        parameters
            .values()
            .map(|p| {
                let shape_size = p.tensor().read().shape().numel();
                shape_size * std::mem::size_of::<f32>() // Assuming f32 parameters
            })
            .sum()
    }

    /// Group parameters by device for efficient updates
    pub fn group_parameters_by_device(
        parameters: &HashMap<String, Parameter>,
    ) -> HashMap<String, Vec<(&String, &Parameter)>> {
        let mut groups = HashMap::new();

        for (name, param) in parameters {
            let device_key = format!("{:?}", param.tensor().read().device());
            groups
                .entry(device_key)
                .or_insert_with(Vec::new)
                .push((name, param));
        }

        groups
    }

    /// Estimate optimal batch size for memory budget
    pub fn estimate_batch_size(
        parameter_memory: usize,
        memory_budget: usize,
        safety_factor: f32,
    ) -> usize {
        if parameter_memory == 0 {
            return 1;
        }

        let effective_budget = (memory_budget as f32 * safety_factor) as usize;
        std::cmp::max(1, effective_budget / parameter_memory)
    }

    /// Check if parameters are compatible for vectorized operations
    pub fn are_parameters_vectorizable(
        params1: &HashMap<String, Parameter>,
        params2: &HashMap<String, Parameter>,
    ) -> bool {
        if params1.len() != params2.len() {
            return false;
        }

        for (name, param1) in params1 {
            if let Some(param2) = params2.get(name) {
                let shape1 = param1.tensor().read().shape();
                let shape2 = param2.tensor().read().shape();
                let device1 = param1.tensor().read().device();
                let device2 = param2.tensor().read().device();

                if shape1 != shape2 || device1 != device2 {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_parameter_updater_creation() {
        let updater = ParameterUpdater::new();
        assert_eq!(updater.config.use_vectorization, true);
        assert_eq!(updater.config.use_inplace_updates, true);
    }

    #[test]
    fn test_update_statistics() {
        let mut stats = UpdateStatistics::new();
        assert_eq!(stats.total_updates, 0);

        stats.total_updates = 10;
        stats.total_time = std::time::Duration::from_secs(1);

        assert_eq!(stats.updates_per_second(), 10.0);
        assert_eq!(
            stats.average_update_time(),
            std::time::Duration::from_millis(100)
        );
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut stats = UpdateStatistics::new();
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        assert_eq!(stats.cache_hit_ratio(), 0.8);
    }

    #[test]
    fn test_memory_calculation() -> Result<()> {
        let mut params = HashMap::new();
        let tensor = randn(&[2, 3])?;
        let param = Parameter::new(tensor);
        params.insert("test_param".to_string(), param);

        let memory_usage = utils::calculate_memory_usage(&params);
        assert_eq!(memory_usage, 2 * 3 * std::mem::size_of::<f32>());

        Ok(())
    }

    #[test]
    fn test_batch_size_estimation() {
        let param_memory = 1000;
        let memory_budget = 10000;
        let safety_factor = 0.8;

        let batch_size = utils::estimate_batch_size(param_memory, memory_budget, safety_factor);
        assert_eq!(batch_size, 8); // (10000 * 0.8) / 1000 = 8
    }

    #[test]
    fn test_gradient_clipping() -> Result<()> {
        let updater = ParameterUpdater::new();
        let mut gradients = HashMap::new();

        let grad1 = randn(&[2, 2])?.mul_op(&torsh_tensor::creation::tensor_scalar(10.0)?)?; // Large gradient
        let grad2 = randn(&[2, 2])?.mul_op(&torsh_tensor::creation::tensor_scalar(10.0)?)?; // Large gradient

        gradients.insert("param1".to_string(), grad1);
        gradients.insert("param2".to_string(), grad2);

        let original_norm = updater.clip_gradients(&mut gradients, 1.0)?;

        // Norm should have been > 1.0 originally, now gradients should be clipped
        assert!(original_norm > 1.0);

        // Check that gradients were actually clipped
        let mut new_norm_squared = 0.0f32;
        for grad in gradients.values() {
            let grad_norm_squared = grad.mul_op(grad)?.sum()?.item()?;
            new_norm_squared += grad_norm_squared;
        }
        let new_norm = new_norm_squared.sqrt();

        // New norm should be approximately 1.0 (within floating point precision)
        assert!((new_norm - 1.0).abs() < 1e-5);

        Ok(())
    }
}
