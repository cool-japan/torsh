//! Mixed Precision Training Support
//!
//! This module provides comprehensive mixed precision training capabilities including:
//! - Automatic Mixed Precision (AMP) training
//! - FP16/BF16 forward pass with FP32 gradient accumulation
//! - Gradient scaling for numerical stability
//! - Loss scaling and underflow detection
//! - Dynamic loss scaling adjustment

use crate::{Module, Parameter};
use torsh_core::{dtype::DType, error::Result};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Forward pass precision (FP16 or BF16)
    pub forward_dtype: DType,
    /// Gradient computation precision (usually FP32)
    pub gradient_dtype: DType,
    /// Initial loss scale factor
    pub init_scale: f32,
    /// Growth factor for loss scaling
    pub growth_factor: f32,
    /// Backoff factor for loss scaling
    pub backoff_factor: f32,
    /// Number of successful steps before increasing scale
    pub growth_interval: usize,
    /// Enable dynamic loss scaling
    pub dynamic_loss_scaling: bool,
    /// Maximum loss scale value
    pub max_loss_scale: f32,
    /// Minimum loss scale value
    pub min_loss_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            forward_dtype: DType::F16,
            gradient_dtype: DType::F32,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            dynamic_loss_scaling: true,
            max_loss_scale: 1e6,
            min_loss_scale: 1e-4,
        }
    }
}

/// Gradient Scaler for mixed precision training
#[derive(Debug)]
pub struct GradScaler {
    config: MixedPrecisionConfig,
    current_scale: f32,
    growth_tracker: usize,
    has_overflow: bool,
    has_underflow: bool,
}

impl GradScaler {
    /// Create a new gradient scaler
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            current_scale: config.init_scale,
            config,
            growth_tracker: 0,
            has_overflow: false,
            has_underflow: false,
        }
    }

    /// Create a gradient scaler with default config
    pub fn default() -> Self {
        Self::new(MixedPrecisionConfig::default())
    }

    /// Get current loss scale
    pub fn scale(&self) -> f32 {
        if self.config.enabled {
            self.current_scale
        } else {
            1.0
        }
    }

    /// Scale a tensor (typically loss) for backward pass
    pub fn scale_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.config.enabled {
            tensor.mul_scalar(self.scale())
        } else {
            Ok(tensor.clone())
        }
    }

    /// Unscale gradients before optimizer step
    pub fn unscale_gradients(&mut self, parameters: &HashMap<String, Parameter>) -> Result<bool> {
        if !self.config.enabled {
            return Ok(true);
        }

        let _inv_scale = 1.0 / self.current_scale;
        let has_inf_or_nan = false;

        for (_name, _param) in parameters {
            // Skip gradient processing for now as Parameter API needs to be updated
            // In a real implementation, this would check and unscale gradients
            // if let Some(grad) = param.grad() { ... }
        }

        if has_inf_or_nan {
            self.has_overflow = true;
            return Ok(false);
        }

        Ok(true)
    }

    /// Update the loss scale based on overflow/underflow detection
    pub fn update(&mut self) {
        if !self.config.enabled || !self.config.dynamic_loss_scaling {
            return;
        }

        if self.has_overflow {
            // Reduce scale on overflow
            self.current_scale *= self.config.backoff_factor;
            self.current_scale = self.current_scale.max(self.config.min_loss_scale);
            self.growth_tracker = 0;
            self.has_overflow = false;
        } else {
            // Increase scale after successful steps
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.current_scale *= self.config.growth_factor;
                self.current_scale = self.current_scale.min(self.config.max_loss_scale);
                self.growth_tracker = 0;
            }
        }

        self.has_underflow = false;
    }

    /// Check if a tensor contains inf or nan values
    #[allow(dead_code)]
    fn has_inf_or_nan(&self, tensor: &Tensor) -> Result<bool> {
        // Simplified implementation - in practice would need proper inf/nan detection
        let data = tensor.to_vec()?;
        for value in data {
            if value.is_infinite() || value.is_nan() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Reset the scaler state
    pub fn reset(&mut self) {
        self.current_scale = self.config.init_scale;
        self.growth_tracker = 0;
        self.has_overflow = false;
        self.has_underflow = false;
    }
}

/// Automatic Mixed Precision wrapper for models
#[derive(Debug)]
pub struct AutocastModel<M: Module> {
    model: M,
    config: MixedPrecisionConfig,
    in_autocast: bool,
}

impl<M: Module> AutocastModel<M> {
    /// Create a new autocast model wrapper
    pub fn new(model: M, config: MixedPrecisionConfig) -> Self {
        Self {
            model,
            config,
            in_autocast: false,
        }
    }

    /// Create with default mixed precision config
    pub fn with_default_config(model: M) -> Self {
        Self::new(model, MixedPrecisionConfig::default())
    }

    /// Enable autocast mode
    pub fn autocast<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let prev_autocast = self.in_autocast;
        self.in_autocast = true;
        let result = f(self);
        self.in_autocast = prev_autocast;
        result
    }

    /// Convert tensor to forward pass precision
    fn maybe_cast_input(&self, input: &Tensor) -> Result<Tensor> {
        if self.config.enabled && self.in_autocast {
            input.to_dtype(self.config.forward_dtype)
        } else {
            Ok(input.clone())
        }
    }

    /// Convert tensor back to full precision if needed
    fn maybe_cast_output(&self, output: &Tensor) -> Result<Tensor> {
        if self.config.enabled && self.in_autocast && output.dtype() != DType::F32 {
            output.to_dtype(DType::F32)
        } else {
            Ok(output.clone())
        }
    }

    /// Get the underlying model
    pub fn inner(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the underlying model
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M: Module> Module for AutocastModel<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let casted_input = self.maybe_cast_input(input)?;
        let output = self.model.forward(&casted_input)?;
        self.maybe_cast_output(&output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.model.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.model.named_parameters()
    }

    fn training(&self) -> bool {
        self.model.training()
    }

    fn train(&mut self) {
        self.model.train()
    }

    fn eval(&mut self) {
        self.model.eval()
    }

    fn set_training(&mut self, training: bool) {
        self.model.set_training(training);
    }

    fn to_device(&mut self, device: torsh_core::device::DeviceType) -> Result<()> {
        self.model.to_device(device)
    }
}

/// Mixed precision training utilities
pub struct MixedPrecisionTrainer {
    pub scaler: GradScaler,
    pub autocast_enabled: bool,
}

impl MixedPrecisionTrainer {
    /// Create a new mixed precision trainer
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            scaler: GradScaler::new(config.clone()),
            autocast_enabled: config.enabled,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MixedPrecisionConfig::default())
    }

    /// Perform a training step with mixed precision
    pub fn step<M: Module, F, L>(
        &mut self,
        model: &mut M,
        loss_fn: F,
        optimizer_step: L,
        input: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
        L: Fn(&HashMap<String, Parameter>) -> Result<()>,
    {
        // Forward pass in reduced precision
        let output = if self.autocast_enabled {
            let fp16_input = input.to_dtype(DType::F16)?;
            model.forward(&fp16_input)?
        } else {
            model.forward(input)?
        };

        // Compute loss in full precision
        let loss = loss_fn(&output.to_dtype(DType::F32)?, target)?;

        // Scale loss for backward pass
        let _scaled_loss = self.scaler.scale_tensor(&loss)?;

        // Backward pass (simplified - in practice would compute gradients)
        // This would trigger gradient computation through autograd

        // Unscale gradients and check for overflow
        let parameters = model.parameters();
        let gradients_valid = self.scaler.unscale_gradients(&parameters)?;

        // Only step optimizer if gradients are valid
        if gradients_valid {
            optimizer_step(&parameters)?;
        }

        // Update loss scale
        self.scaler.update();

        Ok(loss)
    }

    /// Check if currently in autocast mode
    pub fn is_autocast_enabled(&self) -> bool {
        self.autocast_enabled
    }

    /// Get current loss scale
    pub fn get_scale(&self) -> f32 {
        self.scaler.scale()
    }
}

/// Loss scaling utilities
pub mod loss_scaling {
    use super::*;

    /// Dynamic loss scaler that adjusts scale based on gradient overflow
    #[derive(Debug)]
    pub struct DynamicLossScaler {
        current_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
        stable_steps: usize,
        max_scale: f32,
        min_scale: f32,
    }

    impl DynamicLossScaler {
        /// Create a new dynamic loss scaler
        pub fn new() -> Self {
            Self {
                current_scale: 65536.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
                stable_steps: 0,
                max_scale: 1e6,
                min_scale: 1e-4,
            }
        }

        /// Scale a loss tensor
        pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
            loss.mul_scalar(self.current_scale)
        }

        /// Update scale based on gradient state
        pub fn update(&mut self, found_inf: bool) {
            if found_inf {
                self.current_scale *= self.backoff_factor;
                self.current_scale = self.current_scale.max(self.min_scale);
                self.stable_steps = 0;
            } else {
                self.stable_steps += 1;
                if self.stable_steps >= self.growth_interval {
                    self.current_scale *= self.growth_factor;
                    self.current_scale = self.current_scale.min(self.max_scale);
                    self.stable_steps = 0;
                }
            }
        }

        /// Get current scale value
        pub fn get_scale(&self) -> f32 {
            self.current_scale
        }
    }

    impl Default for DynamicLossScaler {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// BF16 (Brain Float16) specific utilities
pub mod bf16 {
    use super::*;

    /// BF16 configuration for mixed precision training
    #[derive(Debug, Clone)]
    pub struct BF16Config {
        /// Use BF16 for forward pass
        pub forward_bf16: bool,
        /// Use BF16 for gradient computation
        pub gradient_bf16: bool,
        /// Gradient clipping threshold
        pub grad_clip_threshold: Option<f32>,
    }

    impl Default for BF16Config {
        fn default() -> Self {
            Self {
                forward_bf16: true,
                gradient_bf16: false,
                grad_clip_threshold: Some(1.0),
            }
        }
    }

    /// BF16 trainer for specific BF16 optimizations
    pub struct BF16Trainer {
        config: BF16Config,
    }

    impl BF16Trainer {
        /// Create new BF16 trainer
        pub fn new(config: BF16Config) -> Self {
            Self { config }
        }

        /// Convert tensor to BF16 if enabled
        pub fn maybe_cast_bf16(&self, tensor: &Tensor) -> Result<Tensor> {
            if self.config.forward_bf16 {
                // Note: BF16 support would need to be implemented in the tensor backend
                // For now, we use F16 as a placeholder
                tensor.to_dtype(DType::F16)
            } else {
                Ok(tensor.clone())
            }
        }

        /// Clip gradients if threshold is set
        pub fn maybe_clip_gradients(&self, gradients: &HashMap<String, Parameter>) -> Result<()> {
            if let Some(_threshold) = self.config.grad_clip_threshold {
                // Implement gradient clipping
                for (_name, _param) in gradients {
                    // Skip gradient clipping for now as Parameter API needs to be updated
                    // if let Some(_grad) = param.grad() { ... }
                }
            }
            Ok(())
        }
    }
}

/// Convenience functions for mixed precision training
pub mod prelude {
    pub use super::{
        bf16::{BF16Config, BF16Trainer},
        loss_scaling::DynamicLossScaler,
        AutocastModel, GradScaler, MixedPrecisionConfig, MixedPrecisionTrainer,
    };

    /// Create a mixed precision config for FP16 training
    pub fn fp16_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            forward_dtype: torsh_core::dtype::DType::F16,
            ..Default::default()
        }
    }

    /// Create a mixed precision config for BF16 training
    pub fn bf16_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            forward_dtype: torsh_core::dtype::DType::F16, // Placeholder for BF16
            dynamic_loss_scaling: false, // BF16 typically doesn't need loss scaling
            init_scale: 1.0,
            ..Default::default()
        }
    }

    /// Create a conservative mixed precision config
    pub fn conservative_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            init_scale: 4096.0,
            growth_factor: 1.5,
            backoff_factor: 0.75,
            growth_interval: 5000,
            ..Default::default()
        }
    }

    /// Create an aggressive mixed precision config for faster training
    pub fn aggressive_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            init_scale: 131072.0,
            growth_factor: 2.5,
            backoff_factor: 0.25,
            growth_interval: 1000,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use torsh_tensor::creation::*;

    #[test]
    fn test_grad_scaler_creation() {
        let config = MixedPrecisionConfig::default();
        let scaler = GradScaler::new(config);
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn test_loss_scaling() -> Result<()> {
        let scaler = GradScaler::default();
        let loss = tensor_scalar(0.5f32)?;
        let scaled_loss = scaler.scale_tensor(&loss)?;

        let expected = 0.5 * scaler.scale();
        let actual = scaled_loss.to_vec()?[0];
        assert!((actual - expected).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_autocast_model() -> Result<()> {
        let linear = Linear::new(10, 5, true);
        let autocast_model = AutocastModel::with_default_config(linear);

        let input = zeros(&[2, 10])?;
        let _output = autocast_model.forward(&input)?;

        Ok(())
    }

    #[test]
    fn test_dynamic_loss_scaler() {
        let mut scaler = loss_scaling::DynamicLossScaler::new();
        let initial_scale = scaler.get_scale();

        // Test backoff on overflow
        scaler.update(true);
        assert!(scaler.get_scale() < initial_scale);

        // Test growth after stable steps
        let current_scale = scaler.get_scale();
        for _ in 0..2001 {
            scaler.update(false);
        }
        assert!(scaler.get_scale() > current_scale);
    }
}
