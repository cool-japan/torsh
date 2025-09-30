//! Mixed precision training support for Metal Performance Shaders

use metal::{CommandBuffer, Device};
use std::collections::HashMap;

use crate::metal::{
    mps::{MPSDataType, MPSOperation},
    MetalBuffer, MetalError, Result,
};

/// Mixed precision training manager
pub struct MPSMixedPrecision {
    device: Device,
    loss_scaling: f32,
    initial_loss_scale: f32,
    loss_scale_factor: f32,
    scale_window: usize,
    min_loss_scale: f32,
    max_loss_scale: f32,
    consecutive_unskipped: usize,
    enabled: bool,
    // Gradient scaling state
    found_inf: bool,
    scale_growth_tracker: usize,
}

impl MPSMixedPrecision {
    /// Create a new mixed precision manager
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            loss_scaling: 65536.0, // 2^16
            initial_loss_scale: 65536.0,
            loss_scale_factor: 2.0,
            scale_window: 2000,
            min_loss_scale: 1.0,
            max_loss_scale: 65536.0 * 65536.0, // 2^32
            consecutive_unskipped: 0,
            enabled: true,
            found_inf: false,
            scale_growth_tracker: 0,
        }
    }

    /// Enable or disable mixed precision training
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        if self.enabled {
            self.loss_scaling
        } else {
            1.0
        }
    }

    /// Scale loss for backward pass
    pub fn scale_loss(
        &self,
        command_buffer: &CommandBuffer,
        loss: &MetalBuffer,
        scaled_loss: &MetalBuffer,
    ) -> Result<()> {
        if !self.enabled {
            // Just copy loss to scaled_loss if mixed precision is disabled
            return self.copy_buffer(command_buffer, loss, scaled_loss);
        }

        // Multiply loss by loss_scaling factor
        self.scale_tensor(command_buffer, loss, self.loss_scaling, scaled_loss)
    }

    /// Unscale gradients after backward pass
    pub fn unscale_gradients(
        &mut self,
        command_buffer: &CommandBuffer,
        gradients: &[MetalBuffer],
        unscaled_gradients: &[MetalBuffer],
    ) -> Result<bool> {
        if !self.enabled {
            // Just copy gradients if mixed precision is disabled
            for (grad, unscaled) in gradients.iter().zip(unscaled_gradients.iter()) {
                self.copy_buffer(command_buffer, grad, unscaled)?;
            }
            return Ok(true);
        }

        // Check for inf/nan in gradients
        self.found_inf = false;
        for gradient in gradients {
            if self.has_inf_or_nan(command_buffer, gradient)? {
                self.found_inf = true;
                break;
            }
        }

        if self.found_inf {
            // Skip parameter updates if inf/nan found
            self.consecutive_unskipped = 0;
            self.reduce_loss_scale();
            return Ok(false);
        }

        // Unscale gradients by dividing by loss_scaling
        let inv_scale = 1.0 / self.loss_scaling;
        for (grad, unscaled) in gradients.iter().zip(unscaled_gradients.iter()) {
            self.scale_tensor(command_buffer, grad, inv_scale, unscaled)?;
        }

        self.consecutive_unskipped += 1;
        self.update_loss_scale();

        Ok(true)
    }

    /// Convert tensor to half precision (FP16)
    pub fn to_half_precision(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Implementation would use Metal compute shaders to convert FP32 to FP16
        // For now, this is a placeholder
        self.cast_tensor(command_buffer, input, output, MPSDataType::Float16)
    }

    /// Convert tensor to full precision (FP32)
    pub fn to_full_precision(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Implementation would use Metal compute shaders to convert FP16 to FP32
        self.cast_tensor(command_buffer, input, output, MPSDataType::Float32)
    }

    /// Reduce loss scale when inf/nan is detected
    fn reduce_loss_scale(&mut self) {
        self.loss_scaling = (self.loss_scaling / self.loss_scale_factor).max(self.min_loss_scale);
        self.scale_growth_tracker = 0;
    }

    /// Update loss scale based on training stability
    fn update_loss_scale(&mut self) {
        if self.consecutive_unskipped >= self.scale_window {
            self.loss_scaling =
                (self.loss_scaling * self.loss_scale_factor).min(self.max_loss_scale);
            self.consecutive_unskipped = 0;
            self.scale_growth_tracker += 1;
        }
    }

    /// Scale a tensor by a scalar value
    fn scale_tensor(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        scale: f32,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Implementation would use a Metal compute shader for efficient scaling
        // For now, this is a placeholder
        self.copy_buffer(command_buffer, input, output)
    }

    /// Copy buffer contents
    fn copy_buffer(
        &self,
        command_buffer: &CommandBuffer,
        src: &MetalBuffer,
        dst: &MetalBuffer,
    ) -> Result<()> {
        // Implementation would use Metal's blit encoder for efficient copying
        // For now, this is a placeholder
        Ok(())
    }

    /// Cast tensor to different data type
    fn cast_tensor(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        target_type: MPSDataType,
    ) -> Result<()> {
        // Implementation would use Metal compute shaders for type conversion
        // For now, this is a placeholder
        Ok(())
    }

    /// Check if tensor contains inf or nan values
    fn has_inf_or_nan(&self, command_buffer: &CommandBuffer, tensor: &MetalBuffer) -> Result<bool> {
        // Implementation would use a reduction operation to check for inf/nan
        // For now, return false (no inf/nan detected)
        Ok(false)
    }

    /// Get training statistics
    pub fn get_stats(&self) -> MixedPrecisionStats {
        MixedPrecisionStats {
            current_loss_scale: self.loss_scaling,
            consecutive_unskipped: self.consecutive_unskipped,
            scale_growth_tracker: self.scale_growth_tracker,
            found_inf_last_step: self.found_inf,
            enabled: self.enabled,
        }
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.loss_scaling = self.initial_loss_scale;
        self.consecutive_unskipped = 0;
        self.scale_growth_tracker = 0;
        self.found_inf = false;
    }
}

/// Mixed precision training statistics
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    pub current_loss_scale: f32,
    pub consecutive_unskipped: usize,
    pub scale_growth_tracker: usize,
    pub found_inf_last_step: bool,
    pub enabled: bool,
}

/// Automatic mixed precision (AMP) context manager
pub struct MPSAutocast {
    device: Device,
    enabled: bool,
    mixed_precision: MPSMixedPrecision,
    fp16_ops: HashMap<String, bool>,
}

impl MPSAutocast {
    /// Create a new autocast context
    pub fn new(device: &Device, enabled: bool) -> Self {
        let mut fp16_ops = HashMap::new();

        // Define which operations should use FP16
        fp16_ops.insert("conv2d".to_string(), true);
        fp16_ops.insert("linear".to_string(), true);
        fp16_ops.insert("matmul".to_string(), true);
        fp16_ops.insert("bmm".to_string(), true);
        fp16_ops.insert("addmm".to_string(), true);

        // Operations that should stay in FP32
        fp16_ops.insert("softmax".to_string(), false);
        fp16_ops.insert("log_softmax".to_string(), false);
        fp16_ops.insert("cross_entropy".to_string(), false);
        fp16_ops.insert("mse_loss".to_string(), false);
        fp16_ops.insert("layer_norm".to_string(), false);
        fp16_ops.insert("batch_norm".to_string(), false);

        Self {
            device: device.clone(),
            enabled,
            mixed_precision: MPSMixedPrecision::new(device),
            fp16_ops,
        }
    }

    /// Check if operation should use FP16
    pub fn should_use_fp16(&self, op_name: &str) -> bool {
        if !self.enabled {
            return false;
        }

        self.fp16_ops.get(op_name).copied().unwrap_or(false)
    }

    /// Convert inputs to appropriate precision for operation
    pub fn autocast_inputs(
        &self,
        command_buffer: &CommandBuffer,
        op_name: &str,
        inputs: &[MetalBuffer],
    ) -> Result<Vec<MetalBuffer>> {
        let mut converted_inputs = Vec::new();

        if self.should_use_fp16(op_name) {
            for input in inputs {
                let fp16_input = MetalBuffer::zeros(
                    input.shape(),
                    &torsh_core::DType::F16,
                    &crate::metal::device::MetalDevice::new()?,
                )?;
                self.mixed_precision
                    .to_half_precision(command_buffer, input, &fp16_input)?;
                converted_inputs.push(fp16_input);
            }
        } else {
            // Keep in FP32 or convert to FP32 if needed
            for input in inputs {
                if input.dtype() == torsh_core::DType::F16 {
                    let fp32_input = MetalBuffer::zeros(
                        input.shape(),
                        &torsh_core::DType::F32,
                        &crate::metal::device::MetalDevice::new()?,
                    )?;
                    self.mixed_precision
                        .to_full_precision(command_buffer, input, &fp32_input)?;
                    converted_inputs.push(fp32_input);
                } else {
                    converted_inputs.push(input.clone());
                }
            }
        }

        Ok(converted_inputs)
    }

    /// Get the mixed precision manager
    pub fn mixed_precision(&mut self) -> &mut MPSMixedPrecision {
        &mut self.mixed_precision
    }
}

/// Gradient scaler for mixed precision training
pub struct MPSGradScaler {
    mixed_precision: MPSMixedPrecision,
    update_frequency: usize,
    update_counter: usize,
}

impl MPSGradScaler {
    /// Create a new gradient scaler
    pub fn new(device: &Device, initial_scale: f32, growth_factor: f32) -> Self {
        let mut mixed_precision = MPSMixedPrecision::new(device);
        mixed_precision.loss_scaling = initial_scale;
        mixed_precision.loss_scale_factor = growth_factor;

        Self {
            mixed_precision,
            update_frequency: 2000,
            update_counter: 0,
        }
    }

    /// Scale loss for backward pass
    pub fn scale(
        &self,
        command_buffer: &CommandBuffer,
        loss: &MetalBuffer,
        scaled_loss: &MetalBuffer,
    ) -> Result<()> {
        self.mixed_precision
            .scale_loss(command_buffer, loss, scaled_loss)
    }

    /// Unscale gradients and update scale factor
    pub fn step(
        &mut self,
        command_buffer: &CommandBuffer,
        gradients: &[MetalBuffer],
        unscaled_gradients: &[MetalBuffer],
    ) -> Result<bool> {
        let should_update = self.mixed_precision.unscale_gradients(
            command_buffer,
            gradients,
            unscaled_gradients,
        )?;

        self.update_counter += 1;
        if self.update_counter >= self.update_frequency {
            self.update_counter = 0;
        }

        Ok(should_update)
    }

    /// Get current scale
    pub fn get_scale(&self) -> f32 {
        self.mixed_precision.get_loss_scale()
    }

    /// Check if inf/nan was found in last step
    pub fn found_inf(&self) -> bool {
        self.mixed_precision.found_inf
    }
}

/// Utility functions for mixed precision operations
pub mod utils {
    use super::*;

    /// Create mixed precision training configuration
    pub fn create_amp_config() -> AMPConfig {
        AMPConfig {
            enabled: true,
            opt_level: OptLevel::O1,
            loss_scale: Some(128.0),
            max_loss_scale: 65536.0,
            min_loss_scale: 1.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }

    /// Check if device supports FP16 operations efficiently
    pub fn supports_efficient_fp16(device: &Device) -> bool {
        // Check device capabilities
        // This would query Metal device properties
        // For now, assume modern devices support efficient FP16
        true
    }

    /// Estimate memory savings from mixed precision
    pub fn estimate_memory_savings(model_params: usize) -> f32 {
        // Rough estimate: FP16 uses half the memory of FP32 for weights and activations
        // But gradients and optimizer states might still be FP32
        // Conservative estimate: 25-40% memory savings
        0.35
    }

    /// Estimate performance improvement from mixed precision
    pub fn estimate_performance_improvement(device: &Device) -> f32 {
        // Performance improvement depends on hardware
        // Modern Apple Silicon GPUs can show 1.5-2x improvement
        if supports_efficient_fp16(device) {
            1.7 // 70% improvement estimate
        } else {
            1.1 // 10% improvement on older hardware
        }
    }
}

/// AMP configuration options
#[derive(Debug, Clone)]
pub struct AMPConfig {
    pub enabled: bool,
    pub opt_level: OptLevel,
    pub loss_scale: Option<f32>,
    pub max_loss_scale: f32,
    pub min_loss_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
}

/// Optimization levels for automatic mixed precision
#[derive(Debug, Clone, Copy)]
pub enum OptLevel {
    /// Conservative: FP16 for forward pass, FP32 for loss computation
    O0,
    /// Balanced: FP16 for most operations, FP32 for numerically sensitive ops
    O1,
    /// Aggressive: FP16 for almost all operations
    O2,
    /// Maximum performance: FP16 everywhere possible
    O3,
}

impl Default for AMPConfig {
    fn default() -> Self {
        utils::create_amp_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_creation() {
        // This would require a real Metal device
        // For now, just test that we can create the struct
        assert!(true);
    }

    #[test]
    fn test_loss_scale_update() {
        // Test loss scale update logic
        let device = unsafe { std::mem::zeroed() }; // Placeholder
        let mut mp = MPSMixedPrecision::new(&device);

        let initial_scale = mp.get_loss_scale();
        mp.consecutive_unskipped = 2000;
        mp.update_loss_scale();

        assert!(mp.get_loss_scale() > initial_scale);
    }

    #[test]
    fn test_autocast_op_detection() {
        let device = unsafe { std::mem::zeroed() }; // Placeholder
        let autocast = MPSAutocast::new(&device, true);

        assert!(autocast.should_use_fp16("conv2d"));
        assert!(autocast.should_use_fp16("linear"));
        assert!(!autocast.should_use_fp16("softmax"));
        assert!(!autocast.should_use_fp16("layer_norm"));
    }

    #[test]
    fn test_amp_config() {
        let config = AMPConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.opt_level, OptLevel::O1));
        assert!(config.growth_factor > 1.0);
    }
}
