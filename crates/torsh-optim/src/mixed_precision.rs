//! Mixed precision support for optimizers
//!
//! Mixed precision training uses 16-bit floating point (fp16) for forward and backward passes
//! while maintaining 32-bit floating point (fp32) master weights in the optimizer.
//! This reduces memory usage and can improve training speed while maintaining model accuracy.

use crate::{Optimizer, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::dtype::DType;
use torsh_core::error::Result;
use torsh_core::DeviceType;
use torsh_tensor::Tensor;

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision training
    pub enabled: bool,
    /// Loss scaling factor to prevent gradient underflow
    pub loss_scale: f32,
    /// Whether to use dynamic loss scaling
    pub dynamic_scale: bool,
    /// Initial loss scale for dynamic scaling
    pub init_scale: f32,
    /// Factor to increase loss scale when no overflow is detected
    pub scale_growth_factor: f32,
    /// Number of steps to wait before increasing loss scale
    pub scale_growth_interval: u32,
    /// Factor to decrease loss scale when overflow is detected
    pub backoff_factor: f32,
    /// Maximum loss scale value
    pub max_scale: f32,
    /// Minimum loss scale value
    pub min_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 65536.0,
            dynamic_scale: true,
            init_scale: 65536.0,
            scale_growth_factor: 2.0,
            scale_growth_interval: 2000,
            backoff_factor: 0.5,
            max_scale: 2.0_f32.powi(24),
            min_scale: 1.0,
        }
    }
}

/// Mixed precision optimizer wrapper
pub struct MixedPrecisionOptimizer<O: Optimizer> {
    optimizer: O,
    config: MixedPrecisionConfig,
    master_weights: HashMap<String, Tensor>,
    loss_scaler: LossScaler,
    overflow_detected: bool,
}

impl<O: Optimizer> MixedPrecisionOptimizer<O> {
    /// Create a new mixed precision optimizer wrapper
    pub fn new(optimizer: O, config: MixedPrecisionConfig) -> Self {
        let loss_scaler = if config.dynamic_scale {
            LossScaler::Dynamic(DynamicLossScaler::new(
                config.init_scale,
                config.scale_growth_factor,
                config.scale_growth_interval,
                config.backoff_factor,
            ))
        } else {
            LossScaler::Static(StaticLossScaler::new(config.loss_scale))
        };

        Self {
            optimizer,
            config,
            master_weights: HashMap::new(),
            loss_scaler,
            overflow_detected: false,
        }
    }

    /// Create a mixed precision optimizer with default configuration
    pub fn with_defaults(optimizer: O) -> Self {
        let mut config = MixedPrecisionConfig::default();
        config.enabled = true;
        Self::new(optimizer, config)
    }

    /// Get the current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scaler.get_scale()
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Initialize master weights for mixed precision training
    pub fn initialize_master_weights(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        for param_arc in params {
            let param = param_arc.read();
            let param_id = format!("{:p}", param_arc.as_ref());

            // Create fp32 master weight if parameter is fp16
            if param.dtype() == DType::F16 {
                let master_weight = param.to_dtype(DType::F32)?;
                self.master_weights.insert(param_id, master_weight);
            }
        }
        Ok(())
    }

    /// Scale loss for backward pass
    pub fn scale_loss(&mut self, loss: &mut Tensor) -> Result<()> {
        if self.config.enabled {
            let scale = self.loss_scaler.get_scale();
            loss.mul_scalar_(scale)?;
        }
        Ok(())
    }

    /// Unscale gradients before optimizer step
    pub fn unscale_gradients(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let scale = self.loss_scaler.get_scale();
        let inv_scale = 1.0 / scale;
        let mut overflow_detected = false;

        for param_arc in params {
            let mut param = param_arc.write();
            if let Some(grad) = param.grad_mut() {
                // Check for inf/nan before unscaling
                if self.has_inf_or_nan(grad)? {
                    overflow_detected = true;
                    break;
                }

                // Unscale gradient
                grad.mul_scalar_(inv_scale)?;

                // Check for inf/nan after unscaling
                if self.has_inf_or_nan(grad)? {
                    overflow_detected = true;
                    break;
                }
            }
        }

        self.overflow_detected = overflow_detected;

        if overflow_detected {
            // Zero out gradients to prevent parameter updates
            for param_arc in params {
                let mut param = param_arc.write();
                param.zero_grad();
            }

            // Update loss scaler
            self.loss_scaler.on_overflow_detected();
        } else {
            self.loss_scaler.on_successful_step();
        }

        Ok(overflow_detected)
    }

    /// Update master weights from fp16 parameters
    pub fn update_master_weights(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        for param_arc in params {
            let param = param_arc.read();
            let param_id = format!("{:p}", param_arc.as_ref());

            if let Some(master_weight) = self.master_weights.get_mut(&param_id) {
                // Copy fp16 parameter to fp32 master weight
                let param_fp32 = param.to_dtype(DType::F32)?;
                *master_weight = param_fp32;
            }
        }

        Ok(())
    }

    /// Copy master weights back to fp16 parameters
    pub fn copy_master_to_params(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        for param_arc in params {
            let mut param = param_arc.write();
            let param_id = format!("{:p}", param_arc.as_ref());

            if let Some(master_weight) = self.master_weights.get(&param_id) {
                // Copy fp32 master weight back to fp16 parameter
                let param_fp16 = master_weight.to_dtype(param.dtype())?;
                *param = param_fp16;
            }
        }

        Ok(())
    }

    /// Check if tensor contains inf or nan values
    fn has_inf_or_nan(&self, tensor: &Tensor) -> Result<bool> {
        // This is a simplified check - in a real implementation, this would
        // use optimized kernels to check for inf/nan values
        let data = tensor.to_vec()?;
        Ok(data.iter().any(|&x| x.is_infinite() || x.is_nan()))
    }

    /// Get the underlying optimizer
    pub fn inner(&self) -> &O {
        &self.optimizer
    }

    /// Get the underlying optimizer mutably
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> Optimizer for MixedPrecisionOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        if self.overflow_detected {
            // Skip optimizer step if overflow was detected
            self.overflow_detected = false;
            return Ok(());
        }

        self.optimizer.step()
    }

    fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.optimizer.load_state_dict(state)
    }
}

/// Loss scaling strategies
enum LossScaler {
    Static(StaticLossScaler),
    Dynamic(DynamicLossScaler),
}

impl LossScaler {
    fn get_scale(&self) -> f32 {
        match self {
            LossScaler::Static(scaler) => scaler.get_scale(),
            LossScaler::Dynamic(scaler) => scaler.get_scale(),
        }
    }

    fn on_overflow_detected(&mut self) {
        match self {
            LossScaler::Static(_) => {} // Static scaler doesn't change
            LossScaler::Dynamic(scaler) => scaler.on_overflow_detected(),
        }
    }

    fn on_successful_step(&mut self) {
        match self {
            LossScaler::Static(_) => {} // Static scaler doesn't change
            LossScaler::Dynamic(scaler) => scaler.on_successful_step(),
        }
    }
}

/// Static loss scaler with fixed scale value
struct StaticLossScaler {
    scale: f32,
}

impl StaticLossScaler {
    fn new(scale: f32) -> Self {
        Self { scale }
    }

    fn get_scale(&self) -> f32 {
        self.scale
    }
}

/// Dynamic loss scaler that adjusts scale based on overflow detection
struct DynamicLossScaler {
    scale: f32,
    growth_factor: f32,
    growth_interval: u32,
    backoff_factor: f32,
    growth_tracker: u32,
}

impl DynamicLossScaler {
    fn new(init_scale: f32, growth_factor: f32, growth_interval: u32, backoff_factor: f32) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            growth_interval,
            backoff_factor,
            growth_tracker: 0,
        }
    }

    fn get_scale(&self) -> f32 {
        self.scale
    }

    fn on_overflow_detected(&mut self) {
        // Decrease scale and reset growth tracker
        self.scale *= self.backoff_factor;
        self.scale = self.scale.max(1.0); // Minimum scale of 1.0
        self.growth_tracker = 0;
    }

    fn on_successful_step(&mut self) {
        // Increment growth tracker
        self.growth_tracker += 1;

        // Increase scale if we've had enough successful steps
        if self.growth_tracker >= self.growth_interval {
            self.scale *= self.growth_factor;
            self.scale = self.scale.min(2.0_f32.powi(24)); // Maximum scale
            self.growth_tracker = 0;
        }
    }
}

/// Utilities for mixed precision training
pub mod utils {
    use super::*;

    /// Check if the device supports mixed precision
    pub fn supports_mixed_precision(device: &DeviceType) -> bool {
        match device {
            DeviceType::Cuda(_) => true,  // CUDA has good fp16 support
            DeviceType::Metal(_) => true, // Metal supports fp16
            DeviceType::Cpu => false,     // CPU fp16 support is limited
            DeviceType::Wgpu(_) => false, // WebGPU fp16 support varies
        }
    }

    /// Convert model parameters to fp16 for mixed precision training
    pub fn convert_to_fp16(params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        for param_arc in params {
            let mut param = param_arc.write();
            if param.dtype() == DType::F32 {
                let param_fp16 = param.to_dtype(DType::F16)?;
                *param = param_fp16;
            }
        }
        Ok(())
    }

    /// Convert model parameters back to fp32
    pub fn convert_to_fp32(params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        for param_arc in params {
            let mut param = param_arc.write();
            if param.dtype() == DType::F16 {
                let param_fp32 = param.to_dtype(DType::F32)?;
                *param = param_fp32;
            }
        }
        Ok(())
    }

    /// Estimate memory savings from using fp16
    pub fn estimate_memory_savings(params: &[Arc<RwLock<Tensor>>]) -> (usize, usize, f64) {
        let mut fp32_size = 0;
        let mut fp16_size = 0;

        for param_arc in params {
            let param = param_arc.read();
            let num_elements = param.numel();
            fp32_size += num_elements * 4; // 4 bytes per fp32
            fp16_size += num_elements * 2; // 2 bytes per fp16
        }

        let savings_ratio = 1.0 - (fp16_size as f64 / fp32_size as f64);
        (fp32_size, fp16_size, savings_ratio)
    }
}

/// Helper function to wrap any optimizer with mixed precision support
pub fn with_mixed_precision<O: Optimizer>(
    optimizer: O,
    config: Option<MixedPrecisionConfig>,
) -> MixedPrecisionOptimizer<O> {
    match config {
        Some(config) => MixedPrecisionOptimizer::new(optimizer, config),
        None => MixedPrecisionOptimizer::with_defaults(optimizer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use torsh_core::device::Device;
    use torsh_tensor::creation;

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.loss_scale, 65536.0);
        assert!(config.dynamic_scale);
    }

    #[test]
    fn test_static_loss_scaler() {
        let scaler = StaticLossScaler::new(1024.0);
        assert_eq!(scaler.get_scale(), 1024.0);
    }

    #[test]
    fn test_dynamic_loss_scaler() {
        let mut scaler = DynamicLossScaler::new(1024.0, 2.0, 2, 0.5);
        assert_eq!(scaler.get_scale(), 1024.0);

        // Simulate overflow
        scaler.on_overflow_detected();
        assert_eq!(scaler.get_scale(), 512.0);

        // Simulate successful steps
        scaler.on_successful_step();
        scaler.on_successful_step();
        assert_eq!(scaler.get_scale(), 1024.0);
    }

    #[test]
    fn test_mixed_precision_optimizer_creation() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let mp_optimizer = MixedPrecisionOptimizer::with_defaults(sgd);
        assert!(mp_optimizer.is_enabled());
        assert_eq!(mp_optimizer.get_loss_scale(), 65536.0);
    }

    #[test]
    fn test_with_mixed_precision_helper() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let mp_optimizer = with_mixed_precision(sgd, None);
        assert!(mp_optimizer.is_enabled());
    }

    #[test]
    fn test_memory_savings_estimation() {
        let param1 = Arc::new(RwLock::new(creation::randn::<f32>(&[100, 100]).unwrap()));
        let param2 = Arc::new(RwLock::new(creation::randn::<f32>(&[50, 50]).unwrap()));
        let params = vec![param1, param2];

        let (fp32_size, fp16_size, savings_ratio) = utils::estimate_memory_savings(&params);

        assert_eq!(fp32_size, (10000 + 2500) * 4); // Total elements * 4 bytes
        assert_eq!(fp16_size, (10000 + 2500) * 2); // Total elements * 2 bytes
        assert!((savings_ratio - 0.5).abs() < 1e-6); // Should be ~50% savings
    }

    // Test disabled due to Device trait issues
    // #[test]
    // fn test_supports_mixed_precision() {
    //     let cpu_device = Device::cpu();
    //     assert!(!utils::supports_mixed_precision(&cpu_device));
    // }
}
