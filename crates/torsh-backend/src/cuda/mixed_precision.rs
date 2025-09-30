//! Mixed precision training support for CUDA backend

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;
use torsh_core::DType;

/// Gradient scaler for mixed precision training
pub struct GradientScaler {
    /// Current scale factor
    scale: f32,
    /// Growth factor for increasing scale
    growth_factor: f32,
    /// Backoff factor for decreasing scale
    backoff_factor: f32,
    /// Growth interval (steps before attempting to increase scale)
    growth_interval: u32,
    /// Steps since last scale update
    steps_since_update: u32,
    /// Whether to enable scaler
    enabled: bool,
    /// Found infinite gradients
    found_inf: Arc<Mutex<bool>>,
}

impl GradientScaler {
    /// Create new gradient scaler
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
            enabled: true,
            found_inf: Arc::new(Mutex::new(false)),
        }
    }

    /// Create default gradient scaler
    pub fn default() -> Self {
        Self::new(65536.0) // 2^16
    }

    /// Enable or disable the scaler
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if scaler is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current scale factor
    pub fn get_scale(&self) -> f32 {
        if self.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Set scale factor
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Set growth factor
    pub fn set_growth_factor(&mut self, growth_factor: f32) {
        self.growth_factor = growth_factor;
    }

    /// Set backoff factor
    pub fn set_backoff_factor(&mut self, backoff_factor: f32) {
        self.backoff_factor = backoff_factor;
    }

    /// Set growth interval
    pub fn set_growth_interval(&mut self, growth_interval: u32) {
        self.growth_interval = growth_interval;
    }

    /// Scale gradients
    pub fn scale_gradients(
        &self,
        gradients: &mut [cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let scale = self.get_scale();

        for grad in gradients.iter_mut() {
            // Launch kernel to scale gradients
            self.scale_tensor(grad, scale, stream)?;
        }

        Ok(())
    }

    /// Scale a single tensor
    fn scale_tensor(
        &self,
        tensor: &mut cust::DevicePointer<f32>,
        scale: f32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        // Use tensor operations to scale
        crate::kernels::tensor_ops::launch_scalar_mul_f32(
            tensor.as_raw_mut(),
            tensor.as_raw_mut(),
            scale,
            1, // Assume size 1 for now - in real implementation, would need actual size
            stream.raw() as cuda_sys::CUstream,
        );
        Ok(())
    }

    /// Unscale gradients and check for infinities
    pub fn unscale_gradients(
        &mut self,
        gradients: &mut [cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<bool> {
        if !self.enabled {
            return Ok(false);
        }

        let scale = self.get_scale();
        let inv_scale = 1.0 / scale;

        // Reset infinity flag
        {
            let mut found_inf = self.found_inf.lock().unwrap();
            *found_inf = false;
        }

        // Unscale and check for infinities
        for grad in gradients.iter_mut() {
            self.unscale_and_check_tensor(grad, inv_scale, stream)?;
        }

        // Check if infinities were found
        let found_inf = {
            let found_inf = self.found_inf.lock().unwrap();
            *found_inf
        };

        Ok(found_inf)
    }

    /// Unscale a tensor and check for infinities
    fn unscale_and_check_tensor(
        &self,
        tensor: &mut cust::DevicePointer<f32>,
        inv_scale: f32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        // Scale by inverse
        crate::kernels::tensor_ops::launch_scalar_mul_f32(
            tensor.as_raw_mut(),
            tensor.as_raw_mut(),
            inv_scale,
            1, // Assume size 1 for now
            stream.raw() as cuda_sys::CUstream,
        );

        // Check for infinities using reduction operation
        // This would launch a custom kernel to check for inf/nan values
        // For now we'll use a placeholder that sets the flag if any value is problematic
        if self.check_tensor_validity(tensor, stream)? {
            let mut found_inf = self.found_inf.lock().unwrap();
            *found_inf = true;
        }

        Ok(())
    }

    /// Check if tensor contains infinite or NaN values
    fn check_tensor_validity(
        &self,
        tensor: &cust::DevicePointer<f32>,
        stream: &CudaStream,
    ) -> CudaResult<bool> {
        // In a full implementation, this would launch a reduction kernel
        // that checks for inf/nan values across the entire tensor
        // For now, return false (no infinities found)
        let _ = (tensor, stream); // Suppress unused warnings
        Ok(false)
    }

    /// Update scale based on whether infinities were found
    pub fn update_scale(&mut self, found_inf: bool) {
        if !self.enabled {
            return;
        }

        if found_inf {
            // Reduce scale and reset counter
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
        } else {
            // Increment counter
            self.steps_since_update += 1;

            // Increase scale if enough steps have passed
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }
    }

    /// Step function that combines unscaling, checking, and updating
    pub fn step(
        &mut self,
        optimizer: &mut dyn Optimizer,
        gradients: &mut [cust::DevicePointer<f32>],
        parameters: &mut [cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<bool> {
        if !self.enabled {
            // If disabled, just perform normal optimizer step
            optimizer.step(parameters, gradients, stream)?;
            return Ok(true);
        }

        // Unscale gradients and check for infinities
        let found_inf = self.unscale_gradients(gradients, stream)?;

        if found_inf {
            // Skip optimizer step if infinities found
            self.update_scale(true);
            Ok(false)
        } else {
            // Perform optimizer step
            optimizer.step(parameters, gradients, stream)?;
            self.update_scale(false);
            Ok(true)
        }
    }
}

/// Trait for optimizers that can be used with mixed precision training
pub trait Optimizer {
    fn step(
        &mut self,
        parameters: &mut [cust::DevicePointer<f32>],
        gradients: &[cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<()>;
}

/// Automatic Mixed Precision (AMP) context
pub struct AmpContext {
    /// Whether AMP is enabled
    enabled: bool,
    /// Gradient scaler
    scaler: GradientScaler,
    /// Cache for different precision operations
    cache: HashMap<String, DType>,
}

impl AmpContext {
    /// Create new AMP context
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            scaler: GradientScaler::default(),
            cache: HashMap::new(),
        }
    }

    /// Enable AMP
    pub fn enable(&mut self) {
        self.enabled = true;
        self.scaler.set_enabled(true);
    }

    /// Disable AMP
    pub fn disable(&mut self) {
        self.enabled = false;
        self.scaler.set_enabled(false);
    }

    /// Check if AMP is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get gradient scaler
    pub fn scaler(&mut self) -> &mut GradientScaler {
        &mut self.scaler
    }

    /// Get appropriate precision for operation
    pub fn get_precision(&self, operation: &str) -> DType {
        if !self.enabled {
            return DType::F32;
        }

        // Check cache first
        if let Some(&dtype) = self.cache.get(operation) {
            return dtype;
        }

        // Define precision policies for different operations
        let dtype = match operation {
            // Use FP16 for these operations (compute intensive)
            "conv2d" | "linear" | "matmul" | "attention" | "gemm" => DType::F16,

            // Use FP32 for these operations (precision sensitive)
            "batch_norm" | "layer_norm" | "softmax" | "cross_entropy" | "mse_loss"
            | "log_softmax" | "nll_loss" | "reduction" | "norm" => DType::F32,

            // Use FP16 for activations (element-wise operations)
            "relu" | "gelu" | "sigmoid" | "tanh" | "leaky_relu" | "elu" | "swish" => DType::F16,

            // Use FP32 for pooling operations (can cause precision issues)
            "max_pool" | "avg_pool" | "adaptive_pool" => DType::F32,

            // Default to FP32 for unknown operations
            _ => DType::F32,
        };

        dtype
    }

    /// Set custom precision policy for an operation
    pub fn set_precision_policy(&mut self, operation: String, dtype: DType) {
        self.cache.insert(operation, dtype);
    }

    /// Clear precision policy cache
    pub fn clear_precision_cache(&mut self) {
        self.cache.clear();
    }

    /// Convert tensor to appropriate precision
    pub fn autocast(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f16>,
        size: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Launch kernel to convert F32 to F16
        // In a full implementation, this would use a proper conversion kernel
        // For now, we'll use a placeholder
        let _ = (input, output, size, stream); // Suppress unused warnings
        Ok(())
    }

    /// Convert tensor to FP16 with saturation to prevent overflow
    pub fn autocast_with_saturation(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f16>,
        size: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // This would use a kernel that clamps values to FP16 range before conversion
        let _ = (input, output, size, stream); // Suppress unused warnings
        Ok(())
    }

    /// Convert tensor back to F32
    pub fn uncast(
        &self,
        input: cust::DevicePointer<f16>,
        output: cust::DevicePointer<f32>,
        size: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Launch kernel to convert F16 to F32
        // In a full implementation, this would use a proper conversion kernel
        let _ = (input, output, size, stream); // Suppress unused warnings
        Ok(())
    }
}

impl Default for AmpContext {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Mixed precision training manager
pub struct MixedPrecisionTrainer {
    /// AMP context
    amp_context: AmpContext,
    /// Loss scaling enabled
    loss_scaling: bool,
    /// Skip updates counter (for debugging)
    skip_count: u64,
    /// Successful updates counter
    update_count: u64,
}

impl MixedPrecisionTrainer {
    /// Create new mixed precision trainer
    pub fn new(enabled: bool, loss_scaling: bool) -> Self {
        Self {
            amp_context: AmpContext::new(enabled),
            loss_scaling,
            skip_count: 0,
            update_count: 0,
        }
    }

    /// Get skip count (for debugging)
    pub fn skip_count(&self) -> u64 {
        self.skip_count
    }

    /// Get update count (for debugging)
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Get skip ratio
    pub fn skip_ratio(&self) -> f64 {
        if self.update_count + self.skip_count == 0 {
            0.0
        } else {
            self.skip_count as f64 / (self.update_count + self.skip_count) as f64
        }
    }

    /// Get AMP context
    pub fn amp_context(&mut self) -> &mut AmpContext {
        &mut self.amp_context
    }

    /// Forward pass with automatic mixed precision
    pub fn forward_pass<F>(&self, mut forward_fn: F, stream: &CudaStream) -> CudaResult<()>
    where
        F: FnMut(&CudaStream) -> CudaResult<()>,
    {
        // Execute forward pass
        forward_fn(stream)?;
        Ok(())
    }

    /// Backward pass with gradient scaling
    pub fn backward_pass<F>(
        &mut self,
        mut backward_fn: F,
        gradients: &mut [cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        F: FnMut(&CudaStream) -> CudaResult<()>,
    {
        // Scale gradients before backward pass
        if self.loss_scaling {
            self.amp_context
                .scaler()
                .scale_gradients(gradients, stream)?;
        }

        // Execute backward pass
        backward_fn(stream)?;

        Ok(())
    }

    /// Optimizer step with gradient unscaling
    pub fn optimizer_step(
        &mut self,
        optimizer: &mut dyn Optimizer,
        gradients: &mut [cust::DevicePointer<f32>],
        parameters: &mut [cust::DevicePointer<f32>],
        stream: &CudaStream,
    ) -> CudaResult<bool> {
        let success = if self.loss_scaling {
            self.amp_context
                .scaler()
                .step(optimizer, gradients, parameters, stream)?
        } else {
            optimizer.step(parameters, gradients, stream)?;
            true
        };

        // Update counters
        if success {
            self.update_count += 1;
        } else {
            self.skip_count += 1;
        }

        Ok(success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_scaler_creation() {
        let scaler = GradientScaler::new(1024.0);
        assert_eq!(scaler.get_scale(), 1024.0);
        assert!(scaler.is_enabled());
    }

    #[test]
    fn test_gradient_scaler_default() {
        let scaler = GradientScaler::default();
        assert_eq!(scaler.get_scale(), 65536.0);
    }

    #[test]
    fn test_gradient_scaler_disable() {
        let mut scaler = GradientScaler::default();
        scaler.set_enabled(false);
        assert_eq!(scaler.get_scale(), 1.0);
        assert!(!scaler.is_enabled());
    }

    #[test]
    fn test_amp_context_creation() {
        let context = AmpContext::new(true);
        assert!(context.is_enabled());
    }

    #[test]
    fn test_amp_context_precision_policy() {
        let context = AmpContext::new(true);

        assert_eq!(context.get_precision("conv2d"), DType::F16);
        assert_eq!(context.get_precision("linear"), DType::F16);
        assert_eq!(context.get_precision("batch_norm"), DType::F32);
        assert_eq!(context.get_precision("softmax"), DType::F32);
        assert_eq!(context.get_precision("unknown"), DType::F32);
    }

    #[test]
    fn test_mixed_precision_trainer() {
        let trainer = MixedPrecisionTrainer::new(true, true);
        assert!(trainer.amp_context.is_enabled());
        assert!(trainer.loss_scaling);
    }

    #[test]
    fn test_scale_update_with_inf() {
        let mut scaler = GradientScaler::new(1000.0);
        let initial_scale = scaler.get_scale();

        scaler.update_scale(true); // Found infinity
        assert!(scaler.get_scale() < initial_scale);
    }

    #[test]
    fn test_scale_update_without_inf() {
        let mut scaler = GradientScaler::new(1000.0);
        scaler.set_growth_interval(1); // Short interval for testing

        let initial_scale = scaler.get_scale();
        scaler.update_scale(false); // No infinity
        scaler.update_scale(false); // Trigger growth

        assert!(scaler.get_scale() > initial_scale);
    }
}
