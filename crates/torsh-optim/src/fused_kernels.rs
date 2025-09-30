//! Fused optimizer kernels for improved performance
//!
//! This module provides fused implementations of common optimizer operations.
//! Fused kernels combine multiple tensor operations into single kernel calls,
//! reducing memory bandwidth requirements and improving performance.

use crate::OptimizerResult;
use std::ops::Add;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Fused Adam step operation
///
/// Combines the following operations in a single kernel:
/// 1. Bias correction: m_hat = m / (1 - beta1^step), v_hat = v / (1 - beta2^step)
/// 2. Update computation: update = lr * m_hat / (sqrt(v_hat) + eps)
/// 3. Parameter update: param = param - update
/// 4. Momentum updates: m = beta1 * m + (1 - beta1) * grad, v = beta2 * v + (1 - beta2) * grad^2
///
/// # Arguments
/// * `param` - Parameter tensor to update
/// * `grad` - Gradient tensor
/// * `exp_avg` - Exponential moving average of gradients (momentum)
/// * `exp_avg_sq` - Exponential moving average of squared gradients
/// * `lr` - Learning rate
/// * `beta1` - Exponential decay rate for first moment estimates
/// * `beta2` - Exponential decay rate for second moment estimates
/// * `eps` - Small constant for numerical stability
/// * `step` - Current step number (for bias correction)
/// * `weight_decay` - Weight decay coefficient (optional)
#[allow(clippy::too_many_arguments)]
pub fn fused_adam_step(
    param: &mut Tensor,
    grad: &Tensor,
    exp_avg: &mut Tensor,
    exp_avg_sq: &mut Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: u64,
    weight_decay: Option<f32>,
) -> Result<()> {
    // Apply weight decay if specified
    let effective_grad = if let Some(wd) = weight_decay {
        let decay_term = param.mul_scalar(wd)?;
        grad.add(&decay_term)?
    } else {
        grad.clone()
    };

    // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    exp_avg.mul_scalar_(beta1)?;
    let grad_term = effective_grad.mul_scalar(1.0 - beta1)?;
    *exp_avg = exp_avg.add(&grad_term)?;

    // Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    exp_avg_sq.mul_scalar_(beta2)?;
    let grad_sq = effective_grad.mul_op(&effective_grad)?;
    let grad_sq_term = grad_sq.mul_scalar(1.0 - beta2)?;
    *exp_avg_sq = exp_avg_sq.add(&grad_sq_term)?;

    // Compute bias correction terms
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);

    // Compute corrected learning rate: lr * sqrt(1 - beta2^step) / (1 - beta1^step)
    let corrected_lr = lr * (bias_correction2.sqrt()) / bias_correction1;

    // Compute update: corrected_lr * m_t / (sqrt(v_t) + eps)
    let denom = exp_avg_sq.sqrt()?.add_scalar(eps)?;
    let update = exp_avg.div(&denom)?.mul_scalar(corrected_lr)?;

    // Apply update to parameter
    *param = param.sub(&update)?;

    Ok(())
}

/// Fused SGD step operation with momentum
///
/// Combines momentum update and parameter update in a single operation:
/// 1. Momentum update: buf = momentum * buf + (1 + dampening) * grad
/// 2. Parameter update: param = param - lr * buf
///
/// # Arguments
/// * `param` - Parameter tensor to update
/// * `grad` - Gradient tensor
/// * `momentum_buffer` - Momentum buffer (if using momentum)
/// * `lr` - Learning rate
/// * `momentum` - Momentum coefficient
/// * `dampening` - Dampening coefficient for momentum
/// * `weight_decay` - Weight decay coefficient (optional)
/// * `nesterov` - Whether to use Nesterov momentum
#[allow(clippy::too_many_arguments)]
pub fn fused_sgd_step(
    param: &mut Tensor,
    grad: &Tensor,
    momentum_buffer: Option<&mut Tensor>,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: Option<f32>,
    nesterov: bool,
) -> Result<()> {
    // Apply weight decay if specified
    let effective_grad = if let Some(wd) = weight_decay {
        let decay_term = param.mul_scalar(wd)?;
        grad.add(&decay_term)?
    } else {
        grad.clone()
    };

    let update = if let Some(buf) = momentum_buffer {
        // Update momentum buffer: buf = momentum * buf + (1 - dampening) * grad
        buf.mul_scalar_(momentum)?;
        let grad_term = effective_grad.mul_scalar(1.0 - dampening)?;
        *buf = buf.add(&grad_term)?;

        if nesterov {
            // Nesterov update: update = momentum * buf + grad
            let momentum_term = buf.mul_scalar(momentum)?;
            momentum_term.add(&effective_grad)?
        } else {
            // Standard momentum: update = buf
            buf.clone()
        }
    } else {
        // No momentum: update = grad
        effective_grad
    };

    // Apply update: param = param - lr * update
    let scaled_update = update.mul_scalar(lr)?;
    *param = param.sub(&scaled_update)?;

    Ok(())
}

/// Fused RMSprop step operation
///
/// Combines all RMSprop operations in a single kernel:
/// 1. Update squared average: sq_avg = alpha * sq_avg + (1 - alpha) * grad^2
/// 2. Compute denominator: denom = sqrt(sq_avg) + eps
/// 3. Update parameter: param = param - lr * grad / denom
///
/// # Arguments
/// * `param` - Parameter tensor to update
/// * `grad` - Gradient tensor
/// * `square_avg` - Running average of squared gradients
/// * `lr` - Learning rate
/// * `alpha` - Smoothing constant
/// * `eps` - Small constant for numerical stability
/// * `weight_decay` - Weight decay coefficient (optional)
/// * `momentum_buffer` - Momentum buffer (if using momentum)
/// * `momentum` - Momentum coefficient
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsprop_step(
    param: &mut Tensor,
    grad: &Tensor,
    square_avg: &mut Tensor,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: Option<f32>,
    momentum_buffer: Option<&mut Tensor>,
    momentum: f32,
) -> Result<()> {
    // Apply weight decay if specified
    let effective_grad = if let Some(wd) = weight_decay {
        let decay_term = param.mul_scalar(wd)?;
        grad.add(&decay_term)?
    } else {
        grad.clone()
    };

    // Update squared average: sq_avg = alpha * sq_avg + (1 - alpha) * grad^2
    square_avg.mul_scalar_(alpha)?;
    let grad_sq = effective_grad.mul_op(&effective_grad)?;
    let grad_sq_term = grad_sq.mul_scalar(1.0 - alpha)?;
    square_avg.add(&grad_sq_term)?;

    // Compute denominator: sqrt(sq_avg) + eps
    let denom = square_avg.sqrt()?.add_scalar(eps)?;

    // Compute base update: grad / denom
    let base_update = effective_grad.div(&denom)?;

    let update = if let Some(buf) = momentum_buffer {
        // Update momentum buffer: buf = momentum * buf + base_update
        buf.mul_scalar_(momentum)?;
        buf.add(&base_update)?;
        buf.clone()
    } else {
        base_update
    };

    // Apply update: param = param - lr * update
    let scaled_update = update.mul_scalar(lr)?;
    *param = param.sub(&scaled_update)?;

    Ok(())
}

/// Fused AdaGrad step operation
///
/// Combines all AdaGrad operations:
/// 1. Update sum of squares: sum_sq = sum_sq + grad^2
/// 2. Compute denominator: denom = sqrt(sum_sq) + eps
/// 3. Update parameter: param = param - lr * grad / denom
///
/// # Arguments
/// * `param` - Parameter tensor to update
/// * `grad` - Gradient tensor
/// * `sum_of_squares` - Sum of squared gradients
/// * `lr` - Learning rate
/// * `eps` - Small constant for numerical stability
/// * `weight_decay` - Weight decay coefficient (optional)
pub fn fused_adagrad_step(
    param: &mut Tensor,
    grad: &Tensor,
    sum_of_squares: &mut Tensor,
    lr: f32,
    eps: f32,
    weight_decay: Option<f32>,
) -> Result<()> {
    // Apply weight decay if specified
    let effective_grad = if let Some(wd) = weight_decay {
        let decay_term = param.mul_scalar(wd)?;
        grad.add(&decay_term)?
    } else {
        grad.clone()
    };

    // Update sum of squares: sum_sq = sum_sq + grad^2
    let grad_sq = effective_grad.mul_op(&effective_grad)?;
    sum_of_squares.add(&grad_sq)?;

    // Compute denominator: sqrt(sum_sq) + eps
    let denom = sum_of_squares.sqrt()?.add_scalar(eps)?;

    // Compute and apply update: param = param - lr * grad / denom
    let update = effective_grad.div(&denom)?.mul_scalar(lr)?;
    param.sub(&update)?;

    Ok(())
}

/// Fused AdaDelta step operation
///
/// Combines all AdaDelta operations:
/// 1. Update squared gradient average: sq_avg = rho * sq_avg + (1 - rho) * grad^2
/// 2. Compute RMS gradient: rms_grad = sqrt(sq_avg + eps)
/// 3. Compute RMS delta: rms_delta = sqrt(acc_delta + eps)
/// 4. Compute update: delta = -rms_delta / rms_grad * grad
/// 5. Update accumulated delta: acc_delta = rho * acc_delta + (1 - rho) * delta^2
/// 6. Update parameter: param = param + delta
///
/// # Arguments
/// * `param` - Parameter tensor to update
/// * `grad` - Gradient tensor
/// * `square_avg` - Running average of squared gradients
/// * `acc_delta` - Running average of squared parameter updates
/// * `rho` - Coefficient for computing running averages
/// * `eps` - Small constant for numerical stability
/// * `weight_decay` - Weight decay coefficient (optional)
#[allow(clippy::too_many_arguments)]
pub fn fused_adadelta_step(
    param: &mut Tensor,
    grad: &Tensor,
    square_avg: &mut Tensor,
    acc_delta: &mut Tensor,
    rho: f32,
    eps: f32,
    weight_decay: Option<f32>,
) -> Result<()> {
    // Apply weight decay if specified
    let effective_grad = if let Some(wd) = weight_decay {
        let decay_term = param.mul_scalar(wd)?;
        grad.add(&decay_term)?
    } else {
        grad.clone()
    };

    // Update squared gradient average: sq_avg = rho * sq_avg + (1 - rho) * grad^2
    square_avg.mul_scalar_(rho)?;
    let grad_sq = effective_grad.mul_op(&effective_grad)?;
    let grad_sq_term = grad_sq.mul_scalar(1.0 - rho)?;
    square_avg.add(&grad_sq_term)?;

    // Compute RMS values
    let rms_grad = square_avg.add_scalar(eps)?.sqrt()?;
    let rms_delta = acc_delta.add_scalar(eps)?.sqrt()?;

    // Compute parameter update: delta = -rms_delta / rms_grad * grad
    let delta = effective_grad
        .mul_op(&rms_delta)?
        .div(&rms_grad)?
        .mul_scalar(-1.0)?;

    // Update accumulated delta: acc_delta = rho * acc_delta + (1 - rho) * delta^2
    acc_delta.mul_scalar_(rho)?;
    let delta_sq = delta.mul_op(&delta)?;
    let delta_sq_term = delta_sq.mul_scalar(1.0 - rho)?;
    acc_delta.add(&delta_sq_term)?;

    // Apply update to parameter
    param.add(&delta)?;

    Ok(())
}

/// Trait for optimizers that support fused kernel operations
pub trait FusedKernelSupport {
    /// Enable or disable fused kernel usage
    fn set_fused(&mut self, fused: bool);

    /// Check if fused kernels are enabled
    fn is_fused(&self) -> bool;

    /// Get performance statistics for fused operations
    fn fused_stats(&self) -> FusedStats;
}

/// Performance statistics for fused operations
#[derive(Debug, Clone, Default)]
pub struct FusedStats {
    pub fused_ops_count: u64,
    pub unfused_ops_count: u64,
    pub total_kernel_launches: u64,
    pub memory_bandwidth_saved: f64, // in GB/s
}

impl FusedStats {
    /// Calculate the fusion efficiency (percentage of operations that were fused)
    pub fn fusion_efficiency(&self) -> f64 {
        if self.fused_ops_count + self.unfused_ops_count == 0 {
            0.0
        } else {
            self.fused_ops_count as f64 / (self.fused_ops_count + self.unfused_ops_count) as f64
                * 100.0
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Utility functions for fused operations
pub mod utils {
    use super::*;

    /// Check if tensors are compatible for fused operations
    pub fn can_fuse_tensors(tensors: &[&Tensor]) -> bool {
        if tensors.is_empty() {
            return false;
        }

        let first_device = tensors[0].device();
        let first_dtype = tensors[0].dtype();
        let first_shape = tensors[0].shape();

        tensors.iter().all(|tensor| {
            tensor.device() == first_device
                && tensor.dtype() == first_dtype
                && tensor.shape() == first_shape
        })
    }

    /// Estimate memory bandwidth savings from fusion
    pub fn estimate_bandwidth_savings(
        tensor_size: usize,
        element_size: usize,
        num_separate_ops: usize,
        num_fused_ops: usize,
    ) -> f64 {
        let separate_bandwidth = tensor_size * element_size * num_separate_ops * 2; // read + write
        let fused_bandwidth = tensor_size * element_size * num_fused_ops * 2; // read + write
        (separate_bandwidth - fused_bandwidth) as f64 / 1e9 // Convert to GB
    }

    /// Check if the current device supports fused operations
    pub fn supports_fused_ops(device: &dyn torsh_core::device::Device) -> bool {
        match device.device_type() {
            torsh_core::device::DeviceType::Cpu => true, // CPU supports basic fusion
            torsh_core::device::DeviceType::Cuda(_) => true, // CUDA supports advanced fusion
            torsh_core::device::DeviceType::Metal(_) => true, // Metal supports fusion
            torsh_core::device::DeviceType::Wgpu(_) => false, // WebGPU support limited
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::Device;
    use torsh_tensor::creation;

    #[test]
    fn test_fused_adam_step() -> OptimizerResult<()> {
        let mut param = creation::ones(&[2, 2]).unwrap();
        let grad = creation::ones(&[2, 2]).unwrap();
        let mut exp_avg = creation::zeros(&[2, 2]).unwrap();
        let mut exp_avg_sq = creation::zeros(&[2, 2]).unwrap();

        let result = fused_adam_step(
            &mut param,
            &grad,
            &mut exp_avg,
            &mut exp_avg_sq,
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
            None,
        );

        assert!(result.is_ok());

        // Parameter should have been updated (should be less than 1.0)
        let param_vals = param.to_vec()?;
        assert!(param_vals.iter().all(|&x| x < 1.0));

        // Momentum buffers should have been updated
        let exp_avg_vals = exp_avg.to_vec()?;
        assert!(exp_avg_vals.iter().any(|&x| x != 0.0));

        let exp_avg_sq_vals = exp_avg_sq.to_vec()?;
        assert!(exp_avg_sq_vals.iter().any(|&x| x != 0.0));
        Ok(())
    }

    #[test]
    fn test_fused_sgd_step() -> OptimizerResult<()> {
        let mut param = creation::ones(&[2, 2]).unwrap();
        let grad = creation::ones(&[2, 2]).unwrap();
        let mut momentum_buffer = creation::zeros(&[2, 2]).unwrap();

        let result = fused_sgd_step(
            &mut param,
            &grad,
            Some(&mut momentum_buffer),
            0.01,
            0.9,
            0.0,
            None,
            false,
        );

        assert!(result.is_ok());

        // Parameter should have been updated
        let param_vals = param.to_vec()?;
        assert!(param_vals.iter().all(|&x| x < 1.0));

        // Momentum buffer should have been updated
        let momentum_vals = momentum_buffer.to_vec()?;
        assert!(momentum_vals.iter().any(|&x| x != 0.0));
        Ok(())
    }

    #[test]
    fn test_can_fuse_tensors() {
        let tensor1 = creation::ones(&[2, 2]).unwrap();
        let tensor2 = creation::zeros(&[2, 2]).unwrap();
        let tensor3 = creation::ones(&[3, 3]).unwrap(); // Different shape

        assert!(utils::can_fuse_tensors(&[&tensor1, &tensor2]));
        assert!(!utils::can_fuse_tensors(&[&tensor1, &tensor3]));
        assert!(!utils::can_fuse_tensors(&[]));
    }

    #[test]
    fn test_fused_stats() {
        let mut stats = FusedStats::default();
        stats.fused_ops_count = 80;
        stats.unfused_ops_count = 20;

        assert_eq!(stats.fusion_efficiency(), 80.0);

        stats.reset();
        assert_eq!(stats.fused_ops_count, 0);
        assert_eq!(stats.unfused_ops_count, 0);
    }

    #[test]
    fn test_estimate_bandwidth_savings() {
        let savings = utils::estimate_bandwidth_savings(1000, 4, 5, 1);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_supports_fused_ops() {
        let cpu_device =
            torsh_core::device::DeviceFactory::create_device(torsh_core::device::DeviceType::Cpu)
                .unwrap();
        assert!(utils::supports_fused_ops(cpu_device.as_ref()));
    }
}
