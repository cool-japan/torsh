//! Numerical stability tests for optimizers
//!
//! This module provides comprehensive tests to ensure optimizers maintain
//! numerical stability under various conditions including:
//! - Extreme gradients (very large/small)
//! - Ill-conditioned optimization landscapes
//! - Different precision levels
//! - Pathological cases

use crate::{
    adagrad::AdaGrad, adam::Adam, rmsprop::RMSprop, sgd::SGD, Optimizer, OptimizerError,
    OptimizerResult,
};
use parking_lot::RwLock;
use std::ops::{Add, Mul, Sub};
use std::sync::Arc;
use torsh_core::{
    device::{CpuDevice, Device, DeviceType},
    DType,
};
use torsh_tensor::{
    creation::{eye, randn, tensor_scalar, zeros},
    Tensor,
};

/// Test configuration for numerical stability
#[derive(Debug, Clone)]
pub struct StabilityTestConfig {
    /// Number of optimization steps to run
    pub num_steps: usize,
    /// Tolerance for checking stability
    pub tolerance: f32,
    /// Maximum allowed parameter magnitude
    pub max_param_magnitude: f32,
    /// Minimum required progress (to avoid stagnation)
    pub min_progress: f32,
    /// Device to run tests on
    pub device: Arc<CpuDevice>,
}

impl Default for StabilityTestConfig {
    fn default() -> Self {
        Self {
            num_steps: 100,
            tolerance: 1e-6,
            max_param_magnitude: 1e10,
            min_progress: 1e-8,
            device: Arc::new(CpuDevice::new()),
        }
    }
}

/// Result of a numerical stability test
#[derive(Debug)]
pub struct StabilityTestResult {
    /// Whether the test passed
    pub passed: bool,
    /// Final loss value
    pub final_loss: f32,
    /// Maximum parameter magnitude encountered
    pub max_param_magnitude: f32,
    /// Number of NaN/infinite values encountered
    pub nan_count: usize,
    /// Detailed error message if test failed
    pub error_message: Option<String>,
}

/// Numerical stability test suite
pub struct NumericalStabilityTests {
    config: StabilityTestConfig,
}

impl NumericalStabilityTests {
    /// Create a new test suite with default configuration
    pub fn new() -> Self {
        Self {
            config: StabilityTestConfig::default(),
        }
    }

    /// Create a new test suite with custom configuration
    pub fn with_config(config: StabilityTestConfig) -> Self {
        Self { config }
    }

    /// Test optimizer with extreme gradients
    pub fn test_extreme_gradients<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<StabilityTestResult> {
        // Create parameters with reasonable initial values
        let mut params = randn::<f32>(&[10, 10])?;
        let mut max_param_magnitude = 0.0f32;
        let mut nan_count = 0;

        for step in 0..self.config.num_steps {
            // Create extreme gradients that increase with steps
            let grad_scale = 10.0f32.powi(step as i32 / 20); // Exponentially increasing
            let grads = randn::<f32>(&[10, 10])?.mul_scalar(grad_scale)?;

            // Check for NaN/infinite gradients
            let grad_data = grads.to_vec()?;
            let has_nan_or_inf = grad_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                nan_count += 1;
                continue;
            }

            // Apply gradients
            params.set_grad(Some(grads));
            optimizer.step()?;

            // Check parameter stability
            let param_norm = params.norm()?.to_vec()?[0];
            max_param_magnitude = max_param_magnitude.max(param_norm);

            // Check for NaN/infinite parameters
            let param_data = params.to_vec()?;
            let has_nan_or_inf = param_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: f32::NAN,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!("Parameters became NaN/infinite at step {}", step)),
                });
            }

            // Check for parameter explosion
            if param_norm > self.config.max_param_magnitude {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: param_norm,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!(
                        "Parameters exploded to magnitude {} at step {}",
                        param_norm, step
                    )),
                });
            }
        }

        Ok(StabilityTestResult {
            passed: true,
            final_loss: params.norm()?.item()?,
            max_param_magnitude,
            nan_count,
            error_message: None,
        })
    }

    /// Test optimizer with ill-conditioned quadratic function
    pub fn test_ill_conditioned_quadratic<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<StabilityTestResult> {
        // Create an ill-conditioned quadratic: f(x) = 0.5 * x^T * A * x where A has poor condition number
        let device = self.config.device.clone();
        let dim = 10;

        // Create a matrix with poor condition number
        let mut hessian_data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            let eigenval = if i == 0 { 1000.0 } else { 0.001 };
            hessian_data[i * dim + i] = eigenval;
        }
        let hessian = Tensor::from_data(hessian_data, vec![dim, dim], DeviceType::Cpu)?;

        let mut params = randn::<f32>(&[dim])?;
        let mut initial_loss = f32::INFINITY;
        let mut max_param_magnitude = 0.0f32;
        let mut nan_count = 0;

        for step in 0..self.config.num_steps {
            // Compute gradients: grad = A * x
            let grads = hessian.matmul(&params.unsqueeze(1)?)?.squeeze(1)?;

            // Check for NaN/infinite gradients
            let grad_data = grads.to_vec()?;
            let has_nan_or_inf = grad_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                nan_count += 1;
                continue;
            }

            // Compute loss: 0.5 * x^T * A * x
            let loss = params
                .unsqueeze(0)?
                .matmul(&grads.unsqueeze(1)?)?
                .squeeze_all()?
                .mul_scalar(0.5)?
                .to_vec()?[0];

            if step == 0 {
                initial_loss = loss;
            }

            // Apply gradients
            params.set_grad(Some(grads));
            optimizer.step()?;

            // Check parameter stability
            let param_norm = params.norm()?.to_vec()?[0];
            max_param_magnitude = max_param_magnitude.max(param_norm);

            // Check for NaN/infinite parameters
            let param_data = params.to_vec()?;
            let has_nan_or_inf = param_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: f32::NAN,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!("Parameters became NaN/infinite at step {}", step)),
                });
            }

            // Check for parameter explosion
            if param_norm > self.config.max_param_magnitude {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: loss,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!(
                        "Parameters exploded to magnitude {} at step {}",
                        param_norm, step
                    )),
                });
            }
        }

        let final_loss = {
            let grads = hessian.matmul(&params.unsqueeze(1)?)?.squeeze(1)?;
            params
                .unsqueeze(0)?
                .matmul(&grads.unsqueeze(1)?)?
                .squeeze_all()?
                .mul_scalar(0.5)?
                .to_vec()?[0]
        };

        // Check if we made sufficient progress
        let progress = (initial_loss - final_loss) / initial_loss.max(1e-8);
        if progress < self.config.min_progress {
            return Ok(StabilityTestResult {
                passed: false,
                final_loss,
                max_param_magnitude,
                nan_count,
                error_message: Some(format!("Insufficient progress: {:.2e}", progress)),
            });
        }

        Ok(StabilityTestResult {
            passed: true,
            final_loss,
            max_param_magnitude,
            nan_count,
            error_message: None,
        })
    }

    /// Test optimizer with noisy gradients
    pub fn test_noisy_gradients<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<StabilityTestResult> {
        let device = self.config.device.clone();
        let mut params = randn::<f32>(&[50])?;
        let target = zeros(&[50])?;

        let mut max_param_magnitude = 0.0f32;
        let mut nan_count = 0;
        let mut initial_loss = f32::INFINITY;

        for step in 0..self.config.num_steps {
            // Compute clean gradients (towards target)
            let clean_grads = params.sub(&target)?;

            // Add high-frequency noise
            let noise_scale = 0.1; // 10% noise
            let noise = randn::<f32>(&[50])?.mul_scalar(noise_scale)?;
            let noisy_grads = clean_grads.add(&noise)?;

            // Check for NaN/infinite gradients
            let noisy_grad_data = noisy_grads.to_vec()?;
            let has_nan_or_inf = noisy_grad_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                nan_count += 1;
                continue;
            }

            // Compute loss
            let loss = params.sub(&target)?.pow(2.0)?.mean(None, false)?.to_vec()?[0];

            if step == 0 {
                initial_loss = loss;
            }

            // Apply gradients
            params.set_grad(Some(noisy_grads));
            optimizer.step()?;

            // Check parameter stability
            let param_norm = params.norm()?.to_vec()?[0];
            max_param_magnitude = max_param_magnitude.max(param_norm);

            // Check for NaN/infinite parameters
            let param_data = params.to_vec()?;
            let has_nan_or_inf = param_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: f32::NAN,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!("Parameters became NaN/infinite at step {}", step)),
                });
            }

            // Check for parameter explosion
            if param_norm > self.config.max_param_magnitude {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: loss,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!(
                        "Parameters exploded to magnitude {} at step {}",
                        param_norm, step
                    )),
                });
            }
        }

        let final_loss = params.sub(&target)?.pow(2.0)?.mean(None, false)?.item()?;

        // Check convergence despite noise
        let progress = (initial_loss - final_loss) / initial_loss.max(1e-8);
        if progress < self.config.min_progress {
            return Ok(StabilityTestResult {
                passed: false,
                final_loss,
                max_param_magnitude,
                nan_count,
                error_message: Some(format!(
                    "Insufficient progress with noisy gradients: {:.2e}",
                    progress
                )),
            });
        }

        Ok(StabilityTestResult {
            passed: true,
            final_loss,
            max_param_magnitude,
            nan_count,
            error_message: None,
        })
    }

    /// Test optimizer with sparse gradients
    pub fn test_sparse_gradients<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<StabilityTestResult> {
        let device = self.config.device.clone();
        let mut params = randn::<f32>(&[100])?;

        let mut max_param_magnitude = 0.0f32;
        let mut nan_count = 0;

        for step in 0..self.config.num_steps {
            // Create sparse gradients (only update 10% of parameters each step)
            let mut grads_data = vec![0.0f32; 100];

            // Set gradients for a random subset of parameters
            let sparsity = 0.1; // 10% non-zero gradients
            for i in 0..100 {
                // Use deterministic pattern for testing (can be replaced with proper RNG)
                if (i * 17 + step) % 10 == 0 {
                    // Deterministic "random" pattern
                    let grad_val = ((i as f32 * 0.1) % 2.0) - 1.0; // Deterministic gradient in [-1, 1]
                    grads_data[i] = grad_val;
                }
            }
            let grads = Tensor::from_data(grads_data, vec![100], DeviceType::Cpu)?;

            // Check for NaN/infinite gradients
            let grad_data = grads.to_vec()?;
            let has_nan_or_inf = grad_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                nan_count += 1;
                continue;
            }

            // Apply gradients
            params.set_grad(Some(grads));
            optimizer.step()?;

            // Check parameter stability
            let param_norm = params.norm()?.to_vec()?[0];
            max_param_magnitude = max_param_magnitude.max(param_norm);

            // Check for NaN/infinite parameters
            let param_data = params.to_vec()?;
            let has_nan_or_inf = param_data
                .iter()
                .any(|&x: &f32| x.is_nan() || x.is_infinite());
            if has_nan_or_inf {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: f32::NAN,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!("Parameters became NaN/infinite at step {}", step)),
                });
            }

            // Check for parameter explosion
            if param_norm > self.config.max_param_magnitude {
                return Ok(StabilityTestResult {
                    passed: false,
                    final_loss: param_norm,
                    max_param_magnitude,
                    nan_count,
                    error_message: Some(format!(
                        "Parameters exploded to magnitude {} at step {}",
                        param_norm, step
                    )),
                });
            }
        }

        Ok(StabilityTestResult {
            passed: true,
            final_loss: params.norm()?.item()?,
            max_param_magnitude,
            nan_count,
            error_message: None,
        })
    }

    /// Run all stability tests for a given optimizer
    /// Note: This consumes the optimizer since optimizers don't implement Clone
    pub fn run_single_test<O: Optimizer>(
        &self,
        optimizer: O,
        test_name: &str,
    ) -> OptimizerResult<StabilityTestResult> {
        match test_name {
            "extreme_gradients" => self.test_extreme_gradients(optimizer),
            "ill_conditioned_quadratic" => self.test_ill_conditioned_quadratic(optimizer),
            "noisy_gradients" => self.test_noisy_gradients(optimizer),
            "sparse_gradients" => self.test_sparse_gradients(optimizer),
            _ => Err(OptimizerError::InvalidParameter(format!(
                "Unknown test: {}",
                test_name
            ))),
        }
    }
}

/// Comprehensive test suite for common optimizers
pub fn run_comprehensive_stability_tests() -> OptimizerResult<()> {
    let test_suite = NumericalStabilityTests::new();

    // Test Adam optimizer with extreme gradients
    let adam_params = randn::<f32>(&[10, 10])?;
    let adam = Adam::new(
        vec![Arc::new(RwLock::new(adam_params))],
        Some(0.001),
        None,
        None,
        None,
        false,
    );

    println!("Testing Adam optimizer stability with extreme gradients...");
    let adam_result = test_suite.run_single_test(adam, "extreme_gradients")?;
    println!(
        "  extreme_gradients: {}",
        if adam_result.passed { "PASS" } else { "FAIL" }
    );
    if let Some(error) = adam_result.error_message {
        println!("    Error: {}", error);
    }

    // Test SGD optimizer with noisy gradients
    let sgd_params = randn::<f32>(&[10, 10])?;
    let sgd = SGD::new(
        vec![Arc::new(RwLock::new(sgd_params))],
        0.01,
        None,
        None,
        None,
        false,
    );

    println!("\nTesting SGD optimizer stability with noisy gradients...");
    let sgd_result = test_suite.run_single_test(sgd, "noisy_gradients")?;
    println!(
        "  noisy_gradients: {}",
        if sgd_result.passed { "PASS" } else { "FAIL" }
    );
    if let Some(error) = sgd_result.error_message {
        println!("    Error: {}", error);
    }

    // Test RMSprop optimizer with sparse gradients
    let rmsprop_params = randn::<f32>(&[10, 10])?;
    let rmsprop = RMSprop::new(
        vec![Arc::new(RwLock::new(rmsprop_params))],
        Some(0.01),
        None,
        None,
        None,
        None,
        false,
    );

    println!("\nTesting RMSprop optimizer stability with sparse gradients...");
    let rmsprop_result = test_suite.run_single_test(rmsprop, "sparse_gradients")?;
    println!(
        "  sparse_gradients: {}",
        if rmsprop_result.passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    if let Some(error) = rmsprop_result.error_message {
        println!("    Error: {}", error);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_test_config() {
        let config = StabilityTestConfig::default();
        assert_eq!(config.num_steps, 100);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.max_param_magnitude, 1e10);
        assert_eq!(config.min_progress, 1e-8);
    }

    #[test]
    fn test_stability_test_result() {
        let result = StabilityTestResult {
            passed: true,
            final_loss: 0.5,
            max_param_magnitude: 10.0,
            nan_count: 0,
            error_message: None,
        };

        assert!(result.passed);
        assert_eq!(result.final_loss, 0.5);
        assert_eq!(result.max_param_magnitude, 10.0);
        assert_eq!(result.nan_count, 0);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_numerical_stability_tests_creation() {
        let test_suite = NumericalStabilityTests::new();
        assert_eq!(test_suite.config.num_steps, 100);

        let custom_config = StabilityTestConfig {
            num_steps: 50,
            tolerance: 1e-5,
            max_param_magnitude: 1e8,
            min_progress: 1e-7,
            device: Arc::new(CpuDevice::new()),
        };

        let custom_test_suite = NumericalStabilityTests::with_config(custom_config);
        assert_eq!(custom_test_suite.config.num_steps, 50);
        assert_eq!(custom_test_suite.config.tolerance, 1e-5);
    }
}
