// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive Examples for torsh-autograd
//!
//! This module provides practical examples demonstrating how to use
//! the autograd features. These examples serve as both documentation
//! and template code for common use cases.
//!
//! # Example Categories
//!
//! - **Basic Gradient Computation**: Simple gradient examples
//! - **Gradient Mode Management**: Using no_grad and enable_grad
//! - **Custom Functions**: Defining custom differentiable operations
//! - **Higher-Order Gradients**: Computing gradients of gradients
//! - **Gradient Clipping**: Preventing gradient explosion
//! - **Checkpointing**: Memory-efficient training
//! - **Hardware Acceleration**: Using GPU/Metal/WebGPU
//! - **Distributed Training**: Multi-device gradient computation

#![allow(dead_code)] // Examples are for documentation

use crate::error_handling::AutogradResult;
use torsh_core::error::Result;

/// Example 1: Basic Gradient Computation
///
/// This example shows how to compute gradients of a simple function.
///
/// ```rust,ignore
/// use torsh_autograd::examples::basic_gradient_example;
///
/// let (loss, grads) = basic_gradient_example().unwrap();
/// println!("Loss: {:?}", loss);
/// println!("Gradients: {:?}", grads);
/// ```
pub fn basic_gradient_example() -> AutogradResult<(f32, Vec<f32>)> {
    // This is a simplified example structure
    // Real implementation would use actual tensor operations

    tracing::info!("Running basic gradient computation example");

    // Example: Compute gradient of y = x^2 at x = 3
    // dy/dx = 2x = 6
    let x = 3.0f32;
    let y = x * x;
    let grad_y = 2.0 * x;

    Ok((y, vec![grad_y]))
}

/// Example 2: Using no_grad for Inference
///
/// Demonstrates how to disable gradient computation during inference
/// to save memory and improve performance.
///
/// ```rust,ignore
/// use torsh_autograd::examples::inference_example;
///
/// let predictions = inference_example().unwrap();
/// ```
pub fn inference_example() -> Result<Vec<f32>> {
    tracing::info!("Running inference example with no_grad");

    // In real usage:
    // let _guard = crate::grad_mode::no_grad();
    // ... perform inference operations ...

    // Simulate inference results
    Ok(vec![0.1, 0.7, 0.2])
}

/// Example 3: Gradient Accumulation
///
/// Shows how to accumulate gradients across multiple batches,
/// useful for training with large batch sizes on limited memory.
///
/// ```rust,ignore
/// use torsh_autograd::examples::gradient_accumulation_example;
///
/// gradient_accumulation_example().unwrap();
/// ```
pub fn gradient_accumulation_example() -> AutogradResult<()> {
    tracing::info!("Running gradient accumulation example");

    let accumulation_steps = 4;
    let mut accumulated_loss = 0.0f32;

    for step in 0..accumulation_steps {
        // Simulate forward pass
        let loss = 0.5f32 / accumulation_steps as f32;
        accumulated_loss += loss;

        tracing::debug!("Step {}: loss = {}", step, loss);

        // In real usage:
        // loss.backward();
        // Don't step optimizer yet
    }

    tracing::info!("Total accumulated loss: {}", accumulated_loss);
    // In real usage: optimizer.step();

    Ok(())
}

/// Example 4: Custom Differentiable Function
///
/// Demonstrates how to create a custom differentiable operation.
///
/// ```rust,ignore
/// use torsh_autograd::examples::custom_function_example;
///
/// let result = custom_function_example().unwrap();
/// ```
pub fn custom_function_example() -> AutogradResult<f32> {
    tracing::info!("Running custom function example");

    // Example: Custom activation function f(x) = x * sigmoid(x) (Swish)
    fn swish(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    fn swish_backward(x: f32, grad_output: f32) -> f32 {
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let grad = sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x);
        grad_output * grad
    }

    let x = 2.0f32;
    let output = swish(x);
    let _grad = swish_backward(x, 1.0);

    Ok(output)
}

/// Example 5: Gradient Clipping
///
/// Shows how to clip gradients to prevent explosion during training.
///
/// ```rust,ignore
/// use torsh_autograd::examples::gradient_clipping_example;
///
/// gradient_clipping_example().unwrap();
/// ```
pub fn gradient_clipping_example() -> AutogradResult<()> {
    tracing::info!("Running gradient clipping example");

    let max_norm = 1.0f32;
    let gradients = vec![2.0, 3.0, 4.0];

    // Compute current norm
    let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        let clipped: Vec<f32> = gradients.iter().map(|g| g * scale).collect();
        tracing::info!("Clipped gradients: {:?}", clipped);
    }

    Ok(())
}

/// Example 6: Higher-Order Gradients
///
/// Demonstrates computing second-order derivatives (Hessian).
///
/// ```rust,ignore
/// use torsh_autograd::examples::higher_order_gradient_example;
///
/// let (grad, grad_grad) = higher_order_gradient_example().unwrap();
/// ```
pub fn higher_order_gradient_example() -> AutogradResult<(f32, f32)> {
    tracing::info!("Running higher-order gradient example");

    // Example: Compute d²y/dx² for y = x³
    // dy/dx = 3x²
    // d²y/dx² = 6x
    let x = 2.0f32;
    let first_derivative = 3.0 * x * x; // 12.0
    let second_derivative = 6.0 * x; // 12.0

    Ok((first_derivative, second_derivative))
}

/// Example 7: Checkpointing for Memory Efficiency
///
/// Shows how to use gradient checkpointing to trade computation for memory.
///
/// ```rust,ignore
/// use torsh_autograd::examples::checkpointing_example;
///
/// checkpointing_example().unwrap();
/// ```
pub fn checkpointing_example() -> AutogradResult<()> {
    tracing::info!("Running gradient checkpointing example");

    // Simulate a large model with many layers
    let num_layers = 100;
    let checkpoint_frequency = 10;

    for layer_idx in 0..num_layers {
        // Perform forward pass
        let _output = layer_idx as f32 * 0.1;

        if layer_idx % checkpoint_frequency == 0 {
            tracing::debug!("Checkpointing at layer {}", layer_idx);
            // In real usage: checkpoint current activations
        }
    }

    tracing::info!(
        "Forward pass complete with {} checkpoints",
        num_layers / checkpoint_frequency
    );

    Ok(())
}

/// Example 8: Mixed Precision Training
///
/// Demonstrates using mixed precision (FP16/FP32) for faster training.
///
/// ```rust,ignore
/// use torsh_autograd::examples::mixed_precision_example;
///
/// mixed_precision_example().unwrap();
/// ```
pub fn mixed_precision_example() -> AutogradResult<()> {
    tracing::info!("Running mixed precision training example");

    let use_fp16 = true;
    let loss_scale = 1024.0f32; // Dynamic loss scaling

    if use_fp16 {
        tracing::info!("Using FP16 for forward/backward pass");
        // Simulate FP16 computation
        let loss_fp16 = 0.5f32 * loss_scale;

        // Scale gradients back
        let _scaled_grads = loss_fp16 / loss_scale;

        tracing::info!("Loss scaling factor: {}", loss_scale);
    }

    Ok(())
}

/// Example 9: Distributed Data Parallel Training
///
/// Shows how to set up distributed gradient computation.
///
/// ```rust,ignore
/// use torsh_autograd::examples::distributed_training_example;
///
/// distributed_training_example().unwrap();
/// ```
pub fn distributed_training_example() -> AutogradResult<()> {
    tracing::info!("Running distributed training example");

    let world_size = 4; // Number of GPUs/processes
    let rank = 0; // Current process rank

    tracing::info!(
        "Process {}/{} starting distributed training",
        rank,
        world_size
    );

    // Simulate gradient all-reduce across processes
    let local_gradients = vec![1.0f32, 2.0, 3.0];

    // In real usage: all_reduce gradients across all processes
    let averaged_gradients: Vec<f32> = local_gradients
        .iter()
        .map(|g| g / world_size as f32)
        .collect();

    tracing::debug!("Averaged gradients: {:?}", averaged_gradients);

    Ok(())
}

/// Example 10: Hardware Acceleration Selection
///
/// Demonstrates how to select and use hardware accelerators.
///
/// ```rust,ignore
/// use torsh_autograd::examples::hardware_acceleration_example;
///
/// hardware_acceleration_example().unwrap();
/// ```
pub fn hardware_acceleration_example() -> AutogradResult<()> {
    tracing::info!("Running hardware acceleration example");

    // In real usage:
    // use crate::hardware_acceleration::get_global_acceleration_manager;
    // let manager = get_global_acceleration_manager();

    tracing::info!("Checking available accelerators:");
    tracing::info!("  - CUDA: Available on NVIDIA GPUs");
    tracing::info!("  - Metal: Available on Apple Silicon");
    tracing::info!("  - WebGPU: Available in browsers");

    // Simulate accelerator selection
    #[cfg(target_os = "macos")]
    {
        tracing::info!("Selected: Metal (Apple Silicon optimized)");
    }

    #[cfg(target_arch = "wasm32")]
    {
        tracing::info!("Selected: WebGPU (Browser deployment)");
    }

    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        tracing::info!("Selected: CPU fallback");
    }

    Ok(())
}

/// Example 11: Anomaly Detection
///
/// Shows how to enable anomaly detection for debugging NaN/Inf gradients.
///
/// ```rust,ignore
/// use torsh_autograd::examples::anomaly_detection_example;
///
/// anomaly_detection_example().unwrap();
/// ```
pub fn anomaly_detection_example() -> AutogradResult<()> {
    tracing::info!("Running anomaly detection example");

    // Simulate gradient computation with potential issues
    let gradients = vec![1.0f32, 2.0, f32::NAN, 3.0];

    for (idx, &grad) in gradients.iter().enumerate() {
        if !grad.is_finite() {
            tracing::error!("Anomaly detected at index {}: {:?}", idx, grad);
            // In real usage: trigger anomaly handler
        }
    }

    Ok(())
}

/// Example 12: Gradient Filtering (Noise Reduction)
///
/// Demonstrates applying filters to gradients for smoother training.
///
/// ```rust,ignore
/// use torsh_autograd::examples::gradient_filtering_example;
///
/// let filtered = gradient_filtering_example().unwrap();
/// ```
pub fn gradient_filtering_example() -> AutogradResult<Vec<f32>> {
    tracing::info!("Running gradient filtering example");

    let noisy_gradients = vec![1.0f32, 1.5, 0.8, 2.0, 1.2];

    // Apply exponential moving average
    let alpha = 0.9;
    let mut filtered = Vec::new();
    let mut ema = noisy_gradients[0];

    for &grad in &noisy_gradients {
        ema = alpha * ema + (1.0 - alpha) * grad;
        filtered.push(ema);
    }

    tracing::info!("Original: {:?}", noisy_gradients);
    tracing::info!("Filtered: {:?}", filtered);

    Ok(filtered)
}

/// Run all examples
///
/// Execute all example functions to verify they work correctly.
///
/// ```rust,ignore
/// use torsh_autograd::examples::run_all_examples;
///
/// run_all_examples().unwrap();
/// ```
pub fn run_all_examples() -> AutogradResult<()> {
    tracing::info!("=== Running All torsh-autograd Examples ===\n");

    basic_gradient_example()?;
    inference_example().ok();
    gradient_accumulation_example()?;
    custom_function_example()?;
    gradient_clipping_example()?;
    higher_order_gradient_example()?;
    checkpointing_example()?;
    mixed_precision_example()?;
    distributed_training_example()?;
    hardware_acceleration_example()?;
    anomaly_detection_example()?;
    gradient_filtering_example()?;

    tracing::info!("\n=== All Examples Completed Successfully ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_gradient_example() {
        let result = basic_gradient_example();
        assert!(result.is_ok());
        let (loss, grads) = result.unwrap();
        assert!(loss > 0.0);
        assert!(!grads.is_empty());
    }

    #[test]
    fn test_inference_example() {
        let result = inference_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradient_accumulation_example() {
        let result = gradient_accumulation_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_function_example() {
        let result = custom_function_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradient_clipping_example() {
        let result = gradient_clipping_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_higher_order_gradient_example() {
        let result = higher_order_gradient_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_checkpointing_example() {
        let result = checkpointing_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixed_precision_example() {
        let result = mixed_precision_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_distributed_training_example() {
        let result = distributed_training_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_hardware_acceleration_example() {
        let result = hardware_acceleration_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_anomaly_detection_example() {
        let result = anomaly_detection_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradient_filtering_example() {
        let result = gradient_filtering_example();
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn test_run_all_examples() {
        let result = run_all_examples();
        assert!(result.is_ok());
    }
}
