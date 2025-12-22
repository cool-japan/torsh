//! Example demonstrating optimized parameter updates in torsh-nn
//!
//! This example shows how to use the parameter update optimizations
//! for better training performance.

use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::layers::Linear;
use torsh_nn::parameter_updates::{LayerSpecificOptimizers, ParameterUpdater, UpdateConfig};
use torsh_nn::{Module, Parameter};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

fn main() -> Result<()> {
    println!("=== ToRSh Parameter Update Optimization Example ===\n");

    // Example 1: Basic parameter update with SGD
    basic_sgd_example()?;

    // Example 2: Adam optimizer with momentum
    adam_optimizer_example()?;

    // Example 3: Gradient clipping
    gradient_clipping_example()?;

    // Example 4: Batch updates for memory efficiency
    batch_update_example()?;

    // Example 5: Layer-specific optimizations
    layer_specific_optimization_example()?;

    // Example 6: Performance comparison
    performance_comparison_example()?;

    Ok(())
}

fn basic_sgd_example() -> Result<()> {
    println!("1. Basic SGD Parameter Updates");
    println!("==============================");

    // Create a simple linear layer
    let linear = Linear::new(784, 128, true);
    let parameters = linear.parameters();

    // Simulate gradients
    let mut gradients = HashMap::new();
    for (name, param) in &parameters {
        let grad = randn(param.tensor().read().shape().dims())?;
        gradients.insert(name.clone(), grad);
    }

    // Create parameter updater
    let mut updater = ParameterUpdater::new();

    println!("Parameters before update:");
    for (name, param) in &parameters {
        let tensor = param.tensor();
        let tensor_guard = tensor.read();
        println!("  {}: shape {:?}", name, tensor_guard.shape().dims());
    }

    // Apply SGD update
    let learning_rate = 0.01;
    updater.sgd_update(&parameters, &gradients, learning_rate)?;

    println!("✓ SGD update completed");
    println!("Learning rate: {}", learning_rate);

    // Show statistics
    let stats = updater.get_statistics();
    println!("Update statistics:");
    println!("  Total updates: {}", stats.total_updates);
    println!("  Average time: {:?}", stats.average_update_time());

    println!();
    Ok(())
}

fn adam_optimizer_example() -> Result<()> {
    println!("2. Adam Optimizer with Momentum");
    println!("===============================");

    // Create parameters
    let linear = Linear::new(256, 64, true);
    let parameters = linear.parameters();

    // Initialize Adam state (momentum estimates)
    let mut m_t = HashMap::new();
    let mut v_t = HashMap::new();

    for (name, param) in &parameters {
        let tensor = param.tensor();
        let tensor_guard = tensor.read();
        let shape_result = tensor_guard.shape();
        let shape = shape_result.dims();
        m_t.insert(name.clone(), zeros(shape)?);
        v_t.insert(name.clone(), zeros(shape)?);
    }

    // Simulate gradients
    let mut gradients = HashMap::new();
    for (name, param) in &parameters {
        let grad = randn(param.tensor().read().shape().dims())?;
        gradients.insert(name.clone(), grad);
    }

    // Create parameter updater with custom configuration
    let config = UpdateConfig {
        use_vectorization: true,
        use_inplace_updates: true,
        use_operation_fusion: true,
        memory_budget: 512 * 1024 * 1024, // 512MB
        use_async_updates: false,
    };

    let mut updater = ParameterUpdater::with_config(config);

    // Adam hyperparameters
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let step = 1;

    println!("Adam hyperparameters:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Beta1: {}", beta1);
    println!("  Beta2: {}", beta2);
    println!("  Epsilon: {}", epsilon);

    // Apply Adam update
    updater.adam_update(
        &parameters,
        &gradients,
        &mut m_t,
        &mut v_t,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        step,
    )?;

    println!("✓ Adam update completed");

    let stats = updater.get_statistics();
    println!("Performance: {:.2} updates/sec", stats.updates_per_second());

    println!();
    Ok(())
}

fn gradient_clipping_example() -> Result<()> {
    println!("3. Gradient Clipping");
    println!("====================");

    let updater = ParameterUpdater::new();

    // Create gradients with large values
    let mut gradients: HashMap<String, Tensor> = HashMap::new();
    let weight_grad =
        randn::<f32>(&[100, 100])?.mul(&torsh_tensor::creation::tensor_scalar(10.0f32)?)?;
    gradients.insert("weight".to_string(), weight_grad);
    let bias_grad = randn::<f32>(&[100])?.mul(&torsh_tensor::creation::tensor_scalar(5.0f32)?)?;
    gradients.insert("bias".to_string(), bias_grad);

    println!("Before clipping:");
    for (name, grad) in &gradients {
        let norm = grad.mul_op(grad)?.sum()?.item()?.sqrt();
        println!("  {} norm: {:.4}", name, norm);
    }

    // Apply gradient clipping
    let max_norm = 1.0;
    let original_norm = updater.clip_gradients(&mut gradients, max_norm)?;

    println!("After clipping (max_norm = {}):", max_norm);
    println!("  Original total norm: {:.4}", original_norm);

    for (name, grad) in &gradients {
        let norm = grad.mul_op(grad)?.sum()?.item()?.sqrt();
        println!("  {} norm: {:.4}", name, norm);
    }

    // Calculate final total norm
    let mut total_norm_squared = 0.0f32;
    for grad in gradients.values() {
        let grad_norm_squared = grad.mul_op(grad)?.sum()?.item()?;
        total_norm_squared += grad_norm_squared;
    }
    let final_total_norm = total_norm_squared.sqrt();

    println!("  Final total norm: {:.4}", final_total_norm);
    println!("✓ Gradient clipping completed");

    println!();
    Ok(())
}

fn batch_update_example() -> Result<()> {
    println!("4. Batch Updates for Memory Efficiency");
    println!("======================================");

    // Create multiple parameter groups (simulating multiple layers)
    let mut parameter_groups = Vec::new();
    let mut gradient_groups = Vec::new();

    for i in 0..5 {
        let linear = Linear::new(128, 64, true);
        let parameters = linear.parameters();

        let mut gradients = HashMap::new();
        for (name, param) in &parameters {
            let grad = randn(param.tensor().read().shape().dims())?;
            gradients.insert(name.clone(), grad);
        }

        parameter_groups.push(parameters);
        gradient_groups.push(gradients);

        println!("Created layer group {}", i + 1);
    }

    let mut updater = ParameterUpdater::new();

    // Calculate total memory usage
    let total_memory: usize = parameter_groups
        .iter()
        .map(|params| torsh_nn::parameter_updates::utils::calculate_memory_usage(params))
        .sum();

    println!("Total memory usage: {} bytes", total_memory);

    // Define update function (SGD in this case)
    let learning_rate = 0.01;
    let update_fn = |parameters: &HashMap<String, Parameter>,
                     gradients: &HashMap<String, Tensor>| {
        for (name, param) in parameters {
            if let Some(grad) = gradients.get(name) {
                let update = grad.mul_op(&torsh_tensor::creation::tensor_scalar(learning_rate)?)?;
                let tensor = param.tensor();
                let param_tensor = tensor.write();
                param_tensor.sub(&update)?;
            }
        }
        Ok(())
    };

    // Apply batch update
    updater.batch_update(&parameter_groups, &gradient_groups, update_fn)?;

    println!(
        "✓ Batch update completed for {} layer groups",
        parameter_groups.len()
    );

    let stats = updater.get_statistics();
    println!(
        "Processed {} updates in {:?}",
        stats.total_updates, stats.total_time
    );

    println!();
    Ok(())
}

fn layer_specific_optimization_example() -> Result<()> {
    println!("5. Layer-Specific Optimizations");
    println!("===============================");

    // Linear layer optimization
    let linear = Linear::new(512, 256, true);
    let linear_params = linear.parameters();

    let weight = linear_params.get("weight").unwrap();
    let bias = linear_params.get("bias");

    let weight_grad = randn(&[256, 512])?;
    let bias_grad = Some(randn(&[256])?);

    println!("Optimizing linear layer:");
    println!(
        "  Weight shape: {:?}",
        weight.tensor().read().shape().dims()
    );
    if let Some(bias) = bias {
        println!("  Bias shape: {:?}", bias.tensor().read().shape().dims());
    }

    let learning_rate = 0.001;
    LayerSpecificOptimizers::update_linear_layer(
        weight,
        bias,
        &weight_grad,
        bias_grad.as_ref(),
        learning_rate,
    )?;

    println!("✓ Linear layer optimization completed");

    // Simulate normalization layer optimization
    let norm_weight = Parameter::new(ones(&[128])?);
    let norm_bias = Parameter::new(zeros(&[128])?);
    let norm_weight_grad = randn(&[128])?;
    let norm_bias_grad = randn(&[128])?;

    println!("Optimizing normalization layer:");
    println!(
        "  Scale (weight) shape: {:?}",
        norm_weight.tensor().read().shape().dims()
    );
    println!(
        "  Shift (bias) shape: {:?}",
        norm_bias.tensor().read().shape().dims()
    );

    LayerSpecificOptimizers::update_norm_layer(
        &norm_weight,
        &norm_bias,
        &norm_weight_grad,
        &norm_bias_grad,
        learning_rate,
    )?;

    println!("✓ Normalization layer optimization completed");

    println!();
    Ok(())
}

fn performance_comparison_example() -> Result<()> {
    println!("6. Performance Comparison");
    println!("========================");

    let linear = Linear::new(1024, 512, true);
    let parameters = linear.parameters();

    let mut gradients = HashMap::new();
    for (name, param) in &parameters {
        let grad = randn(param.tensor().read().shape().dims())?;
        gradients.insert(name.clone(), grad);
    }

    // Test different configurations
    let configs = vec![
        (
            "Standard",
            UpdateConfig {
                use_vectorization: false,
                use_inplace_updates: false,
                use_operation_fusion: false,
                memory_budget: 1024 * 1024 * 1024,
                use_async_updates: false,
            },
        ),
        (
            "Optimized",
            UpdateConfig {
                use_vectorization: true,
                use_inplace_updates: true,
                use_operation_fusion: true,
                memory_budget: 1024 * 1024 * 1024,
                use_async_updates: false,
            },
        ),
    ];

    for (name, config) in configs {
        let mut updater = ParameterUpdater::with_config(config);

        // Run multiple updates to get accurate timing
        let iterations = 10;
        let start_time = std::time::Instant::now();

        for _ in 0..iterations {
            updater.sgd_update(&parameters, &gradients, 0.01)?;
        }

        let elapsed = start_time.elapsed();
        let stats = updater.get_statistics();

        println!("{} configuration:", name);
        println!("  Total time for {} iterations: {:?}", iterations, elapsed);
        println!("  Average time per update: {:?}", elapsed / iterations);
        println!("  Updates per second: {:.2}", stats.updates_per_second());
    }

    println!("✓ Performance comparison completed");

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_update_integration() -> Result<()> {
        let linear = Linear::new(10, 5, true);
        let parameters = linear.parameters();

        let mut gradients = HashMap::new();
        for (name, param) in &parameters {
            let grad = randn(param.tensor().read().shape().dims())?;
            gradients.insert(name.clone(), grad);
        }

        let mut updater = ParameterUpdater::new();

        // Should not panic
        updater.sgd_update(&parameters, &gradients, 0.01)?;

        let stats = updater.get_statistics();
        assert_eq!(stats.total_updates, 1);

        Ok(())
    }

    #[test]
    fn test_gradient_clipping_integration() -> Result<()> {
        let mut updater = ParameterUpdater::new();
        let mut gradients = HashMap::new();

        let large_grad = randn(&[5, 5])?.mul_(&torsh_tensor::creation::tensor_scalar(100.0f32)?)?;
        gradients.insert("test_grad".to_string(), large_grad);

        let original_norm = updater.clip_gradients(&mut gradients, 1.0)?;

        assert!(original_norm > 1.0);

        // Check that gradient was actually clipped
        let grad = gradients.get("test_grad").unwrap();
        let new_norm = grad.mul_(grad)?.sum(None, false)?.item::<f32>()?.sqrt();

        assert!((new_norm - 1.0).abs() < 1e-4);

        Ok(())
    }
}
