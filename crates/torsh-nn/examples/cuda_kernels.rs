//! Example usage of custom CUDA kernels in torsh-nn
//!
//! This example demonstrates how to use the custom CUDA kernels integration
//! for high-performance neural network operations.

#[cfg(feature = "serialize")]
use serde_json;
use torsh_core::error::Result;
use torsh_nn::cuda_kernels::{
    global_kernel_registry, CudaNeuralOps, CudaOptimizations, CustomActivations,
};
use torsh_tensor::creation::*;

fn main() -> Result<()> {
    println!("=== ToRSh Custom CUDA Kernels Example ===\n");

    // Example 1: Using custom activation functions
    custom_activations_example()?;

    // Example 2: Fused operations for better performance
    fused_operations_example()?;

    // Example 3: Custom kernel registration
    custom_kernel_registration_example()?;

    // Example 4: Kernel benchmarking and optimization
    kernel_optimization_example()?;

    // Example 5: Memory-efficient operations
    memory_efficient_operations_example()?;

    Ok(())
}

fn custom_activations_example() -> Result<()> {
    println!("1. Custom Activation Functions");
    println!("==============================");

    let activations = CustomActivations::new();
    let input = randn(&[2, 1024])?; // Batch of 2, 1024 features

    println!(
        "Available activations: {:?}",
        activations.list_activations()
    );

    // Test different activation functions
    let swish_output = activations.apply("swish", &input)?;
    println!(
        "✓ Swish activation applied - shape: {:?}",
        swish_output.shape().dims()
    );

    let gelu_output = activations.apply("gelu", &input)?;
    println!(
        "✓ GELU activation applied - shape: {:?}",
        gelu_output.shape().dims()
    );

    let mish_output = activations.apply("mish", &input)?;
    println!(
        "✓ Mish activation applied - shape: {:?}",
        mish_output.shape().dims()
    );

    println!();
    Ok(())
}

fn fused_operations_example() -> Result<()> {
    println!("2. Fused Operations for Better Performance");
    println!("==========================================");

    // Simulate a typical CNN layer: Conv + BatchNorm + ReLU
    let input = randn(&[4, 64, 32, 32])?; // 4 batch, 64 channels, 32x32 image
    let weight = randn(&[128, 64, 3, 3])?; // 128 output channels, 3x3 kernel
    let bias = Some(randn(&[128])?);

    // BatchNorm parameters
    let bn_weight = randn(&[128])?;
    let bn_bias = randn(&[128])?;
    let bn_mean = randn(&[128])?;
    let bn_var = randn(&[128])?;
    let eps = 1e-5;

    println!("Input shape: {:?}", input.shape().dims());
    println!("Weight shape: {:?}", weight.shape().dims());

    // Use fused kernel for better performance
    let fused_output = CudaNeuralOps::fused_conv_bn_relu(
        &input,
        &weight,
        bias.as_ref(),
        &bn_weight,
        &bn_bias,
        &bn_mean,
        &bn_var,
        eps,
        (1, 1), // stride
        (1, 1), // padding
    )?;

    println!(
        "✓ Fused Conv+BN+ReLU completed - output shape: {:?}",
        fused_output.shape().dims()
    );
    println!();
    Ok(())
}

fn custom_kernel_registration_example() -> Result<()> {
    println!("3. Custom Kernel Registration");
    println!("=============================");

    let registry = global_kernel_registry();

    // Register a custom element-wise square operation
    registry.register_kernel(
        "element_square".to_string(),
        |inputs: &[&torsh_tensor::Tensor], outputs: &mut [&mut torsh_tensor::Tensor]| {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(torsh_core::error::TorshError::Other(
                    "Square operation requires 1 input and 1 output".to_string(),
                ));
            }

            let input = inputs[0];
            let result = input.mul_op(input)?;
            *outputs[0] = result;
            Ok(())
        },
    )?;

    println!("✓ Custom 'element_square' kernel registered");

    // Test the custom kernel
    let input = randn(&[3, 4])?;
    let mut output = zeros(&[3, 4])?;

    registry.execute_kernel("element_square", &[&input], &mut [&mut output])?;
    println!("✓ Custom kernel executed successfully");
    println!("Available kernels: {:?}", registry.list_kernels());

    println!();
    Ok(())
}

fn kernel_optimization_example() -> Result<()> {
    println!("4. Kernel Benchmarking and Optimization");
    println!("=======================================");

    let input = randn(&[1024, 1024])?;
    let inputs = vec![&input];

    // Auto-tune kernel parameters
    let config = CudaOptimizations::auto_tune_kernel(
        "element_square",
        &[vec![1024, 1024]],
        10, // iterations for tuning
    )?;

    println!("✓ Auto-tuning completed");
    #[cfg(feature = "serialize")]
    println!(
        "Optimal configuration: {:#}",
        serde_json::to_string_pretty(&config)?
    );
    #[cfg(not(feature = "serialize"))]
    println!("Optimal configuration: {:?}", config);

    // Benchmark kernel performance
    let benchmark_result = CudaOptimizations::benchmark_kernel(
        "element_square",
        &inputs,
        100, // benchmark iterations
    )?;

    println!("✓ Benchmark completed");
    println!("Kernel: {}", benchmark_result.kernel_name);
    println!("Iterations: {}", benchmark_result.iterations);
    println!("Average time: {:?}", benchmark_result.avg_time);
    println!("Performance: {:.2} GFLOPS", benchmark_result.gflops);

    // Profile memory usage
    let memory_profile = CudaOptimizations::profile_memory_usage("element_square", &inputs)?;
    println!("✓ Memory profiling completed");
    println!("Input memory: {} bytes", memory_profile.input_memory_bytes);
    println!("Peak memory: {} bytes", memory_profile.peak_memory_bytes);
    println!(
        "Memory efficiency: {:.2}%",
        memory_profile.memory_efficiency * 100.0
    );

    println!();
    Ok(())
}

fn memory_efficient_operations_example() -> Result<()> {
    println!("5. Memory-Efficient Operations");
    println!("==============================");

    // Flash Attention example
    let batch_size = 4;
    let seq_len = 512;
    let head_dim = 64;

    let query = randn(&[batch_size, seq_len, head_dim])?;
    let key = randn(&[batch_size, seq_len, head_dim])?;
    let value = randn(&[batch_size, seq_len, head_dim])?;

    println!("Query shape: {:?}", query.shape().dims());
    println!("Key shape: {:?}", key.shape().dims());
    println!("Value shape: {:?}", value.shape().dims());

    let attention_output = CudaNeuralOps::flash_attention(
        &query,
        &key,
        &value,
        None,                           // no mask
        1.0 / (head_dim as f32).sqrt(), // scale factor
        64,                             // block size
    )?;

    println!(
        "✓ Flash Attention completed - output shape: {:?}",
        attention_output.shape().dims()
    );

    // Optimized matrix multiplication with Tensor Cores
    let a = randn(&[512, 1024])?;
    let b = randn(&[1024, 2048])?;

    let matmul_output = CudaNeuralOps::optimized_matmul(
        &a, &b, false, // don't transpose A
        false, // don't transpose B
        true,  // use Tensor Cores if available
    )?;

    println!(
        "✓ Optimized MatMul completed - output shape: {:?}",
        matmul_output.shape().dims()
    );

    // Memory-efficient layer normalization
    let input = randn(&[32, 768])?; // Transformer-like input
    let weight = ones(&[768])?;
    let bias = zeros(&[768])?;

    let layernorm_output =
        CudaNeuralOps::memory_efficient_layer_norm(&input, &weight, Some(&bias), 1e-5, &[768])?;

    println!(
        "✓ Memory-efficient LayerNorm completed - output shape: {:?}",
        layernorm_output.shape().dims()
    );

    // Grouped convolution
    let input = randn(&[8, 64, 56, 56])?; // ResNet-like input
    let weight = randn(&[64, 32, 3, 3])?; // 3x3 conv, groups=2
    let bias = Some(randn(&[64])?);

    let grouped_conv_output = CudaNeuralOps::grouped_conv2d(
        &input,
        &weight,
        bias.as_ref(),
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        2,      // groups
    )?;

    println!(
        "✓ Grouped Convolution completed - output shape: {:?}",
        grouped_conv_output.shape().dims()
    );

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_activations() -> Result<()> {
        let activations = CustomActivations::new();
        let input = randn(&[2, 4])?;

        let swish_output = activations.apply("swish", &input)?;
        assert_eq!(swish_output.shape().dims(), input.shape().dims());

        Ok(())
    }

    #[test]
    fn test_kernel_registration() -> Result<()> {
        let registry = global_kernel_registry();

        let initial_count = registry.list_kernels().len();

        registry.register_kernel(
            "test_kernel".to_string(),
            |inputs: &[&torsh_tensor::Tensor], outputs: &mut [&mut torsh_tensor::Tensor]| {
                *outputs[0] = inputs[0].clone();
                Ok(())
            },
        )?;

        assert_eq!(registry.list_kernels().len(), initial_count + 1);
        assert!(registry.has_kernel("test_kernel"));

        Ok(())
    }
}
