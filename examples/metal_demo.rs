//! Demonstration of Metal GPU backend for Apple Silicon
//!
//! This example shows how to use the Metal backend for GPU-accelerated
//! tensor operations on macOS with Apple Silicon (M1/M2/M3).

use std::time::Instant;
use torsh_backend::{Backend, BackendType, MetalBackend};
use torsh_backend::metal::{MetalBuffer, MetalDevice};
use torsh_core::{DType, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ToRSh Metal Backend Demo");
    println!("========================\n");

    // Check if Metal is available using unified backend
    let backend = match Backend::new(BackendType::Metal) {
        Ok(backend) => backend,
        Err(_) => {
            println!("❌ No Metal device found. This example requires macOS with Metal support.");
            return Ok(());
        }
    };

    // Initialize Metal device through unified backend
    let device = backend.as_metal().unwrap().device();
    println!("✅ Metal Device: {}", device.info().name);
    println!("   Unified Memory: {}", device.info().has_unified_memory);
    println!(
        "   Max Threadgroup Memory: {} KB",
        device.info().max_threadgroup_memory / 1024
    );
    println!();

    // Basic tensor operations
    basic_operations(&device)?;

    // Advanced operations demo
    advanced_operations(&device)?;

    // Matrix multiplication benchmark
    matmul_benchmark(&device)?;

    // Convolution demo
    conv2d_demo(&device)?;

    // Reduction operations demo
    reduction_operations(&device)?;

    // Memory transfer benchmark
    memory_benchmark(&device)?;

    println!("\n✅ All demos completed successfully!");

    Ok(())
}

fn basic_operations(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("Basic Tensor Operations");
    println!("-----------------------");

    let shape = Shape::from(vec![4, 4]);

    // Create tensors
    let a = MetalBuffer::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &shape,
        device,
    )?;

    let b = MetalBuffer::ones(&shape, &DType::F32, device)?;

    // Addition
    let c = backend.as_metal().unwrap().add(&a, &b)?;
    println!(
        "Addition result (first 4 elements): {:?}",
        &c.to_vec::<f32>()?[..4]
    );

    // Element-wise multiplication
    let d = torsh_backend_metal::ops::mul(&a, &c)?;
    println!(
        "Multiplication result (first 4 elements): {:?}",
        &d.to_vec::<f32>()?[..4]
    );

    // ReLU activation
    let neg_data = vec![
        -1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0,
        16.0,
    ];
    let neg_tensor = MetalBuffer::from_slice(&neg_data, &shape, device)?;
    let relu_result = torsh_backend_metal::ops::relu(&neg_tensor)?;
    println!(
        "ReLU result (first 4 elements): {:?}",
        &relu_result.to_vec::<f32>()?[..4]
    );

    Ok(())
}

fn advanced_operations(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nAdvanced Tensor Operations");
    println!("---------------------------");

    let shape = Shape::from(vec![2, 3]);
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor = MetalBuffer::from_slice(&data, &shape, device)?;

    // Activation functions
    println!("Input: {:?}", data);

    // Sigmoid
    let sigmoid_result = torsh_backend_metal::ops::sigmoid(&tensor)?;
    println!("Sigmoid: {:?}", sigmoid_result.to_vec::<f32>()?);

    // Tanh
    let tanh_result = torsh_backend_metal::ops::tanh(&tensor)?;
    println!("Tanh: {:?}", tanh_result.to_vec::<f32>()?);

    // GELU
    let gelu_result = torsh_backend_metal::ops::gelu(&tensor)?;
    println!("GELU: {:?}", gelu_result.to_vec::<f32>()?);

    // Mathematical operations
    let shape2 = Shape::from(vec![4]);
    let angles = vec![
        0.0f32,
        std::f32::consts::PI / 4.0,
        std::f32::consts::PI / 2.0,
        std::f32::consts::PI,
    ];
    let angle_tensor = MetalBuffer::from_slice(&angles, &shape2, device)?;

    // Trigonometric functions
    let sin_result = torsh_backend_metal::ops::sin(&angle_tensor)?;
    let cos_result = torsh_backend_metal::ops::cos(&angle_tensor)?;
    println!("\nAngles: {:?}", angles);
    println!("Sin: {:?}", sin_result.to_vec::<f32>()?);
    println!("Cos: {:?}", cos_result.to_vec::<f32>()?);

    // Exponential and logarithm
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let value_tensor = MetalBuffer::from_slice(&values, &shape2, device)?;
    let exp_result = torsh_backend_metal::ops::exp(&value_tensor)?;
    let log_result = torsh_backend_metal::ops::log(&value_tensor)?;
    println!("\nValues: {:?}", values);
    println!("Exp: {:?}", exp_result.to_vec::<f32>()?);
    println!("Log: {:?}", log_result.to_vec::<f32>()?);

    Ok(())
}

fn matmul_benchmark(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nMatrix Multiplication Benchmark");
    println!("--------------------------------");

    for size in [128, 256, 512, 1024] {
        let shape_a = Shape::from(vec![size, size]);
        let shape_b = Shape::from(vec![size, size]);

        // Create random matrices
        let a = MetalBuffer::rand(&shape_a, &DType::F32, device)?;
        let b = MetalBuffer::rand(&shape_b, &DType::F32, device)?;

        // Warmup
        let _ = torsh_backend_metal::ops::matmul(&a, &b)?;
        device.synchronize()?;

        // Benchmark
        let start = Instant::now();
        let n_iters = 10;

        for _ in 0..n_iters {
            let _ = torsh_backend_metal::ops::matmul(&a, &b)?;
        }
        device.synchronize()?;

        let elapsed = start.elapsed();
        let avg_time = elapsed.as_secs_f64() / n_iters as f64;
        let gflops = (2.0 * size as f64 * size as f64 * size as f64) / (avg_time * 1e9);

        println!(
            "  {}x{} @ {}x{}: {:.2} ms/iter, {:.1} GFLOPS",
            size,
            size,
            size,
            size,
            avg_time * 1000.0,
            gflops
        );
    }

    Ok(())
}

fn conv2d_demo(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2D Convolution Demo");
    println!("-------------------");

    // Input: 1 image, 3 channels (RGB), 32x32
    let input_shape = Shape::from(vec![1, 3, 32, 32]);
    let input = MetalBuffer::rand(&input_shape, &DType::F32, device)?;

    // Weight: 16 filters, 3 input channels, 3x3 kernel
    let weight_shape = Shape::from(vec![16, 3, 3, 3]);
    let weight = MetalBuffer::randn(&weight_shape, &DType::F32, device)?;

    // Bias: 16 output channels
    let bias_shape = Shape::from(vec![16]);
    let bias = MetalBuffer::zeros(&bias_shape, &DType::F32, device)?;

    // Perform convolution
    let config = torsh_backend_metal::ops::Conv2dConfig {
        stride: (1, 1),
        padding: (1, 1),
        dilation: (1, 1),
        groups: 1,
    };

    let start = Instant::now();
    let output = torsh_backend_metal::ops::conv2d(&input, &weight, Some(&bias), config)?;
    device.synchronize()?;
    let elapsed = start.elapsed();

    println!("  Input shape: {:?}", input.shape().dims());
    println!("  Weight shape: {:?}", weight.shape().dims());
    println!("  Output shape: {:?}", output.shape().dims());
    println!(
        "  Convolution time: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn reduction_operations(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nReduction Operations Demo");
    println!("-------------------------");

    // 1D reductions
    let shape_1d = Shape::from(vec![10]);
    let data_1d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let tensor_1d = MetalBuffer::from_slice(&data_1d, &shape_1d, device)?;

    println!("1D Tensor: {:?}", data_1d);

    let sum_result = torsh_backend_metal::ops::sum(&tensor_1d, None, false)?;
    println!("Sum: {:?}", sum_result.to_vec::<f32>()?);

    let mean_result = torsh_backend_metal::ops::mean(&tensor_1d, None, false)?;
    println!("Mean: {:?}", mean_result.to_vec::<f32>()?);

    let max_result = torsh_backend_metal::ops::max(&tensor_1d, None, false)?;
    println!("Max: {:?}", max_result.to_vec::<f32>()?);

    let min_result = torsh_backend_metal::ops::min(&tensor_1d, None, false)?;
    println!("Min: {:?}", min_result.to_vec::<f32>()?);

    // 2D reductions
    let shape_2d = Shape::from(vec![3, 4]);
    let data_2d = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor_2d = MetalBuffer::from_slice(&data_2d, &shape_2d, device)?;

    println!("\n2D Tensor shape: {:?}", shape_2d.dims());
    let sum_2d = torsh_backend_metal::ops::sum(&tensor_2d, None, false)?;
    println!("Sum (all): {:?}", sum_2d.to_vec::<f32>()?);

    // Performance test
    let large_shape = Shape::from(vec![1000, 1000]);
    let large_tensor = MetalBuffer::rand(&large_shape, &DType::F32, device)?;

    let start = Instant::now();
    let _ = torsh_backend_metal::ops::sum(&large_tensor, None, false)?;
    device.synchronize()?;
    let elapsed = start.elapsed();

    println!("\nLarge tensor reduction (1M elements):");
    println!("Time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

fn memory_benchmark(device: &MetalDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nMemory Transfer Benchmark");
    println!("-------------------------");

    for mb in [1, 10, 100, 500] {
        let size = mb * 1024 * 1024 / 4; // Number of f32 elements
        let shape = Shape::from(vec![size]);
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Host to Device
        let start = Instant::now();
        let buffer = MetalBuffer::from_slice(&data, &shape, device)?;
        device.synchronize()?;
        let h2d_time = start.elapsed();

        // Device to Host
        let start = Instant::now();
        let _ = buffer.to_vec::<f32>()?;
        let d2h_time = start.elapsed();

        let mb_size = mb as f64;
        let h2d_bandwidth = mb_size / h2d_time.as_secs_f64();
        let d2h_bandwidth = mb_size / d2h_time.as_secs_f64();

        println!(
            "  {} MB: H2D {:.1} MB/s, D2H {:.1} MB/s",
            mb, h2d_bandwidth, d2h_bandwidth
        );
    }

    Ok(())
}
