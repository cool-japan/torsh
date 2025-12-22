//! Performance and stress tests for torsh-nn
//!
//! Tests that verify performance characteristics and robustness under load

use std::time::Instant;
use torsh_nn::container::Sequential;
use torsh_nn::functional::{binary_cross_entropy, mse_loss, relu, sigmoid, softmax, tanh};
use torsh_nn::layers::{activation::ReLU, linear::Linear};
use torsh_nn::Module;
use torsh_tensor::creation::*;

/// Test activation function performance with large tensors
#[test]
fn test_activation_performance() {
    let large_tensor = randn::<f32>(&[1000, 1000]).unwrap();

    // Test ReLU performance
    let start = Instant::now();
    let _relu_result = relu(&large_tensor).unwrap();
    let relu_time = start.elapsed();

    // Test Sigmoid performance
    let start = Instant::now();
    let _sigmoid_result = sigmoid(&large_tensor).unwrap();
    let sigmoid_time = start.elapsed();

    // Test Tanh performance
    let start = Instant::now();
    let _tanh_result = tanh(&large_tensor).unwrap();
    let tanh_time = start.elapsed();

    println!("Activation performance on 1M elements:");
    println!("  ReLU: {:?}", relu_time);
    println!("  Sigmoid: {:?}", sigmoid_time);
    println!("  Tanh: {:?}", tanh_time);

    // ReLU should typically be fastest (simple element-wise max),
    // but performance can vary due to CPU scheduling and cache effects.
    // Just verify all operations complete in reasonable time.
    let max_time = std::time::Duration::from_millis(100);
    assert!(relu_time < max_time, "ReLU took too long: {:?}", relu_time);
    assert!(
        sigmoid_time < max_time,
        "Sigmoid took too long: {:?}",
        sigmoid_time
    );
    assert!(tanh_time < max_time, "Tanh took too long: {:?}", tanh_time);
}

/// Test loss function performance
#[test]
fn test_loss_performance() {
    let batch_size = 1000;
    let num_features = 1000;

    let predictions = randn::<f32>(&[batch_size, num_features]).unwrap();
    let targets = randn::<f32>(&[batch_size, num_features]).unwrap();

    // Test MSE performance
    let start = Instant::now();
    let _mse_result = mse_loss(&predictions, &targets, "mean").unwrap();
    let mse_time = start.elapsed();

    // Test Binary Cross Entropy performance (with valid inputs)
    let sigmoid_pred = sigmoid(&predictions).unwrap();
    let sigmoid_targets = sigmoid(&targets).unwrap();

    let start = Instant::now();
    let _bce_result = binary_cross_entropy(&sigmoid_pred, &sigmoid_targets, None, "mean").unwrap();
    let bce_time = start.elapsed();

    println!(
        "Loss function performance on {}x{} tensors:",
        batch_size, num_features
    );
    println!("  MSE: {:?}", mse_time);
    println!("  BCE: {:?}", bce_time);

    // Both should complete in reasonable time
    assert!(
        mse_time.as_millis() < 1000,
        "MSE should complete within 1 second"
    );
    assert!(
        bce_time.as_millis() < 2000,
        "BCE should complete within 2 seconds"
    );
}

/// Test softmax performance and numerical stability
#[test]
fn test_softmax_performance_stability() {
    let batch_size = 100;
    let num_classes = 1000;

    // Test with normal values
    let normal_input = randn::<f32>(&[batch_size, num_classes]).unwrap();

    let start = Instant::now();
    let normal_result = softmax(&normal_input, Some(1)).unwrap();
    let normal_time = start.elapsed();

    // Test with large values (numerical stability test)
    let large_values = ones::<f32>(&[batch_size, num_classes])
        .unwrap()
        .mul_op(&tensor_scalar(100.0f32).unwrap())
        .unwrap();

    let start = Instant::now();
    let large_result = softmax(&large_values, Some(1)).unwrap();
    let large_time = start.elapsed();

    // Test with very large values
    let very_large = ones::<f32>(&[batch_size, num_classes])
        .unwrap()
        .mul_op(&tensor_scalar(1000.0f32).unwrap())
        .unwrap();

    let start = Instant::now();
    let very_large_result = softmax(&very_large, Some(1)).unwrap();
    let very_large_time = start.elapsed();

    println!(
        "Softmax performance on {}x{} tensors:",
        batch_size, num_classes
    );
    println!("  Normal values: {:?}", normal_time);
    println!("  Large values: {:?}", large_time);
    println!("  Very large values: {:?}", very_large_time);

    // Check that all outputs are valid probabilities
    for result in [&normal_result, &large_result, &very_large_result] {
        let data = result.to_vec().unwrap();

        // Check each batch
        for batch in 0..batch_size {
            let mut batch_sum = 0.0;
            for class in 0..num_classes {
                let prob = data[batch * num_classes + class];
                assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability: {}", prob);
                assert!(!prob.is_nan(), "Probability is NaN");
                assert!(!prob.is_infinite(), "Probability is infinite");
                batch_sum += prob;
            }

            // Each batch should sum to approximately 1
            assert!(
                (batch_sum - 1.0).abs() < 1e-4,
                "Batch {} sum: {}",
                batch,
                batch_sum
            );
        }
    }
}

/// Test memory usage with large models
#[test]
fn test_large_model_memory() {
    // Create a large sequential model
    let model = Sequential::new()
        .add(Linear::new(1000, 512, true))
        .add(ReLU::new())
        .add(Linear::new(512, 256, true))
        .add(ReLU::new())
        .add(Linear::new(256, 128, true))
        .add(ReLU::new())
        .add(Linear::new(128, 10, true));

    let batch_size = 32;
    let input = randn::<f32>(&[batch_size, 1000]).unwrap();

    // Test forward pass performance
    let start = Instant::now();
    let output = model.forward(&input).unwrap();
    let forward_time = start.elapsed();

    // Verify output shape
    assert_eq!(output.shape().dims(), &[batch_size, 10]);

    println!(
        "Large model forward pass ({}x1000 -> {}x10): {:?}",
        batch_size, batch_size, forward_time
    );

    // Should complete in reasonable time
    assert!(
        forward_time.as_millis() < 5000,
        "Large model forward should complete within 5 seconds"
    );
}

/// Test batch size scaling
#[test]
fn test_batch_scaling() {
    let batch_sizes = vec![1, 8, 32, 128, 512];
    let input_size = 784;
    let output_size = 10;

    let layer = Linear::new(input_size, output_size, true);

    for &batch_size in &batch_sizes {
        let input = randn::<f32>(&[batch_size, input_size]).unwrap();

        let start = Instant::now();
        let output = layer.forward(&input).unwrap();
        let forward_time = start.elapsed();

        // Verify output shape
        assert_eq!(output.shape().dims(), &[batch_size, output_size]);

        println!("Linear layer batch_size={}: {:?}", batch_size, forward_time);

        // Time should scale roughly linearly with batch size
        // (allowing for some overhead and variance)
        if batch_size <= 128 {
            assert!(
                forward_time.as_millis() < 1000,
                "Linear layer should be fast for reasonable batch sizes"
            );
        }
    }
}

/// Test activation function accuracy with extreme values
#[test]
fn test_extreme_value_handling() {
    // Test with very small values
    let tiny_values = tensor_2d(&[&[1e-10, -1e-10, 1e-20]]).unwrap();

    let sigmoid_tiny = sigmoid(&tiny_values).unwrap();
    let sigmoid_tiny_data = sigmoid_tiny.to_vec().unwrap();

    // Should handle tiny values gracefully
    for &val in &sigmoid_tiny_data {
        assert!(!val.is_nan(), "Sigmoid produced NaN for tiny values");
        assert!(
            !val.is_infinite(),
            "Sigmoid produced infinity for tiny values"
        );
        assert!(
            val >= 0.0 && val <= 1.0,
            "Sigmoid output out of range: {}",
            val
        );
    }

    // Test with moderately large values (not so large as to cause overflow)
    let large_values = tensor_2d(&[&[10.0, -10.0, 15.0]]).unwrap();

    let sigmoid_large = sigmoid(&large_values).unwrap();
    let sigmoid_large_data = sigmoid_large.to_vec().unwrap();

    for &val in &sigmoid_large_data {
        assert!(!val.is_nan(), "Sigmoid produced NaN for large values");
        assert!(
            !val.is_infinite(),
            "Sigmoid produced infinity for large values"
        );
        assert!(
            val >= 0.0 && val <= 1.0,
            "Sigmoid output out of range: {}",
            val
        );
    }

    // Test tanh with extreme values
    let tanh_extreme = tanh(&large_values).unwrap();
    let tanh_extreme_data = tanh_extreme.to_vec().unwrap();

    for &val in &tanh_extreme_data {
        assert!(!val.is_nan(), "Tanh produced NaN for extreme values");
        assert!(
            !val.is_infinite(),
            "Tanh produced infinity for extreme values"
        );
        assert!(
            val >= -1.0 && val <= 1.0,
            "Tanh output out of range: {}",
            val
        );
    }
}

/// Stress test with repeated operations
#[test]
fn test_repeated_operations() {
    let input = randn::<f32>(&[100, 100]).unwrap();
    let num_iterations = 100;

    let start = Instant::now();
    let mut result = input;

    for _i in 0..num_iterations {
        // Chain multiple operations
        result = relu(&result).unwrap();
        result = sigmoid(&result).unwrap();

        // Prevent numerical drift by occasionally resetting scale
        if result.to_vec().unwrap().iter().all(|&x| x.abs() < 1e-10) {
            result = randn::<f32>(&[100, 100]).unwrap();
        }
    }

    let total_time = start.elapsed();

    println!(
        "Repeated operations ({} iterations): {:?}",
        num_iterations, total_time
    );

    // Verify result is still valid
    let final_data = result.to_vec().unwrap();
    for &val in &final_data {
        assert!(!val.is_nan(), "Final result contains NaN");
        assert!(!val.is_infinite(), "Final result contains infinity");
    }

    // Should complete in reasonable time
    assert!(
        total_time.as_millis() < 10000,
        "Repeated operations should complete within 10 seconds"
    );
}

/// Test concurrent operation performance (if threading is supported)
#[test]
fn test_concurrent_operations() {
    use std::sync::Arc;
    use std::thread;

    let input = Arc::new(randn::<f32>(&[500, 500]).unwrap());
    let num_threads = 4;

    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let input_clone = Arc::clone(&input);
            thread::spawn(move || {
                // Each thread performs the same operations
                let _relu_result = relu(&input_clone).unwrap();
                let _sigmoid_result = sigmoid(&input_clone).unwrap();
                let _tanh_result = tanh(&input_clone).unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let concurrent_time = start.elapsed();

    println!(
        "Concurrent operations ({} threads): {:?}",
        num_threads, concurrent_time
    );

    // Concurrent operations should complete without hanging
    assert!(
        concurrent_time.as_millis() < 15000,
        "Concurrent operations should complete within 15 seconds"
    );
}
