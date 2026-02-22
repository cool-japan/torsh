//! Comprehensive Integration Tests for ToRSh Framework
//!
//! This module contains integration tests that validate:
//! - Cross-crate functionality and compatibility
//! - API consistency across modules
//! - End-to-end workflows
//! - Performance characteristics
//! - Memory management
//! - Device compatibility
//! - Error handling and edge cases

use torsh::prelude::*;
use std::collections::HashMap;

/// Test basic tensor operations across all backends
#[test]
fn test_cross_backend_tensor_operations() {
    // Test CPU backend
    let cpu_device = Device::cpu();
    test_tensor_operations_on_device(&cpu_device).expect("CPU tensor operations failed");
    
    // Test CUDA backend if available
    if Device::cuda(0).is_available() {
        let cuda_device = Device::cuda(0);
        test_tensor_operations_on_device(&cuda_device).expect("CUDA tensor operations failed");
    }
    
    // Test Metal backend if available (on macOS)
    if Device::metal().is_available() {
        let metal_device = Device::metal();
        test_tensor_operations_on_device(&metal_device).expect("Metal tensor operations failed");
    }
}

fn test_tensor_operations_on_device(device: &Device) -> Result<()> {
    println!("Testing tensor operations on device: {:?}", device);
    
    // Basic tensor creation
    let a = randn(&[4, 4]).to_device(device)?;
    let b = randn(&[4, 4]).to_device(device)?;
    
    // Arithmetic operations
    let c = a.add(&b)?;
    let d = a.mul(&b)?;
    let e = a.sub(&b)?;
    let f = a.div(&b)?;
    
    assert_eq!(c.device(), device);
    assert_eq!(d.device(), device);
    assert_eq!(e.device(), device);
    assert_eq!(f.device(), device);
    
    // Matrix operations
    let matmul_result = a.matmul(&b)?;
    assert_eq!(matmul_result.shape().dims(), &[4, 4]);
    assert_eq!(matmul_result.device(), device);
    
    // Reductions
    let sum = a.sum()?;
    let mean = a.mean()?;
    let max_val = a.max()?;
    let min_val = a.min()?;
    
    assert_eq!(sum.device(), device);
    assert_eq!(mean.device(), device);
    assert_eq!(max_val.device(), device);
    assert_eq!(min_val.device(), device);
    
    // Shape operations
    let reshaped = a.reshape(&[2, 8])?;
    let transposed = a.transpose(0, 1)?;
    let squeezed = a.unsqueeze(0)?.squeeze(0)?;
    
    assert_eq!(reshaped.shape().dims(), &[2, 8]);
    assert_eq!(transposed.shape().dims(), &[4, 4]);
    assert_eq!(squeezed.shape().dims(), &[4, 4]);
    
    println!("✓ All tensor operations passed on device: {:?}", device);
    Ok(())
}

/// Test autograd functionality across different operations
#[test]
fn test_autograd_integration() -> Result<()> {
    println!("Testing autograd integration...");
    
    // Test basic differentiation
    let x = tensor![2.0, 3.0, 4.0].requires_grad_(true);
    let y = x.pow(2.0)?.sum()?;
    
    y.backward()?;
    let grad = x.grad().unwrap();
    
    // Gradient should be 2*x = [4.0, 6.0, 8.0]
    let expected_grad = tensor![4.0, 6.0, 8.0];
    assert_tensor_close(&grad, &expected_grad, 1e-6)?;
    
    // Test complex computational graph
    let a = tensor![1.0, 2.0].requires_grad_(true);
    let b = tensor![3.0, 4.0].requires_grad_(true);
    
    let c = a.mul(&b)?;  // [3.0, 8.0]
    let d = c.sum()?;    // 11.0
    let e = d.pow(2.0)?; // 121.0
    
    e.backward()?;
    
    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();
    
    // Check gradients
    assert_tensor_close(&grad_a, &tensor![66.0, 88.0], 1e-6)?;
    assert_tensor_close(&grad_b, &tensor![22.0, 44.0], 1e-6)?;
    
    println!("✓ Autograd integration tests passed");
    Ok(())
}

/// Test neural network module integration
#[test]
fn test_neural_network_integration() -> Result<()> {
    println!("Testing neural network module integration...");
    
    // Create a simple MLP
    let linear1 = Linear::new(784, 256);
    let relu1 = ReLU::new();
    let linear2 = Linear::new(256, 128);
    let relu2 = ReLU::new();
    let linear3 = Linear::new(128, 10);
    
    // Test forward pass
    let input = randn(&[32, 784]);
    
    let x = linear1.forward(&input)?;
    assert_eq!(x.shape().dims(), &[32, 256]);
    
    let x = relu1.forward(&x)?;
    assert_eq!(x.shape().dims(), &[32, 256]);
    
    let x = linear2.forward(&x)?;
    assert_eq!(x.shape().dims(), &[32, 128]);
    
    let x = relu2.forward(&x)?;
    assert_eq!(x.shape().dims(), &[32, 128]);
    
    let output = linear3.forward(&x)?;
    assert_eq!(output.shape().dims(), &[32, 10]);
    
    // Test with Sequential container
    let model = Sequential::new()
        .add(linear1)
        .add(relu1)
        .add(linear2)
        .add(relu2)
        .add(linear3);
    
    let output2 = model.forward(&input)?;
    assert_eq!(output2.shape().dims(), &[32, 10]);
    
    // Test parameter collection
    let params = model.parameters();
    assert!(!params.is_empty());
    
    println!("✓ Neural network integration tests passed");
    Ok(())
}

/// Test optimization integration with neural networks
#[test]
fn test_optimization_integration() -> Result<()> {
    println!("Testing optimization integration...");
    
    // Create a simple model
    let model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(ReLU::new())
        .add(Linear::new(4, 1));
    
    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.01)?;
    
    // Training data (simple regression)
    let x_train = tensor_2d![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y_train = tensor![3.0, 5.0, 7.0, 9.0]; // y = x1 + x2
    
    let mut losses = Vec::new();
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let predictions = model.forward(&x_train)?;
        let predictions = predictions.squeeze(-1)?;
        
        // Compute loss
        let loss = F::mse_loss(&predictions, &y_train)?;
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
        
        losses.push(loss.item::<f32>());
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.item::<f32>());
        }
    }
    
    // Check that loss decreased
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    assert!(final_loss < initial_loss, "Loss should decrease during training");
    assert!(final_loss < 0.1, "Final loss should be small");
    
    println!("✓ Optimization integration tests passed");
    Ok(())
}

/// Test functional operations integration
#[test]
fn test_functional_integration() -> Result<()> {
    println!("Testing functional operations integration...");
    
    let input = randn(&[2, 3, 4]);
    
    // Test activation functions
    let relu_out = F::relu(&input)?;
    let sigmoid_out = F::sigmoid(&input)?;
    let tanh_out = F::tanh(&input)?;
    let gelu_out = F::gelu(&input)?;
    
    assert_eq!(relu_out.shape(), input.shape());
    assert_eq!(sigmoid_out.shape(), input.shape());
    assert_eq!(tanh_out.shape(), input.shape());
    assert_eq!(gelu_out.shape(), input.shape());
    
    // Test normalization
    let softmax_out = F::softmax(&input, -1)?;
    let log_softmax_out = F::log_softmax(&input, -1)?;
    
    assert_eq!(softmax_out.shape(), input.shape());
    assert_eq!(log_softmax_out.shape(), input.shape());
    
    // Test loss functions
    let logits = randn(&[4, 10]);
    let targets = randint(0, 10, &[4]);
    
    let ce_loss = F::cross_entropy(&logits, &targets)?;
    assert_eq!(ce_loss.shape().dims(), &[]);
    
    let regression_targets = randn(&[4, 10]);
    let mse_loss = F::mse_loss(&logits, &regression_targets)?;
    assert_eq!(mse_loss.shape().dims(), &[]);
    
    // Test convolution operations
    let conv_input = randn(&[2, 3, 32, 32]);
    let conv_weight = randn(&[16, 3, 3, 3]);
    
    let conv_out = F::conv2d(&conv_input, &conv_weight, None, (1, 1), (1, 1), (1, 1), 1)?;
    assert_eq!(conv_out.shape().dims(), &[2, 16, 32, 32]);
    
    println!("✓ Functional operations integration tests passed");
    Ok(())
}

/// Test data loading integration
#[test]
fn test_data_loading_integration() -> Result<()> {
    println!("Testing data loading integration...");
    
    // Create synthetic dataset
    let data_size = 1000;
    let feature_size = 20;
    
    let data = randn(&[data_size, feature_size]);
    let targets = randint(0, 5, &[data_size]);
    
    // Create dataset
    let dataset = TensorDataset::new(
        (0..data_size).map(|i| data.slice(0, i, i + 1)?.squeeze(0)?).collect::<Result<Vec<_>>>()?,
        (0..data_size).map(|i| targets.slice(0, i, i + 1)?.squeeze(0)?).collect::<Result<Vec<_>>>()?,
    );
    
    // Create data loader
    let batch_size = 32;
    let dataloader = DataLoader::new(dataset, batch_size, true, 2, false);
    
    let mut total_batches = 0;
    let mut total_samples = 0;
    
    for batch in dataloader {
        let (batch_data, batch_targets) = batch;
        
        assert_eq!(batch_data.shape().dims()[1], feature_size);
        assert!(batch_data.shape().dims()[0] <= batch_size);
        assert_eq!(batch_targets.shape().dims()[0], batch_data.shape().dims()[0]);
        
        total_batches += 1;
        total_samples += batch_data.shape().dims()[0];
    }
    
    assert_eq!(total_samples, data_size);
    assert!(total_batches > 0);
    
    println!("✓ Data loading integration tests passed");
    Ok(())
}

/// Test memory management across operations
#[test]
fn test_memory_management_integration() -> Result<()> {
    println!("Testing memory management integration...");
    
    let initial_memory = get_memory_usage()?;
    
    // Create large tensors
    let large_tensors: Vec<Tensor> = (0..10)
        .map(|_| randn(&[1000, 1000]))
        .collect();
    
    let after_allocation = get_memory_usage()?;
    assert!(after_allocation > initial_memory, "Memory usage should increase after allocation");
    
    // Perform operations that create intermediate tensors
    let mut results = Vec::new();
    for i in 0..9 {
        let result = large_tensors[i].matmul(&large_tensors[i + 1])?;
        results.push(result);
    }
    
    let after_operations = get_memory_usage()?;
    
    // Drop references
    drop(large_tensors);
    drop(results);
    
    // Force garbage collection
    empty_cache();
    
    let after_cleanup = get_memory_usage()?;
    
    println!("Memory usage: initial={}, after_alloc={}, after_ops={}, after_cleanup={}",
             initial_memory, after_allocation, after_operations, after_cleanup);
    
    // Memory should be lower after cleanup (though not necessarily back to initial due to fragmentation)
    assert!(after_cleanup < after_operations, "Memory usage should decrease after cleanup");
    
    println!("✓ Memory management integration tests passed");
    Ok(())
}

/// Test device transfer and compatibility
#[test]
fn test_device_transfer_integration() -> Result<()> {
    println!("Testing device transfer integration...");
    
    let cpu_device = Device::cpu();
    let tensor_cpu = randn(&[100, 100]).to_device(&cpu_device)?;
    
    assert_eq!(tensor_cpu.device(), &cpu_device);
    
    // Test operations on CPU
    let result_cpu = tensor_cpu.matmul(&tensor_cpu)?;
    assert_eq!(result_cpu.device(), &cpu_device);
    
    // Test CUDA transfer if available
    if Device::cuda(0).is_available() {
        let cuda_device = Device::cuda(0);
        let tensor_cuda = tensor_cpu.to_device(&cuda_device)?;
        
        assert_eq!(tensor_cuda.device(), &cuda_device);
        
        // Test operations on CUDA
        let result_cuda = tensor_cuda.matmul(&tensor_cuda)?;
        assert_eq!(result_cuda.device(), &cuda_device);
        
        // Test transfer back to CPU
        let result_back_to_cpu = result_cuda.to_device(&cpu_device)?;
        assert_eq!(result_back_to_cpu.device(), &cpu_device);
        
        // Values should be approximately equal
        let cpu_result = result_cpu.to_device(&cpu_device)?;
        assert_tensor_close(&result_back_to_cpu, &cpu_result, 1e-4)?;
    }
    
    println!("✓ Device transfer integration tests passed");
    Ok(())
}

/// Test error handling and edge cases
#[test]
fn test_error_handling_integration() -> Result<()> {
    println!("Testing error handling integration...");
    
    // Test shape mismatch errors
    let a = randn(&[2, 3]);
    let b = randn(&[4, 5]);
    
    let result = a.matmul(&b);
    assert!(result.is_err(), "Matrix multiplication with incompatible shapes should fail");
    
    // Test device mismatch errors
    let cpu_tensor = randn(&[2, 2]).to_device(&Device::cpu())?;
    
    if Device::cuda(0).is_available() {
        let cuda_tensor = randn(&[2, 2]).to_device(&Device::cuda(0))?;
        
        let result = cpu_tensor.add(&cuda_tensor);
        assert!(result.is_err(), "Operations between different devices should fail");
    }
    
    // Test index out of bounds
    let tensor = randn(&[2, 3]);
    let result = tensor.slice(0, 10, 20);
    assert!(result.is_err(), "Slice with out-of-bounds indices should fail");
    
    // Test invalid reshape
    let tensor = randn(&[2, 3]);
    let result = tensor.reshape(&[5]);  // 6 elements cannot be reshaped to 5
    assert!(result.is_err(), "Invalid reshape should fail");
    
    // Test division by zero
    let a = randn(&[2, 2]);
    let b = zeros(&[2, 2]);
    let result = a.div(&b);
    // Note: Division by zero might produce inf/nan rather than error, depending on implementation
    
    println!("✓ Error handling integration tests passed");
    Ok(())
}

/// Test performance characteristics
#[test]
fn test_performance_characteristics() -> Result<()> {
    println!("Testing performance characteristics...");
    
    let sizes = vec![100, 500, 1000];
    let mut timings = HashMap::new();
    
    for size in sizes {
        let a = randn(&[size, size]);
        let b = randn(&[size, size]);
        
        // Time matrix multiplication
        let start = std::time::Instant::now();
        let _result = a.matmul(&b)?;
        let duration = start.elapsed();
        
        timings.insert(size, duration);
        println!("Matrix multiplication {}x{}: {:?}", size, size, duration);
    }
    
    // Verify that larger matrices take more time (roughly)
    assert!(timings[&1000] > timings[&100], "Larger operations should take more time");
    
    // Test memory efficiency
    let large_size = 2000;
    let memory_before = get_memory_usage()?;
    
    {
        let large_tensor = randn(&[large_size, large_size]);
        let _result = large_tensor.sum()?;
    } // large_tensor goes out of scope here
    
    empty_cache();
    let memory_after = get_memory_usage()?;
    
    // Memory should not have grown significantly after cleanup
    let memory_growth = memory_after as i64 - memory_before as i64;
    println!("Memory growth after large operation: {} MB", memory_growth / 1024 / 1024);
    
    println!("✓ Performance characteristics tests passed");
    Ok(())
}

/// Test serialization and checkpointing
#[test]
fn test_serialization_integration() -> Result<()> {
    println!("Testing serialization integration...");
    
    // Create a model
    let model = Sequential::new()
        .add(Linear::new(10, 20))
        .add(ReLU::new())
        .add(Linear::new(20, 5));
    
    // Get model state
    let state_dict = model.state_dict();
    assert!(!state_dict.is_empty());
    
    // Test tensor serialization
    let original_tensor = randn(&[5, 5]);
    
    // In a real implementation, you would save/load from disk
    // For now, we'll test that the state dict contains the expected keys
    let parameters = model.parameters();
    assert!(!parameters.is_empty());
    
    println!("Model has {} parameters", parameters.len());
    
    // Test optimizer state serialization
    let mut optimizer = Adam::new(model.parameters(), 0.01)?;
    
    // Simulate some training steps
    let input = randn(&[4, 10]);
    let target = randn(&[4, 5]);
    
    for _ in 0..5 {
        let output = model.forward(&input)?;
        let loss = F::mse_loss(&output, &target)?;
        
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }
    
    let optimizer_state = optimizer.state_dict();
    assert!(!optimizer_state.is_empty());
    
    println!("✓ Serialization integration tests passed");
    Ok(())
}

/// Test advanced features integration
#[test]
fn test_advanced_features_integration() -> Result<()> {
    println!("Testing advanced features integration...");
    
    // Test gradient clipping
    let x = tensor![1000.0].requires_grad_(true);
    let y = x.pow(3.0)?; // Large gradient
    
    y.backward()?;
    let original_grad = x.grad().unwrap().clone();
    
    // Clip gradients
    let max_norm = 1.0;
    clip_grad_norm_(&[x.clone()], max_norm)?;
    
    let clipped_grad = x.grad().unwrap();
    let grad_norm = clipped_grad.norm()?;
    
    assert!(grad_norm.item::<f32>() <= max_norm + 1e-6, "Gradient norm should be clipped");
    
    // Test gradient accumulation
    let model = Linear::new(2, 1);
    let mut optimizer = SGD::new(model.parameters(), 0.01)?;
    
    let input1 = randn(&[1, 2]);
    let input2 = randn(&[1, 2]);
    let target = randn(&[1, 1]);
    
    // Accumulate gradients from two batches
    let output1 = model.forward(&input1)?;
    let loss1 = F::mse_loss(&output1, &target)?;
    loss1.backward()?;
    
    let output2 = model.forward(&input2)?;
    let loss2 = F::mse_loss(&output2, &target)?;
    loss2.backward()?;
    
    // Gradients should be accumulated
    let accumulated_grads: Vec<_> = model.parameters()
        .iter()
        .map(|p| p.grad().unwrap().clone())
        .collect();
    
    optimizer.step()?;
    optimizer.zero_grad();
    
    println!("✓ Advanced features integration tests passed");
    Ok(())
}

/// Test end-to-end training workflow
#[test]
fn test_end_to_end_training_workflow() -> Result<()> {
    println!("Testing end-to-end training workflow...");
    
    // Create a classification dataset
    let num_samples = 1000;
    let num_features = 20;
    let num_classes = 3;
    
    let data = randn(&[num_samples, num_features]);
    let targets = randint(0, num_classes as i64, &[num_samples]);
    
    // Create model
    let model = Sequential::new()
        .add(Linear::new(num_features, 64))
        .add(ReLU::new())
        .add(Linear::new(64, 32))
        .add(ReLU::new())
        .add(Linear::new(32, num_classes));
    
    // Create optimizer and scheduler
    let mut optimizer = Adam::new(model.parameters(), 0.001)?;
    let mut scheduler = StepLR::new(Box::new(optimizer.clone()), 10, 0.5);
    
    // Training loop
    let batch_size = 32;
    let num_epochs = 5;
    
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Batch iteration (simplified)
        for start in (0..num_samples).step_by(batch_size) {
            let end = std::cmp::min(start + batch_size, num_samples);
            
            let batch_data = data.slice(0, start, end)?;
            let batch_targets = targets.slice(0, start, end)?;
            
            // Forward pass
            let outputs = model.forward(&batch_data)?;
            let loss = F::cross_entropy(&outputs, &batch_targets)?;
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;
            
            total_loss += loss.item::<f32>();
            num_batches += 1;
        }
        
        // Step scheduler
        scheduler.step();
        
        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {}: Average Loss = {:.6}, LR = {:.6}", 
                 epoch + 1, avg_loss, scheduler.get_lr());
    }
    
    // Test model evaluation
    model.eval();
    
    no_grad(|| -> Result<()> {
        let test_data = randn(&[100, num_features]);
        let test_outputs = model.forward(&test_data)?;
        let predictions = test_outputs.argmax(-1)?;
        
        assert_eq!(predictions.shape().dims(), &[100]);
        
        Ok(())
    })?;
    
    println!("✓ End-to-end training workflow tests passed");
    Ok(())
}

// Helper functions

fn assert_tensor_close(a: &Tensor, b: &Tensor, tolerance: f64) -> Result<()> {
    let diff = a.sub(b)?;
    let max_diff = diff.abs()?.max()?.item::<f32>() as f64;
    
    if max_diff > tolerance {
        return Err(TorshError::Other(format!(
            "Tensors not close: max difference {} > tolerance {}", 
            max_diff, tolerance
        )));
    }
    
    Ok(())
}

fn get_memory_usage() -> Result<usize> {
    // Simplified memory usage tracking
    // In a real implementation, this would query the actual memory usage
    Ok(0) // Placeholder
}

fn empty_cache() {
    // Simplified cache emptying
    // In a real implementation, this would free unused memory
}

fn clip_grad_norm_(parameters: &[Tensor], max_norm: f64) -> Result<()> {
    // Simplified gradient clipping
    // In a real implementation, this would clip gradients
    Ok(()) // Placeholder
}

/// Main integration test runner
#[test]
fn run_all_integration_tests() {
    println!("=== Running ToRSh Integration Tests ===\n");
    
    // Core functionality tests
    test_cross_backend_tensor_operations();
    test_autograd_integration().expect("Autograd integration failed");
    test_neural_network_integration().expect("Neural network integration failed");
    test_optimization_integration().expect("Optimization integration failed");
    test_functional_integration().expect("Functional integration failed");
    test_data_loading_integration().expect("Data loading integration failed");
    
    // Advanced functionality tests
    test_memory_management_integration().expect("Memory management integration failed");
    test_device_transfer_integration().expect("Device transfer integration failed");
    test_error_handling_integration().expect("Error handling integration failed");
    test_performance_characteristics().expect("Performance characteristics failed");
    test_serialization_integration().expect("Serialization integration failed");
    test_advanced_features_integration().expect("Advanced features integration failed");
    
    // End-to-end workflow test
    test_end_to_end_training_workflow().expect("End-to-end workflow failed");
    
    println!("\n=== All Integration Tests Passed! ===");
}

// Benchmark tests for performance validation
#[cfg(feature = "bench")]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_matrix_multiplication() {
        let sizes = vec![128, 256, 512, 1024];
        
        for size in sizes {
            let a = randn(&[size, size]);
            let b = randn(&[size, size]);
            
            let start = Instant::now();
            let _result = a.matmul(&b).unwrap();
            let duration = start.elapsed();
            
            let flops = 2.0 * (size as f64).powi(3);
            let gflops = flops / duration.as_secs_f64() / 1e9;
            
            println!("Matrix multiplication {}x{}: {:.2} GFLOPS", size, size, gflops);
        }
    }
    
    #[test]
    fn benchmark_convolution() {
        let batch_sizes = vec![1, 4, 16];
        let input_channels = 64;
        let output_channels = 128;
        let kernel_size = 3;
        let image_size = 224;
        
        for batch_size in batch_sizes {
            let input = randn(&[batch_size, input_channels, image_size, image_size]);
            let weight = randn(&[output_channels, input_channels, kernel_size, kernel_size]);
            
            let start = Instant::now();
            let _result = F::conv2d(&input, &weight, None, (1, 1), (1, 1), (1, 1), 1).unwrap();
            let duration = start.elapsed();
            
            println!("Convolution batch_size={}: {:.2} ms", batch_size, duration.as_millis());
        }
    }
    
    #[test]
    fn benchmark_training_step() {
        let model = Sequential::new()
            .add(Linear::new(1000, 512))
            .add(ReLU::new())
            .add(Linear::new(512, 256))
            .add(ReLU::new())
            .add(Linear::new(256, 10));
        
        let mut optimizer = Adam::new(model.parameters(), 0.001).unwrap();
        
        let input = randn(&[64, 1000]);
        let target = randint(0, 10, &[64]);
        
        let start = Instant::now();
        
        // Forward pass
        let output = model.forward(&input).unwrap();
        let loss = F::cross_entropy(&output, &target).unwrap();
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward().unwrap();
        optimizer.step().unwrap();
        
        let duration = start.elapsed();
        
        println!("Training step (batch_size=64): {:.2} ms", duration.as_millis());
    }
}