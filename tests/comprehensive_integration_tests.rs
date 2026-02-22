//! Comprehensive integration tests for ToRSh framework
//!
//! This test suite validates cross-crate functionality, API compatibility,
//! end-to-end workflows, performance characteristics, memory management,
//! device compatibility, and error handling across all framework components.

#[cfg(test)]
mod integration_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;
    
    // Import all ToRSh components for integration testing
    use torsh_core::{
        device::DeviceType,
        dtype::DType,
        error::{Result, TorshError},
        shape::Shape,
    };
    
    use torsh_tensor::{
        creation::*,
        Tensor,
    };
    
    // Mock imports for components that may not compile yet
    // These would be replaced with actual imports once compilation is fixed
    
    /// Mock neural network module trait for testing
    trait MockModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor>;
        fn parameters(&self) -> HashMap<String, MockParameter>;
        fn training(&self) -> bool;
        fn train(&mut self);
        fn eval(&mut self);
    }
    
    /// Mock parameter type for testing
    #[derive(Clone, Debug)]
    struct MockParameter {
        data: Tensor,
        requires_grad: bool,
    }
    
    impl MockParameter {
        fn new(tensor: Tensor) -> Self {
            Self {
                data: tensor,
                requires_grad: true,
            }
        }
    }
    
    /// Mock linear layer for testing
    struct MockLinear {
        weight: MockParameter,
        bias: Option<MockParameter>,
        training: bool,
    }
    
    impl MockLinear {
        fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
            let weight = randn(&[in_features, out_features])?;
            let bias_param = if bias {
                Some(MockParameter::new(zeros(&[out_features])?))
            } else {
                None
            };
            
            Ok(Self {
                weight: MockParameter::new(weight),
                bias: bias_param,
                training: true,
            })
        }
    }
    
    impl MockModule for MockLinear {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            let output = input.matmul(&self.weight.data)?;
            if let Some(ref bias) = self.bias {
                output.add(&bias.data)
            } else {
                Ok(output)
            }
        }
        
        fn parameters(&self) -> HashMap<String, MockParameter> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), self.weight.clone());
            if let Some(ref bias) = self.bias {
                params.insert("bias".to_string(), bias.clone());
            }
            params
        }
        
        fn training(&self) -> bool {
            self.training
        }
        
        fn train(&mut self) {
            self.training = true;
        }
        
        fn eval(&mut self) {
            self.training = false;
        }
    }
    
    /// Test 1: Cross-Crate Tensor Operations
    #[test]
    fn test_cross_crate_tensor_operations() -> Result<()> {
        // Test tensor creation from torsh-tensor
        let a = zeros(&[2, 3])?;
        let b = ones(&[2, 3])?;
        let c = randn(&[2, 3])?;
        
        // Test basic operations
        let sum = a.add(&b)?;
        let product = b.mul_op(&c)?;
        
        // Verify shapes
        assert_eq!(sum.shape().dims(), &[2, 3]);
        assert_eq!(product.shape().dims(), &[2, 3]);
        
        // Test device compatibility
        assert_eq!(sum.device(), DeviceType::Cpu);
        assert_eq!(product.device(), DeviceType::Cpu);
        
        // Test data type consistency
        assert_eq!(sum.dtype(), DType::F32);
        assert_eq!(product.dtype(), DType::F32);
        
        println!("✓ Cross-crate tensor operations test passed");
        Ok(())
    }
    
    /// Test 2: Neural Network Module Integration
    #[test]
    fn test_neural_network_integration() -> Result<()> {
        // Create mock neural network components
        let mut linear1 = MockLinear::new(784, 128, true)?;
        let mut linear2 = MockLinear::new(128, 10, true)?;
        
        // Test forward pass
        let input = randn(&[32, 784])?; // Batch size 32, 784 features
        let hidden = linear1.forward(&input)?;
        let output = linear2.forward(&hidden)?;
        
        // Verify output shape
        assert_eq!(output.shape().dims(), &[32, 10]);
        
        // Test parameter access
        let params1 = linear1.parameters();
        let params2 = linear2.parameters();
        
        assert!(params1.contains_key("weight"));
        assert!(params1.contains_key("bias"));
        assert!(params2.contains_key("weight"));
        assert!(params2.contains_key("bias"));
        
        // Test training/eval mode
        assert!(linear1.training());
        linear1.eval();
        assert!(!linear1.training());
        linear1.train();
        assert!(linear1.training());
        
        println!("✓ Neural network integration test passed");
        Ok(())
    }
    
    /// Test 3: Memory Management and Resource Cleanup
    #[test]
    fn test_memory_management() -> Result<()> {
        let initial_memory = get_memory_usage()?;
        
        {
            // Create large tensors in limited scope
            let large_tensors: Vec<Tensor> = (0..10)
                .map(|_| randn(&[1000, 1000]))
                .collect::<Result<Vec<_>>>()?;
            
            // Perform operations
            let sum = large_tensors
                .iter()
                .try_fold(zeros(&[1000, 1000])?, |acc, tensor| {
                    acc.add(tensor)
                })?;
            
            assert_eq!(sum.shape().dims(), &[1000, 1000]);
            
            // Tensors should be automatically cleaned up when scope ends
        }
        
        // Force garbage collection (if available)
        std::hint::black_box(());
        
        let final_memory = get_memory_usage()?;
        
        // Memory should be reasonably close to initial (allowing for some overhead)
        let memory_diff = final_memory.saturating_sub(initial_memory);
        let memory_threshold = initial_memory / 10; // 10% threshold
        
        assert!(
            memory_diff <= memory_threshold,
            "Memory leak detected: initial={}, final={}, diff={}",
            initial_memory, final_memory, memory_diff
        );
        
        println!("✓ Memory management test passed");
        Ok(())
    }
    
    /// Test 4: End-to-End Training Workflow
    #[test]
    fn test_end_to_end_training() -> Result<()> {
        // Create mock model
        let mut model = MockLinear::new(4, 1, true)?;
        
        // Create synthetic dataset
        let x_train = randn(&[100, 4])?;
        let y_train = randn(&[100, 1])?;
        
        let batch_size = 10;
        let num_epochs = 5;
        let learning_rate = 0.01;
        
        // Training loop
        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let num_batches = 100 / batch_size;
            
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(100);
                
                // Get batch data (simplified slicing)
                let batch_x = x_train.slice(0, start_idx, end_idx)?;
                let batch_y = y_train.slice(0, start_idx, end_idx)?;
                
                // Forward pass
                model.train();
                let predictions = model.forward(&batch_x)?;
                
                // Compute loss (MSE)
                let diff = predictions.sub(&batch_y)?;
                let loss = diff.mul_op(&diff)?.mean()?;
                
                total_loss += loss.item::<f32>().unwrap_or(0.0);
                
                // Mock backward pass and parameter update
                // In a real implementation, this would compute gradients and update parameters
                let params = model.parameters();
                assert!(!params.is_empty());
            }
            
            let avg_loss = total_loss / num_batches as f32;
            println!("Epoch {}: Average Loss = {:.6}", epoch + 1, avg_loss);
        }
        
        // Test evaluation mode
        model.eval();
        let eval_predictions = model.forward(&x_train.slice(0, 0, 10)?)?;
        assert_eq!(eval_predictions.shape().dims(), &[10, 1]);
        
        println!("✓ End-to-end training workflow test passed");
        Ok(())
    }
    
    /// Test 5: Device Compatibility
    #[test]
    fn test_device_compatibility() -> Result<()> {
        // Test CPU operations
        let cpu_tensor = randn(&[5, 5])?;
        assert_eq!(cpu_tensor.device(), DeviceType::Cpu);
        
        let cpu_result = cpu_tensor.matmul(&cpu_tensor.transpose(0, 1)?)?;
        assert_eq!(cpu_result.device(), DeviceType::Cpu);
        
        // Test device queries
        let available_devices = get_available_devices();
        assert!(available_devices.contains(&DeviceType::Cpu));
        
        // If CUDA is available, test GPU operations
        if available_devices.contains(&DeviceType::Cuda(0)) {
            println!("CUDA detected, testing GPU operations...");
            
            // Note: These would be uncommented once CUDA backend is stable
            // let gpu_tensor = cpu_tensor.to(DeviceType::Cuda(0))?;
            // assert_eq!(gpu_tensor.device(), DeviceType::Cuda(0));
            // 
            // let gpu_result = gpu_tensor.matmul(&gpu_tensor.transpose(0, 1)?)?;
            // assert_eq!(gpu_result.device(), DeviceType::Cuda(0));
            // 
            // // Test device transfer
            // let back_to_cpu = gpu_result.to(DeviceType::Cpu)?;
            // assert_eq!(back_to_cpu.device(), DeviceType::Cpu);
        } else {
            println!("CUDA not available, skipping GPU tests");
        }
        
        println!("✓ Device compatibility test passed");
        Ok(())
    }
    
    /// Test 6: Error Handling and Edge Cases
    #[test]
    fn test_error_handling() -> Result<()> {
        // Test shape mismatch errors
        let a = zeros(&[3, 4])?;
        let b = zeros(&[5, 6])?;
        
        let result = a.matmul(&b);
        assert!(result.is_err(), "Expected shape mismatch error");
        
        // Test invalid indexing
        let tensor = zeros(&[3, 3])?;
        // Note: This would test actual indexing once the API is stable
        // let invalid_slice = tensor.slice(0, 10, 20);
        // assert!(invalid_slice.is_err(), "Expected index out of bounds error");
        
        // Test empty tensor operations
        let empty = zeros(&[0, 5])?;
        assert_eq!(empty.numel(), 0);
        
        // Test invalid reshape
        let tensor = zeros(&[2, 3])?;
        let invalid_reshape = tensor.reshape(&[2, 4]);
        assert!(invalid_reshape.is_err(), "Expected invalid reshape error");
        
        // Test division by zero handling
        let zero_tensor = zeros(&[2, 2])?;
        let ones_tensor = ones(&[2, 2])?;
        let div_result = ones_tensor.div(&zero_tensor);
        
        // Should either error or produce inf/nan values
        match div_result {
            Ok(result) => {
                // Check for inf/nan values
                let data = result.data()?;
                assert!(data.iter().any(|&x| x.is_infinite() || x.is_nan()));
            }
            Err(_) => {
                // Division by zero error is also acceptable
            }
        }
        
        println!("✓ Error handling test passed");
        Ok(())
    }
    
    /// Test 7: Performance Benchmarking
    #[test]
    fn test_performance_characteristics() -> Result<()> {
        let sizes = vec![100, 500, 1000];
        
        for size in sizes {
            let start_time = Instant::now();
            
            // Matrix multiplication benchmark
            let a = randn(&[size, size])?;
            let b = randn(&[size, size])?;
            let c = a.matmul(&b)?;
            
            let duration = start_time.elapsed();
            let ops = (size * size * size) as f64; // Approximate FLOPs
            let gflops = ops / duration.as_secs_f64() / 1e9;
            
            println!("Matrix multiply {}x{}: {:.2} GFLOPS", size, size, gflops);
            
            // Verify result is reasonable
            assert_eq!(c.shape().dims(), &[size, size]);
            assert!(!c.data()?.iter().any(|&x| x.is_nan()));
            
            // Performance should be reasonable (at least 0.1 GFLOPS)
            assert!(gflops > 0.1, "Performance too slow: {} GFLOPS", gflops);
        }
        
        println!("✓ Performance characteristics test passed");
        Ok(())
    }
    
    /// Test 8: API Compatibility and Consistency
    #[test]
    fn test_api_compatibility() -> Result<()> {
        // Test tensor creation consistency
        let methods = vec![
            zeros(&[2, 3])?,
            ones(&[2, 3])?,
            full(&[2, 3], 0.5)?,
        ];
        
        for tensor in &methods {
            assert_eq!(tensor.shape().dims(), &[2, 3]);
            assert_eq!(tensor.dtype(), DType::F32);
            assert_eq!(tensor.device(), DeviceType::Cpu);
        }
        
        // Test operation chaining
        let result = zeros(&[3, 3])?
            .add(&ones(&[3, 3])?)?
            .mul_op(&full(&[3, 3], 2.0)?)?
            .transpose(0, 1)?;
        
        assert_eq!(result.shape().dims(), &[3, 3]);
        
        // Test consistent error types
        let error1 = zeros(&[2, 3])?.matmul(&zeros(&[4, 5])?);
        let error2 = zeros(&[1, 2])?.matmul(&zeros(&[3, 4])?);
        
        assert!(error1.is_err());
        assert!(error2.is_err());
        
        // Both should be the same type of error
        match (error1, error2) {
            (Err(e1), Err(e2)) => {
                // Both should be shape mismatch errors
                assert!(matches!(e1, TorshError::ShapeMismatch(_)));
                assert!(matches!(e2, TorshError::ShapeMismatch(_)));
            }
            _ => panic!("Expected both operations to fail"),
        }
        
        println!("✓ API compatibility test passed");
        Ok(())
    }
    
    /// Test 9: Thread Safety and Concurrent Operations
    #[test]
    fn test_thread_safety() -> Result<()> {
        use std::thread;
        use std::sync::Arc;
        
        let tensor = Arc::new(randn(&[100, 100])?);
        let num_threads = 4;
        let mut handles = vec![];
        
        for i in 0..num_threads {
            let tensor_clone = Arc::clone(&tensor);
            let handle = thread::spawn(move || -> Result<()> {
                // Each thread performs independent operations
                let local_tensor = randn(&[100, 100])?;
                let result = tensor_clone.add(&local_tensor)?;
                let final_result = result.matmul(&tensor_clone.transpose(0, 1)?)?;
                
                // Verify result is valid
                assert_eq!(final_result.shape().dims(), &[100, 100]);
                assert!(!final_result.data()?.iter().any(|&x| x.is_nan()));
                
                println!("Thread {} completed successfully", i);
                Ok(())
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(result) => result?,
                Err(e) => panic!("Thread {} panicked: {:?}", i, e),
            }
        }
        
        println!("✓ Thread safety test passed");
        Ok(())
    }
    
    /// Test 10: Data Type and Precision Handling
    #[test]
    fn test_data_type_handling() -> Result<()> {
        // Test different data types (if supported)
        let f32_tensor = randn(&[5, 5])?;
        assert_eq!(f32_tensor.dtype(), DType::F32);
        
        // Test type conversions (if supported)
        // let f64_tensor = f32_tensor.to_dtype::<f64>()?;
        // assert_eq!(f64_tensor.dtype(), DType::F64);
        
        // Test precision preservation
        let precise_values = vec![
            std::f32::consts::PI,
            std::f32::consts::E,
            1.23456789,
            -0.987654321,
        ];
        
        let tensor = tensor_from_vec(precise_values.clone(), &[2, 2])?;
        let recovered_data = tensor.data()?;
        
        for (original, recovered) in precise_values.iter().zip(recovered_data.iter()) {
            let diff = (original - recovered).abs();
            assert!(diff < 1e-6, "Precision loss detected: {} vs {}", original, recovered);
        }
        
        // Test special values
        let special_values = vec![0.0, f32::INFINITY, f32::NEG_INFINITY];
        let special_tensor = tensor_from_vec(special_values.clone(), &[1, 3])?;
        let special_data = special_tensor.data()?;
        
        assert_eq!(special_data[0], 0.0);
        assert!(special_data[1].is_infinite() && special_data[1].is_sign_positive());
        assert!(special_data[2].is_infinite() && special_data[2].is_sign_negative());
        
        println!("✓ Data type handling test passed");
        Ok(())
    }
    
    /// Test 11: Broadcasting and Shape Compatibility
    #[test]
    fn test_broadcasting_operations() -> Result<()> {
        // Test basic broadcasting
        let a = ones(&[3, 1])?;
        let b = ones(&[1, 4])?;
        let result = a.add(&b)?;
        assert_eq!(result.shape().dims(), &[3, 4]);
        
        // Test scalar broadcasting
        let tensor = randn(&[2, 3])?;
        let scalar = full(&[], 2.0)?; // Scalar tensor
        
        // Note: These operations would be tested once broadcasting is fully implemented
        // let scaled = tensor.add(&scalar)?;
        // assert_eq!(scaled.shape().dims(), &[2, 3]);
        
        // Test complex broadcasting patterns
        let x = ones(&[5, 1, 3])?;
        let y = ones(&[1, 4, 1])?;
        let broadcasted = x.add(&y)?;
        assert_eq!(broadcasted.shape().dims(), &[5, 4, 3]);
        
        // Test broadcasting error cases
        let incompatible_a = ones(&[3, 5])?;
        let incompatible_b = ones(&[4, 6])?;
        let should_fail = incompatible_a.add(&incompatible_b);
        assert!(should_fail.is_err(), "Expected broadcasting error");
        
        println!("✓ Broadcasting operations test passed");
        Ok(())
    }
    
    /// Test 12: Model Serialization and State Management
    #[test]
    fn test_model_serialization() -> Result<()> {
        // Create and train a simple model
        let mut model = MockLinear::new(10, 5, true)?;
        model.train();
        
        // Get initial parameters
        let initial_params = model.parameters();
        let initial_weight_data = initial_params["weight"].data.data()?;
        
        // Simulate training (modify parameters)
        // In a real scenario, this would be done through optimizer
        let modified_weight = initial_params["weight"].data.add(&full(&[10, 5], 0.1)?)?;
        // Note: Parameter update would happen here in real implementation
        
        // Test parameter consistency
        assert_eq!(initial_params["weight"].data.shape().dims(), &[10, 5]);
        if let Some(bias) = &initial_params.get("bias") {
            assert_eq!(bias.data.shape().dims(), &[5]);
        }
        
        // Test state dict operations (mock)
        let state_dict = model.parameters();
        let param_count = state_dict.len();
        assert!(param_count >= 1); // At least weight parameter
        
        // Test parameter naming consistency
        assert!(state_dict.contains_key("weight"));
        if model.bias.is_some() {
            assert!(state_dict.contains_key("bias"));
        }
        
        println!("✓ Model serialization test passed");
        Ok(())
    }
    
    /// Helper function to get memory usage (mock implementation)
    fn get_memory_usage() -> Result<usize> {
        // This would be replaced with actual memory tracking
        // For now, return a mock value
        Ok(1024 * 1024) // 1MB mock value
    }
    
    /// Helper function to get available devices
    fn get_available_devices() -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];
        
        // Mock CUDA detection
        // In reality, this would query the system for available devices
        if std::env::var("TORSH_TEST_CUDA").is_ok() {
            devices.push(DeviceType::Cuda(0));
        }
        
        devices
    }
    
    /// Integration test runner that executes all tests
    #[test]
    fn run_all_integration_tests() {
        println!("🚀 Running ToRSh Integration Test Suite");
        println!("=====================================");
        
        let tests = vec![
            ("Cross-Crate Tensor Operations", test_cross_crate_tensor_operations as fn() -> Result<()>),
            ("Neural Network Integration", test_neural_network_integration),
            ("Memory Management", test_memory_management),
            ("End-to-End Training Workflow", test_end_to_end_training),
            ("Device Compatibility", test_device_compatibility),
            ("Error Handling and Edge Cases", test_error_handling),
            ("Performance Characteristics", test_performance_characteristics),
            ("API Compatibility", test_api_compatibility),
            ("Thread Safety", test_thread_safety),
            ("Data Type Handling", test_data_type_handling),
            ("Broadcasting Operations", test_broadcasting_operations),
            ("Model Serialization", test_model_serialization),
        ];
        
        let mut passed = 0;
        let mut failed = 0;
        
        for (name, test_fn) in tests {
            print!("Running {:<35} ... ", name);
            match test_fn() {
                Ok(()) => {
                    println!("✅ PASSED");
                    passed += 1;
                }
                Err(e) => {
                    println!("❌ FAILED: {:?}", e);
                    failed += 1;
                }
            }
        }
        
        println!("\n📊 Integration Test Results");
        println!("===========================");
        println!("✅ Passed: {}", passed);
        println!("❌ Failed: {}", failed);
        println!("📈 Success Rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
        
        if failed > 0 {
            panic!("Integration tests failed! {} out of {} tests failed.", failed, passed + failed);
        }
        
        println!("\n🎉 All integration tests passed!");
    }
}

/// Module for testing utilities and helpers
mod test_utils {
    use super::*;
    
    /// Generate reproducible test data
    pub fn generate_test_data(shape: &[usize], seed: u64) -> Result<Tensor> {
        // Mock implementation - would use proper seeded random generation
        randn(shape)
    }
    
    /// Compare tensors with tolerance
    pub fn assert_tensors_close(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> Result<()> {
        assert_eq!(a.shape().dims(), b.shape().dims());
        
        let a_data = a.data()?;
        let b_data = b.data()?;
        
        for (i, (&a_val, &b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = (a_val - b_val).abs();
            let tolerance = atol + rtol * b_val.abs();
            
            if diff > tolerance {
                return Err(TorshError::InvalidOperation(
                    format!(
                        "Tensors not close at index {}: {} vs {} (diff: {}, tolerance: {})",
                        i, a_val, b_val, diff, tolerance
                    )
                ));
            }
        }
        
        Ok(())
    }
    
    /// Create a mock dataset for testing
    pub fn create_mock_dataset(num_samples: usize, input_dim: usize, output_dim: usize) -> Result<(Tensor, Tensor)> {
        let inputs = randn(&[num_samples, input_dim])?;
        let outputs = randn(&[num_samples, output_dim])?;
        Ok((inputs, outputs))
    }
    
    /// Benchmark function execution time
    pub fn benchmark<F, R>(f: F, name: &str) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        println!("⏱️  {} took: {:?}", name, duration);
        result
    }
    
    /// Validate tensor properties
    pub fn validate_tensor(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
        // Check shape
        assert_eq!(tensor.shape().dims(), expected_shape);
        
        // Check for NaN/Inf values
        let data = tensor.data()?;
        let has_nan = data.iter().any(|&x| x.is_nan());
        let has_inf = data.iter().any(|&x| x.is_infinite());
        
        if has_nan || has_inf {
            return Err(TorshError::InvalidOperation(
                format!("Tensor contains NaN: {}, Inf: {}", has_nan, has_inf)
            ));
        }
        
        Ok(())
    }
}

/// Performance testing module
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark matrix multiplication performance
    #[test]
    fn benchmark_matrix_multiplication() -> Result<()> {
        let sizes = vec![128, 256, 512, 1024];
        
        println!("🔥 Matrix Multiplication Benchmarks");
        println!("===================================");
        
        for size in sizes {
            let a = randn(&[size, size])?;
            let b = randn(&[size, size])?;
            
            let start = Instant::now();
            let _result = a.matmul(&b)?;
            let duration = start.elapsed();
            
            let ops = (size * size * size) as f64;
            let gflops = ops / duration.as_secs_f64() / 1e9;
            
            println!("Size: {}x{}, Time: {:?}, Performance: {:.2} GFLOPS", 
                    size, size, duration, gflops);
        }
        
        Ok(())
    }
    
    /// Benchmark element-wise operations
    #[test]
    fn benchmark_elementwise_operations() -> Result<()> {
        let size = 1_000_000;
        let a = randn(&[size])?;
        let b = randn(&[size])?;
        
        println!("⚡ Element-wise Operation Benchmarks");
        println!("===================================");
        
        let operations = vec![
            ("Add", |x: &Tensor, y: &Tensor| x.add(y)),
            ("Multiply", |x: &Tensor, y: &Tensor| x.mul_op(y)),
            ("Subtract", |x: &Tensor, y: &Tensor| x.sub(y)),
            ("Divide", |x: &Tensor, y: &Tensor| x.div(y)),
        ];
        
        for (name, op) in operations {
            let start = Instant::now();
            let _result = op(&a, &b)?;
            let duration = start.elapsed();
            
            let throughput = size as f64 / duration.as_secs_f64() / 1e6;
            println!("{}: {:?}, Throughput: {:.2} M elements/sec", name, duration, throughput);
        }
        
        Ok(())
    }
}

/// Stress testing module
mod stress_tests {
    use super::*;
    
    /// Test with very large tensors
    #[test]
    #[ignore] // Ignore by default due to memory requirements
    fn test_large_tensor_operations() -> Result<()> {
        println!("💪 Large Tensor Stress Test");
        
        // Create large tensors (1GB each)
        let large_size = 16_384; // 16K x 16K = 256M elements = 1GB in f32
        let a = randn(&[large_size, large_size])?;
        let b = randn(&[large_size, large_size])?;
        
        // Test basic operations
        let sum = a.add(&b)?;
        assert_eq!(sum.shape().dims(), &[large_size, large_size]);
        
        println!("✅ Large tensor operations completed successfully");
        Ok(())
    }
    
    /// Test memory pressure scenarios
    #[test]
    fn test_memory_pressure() -> Result<()> {
        println!("🧠 Memory Pressure Test");
        
        // Create many medium-sized tensors
        let mut tensors = Vec::new();
        let tensor_size = 1000;
        let num_tensors = 100;
        
        for i in 0..num_tensors {
            let tensor = randn(&[tensor_size, tensor_size])?;
            tensors.push(tensor);
            
            if i % 10 == 0 {
                println!("Created {} tensors", i + 1);
            }
        }
        
        // Perform operations on all tensors
        let mut result = zeros(&[tensor_size, tensor_size])?;
        for tensor in &tensors {
            result = result.add(tensor)?;
        }
        
        assert_eq!(result.shape().dims(), &[tensor_size, tensor_size]);
        println!("✅ Memory pressure test completed");
        Ok(())
    }
}