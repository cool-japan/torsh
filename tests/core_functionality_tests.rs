//! Core functionality tests for ToRSh framework
//!
//! This test suite focuses on testing the core tensor operations
//! and basic functionality without depending on neural network components.

#[cfg(test)]
mod core_tests {
    use torsh_core::{
        device::DeviceType,
        dtype::DType,
        error::Result,
        shape::Shape,
    };
    
    use torsh_tensor::{
        creation::*,
        Tensor,
    };
    
    /// Test 1: Basic Tensor Creation and Properties
    #[test]
    fn test_tensor_creation() -> Result<()> {
        // Test zeros creation
        let zeros_tensor = zeros(&[2, 3])?;
        assert_eq!(zeros_tensor.shape().dims(), &[2, 3]);
        assert_eq!(zeros_tensor.dtype(), DType::F32);
        assert_eq!(zeros_tensor.device(), DeviceType::Cpu);
        assert_eq!(zeros_tensor.numel(), 6);
        
        // Test ones creation
        let ones_tensor = ones(&[3, 2])?;
        assert_eq!(ones_tensor.shape().dims(), &[3, 2]);
        assert_eq!(ones_tensor.numel(), 6);
        
        // Test full creation
        let full_tensor = full(&[2, 2], 5.0)?;
        assert_eq!(full_tensor.shape().dims(), &[2, 2]);
        
        // Verify data
        let data = full_tensor.data()?;
        for &value in &data {
            assert_eq!(value, 5.0);
        }
        
        println!("✓ Tensor creation test passed");
        Ok(())
    }
    
    /// Test 2: Tensor Operations
    #[test]
    fn test_tensor_operations() -> Result<()> {
        let a = full(&[2, 2], 2.0)?;
        let b = full(&[2, 2], 3.0)?;
        
        // Test addition
        let sum = a.add(&b)?;
        let sum_data = sum.data()?;
        for &value in &sum_data {
            assert_eq!(value, 5.0);
        }
        
        // Test multiplication
        let product = a.mul_op(&b)?;
        let product_data = product.data()?;
        for &value in &product_data {
            assert_eq!(value, 6.0);
        }
        
        // Test subtraction
        let diff = b.sub(&a)?;
        let diff_data = diff.data()?;
        for &value in &diff_data {
            assert_eq!(value, 1.0);
        }
        
        println!("✓ Tensor operations test passed");
        Ok(())
    }
    
    /// Test 3: Shape Operations
    #[test]
    fn test_shape_operations() -> Result<()> {
        let tensor = ones(&[2, 3])?;
        
        // Test transpose
        let transposed = tensor.transpose(0, 1)?;
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        
        // Test reshape
        let reshaped = tensor.reshape(&[3, 2])?;
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        
        // Test that data is preserved
        let original_data = tensor.data()?;
        let reshaped_data = reshaped.data()?;
        assert_eq!(original_data, reshaped_data);
        
        println!("✓ Shape operations test passed");
        Ok(())
    }
    
    /// Test 4: Matrix Multiplication
    #[test]
    fn test_matrix_multiplication() -> Result<()> {
        // Create test matrices
        let a = tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = tensor_from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        
        // Perform matrix multiplication
        let result = a.matmul(&b)?;
        assert_eq!(result.shape().dims(), &[2, 2]);
        
        // Verify result
        let result_data = result.data()?;
        // Expected: [19, 22, 43, 50]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        
        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            let diff = (actual - expected).abs();
            assert!(diff < 1e-6, "Expected {}, got {}", expected, actual);
        }
        
        println!("✓ Matrix multiplication test passed");
        Ok(())
    }
    
    /// Test 5: Broadcasting
    #[test]
    fn test_broadcasting() -> Result<()> {
        let a = ones(&[3, 1])?;
        let b = full(&[1, 4], 2.0)?;
        
        // Test broadcasting addition
        let result = a.add(&b)?;
        assert_eq!(result.shape().dims(), &[3, 4]);
        
        // Verify all values are 3.0 (1.0 + 2.0)
        let result_data = result.data()?;
        for &value in &result_data {
            assert_eq!(value, 3.0);
        }
        
        println!("✓ Broadcasting test passed");
        Ok(())
    }
    
    /// Test 6: Error Handling
    #[test]
    fn test_error_handling() -> Result<()> {
        // Test shape mismatch in matrix multiplication
        let a = ones(&[2, 3])?;
        let b = ones(&[4, 5])?;
        
        let result = a.matmul(&b);
        assert!(result.is_err(), "Expected shape mismatch error");
        
        // Test invalid reshape
        let tensor = ones(&[2, 3])?;
        let invalid_reshape = tensor.reshape(&[2, 4]);
        assert!(invalid_reshape.is_err(), "Expected invalid reshape error");
        
        println!("✓ Error handling test passed");
        Ok(())
    }
    
    /// Test 7: Data Type Consistency
    #[test]
    fn test_data_type_consistency() -> Result<()> {
        let tensor = randn(&[5, 5])?;
        assert_eq!(tensor.dtype(), DType::F32);
        
        // Test that operations preserve data type
        let doubled = tensor.mul_scalar(2.0)?;
        assert_eq!(doubled.dtype(), DType::F32);
        
        let sum = tensor.add(&doubled)?;
        assert_eq!(sum.dtype(), DType::F32);
        
        println!("✓ Data type consistency test passed");
        Ok(())
    }
    
    /// Test 8: Memory and Performance
    #[test]
    fn test_memory_performance() -> Result<()> {
        use std::time::Instant;
        
        // Create reasonably sized tensors
        let size = 100;
        let a = randn(&[size, size])?;
        let b = randn(&[size, size])?;
        
        // Time matrix multiplication
        let start = Instant::now();
        let result = a.matmul(&b)?;
        let duration = start.elapsed();
        
        // Verify result
        assert_eq!(result.shape().dims(), &[size, size]);
        
        // Performance should be reasonable (less than 1 second for 100x100)
        assert!(duration.as_secs() < 1, "Matrix multiplication took too long: {:?}", duration);
        
        println!("✓ Memory and performance test passed ({}x{} in {:?})", size, size, duration);
        Ok(())
    }
    
    /// Test 9: Large Tensor Operations
    #[test]
    fn test_large_tensor_operations() -> Result<()> {
        // Test with larger tensors to ensure scalability
        let size = 1000;
        let a = zeros(&[size])?;
        let b = ones(&[size])?;
        
        // Test operations
        let sum = a.add(&b)?;
        let product = a.mul_op(&b)?;
        
        // Verify results
        assert_eq!(sum.numel(), size);
        assert_eq!(product.numel(), size);
        
        // Check some values
        let sum_data = sum.data()?;
        let product_data = product.data()?;
        
        assert_eq!(sum_data[0], 1.0);
        assert_eq!(sum_data[size - 1], 1.0);
        assert_eq!(product_data[0], 0.0);
        assert_eq!(product_data[size - 1], 0.0);
        
        println!("✓ Large tensor operations test passed");
        Ok(())
    }
    
    /// Test 10: Edge Cases
    #[test]
    fn test_edge_cases() -> Result<()> {
        // Test scalar tensors
        let scalar = full(&[], 42.0)?;
        assert_eq!(scalar.numel(), 1);
        assert_eq!(scalar.shape().dims(), &[]);
        
        let scalar_data = scalar.data()?;
        assert_eq!(scalar_data[0], 42.0);
        
        // Test 1D tensors
        let vector = ones(&[5])?;
        assert_eq!(vector.numel(), 5);
        assert_eq!(vector.shape().dims(), &[5]);
        
        // Test high-dimensional tensors
        let high_dim = zeros(&[2, 3, 4, 5])?;
        assert_eq!(high_dim.numel(), 120);
        assert_eq!(high_dim.shape().dims(), &[2, 3, 4, 5]);
        
        println!("✓ Edge cases test passed");
        Ok(())
    }
    
    /// Integration test runner
    #[test]
    fn run_all_core_tests() {
        println!("🧪 Running ToRSh Core Functionality Tests");
        println!("==========================================");
        
        let tests = vec![
            ("Tensor Creation", test_tensor_creation as fn() -> Result<()>),
            ("Tensor Operations", test_tensor_operations),
            ("Shape Operations", test_shape_operations),
            ("Matrix Multiplication", test_matrix_multiplication),
            ("Broadcasting", test_broadcasting),
            ("Error Handling", test_error_handling),
            ("Data Type Consistency", test_data_type_consistency),
            ("Memory and Performance", test_memory_performance),
            ("Large Tensor Operations", test_large_tensor_operations),
            ("Edge Cases", test_edge_cases),
        ];
        
        let mut passed = 0;
        let mut failed = 0;
        
        for (name, test_fn) in tests {
            print!("Running {:<25} ... ", name);
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
        
        println!("\n📊 Core Test Results");
        println!("====================");
        println!("✅ Passed: {}", passed);
        println!("❌ Failed: {}", failed);
        println!("📈 Success Rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
        
        if failed == 0 {
            println!("\n🎉 All core functionality tests passed!");
        } else {
            println!("\n⚠️  Some tests failed. Please review the errors above.");
        }
    }
}

/// Utility functions for testing
mod test_utilities {
    use super::*;
    
    /// Compare floating point values with tolerance
    pub fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }
    
    /// Validate tensor shape and basic properties
    pub fn validate_tensor_basic(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
        assert_eq!(tensor.shape().dims(), expected_shape);
        assert_eq!(tensor.numel(), expected_shape.iter().product::<usize>());
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.device(), DeviceType::Cpu);
        Ok(())
    }
    
    /// Create a test tensor with known values
    pub fn create_test_tensor(shape: &[usize], value: f32) -> Result<Tensor> {
        full(shape, value)
    }
    
    /// Check if tensor contains any NaN or infinite values
    pub fn check_tensor_validity(tensor: &Tensor) -> Result<bool> {
        let data = tensor.data()?;
        let is_valid = !data.iter().any(|&x| x.is_nan() || x.is_infinite());
        Ok(is_valid)
    }
}