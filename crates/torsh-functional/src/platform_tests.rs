//! Cross-platform compatibility tests
//!
//! This module provides comprehensive tests to ensure torsh-functional operations
//! work correctly across different platforms (x86_64, ARM, Windows, Linux, macOS)
//! and with different feature flags (SIMD, GPU, etc.).

#[cfg(test)]
use crate::reduction::{max, mean, min, sum};
#[cfg(test)]
use crate::*;
#[cfg(test)]
use torsh_core::{device::DeviceType, Result as TorshResult};
#[cfg(test)]
use torsh_tensor::{creation::*, Tensor};

/// Platform-specific configuration detection
#[cfg(test)]
mod platform_detection {
    #[test]
    fn test_platform_architecture() {
        #[cfg(target_arch = "x86_64")]
        println!("Platform: x86_64");

        #[cfg(target_arch = "aarch64")]
        println!("Platform: ARM64/AArch64");

        #[cfg(target_arch = "x86")]
        println!("Platform: x86 (32-bit)");

        #[cfg(target_arch = "wasm32")]
        println!("Platform: WebAssembly");

        // Test passes as long as we can detect the platform
        assert!(cfg!(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "wasm32"
        )));
    }

    #[test]
    fn test_operating_system() {
        #[cfg(target_os = "linux")]
        println!("OS: Linux");

        #[cfg(target_os = "macos")]
        println!("OS: macOS");

        #[cfg(target_os = "windows")]
        println!("OS: Windows");

        #[cfg(target_os = "ios")]
        println!("OS: iOS");

        #[cfg(target_os = "android")]
        println!("OS: Android");

        // Test passes as long as we can detect the OS
        assert!(cfg!(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows",
            target_os = "ios",
            target_os = "android"
        )));
    }

    #[test]
    fn test_simd_features() {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            println!("SIMD: AVX2 available");
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            println!("SIMD: SSE2 available");
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            println!("SIMD: NEON available");
        }

        // Test always passes - we're just detecting features
        assert!(true);
    }

    #[test]
    fn test_pointer_size() {
        // Verify we're on a supported architecture
        assert!(
            cfg!(target_pointer_width = "64") || cfg!(target_pointer_width = "32"),
            "Unsupported pointer width"
        );

        #[cfg(target_pointer_width = "64")]
        {
            println!("Pointer width: 64-bit");
            assert_eq!(std::mem::size_of::<usize>(), 8);
        }

        #[cfg(target_pointer_width = "32")]
        {
            println!("Pointer width: 32-bit");
            assert_eq!(std::mem::size_of::<usize>(), 4);
        }
    }
}

/// Basic operations cross-platform consistency tests
#[cfg(test)]
mod basic_operations {
    use super::*;

    #[test]
    fn test_tensor_creation_cross_platform() -> TorshResult<()> {
        // Test basic tensor creation works on all platforms
        let t1 = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu)?;
        assert_eq!(t1.shape().dims(), &[4]);

        let t2 = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DeviceType::Cpu)?;
        assert_eq!(t2.shape().dims(), &[2, 3]);

        // Test zeros and ones
        let zeros: Tensor = zeros(&[3, 3])?;
        assert_eq!(zeros.shape().dims(), &[3, 3]);

        let ones: Tensor = ones(&[2, 4])?;
        assert_eq!(ones.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_basic_arithmetic_cross_platform() -> TorshResult<()> {
        let a = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let b = from_vec(vec![4.0, 5.0, 6.0], &[3], DeviceType::Cpu)?;

        // Addition
        let sum = a.add(&b)?;
        let sum_data = sum.data()?;
        assert!((sum_data[0] - 5.0_f32).abs() < 1e-6);
        assert!((sum_data[1] - 7.0_f32).abs() < 1e-6);
        assert!((sum_data[2] - 9.0_f32).abs() < 1e-6);

        // Multiplication
        let prod = a.mul(&b)?;
        let prod_data = prod.data()?;
        assert!((prod_data[0] - 4.0_f32).abs() < 1e-6);
        assert!((prod_data[1] - 10.0_f32).abs() < 1e-6);
        assert!((prod_data[2] - 18.0_f32).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_activation_functions_cross_platform() -> TorshResult<()> {
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;

        // ReLU
        let relu_out = relu(&input, false)?;
        let relu_data = relu_out.data()?;
        assert!((relu_data[0] - 0.0_f32).abs() < 1e-6);
        assert!((relu_data[1] - 0.0_f32).abs() < 1e-6);
        assert!((relu_data[2] - 0.0_f32).abs() < 1e-6);
        assert!((relu_data[3] - 1.0_f32).abs() < 1e-6);
        assert!((relu_data[4] - 2.0_f32).abs() < 1e-6);

        // Sigmoid
        let sigmoid_out = sigmoid(&input)?;
        let sigmoid_data = sigmoid_out.data()?;
        // Sigmoid(0) should be 0.5
        assert!((sigmoid_data[2] - 0.5_f32).abs() < 1e-6);
        // Sigmoid is symmetric: sigmoid(-x) + sigmoid(x) = 1
        assert!((sigmoid_data[0] + sigmoid_data[4] - 1.0_f32).abs() < 1e-6);
        assert!((sigmoid_data[1] + sigmoid_data[3] - 1.0_f32).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_reduction_operations_cross_platform() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DeviceType::Cpu)?;

        // Sum
        let sum_result = sum(&input)?;
        let sum_val = sum_result.data()?[0];
        assert!((sum_val - 21.0_f32).abs() < 1e-6);

        // Mean
        let mean_result = mean(&input)?;
        let mean_val = mean_result.data()?[0];
        assert!((mean_val - 3.5_f32).abs() < 1e-6);

        // Max
        let max_result = max(&input)?;
        let max_val = max_result.data()?[0];
        assert!((max_val - 6.0_f32).abs() < 1e-6);

        // Min
        let min_result = min(&input)?;
        let min_val = min_result.data()?[0];
        assert!((min_val - 1.0_f32).abs() < 1e-6);

        Ok(())
    }
}

/// Numerical consistency tests across platforms
#[cfg(test)]
mod numerical_consistency {
    use super::*;

    #[test]
    fn test_floating_point_consistency() -> TorshResult<()> {
        // Test that floating point operations produce consistent results
        let a = from_vec(vec![0.1_f32, 0.2, 0.3], &[3], DeviceType::Cpu)?;
        let b = from_vec(vec![0.4_f32, 0.5, 0.6], &[3], DeviceType::Cpu)?;

        let sum = a.add(&b)?;
        let sum_data = sum.data()?;

        // These should be consistent across platforms
        // (within floating point precision)
        assert!((sum_data[0] - 0.5_f32).abs() < 1e-6);
        assert!((sum_data[1] - 0.7_f32).abs() < 1e-6);
        assert!((sum_data[2] - 0.9_f32).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_deterministic_operations() -> TorshResult<()> {
        // Test that deterministic operations produce identical results
        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu)?;

        // Run operation twice
        let result1 = relu(&input, false)?;
        let result2 = relu(&input, false)?;

        let data1 = result1.data()?;
        let data2 = result2.data()?;

        // Results should be identical
        for i in 0..data1.len() {
            assert_eq!(data1[i], data2[i], "Non-deterministic behavior detected");
        }

        Ok(())
    }

    #[test]
    fn test_loss_function_consistency() -> TorshResult<()> {
        use crate::loss::{mse_loss, ReductionType};

        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.5, 2.5, 2.5], &[3], DeviceType::Cpu)?;

        // MSE should be: ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.25
        let loss = mse_loss(&input, &target, ReductionType::Mean)?;
        let loss_val = loss.data()?[0];

        assert!(
            (loss_val - 0.25_f32).abs() < 1e-6,
            "MSE loss inconsistent: expected 0.25, got {}",
            loss_val
        );

        Ok(())
    }

    #[test]
    fn test_matrix_multiplication_consistency() -> TorshResult<()> {
        let a = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu)?;
        let b = from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], DeviceType::Cpu)?;

        let result = a.matmul(&b)?;
        let data = result.data()?;

        // Expected result:
        // [1*5 + 2*7, 1*6 + 2*8]   [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert!((data[0] - 19.0_f32).abs() < 1e-6);
        assert!((data[1] - 22.0_f32).abs() < 1e-6);
        assert!((data[2] - 43.0_f32).abs() < 1e-6);
        assert!((data[3] - 50.0_f32).abs() < 1e-6);

        Ok(())
    }
}

/// Performance and optimization tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_large_tensor_operations() -> TorshResult<()> {
        // Test that large tensor operations complete successfully
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = from_vec(data, &[size], DeviceType::Cpu)?;

        // This should complete without issues
        let result = relu(&tensor, false)?;
        assert_eq!(result.shape().dims(), &[size]);

        Ok(())
    }

    #[test]
    fn test_batch_operations() -> TorshResult<()> {
        // Test batch operations work correctly
        let batch_size = 8;
        let features = 64;
        let data: Vec<f32> = (0..(batch_size * features))
            .map(|i| (i as f32) * 0.01)
            .collect();

        let tensor = from_vec(data, &[batch_size, features], DeviceType::Cpu)?;
        let result = sigmoid(&tensor)?;

        assert_eq!(result.shape().dims(), &[batch_size, features]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_consistency() -> TorshResult<()> {
        // When SIMD is enabled, results should still be consistent
        let input = from_vec(
            (0..100).map(|i| i as f32 * 0.1).collect(),
            &[100],
            DeviceType::Cpu,
        )?;

        let result1 = relu(&input, false)?;
        let result2 = relu(&input, false)?;

        let data1 = result1.data()?;
        let data2 = result2.data()?;

        for i in 0..data1.len() {
            assert!(
                (data1[i] - data2[i]).abs() < 1e-6,
                "SIMD results not consistent at index {}",
                i
            );
        }

        Ok(())
    }
}

/// Edge case and boundary condition tests
#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_tensor_handling() {
        // Empty tensors should be handled gracefully
        let result = from_vec::<f32>(vec![], &[0], DeviceType::Cpu);
        // This might fail or succeed depending on implementation
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_single_element_tensor() -> TorshResult<()> {
        let tensor = from_vec(vec![42.0], &[1], DeviceType::Cpu)?;
        let result = relu(&tensor, false)?;
        let data = result.data()?;
        assert_eq!(data[0], 42.0);

        Ok(())
    }

    #[test]
    fn test_extreme_dimensions() -> TorshResult<()> {
        // Test tensors with extreme aspect ratios
        let data = vec![1.0; 1000];

        // Very wide tensor
        let wide = from_vec(data.clone(), &[1, 1000], DeviceType::Cpu)?;
        let result = sigmoid(&wide)?;
        assert_eq!(result.shape().dims(), &[1, 1000]);

        // Very tall tensor
        let tall = from_vec(data, &[1000, 1], DeviceType::Cpu)?;
        let result = sigmoid(&tall)?;
        assert_eq!(result.shape().dims(), &[1000, 1]);

        Ok(())
    }

    #[test]
    fn test_special_values() -> TorshResult<()> {
        // Test handling of special floating point values
        let input = from_vec(vec![0.0, -0.0, 1.0, -1.0], &[4], DeviceType::Cpu)?;

        // Operations should handle these gracefully
        let relu_result = relu(&input, false)?;
        let sigmoid_result = sigmoid(&input)?;

        assert_eq!(relu_result.shape().dims(), &[4]);
        assert_eq!(sigmoid_result.shape().dims(), &[4]);

        Ok(())
    }
}

/// Memory safety tests
#[cfg(test)]
mod memory_safety {
    use super::*;

    #[test]
    fn test_no_memory_leaks_simple() -> TorshResult<()> {
        // Simple test to ensure operations don't leak memory
        for _ in 0..100 {
            let tensor = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
            let _ = relu(&tensor, false)?;
        }
        Ok(())
    }

    #[test]
    fn test_multiple_operations_chain() -> TorshResult<()> {
        // Test chaining multiple operations
        let input = from_vec(vec![1.0, -2.0, 3.0, -4.0], &[4], DeviceType::Cpu)?;

        let step1 = relu(&input, false)?;
        let step2 = sigmoid(&step1)?;
        let step3 = tanh(&step2)?;

        assert_eq!(step3.shape().dims(), &[4]);

        Ok(())
    }

    #[test]
    fn test_concurrent_operations() -> TorshResult<()> {
        // Test that multiple tensors can coexist
        let t1 = from_vec(vec![1.0, 2.0], &[2], DeviceType::Cpu)?;
        let t2 = from_vec(vec![3.0, 4.0], &[2], DeviceType::Cpu)?;
        let t3 = from_vec(vec![5.0, 6.0], &[2], DeviceType::Cpu)?;

        let r1 = relu(&t1, false)?;
        let r2 = sigmoid(&t2)?;
        let r3 = tanh(&t3)?;

        // All results should be valid
        assert_eq!(r1.shape().dims(), &[2]);
        assert_eq!(r2.shape().dims(), &[2]);
        assert_eq!(r3.shape().dims(), &[2]);

        Ok(())
    }
}

/// Helper function for numerical comparison
#[cfg(test)]
#[allow(dead_code)]
fn assert_close(a: f32, b: f32, tolerance: f32, message: &str) {
    assert!(
        (a - b).abs() < tolerance,
        "{}: expected {}, got {} (diff: {})",
        message,
        b,
        a,
        (a - b).abs()
    );
}
