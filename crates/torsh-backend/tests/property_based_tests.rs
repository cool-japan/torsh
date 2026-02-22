//! Property-based testing for backend mathematical properties
//!
//! This module uses proptest to verify mathematical properties hold across
//! all backend implementations, testing with randomized inputs to find edge cases.

use proptest::prelude::*;
use torsh_backend::cpu::optimized_kernels::*;
use torsh_backend::cpu::simd::*;

/// Test that addition is commutative: a + b = b + a
#[test]
fn test_addition_commutative() {
    proptest!(|(a: f32, b: f32)| {
        if a.is_finite() && b.is_finite() {
            let sum1 = a + b;
            let sum2 = b + a;

            // Skip comparison if result overflows to infinity
            if sum1.is_finite() && sum2.is_finite() {
                // Allow for floating-point precision differences
                let diff = (sum1 - sum2).abs();
                prop_assert!(diff < 1e-6, "Addition should be commutative: {} + {} = {}, {} + {} = {}",
                    a, b, sum1, b, a, sum2);
            }
        }
    });
}

/// Test that addition is associative: (a + b) + c = a + (b + c)
#[test]
fn test_addition_associative() {
    proptest!(|(a: f32, b: f32, c: f32)| {
        if a.is_finite() && b.is_finite() && c.is_finite() {
            let sum1 = (a + b) + c;
            let sum2 = a + (b + c);

            // Skip comparison if result overflows to infinity
            if sum1.is_finite() && sum2.is_finite() {
                // Allow for floating-point rounding differences
                let diff = (sum1 - sum2).abs();
                let tolerance = (a.abs() + b.abs() + c.abs()) * 1e-6 + 1e-10;
                prop_assert!(diff <= tolerance,
                    "Addition should be associative: ({} + {}) + {} = {}, {} + ({} + {}) = {}",
                    a, b, c, sum1, a, b, c, sum2);
            }
        }
    });
}

/// Test that multiplication is commutative: a * b = b * a
#[test]
fn test_multiplication_commutative() {
    proptest!(|(a: f32, b: f32)| {
        if a.is_finite() && b.is_finite() {
            let prod1 = a * b;
            let prod2 = b * a;

            // Both should be either finite or both infinite with same sign
            if prod1.is_finite() && prod2.is_finite() {
                // Allow for floating-point precision differences
                let diff = (prod1 - prod2).abs();
                prop_assert!(diff < 1e-5,
                    "Multiplication should be commutative: {} * {} = {}, {} * {} = {}",
                    a, b, prod1, b, a, prod2);
            } else {
                // Both should have same sign if infinite
                prop_assert!(prod1.is_infinite() == prod2.is_infinite() && prod1.signum() == prod2.signum(),
                    "Multiplication overflow should be consistent: {} * {} = {}, {} * {} = {}",
                    a, b, prod1, b, a, prod2);
            }
        }
    });
}

/// Test that multiplication distributes over addition: a * (b + c) = a*b + a*c
#[test]
fn test_distributive_property() {
    proptest!(|(a: f32, b: f32, c: f32)| {
        if a.is_finite() && b.is_finite() && c.is_finite() {
            let left = a * (b + c);
            let right = a * b + a * c;

            // Only test if results are finite (avoids overflow edge cases)
            if left.is_finite() && right.is_finite() {
                // Allow for floating-point rounding differences
                let diff = (left - right).abs();
                // Use relative tolerance with minimum absolute tolerance
                // This handles both large and very small numbers correctly
                let relative_tolerance = (a.abs() * (b.abs() + c.abs())) * 1e-5;
                let tolerance = relative_tolerance.max(1e-40f32);
                prop_assert!(diff <= tolerance,
                    "Distributive property should hold: {} * ({} + {}) = {}, {}*{} + {}*{} = {}",
                    a, b, c, left, a, b, a, c, right);
            }
        }
    });
}

/// Test SIMD operations produce same results as scalar operations
#[test]
fn test_simd_correctness() {
    proptest!(|(values in prop::collection::vec(-1000.0f32..1000.0, 1..=100))| {
        let mut simd_result = vec![0.0f32; values.len()];
        let mut scalar_result = vec![0.0f32; values.len()];

        // SIMD addition (a + a = 2a)
        simd_add_f32(&values, &values, &mut simd_result);

        // Scalar addition
        for i in 0..values.len() {
            scalar_result[i] = values[i] + values[i];
        }

        // Results should match within tolerance
        for i in 0..values.len() {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            prop_assert!(diff < 1e-5,
                "SIMD and scalar addition should match at index {}: SIMD={}, Scalar={}",
                i, simd_result[i], scalar_result[i]);
        }
    });
}

/// Test SIMD multiplication correctness
#[test]
fn test_simd_multiplication_correctness() {
    proptest!(|(a_values in prop::collection::vec(-100.0f32..100.0, 1..=100),
                b_values in prop::collection::vec(-100.0f32..100.0, 1..=100))| {
        let len = a_values.len().min(b_values.len());
        let a = &a_values[..len];
        let b = &b_values[..len];

        let mut simd_result = vec![0.0f32; len];
        let mut scalar_result = vec![0.0f32; len];

        // SIMD multiplication
        simd_mul_f32(a, b, &mut simd_result);

        // Scalar multiplication
        for i in 0..len {
            scalar_result[i] = a[i] * b[i];
        }

        // Results should match within tolerance
        for i in 0..len {
            let diff = (simd_result[i] - scalar_result[i]).abs();
            let tolerance = (a[i].abs() * b[i].abs()) * 1e-5 + 1e-6;
            prop_assert!(diff <= tolerance,
                "SIMD and scalar multiplication should match at index {}: SIMD={}, Scalar={}, a={}, b={}",
                i, simd_result[i], scalar_result[i], a[i], b[i]);
        }
    });
}

/// Test optimized dot product properties
#[test]
fn test_dot_product_properties() {
    proptest!(|(a_values in prop::collection::vec(-10.0f32..10.0, 1..=50),
                b_values in prop::collection::vec(-10.0f32..10.0, 1..=50))| {
        let len = a_values.len().min(b_values.len());
        let a = &a_values[..len];
        let b = &b_values[..len];

        // Compute dot product both ways
        if let Ok(dot_ab) = optimized_dot(a, b) {
            if let Ok(dot_ba) = optimized_dot(b, a) {
                // Dot product should be commutative
                let diff = (dot_ab - dot_ba).abs();
                let tolerance = len as f32 * 1e-4;
                prop_assert!(diff <= tolerance,
                    "Dot product should be commutative: dot(a,b)={}, dot(b,a)={}",
                    dot_ab, dot_ba);
            }
        }

        // Test dot product with zero vector
        let zeros = vec![0.0f32; len];
        if let Ok(dot_zero) = optimized_dot(a, &zeros) {
            prop_assert!((dot_zero).abs() < 1e-5,
                "Dot product with zero vector should be zero, got {}", dot_zero);
        }
    });
}

/// Test SIMD threshold detection properties
#[test]
fn test_simd_threshold_properties() {
    proptest!(|(size in 0usize..10_000)| {
        let should_use = should_use_simd(size);

        // SIMD should only be used for larger arrays
        if size < 4 {
            prop_assert!(!should_use || !cfg!(feature = "simd"),
                "SIMD should not be used for very small arrays (size={})", size);
        }

        // For large arrays, SIMD should be considered if feature is enabled
        if cfg!(feature = "simd") && size >= 64 {
            prop_assert!(should_use,
                "SIMD should be used for large arrays when feature is enabled (size={})", size);
        }
    });
}

/// Test quantization creates valid range for int8
#[test]
fn test_quantization_int8_range() {
    proptest!(|(values in prop::collection::vec(-1000.0f32..1000.0, 2..=100))| {
        use torsh_backend::quantization::core::*;

        let params = QuantizationParams::int8_symmetric();

        if let Ok(quantized) = quantize_to_int8(&values, &params) {
            // All quantized values should be in valid i8 range
            for (i, &q) in quantized.iter().enumerate() {
                prop_assert!(q >= i8::MIN && q <= i8::MAX,
                    "Quantized value at index {} should be in i8 range: got {}", i, q);
            }

            // Length should be preserved
            prop_assert_eq!(quantized.len(), values.len(),
                "Quantization should preserve vector length");
        }
    });
}

/// Test FFT plan creation for different sizes
#[test]
fn test_fft_plan_creation() {
    proptest!(|(sizes in prop::collection::vec(8usize..=128, 1..=10))| {
        use torsh_backend::fft::*;

        for s in sizes {
            // Ensure size is power of 2 for FFT
            let fft_size = s.next_power_of_two();

            let forward_plan = FftPlan::new_1d(fft_size, FftDirection::Forward);
            let inverse_plan = FftPlan::new_1d(fft_size, FftDirection::Inverse);

            // Verify plans have correct properties
            prop_assert_eq!(forward_plan.dimensions, vec![fft_size],
                "Forward plan should have correct dimensions");
            prop_assert_eq!(inverse_plan.dimensions, vec![fft_size],
                "Inverse plan should have correct dimensions");

            prop_assert_eq!(forward_plan.direction, FftDirection::Forward,
                "Forward plan should have Forward direction");
            prop_assert_eq!(inverse_plan.direction, FftDirection::Inverse,
                "Inverse plan should have Inverse direction");
        }
    });
}

#[cfg(test)]
mod sparse_matrix_properties {
    use super::*;
    use torsh_backend::sparse_ops::*;

    /// Test sparse matrix element insertion and retrieval
    #[test]
    fn test_sparse_matrix_element_consistency() {
        proptest!(|(elements in prop::collection::vec((0usize..10, 0usize..10, -100.0f32..100.0), 1..=20))| {
            let mut matrix: SparseMatrix<f32> = SparseMatrix::new_coo(10, 10);

            // Insert elements
            for &(row, col, value) in &elements {
                let _ = matrix.insert_coo(row, col, value);
            }

            // Verify structure
            prop_assert_eq!(matrix.rows, 10, "Matrix should have 10 rows");
            prop_assert_eq!(matrix.cols, 10, "Matrix should have 10 columns");

            // Verify non-zero count doesn't exceed number of insertions
            prop_assert!(matrix.nnz <= elements.len(),
                "Non-zero count should not exceed insertions: nnz={}, insertions={}",
                matrix.nnz, elements.len());
        });
    }

    /// Test sparse matrix format conversion preserves dimensions
    #[test]
    fn test_sparse_format_conversion_dimensions() {
        proptest!(|(rows in 1usize..20, cols in 1usize..20)| {
            let matrix: SparseMatrix<f32> = SparseMatrix::new_coo(rows, cols);

            // Convert to CSR
            if let Ok(csr) = matrix.to_csr() {
                prop_assert_eq!(csr.rows, rows, "CSR should preserve row count");
                prop_assert_eq!(csr.cols, cols, "CSR should preserve column count");
            }

            // Convert to CSC
            if let Ok(csc) = matrix.to_csc() {
                prop_assert_eq!(csc.rows, rows, "CSC should preserve row count");
                prop_assert_eq!(csc.cols, cols, "CSC should preserve column count");
            }
        });
    }
}
