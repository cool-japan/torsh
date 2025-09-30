//! SIMD specialized operations
//!
//! This module provides SIMD-accelerated specialized operations including
//! matrix multiplication, complex numbers, quantized operations, and adaptive SIMD.

use crate::cpu::simd::should_use_simd;
#[cfg(feature = "simd")]
use wide::f32x8;

/// SIMD-accelerated matrix multiplication for f32 (basic row-major layout)
#[cfg(feature = "simd")]
pub fn simd_matmul_f32(a: &[f32], b: &[f32], result: &mut [f32], m: usize, n: usize, k: usize) {
    // A is m x k, B is k x n, result is m x n
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(result.len(), m * n);

    // Initialize result to zero
    result.fill(0.0);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let simd_len = k / 8;
            let remainder_start = simd_len * 8;

            // SIMD part
            let mut sum_simd = f32x8::splat(0.0);
            for l_chunk in 0..simd_len {
                let l = l_chunk * 8;
                let a_base = i * k + l;
                let b_base = l * n + j;

                let a_simd = f32x8::from([
                    a[a_base],
                    a[a_base + 1],
                    a[a_base + 2],
                    a[a_base + 3],
                    a[a_base + 4],
                    a[a_base + 5],
                    a[a_base + 6],
                    a[a_base + 7],
                ]);

                let b_simd = f32x8::from([
                    b[b_base],
                    b[b_base + n],
                    b[b_base + 2 * n],
                    b[b_base + 3 * n],
                    b[b_base + 4 * n],
                    b[b_base + 5 * n],
                    b[b_base + 6 * n],
                    b[b_base + 7 * n],
                ]);

                sum_simd += a_simd * b_simd;
            }

            // Sum up SIMD results
            let sum_array: [f32; 8] = sum_simd.into();
            sum += sum_array.iter().sum::<f32>();

            // Handle remaining elements
            for l in remainder_start..k {
                sum += a[i * k + l] * b[l * n + j];
            }

            result[i * n + j] = sum;
        }
    }
}

/// SIMD-accelerated complex number addition for f32
#[cfg(feature = "simd")]
pub fn simd_complex_add_f32(
    a_real: &[f32],
    a_imag: &[f32],
    b_real: &[f32],
    b_imag: &[f32],
    result_real: &mut [f32],
    result_imag: &mut [f32],
) {
    let len = a_real
        .len()
        .min(a_imag.len())
        .min(b_real.len())
        .min(b_imag.len())
        .min(result_real.len())
        .min(result_imag.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    for i in 0..simd_len {
        let idx = i * 8;

        // Load real parts
        let a_real_simd = f32x8::from([
            a_real[idx],
            a_real[idx + 1],
            a_real[idx + 2],
            a_real[idx + 3],
            a_real[idx + 4],
            a_real[idx + 5],
            a_real[idx + 6],
            a_real[idx + 7],
        ]);
        let b_real_simd = f32x8::from([
            b_real[idx],
            b_real[idx + 1],
            b_real[idx + 2],
            b_real[idx + 3],
            b_real[idx + 4],
            b_real[idx + 5],
            b_real[idx + 6],
            b_real[idx + 7],
        ]);

        // Load imaginary parts
        let a_imag_simd = f32x8::from([
            a_imag[idx],
            a_imag[idx + 1],
            a_imag[idx + 2],
            a_imag[idx + 3],
            a_imag[idx + 4],
            a_imag[idx + 5],
            a_imag[idx + 6],
            a_imag[idx + 7],
        ]);
        let b_imag_simd = f32x8::from([
            b_imag[idx],
            b_imag[idx + 1],
            b_imag[idx + 2],
            b_imag[idx + 3],
            b_imag[idx + 4],
            b_imag[idx + 5],
            b_imag[idx + 6],
            b_imag[idx + 7],
        ]);

        // Complex addition: (a + bi) + (c + di) = (a + c) + (b + d)i
        let result_real_simd = a_real_simd + b_real_simd;
        let result_imag_simd = a_imag_simd + b_imag_simd;

        let real_array: [f32; 8] = result_real_simd.into();
        let imag_array: [f32; 8] = result_imag_simd.into();

        result_real[idx..idx + 8].copy_from_slice(&real_array);
        result_imag[idx..idx + 8].copy_from_slice(&imag_array);
    }

    // Handle remaining elements
    for i in remainder_start..len {
        result_real[i] = a_real[i] + b_real[i];
        result_imag[i] = a_imag[i] + b_imag[i];
    }
}

/// SIMD-accelerated complex number multiplication for f32
#[cfg(feature = "simd")]
pub fn simd_complex_mul_f32(
    a_real: &[f32],
    a_imag: &[f32],
    b_real: &[f32],
    b_imag: &[f32],
    result_real: &mut [f32],
    result_imag: &mut [f32],
) {
    let len = a_real
        .len()
        .min(a_imag.len())
        .min(b_real.len())
        .min(b_imag.len())
        .min(result_real.len())
        .min(result_imag.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    for i in 0..simd_len {
        let idx = i * 8;

        let a_real_simd = f32x8::from([
            a_real[idx],
            a_real[idx + 1],
            a_real[idx + 2],
            a_real[idx + 3],
            a_real[idx + 4],
            a_real[idx + 5],
            a_real[idx + 6],
            a_real[idx + 7],
        ]);
        let a_imag_simd = f32x8::from([
            a_imag[idx],
            a_imag[idx + 1],
            a_imag[idx + 2],
            a_imag[idx + 3],
            a_imag[idx + 4],
            a_imag[idx + 5],
            a_imag[idx + 6],
            a_imag[idx + 7],
        ]);
        let b_real_simd = f32x8::from([
            b_real[idx],
            b_real[idx + 1],
            b_real[idx + 2],
            b_real[idx + 3],
            b_real[idx + 4],
            b_real[idx + 5],
            b_real[idx + 6],
            b_real[idx + 7],
        ]);
        let b_imag_simd = f32x8::from([
            b_imag[idx],
            b_imag[idx + 1],
            b_imag[idx + 2],
            b_imag[idx + 3],
            b_imag[idx + 4],
            b_imag[idx + 5],
            b_imag[idx + 6],
            b_imag[idx + 7],
        ]);

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let result_real_simd = a_real_simd * b_real_simd - a_imag_simd * b_imag_simd;
        let result_imag_simd = a_real_simd * b_imag_simd + a_imag_simd * b_real_simd;

        let real_array: [f32; 8] = result_real_simd.into();
        let imag_array: [f32; 8] = result_imag_simd.into();

        result_real[idx..idx + 8].copy_from_slice(&real_array);
        result_imag[idx..idx + 8].copy_from_slice(&imag_array);
    }

    // Handle remaining elements
    for i in remainder_start..len {
        let ac = a_real[i] * b_real[i];
        let bd = a_imag[i] * b_imag[i];
        let ad = a_real[i] * b_imag[i];
        let bc = a_imag[i] * b_real[i];

        result_real[i] = ac - bd;
        result_imag[i] = ad + bc;
    }
}

/// SIMD-accelerated quantized multiplication for u8
#[cfg(feature = "simd")]
pub fn simd_quantized_mul_u8(
    a: &[u8],
    b: &[u8],
    result: &mut [u8],
    scale_a: f32,
    scale_b: f32,
    scale_result: f32,
) {
    let len = a.len().min(b.len()).min(result.len());

    for i in 0..len {
        // Dequantize, multiply, and requantize
        let val_a = a[i] as f32 * scale_a;
        let val_b = b[i] as f32 * scale_b;
        let product = val_a * val_b;
        let quantized = (product / scale_result).round().clamp(0.0, 255.0) as u8;
        result[i] = quantized;
    }
}

/// SIMD-accelerated integer addition for i8
#[cfg(feature = "simd")]
pub fn simd_add_i8(a: &[i8], b: &[i8], result: &mut [i8]) {
    let len = a.len().min(b.len()).min(result.len());
    // Simplified implementation - in practice would use SIMD for i8
    for i in 0..len {
        result[i] = a[i].saturating_add(b[i]);
    }
}

/// SIMD-accelerated unsigned addition for u8
#[cfg(feature = "simd")]
pub fn simd_add_u8(a: &[u8], b: &[u8], result: &mut [u8]) {
    let len = a.len().min(b.len()).min(result.len());
    // Simplified implementation - in practice would use SIMD for u8
    for i in 0..len {
        result[i] = a[i].saturating_add(b[i]);
    }
}

/// Adaptive SIMD functions that choose the best implementation
pub mod adaptive {
    use super::*;

    /// Adaptive SIMD addition that selects the best implementation
    pub fn adaptive_simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
        if should_use_simd(a.len()) {
            crate::cpu::simd::arithmetic::simd_add_f32(a, b, result);
        } else {
            // Use scalar fallback for small arrays
            let len = a.len().min(b.len()).min(result.len());
            for i in 0..len {
                result[i] = a[i] + b[i];
            }
        }
    }

    /// Adaptive SIMD multiplication that selects the best implementation
    pub fn adaptive_simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
        if should_use_simd(a.len()) {
            crate::cpu::simd::arithmetic::simd_mul_f32(a, b, result);
        } else {
            let len = a.len().min(b.len()).min(result.len());
            for i in 0..len {
                result[i] = a[i] * b[i];
            }
        }
    }

    /// Adaptive SIMD dot product
    pub fn adaptive_simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        if should_use_simd(a.len()) {
            crate::cpu::simd::arithmetic::simd_dot_f32(a, b)
        } else {
            let len = a.len().min(b.len());
            let mut sum = 0.0;
            for i in 0..len {
                sum += a[i] * b[i];
            }
            sum
        }
    }

    /// Adaptive SIMD sum reduction
    pub fn adaptive_simd_sum_f32(input: &[f32]) -> f32 {
        if should_use_simd(input.len()) {
            crate::cpu::simd::arithmetic::simd_sum_f32(input)
        } else {
            input.iter().sum()
        }
    }

    /// Adaptive SIMD ReLU activation
    pub fn adaptive_simd_relu_f32(input: &[f32], output: &mut [f32]) {
        if should_use_simd(input.len()) {
            crate::cpu::simd::activation::simd_relu_f32(input, output);
        } else {
            let len = input.len().min(output.len());
            for i in 0..len {
                output[i] = input[i].max(0.0);
            }
        }
    }

    /// Adaptive SIMD Sigmoid activation
    pub fn adaptive_simd_sigmoid_f32(input: &[f32], output: &mut [f32]) {
        if should_use_simd(input.len()) {
            crate::cpu::simd::activation::simd_sigmoid_f32(input, output);
        } else {
            let len = input.len().min(output.len());
            for i in 0..len {
                output[i] = 1.0 / (1.0 + (-input[i]).exp());
            }
        }
    }

    /// Adaptive SIMD matrix multiplication
    pub fn adaptive_simd_matmul_f32(
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        if should_use_simd(a.len()) && should_use_simd(b.len()) {
            simd_matmul_f32(a, b, result, m, n, k);
        } else {
            // Scalar fallback
            result.fill(0.0);
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        result[i * n + j] += a[i * k + l] * b[l * n + j];
                    }
                }
            }
        }
    }
}

// Fallback implementations when SIMD is not available
#[cfg(not(feature = "simd"))]
pub fn simd_matmul_f32(a: &[f32], b: &[f32], result: &mut [f32], m: usize, n: usize, k: usize) {
    result.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                result[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_complex_add_f32(
    a_real: &[f32],
    a_imag: &[f32],
    b_real: &[f32],
    b_imag: &[f32],
    result_real: &mut [f32],
    result_imag: &mut [f32],
) {
    let len = a_real
        .len()
        .min(a_imag.len())
        .min(b_real.len())
        .min(b_imag.len())
        .min(result_real.len())
        .min(result_imag.len());
    for i in 0..len {
        result_real[i] = a_real[i] + b_real[i];
        result_imag[i] = a_imag[i] + b_imag[i];
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_complex_mul_f32(
    a_real: &[f32],
    a_imag: &[f32],
    b_real: &[f32],
    b_imag: &[f32],
    result_real: &mut [f32],
    result_imag: &mut [f32],
) {
    let len = a_real
        .len()
        .min(a_imag.len())
        .min(b_real.len())
        .min(b_imag.len())
        .min(result_real.len())
        .min(result_imag.len());
    for i in 0..len {
        let ac = a_real[i] * b_real[i];
        let bd = a_imag[i] * b_imag[i];
        let ad = a_real[i] * b_imag[i];
        let bc = a_imag[i] * b_real[i];

        result_real[i] = ac - bd;
        result_imag[i] = ad + bc;
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_quantized_mul_u8(
    a: &[u8],
    b: &[u8],
    result: &mut [u8],
    scale_a: f32,
    scale_b: f32,
    scale_result: f32,
) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        let val_a = a[i] as f32 * scale_a;
        let val_b = b[i] as f32 * scale_b;
        let product = val_a * val_b;
        let quantized = (product / scale_result).round().clamp(0.0, 255.0) as u8;
        result[i] = quantized;
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_add_i8(a: &[i8], b: &[i8], result: &mut [i8]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i].saturating_add(b[i]);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_add_u8(a: &[u8], b: &[u8], result: &mut [u8]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i].saturating_add(b[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matmul_f32() {
        // Test 2x2 matrix multiplication
        let a = [1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = [2.0, 0.0, 1.0, 2.0]; // [[2, 0], [1, 2]]
        let mut result = [0.0; 4];

        simd_matmul_f32(&a, &b, &mut result, 2, 2, 2);

        // Expected: [[4, 4], [10, 8]]
        assert_eq!(result[0], 4.0); // 1*2 + 2*1
        assert_eq!(result[1], 4.0); // 1*0 + 2*2
        assert_eq!(result[2], 10.0); // 3*2 + 4*1
        assert_eq!(result[3], 8.0); // 3*0 + 4*2
    }

    #[test]
    fn test_simd_complex_add_f32() {
        let a_real = [1.0, 3.0];
        let a_imag = [2.0, 4.0];
        let b_real = [5.0, 7.0];
        let b_imag = [6.0, 8.0];
        let mut result_real = [0.0; 2];
        let mut result_imag = [0.0; 2];

        simd_complex_add_f32(
            &a_real,
            &a_imag,
            &b_real,
            &b_imag,
            &mut result_real,
            &mut result_imag,
        );

        assert_eq!(result_real[0], 6.0); // 1 + 5
        assert_eq!(result_imag[0], 8.0); // 2 + 6
        assert_eq!(result_real[1], 10.0); // 3 + 7
        assert_eq!(result_imag[1], 12.0); // 4 + 8
    }

    #[test]
    fn test_simd_complex_mul_f32() {
        let a_real = [1.0];
        let a_imag = [2.0];
        let b_real = [3.0];
        let b_imag = [4.0];
        let mut result_real = [0.0; 1];
        let mut result_imag = [0.0; 1];

        simd_complex_mul_f32(
            &a_real,
            &a_imag,
            &b_real,
            &b_imag,
            &mut result_real,
            &mut result_imag,
        );

        // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert_eq!(result_real[0], -5.0); // 1*3 - 2*4
        assert_eq!(result_imag[0], 10.0); // 1*4 + 2*3
    }
}
