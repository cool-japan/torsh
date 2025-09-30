//! ARM NEON SIMD optimizations for ToRSh core operations
//!
//! This module provides optimized implementations of common operations using
//! ARM NEON SIMD instructions for improved performance on ARM64 platforms.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::error::{Result, TorshError};

/// ARM NEON optimized operations
pub struct ArmSimdOps;

#[cfg(target_arch = "aarch64")]
impl ArmSimdOps {
    /// Check if NEON is available at runtime
    pub fn is_neon_available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    /// Check if Advanced SIMD is available
    pub fn is_asimd_available() -> bool {
        std::arch::is_aarch64_feature_detected!("asimd")
    }

    /// Check if FP16 arithmetic is supported
    pub fn is_fp16_available() -> bool {
        std::arch::is_aarch64_feature_detected!("fp16")
    }

    /// Check if dot product instructions are supported
    pub fn is_dotprod_available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    /// Vectorized addition of f32 arrays using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn add_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "add_f32_neon",
            ));
        }

        let len = a.len();
        let simd_len = len & !3; // Process 4 elements at a time

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        // Process 4 f32 elements at a time using NEON
        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let vresult = vaddq_f32(va, vb);
            vst1q_f32(result_ptr.add(i), vresult);
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Vectorized subtraction of f32 arrays using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn sub_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = a.len();
        let simd_len = len & !3;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let vresult = vsubq_f32(va, vb);
            vst1q_f32(result_ptr.add(i), vresult);
        }

        for i in simd_len..len {
            result[i] = a[i] - b[i];
        }

        Ok(())
    }

    /// Vectorized multiplication of f32 arrays using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn mul_f32_neon(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = a.len();
        let simd_len = len & !3;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let vresult = vmulq_f32(va, vb);
            vst1q_f32(result_ptr.add(i), vresult);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Vectorized fused multiply-add operation using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = a.len();
        let simd_len = len & !3;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let vc = vld1q_f32(c_ptr.add(i));
            let vresult = vfmaq_f32(vc, va, vb); // c + a * b
            vst1q_f32(result_ptr.add(i), vresult);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i] + c[i];
        }

        Ok(())
    }

    /// Vectorized dot product using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = a.len();
        let simd_len = len & !3;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum_vec = vdupq_n_f32(0.0);

        // SIMD accumulation
        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let vmul = vmulq_f32(va, vb);
            sum_vec = vaddq_f32(sum_vec, vmul);
        }

        // Horizontal sum of the vector
        let sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        let sum_scalar = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(sum_scalar, 0);

        // Handle remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    /// Optimized dot product for small vectors
    /// Note: dotprod instructions are currently unstable, using manual implementation
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_i8_dotprod(a: &[i8], b: &[i8]) -> Result<i32> {
        if a.len() != b.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = a.len();
        let simd_len = len & !15; // Process 16 elements at a time

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum_vec = vdupq_n_s32(0);

        // SIMD accumulation using manual multiply-accumulate
        // TODO: Re-enable vdotq_s32 when it becomes stable
        for i in (0..simd_len).step_by(16) {
            // Load 16 i8 values and process in groups of 4
            for j in 0..4 {
                let offset = i + j * 4;
                if offset < len {
                    // Load 4 i8 values and convert to i32
                    let a_vals = [
                        *a_ptr.add(offset) as i32,
                        *a_ptr.add(offset + 1) as i32,
                        *a_ptr.add(offset + 2) as i32,
                        *a_ptr.add(offset + 3) as i32,
                    ];
                    let b_vals = [
                        *b_ptr.add(offset) as i32,
                        *b_ptr.add(offset + 1) as i32,
                        *b_ptr.add(offset + 2) as i32,
                        *b_ptr.add(offset + 3) as i32,
                    ];

                    let va = vld1q_s32(a_vals.as_ptr());
                    let vb = vld1q_s32(b_vals.as_ptr());
                    sum_vec = vmlaq_s32(sum_vec, va, vb);
                }
            }
        }

        // Horizontal sum
        let sum_pair = vadd_s32(vget_low_s32(sum_vec), vget_high_s32(sum_vec));
        let sum_scalar = vpadd_s32(sum_pair, sum_pair);
        let mut result = vget_lane_s32(sum_scalar, 0);

        // Handle remaining elements
        for i in simd_len..len {
            result += a[i] as i32 * b[i] as i32;
        }

        Ok(result)
    }

    /// Vectorized sum reduction using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn sum_f32_neon(data: &[f32]) -> f32 {
        let len = data.len();
        let simd_len = len & !3;
        let data_ptr = data.as_ptr();

        let mut sum_vec = vdupq_n_f32(0.0);

        // SIMD accumulation
        for i in (0..simd_len).step_by(4) {
            let vdata = vld1q_f32(data_ptr.add(i));
            sum_vec = vaddq_f32(sum_vec, vdata);
        }

        // Horizontal sum
        let sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        let sum_scalar = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(sum_scalar, 0);

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Indexing is clearer for accumulation
        for i in simd_len..len {
            result += data[i];
        }

        result
    }

    /// Vectorized ReLU activation using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn relu_f32_neon(data: &[f32], result: &mut [f32]) -> Result<()> {
        if data.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let len = data.len();
        let simd_len = len & !3;

        let data_ptr = data.as_ptr();
        let result_ptr = result.as_mut_ptr();
        let zero_vec = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(4) {
            let vdata = vld1q_f32(data_ptr.add(i));
            let vresult = vmaxq_f32(vdata, zero_vec);
            vst1q_f32(result_ptr.add(i), vresult);
        }

        for i in simd_len..len {
            result[i] = data[i].max(0.0);
        }

        Ok(())
    }

    /// Vectorized matrix multiplication for small matrices using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn matmul_f32_4x4_neon(
        a: &[f32; 16],
        b: &[f32; 16],
        result: &mut [f32; 16],
    ) -> Result<()> {
        // Load matrix A rows
        let a_row0 = vld1q_f32(a.as_ptr());
        let a_row1 = vld1q_f32(a.as_ptr().add(4));
        let a_row2 = vld1q_f32(a.as_ptr().add(8));
        let a_row3 = vld1q_f32(a.as_ptr().add(12));

        // Load matrix B columns (transposed for efficient access)
        let b_col0_arr = [b[0], b[4], b[8], b[12]];
        let b_col1_arr = [b[1], b[5], b[9], b[13]];
        let b_col2_arr = [b[2], b[6], b[10], b[14]];
        let b_col3_arr = [b[3], b[7], b[11], b[15]];

        let b_col0 = vld1q_f32(b_col0_arr.as_ptr());
        let b_col1 = vld1q_f32(b_col1_arr.as_ptr());
        let b_col2 = vld1q_f32(b_col2_arr.as_ptr());
        let b_col3 = vld1q_f32(b_col3_arr.as_ptr());

        // Compute result matrix
        let a_rows = [a_row0, a_row1, a_row2, a_row3];
        let b_cols = [b_col0, b_col1, b_col2, b_col3];

        for i in 0..4 {
            for j in 0..4 {
                let dot = vmulq_f32(a_rows[i], b_cols[j]);
                let sum_pair = vadd_f32(vget_low_f32(dot), vget_high_f32(dot));
                let sum_scalar = vpadd_f32(sum_pair, sum_pair);
                let final_sum = vget_lane_f32(sum_scalar, 0);
                result[i * 4 + j] = final_sum;
            }
        }

        Ok(())
    }

    // Note: f16 NEON operations require unstable Rust features
    // Commented out until f16 support is stabilized
    // #[cfg(feature = "fp16")]
    // #[target_feature(enable = "neon", enable = "fp16")]
    // pub unsafe fn add_f16_neon(a: &[f16], b: &[f16], result: &mut [f16]) -> Result<()> {
    //     if a.len() != b.len() || a.len() != result.len() {
    //         return Err(TorshError::dimension_error_with_context(
    //             "Array lengths must match",
    //             "simd_operation",
    //         ));
    //     }
    //
    //     let len = a.len();
    //     let simd_len = len & !7; // Process 8 f16 elements at a time
    //
    //     let a_ptr = a.as_ptr() as *const __fp16;
    //     let b_ptr = b.as_ptr() as *const __fp16;
    //     let result_ptr = result.as_mut_ptr() as *mut __fp16;
    //
    //     for i in (0..simd_len).step_by(8) {
    //         let va = vld1q_f16(a_ptr.add(i));
    //         let vb = vld1q_f16(b_ptr.add(i));
    //         let vresult = vaddq_f16(va, vb);
    //         vst1q_f16(result_ptr.add(i), vresult);
    //     }
    //
    //     // Handle remaining elements (fallback to scalar)
    //     for i in simd_len..len {
    //         result[i] = a[i] + b[i];
    //     }
    //
    //     Ok(())
    // }

    /// Optimized memcpy using NEON for large data transfers
    #[target_feature(enable = "neon")]
    pub unsafe fn memcpy_neon(src: &[u8], dest: &mut [u8]) -> Result<()> {
        if src.len() != dest.len() {
            return Err(TorshError::dimension_error_with_context(
                "Source and destination lengths must match",
                "memcpy_neon",
            ));
        }

        let len = src.len();
        let simd_len = len & !31; // Process 32 bytes at a time

        let src_ptr = src.as_ptr();
        let dest_ptr = dest.as_mut_ptr();

        // NEON optimized copy for large blocks
        for i in (0..simd_len).step_by(32) {
            let v0 = vld1q_u8(src_ptr.add(i));
            let v1 = vld1q_u8(src_ptr.add(i + 16));
            vst1q_u8(dest_ptr.add(i), v0);
            vst1q_u8(dest_ptr.add(i + 16), v1);
        }

        // Handle remaining bytes
        dest[simd_len..len].copy_from_slice(&src[simd_len..len]);

        Ok(())
    }
}

/// Safe wrapper functions for ARM SIMD operations
impl ArmSimdOps {
    /// Safe wrapper for NEON f32 addition
    pub fn add_f32_safe(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_neon_available() {
                unsafe { Self::add_f32_neon(a, b, result) }
            } else {
                Self::add_f32_scalar(a, b, result)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::add_f32_scalar(a, b, result)
        }
    }

    /// Safe wrapper for NEON f32 multiplication
    pub fn mul_f32_safe(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_neon_available() {
                unsafe { Self::mul_f32_neon(a, b, result) }
            } else {
                Self::mul_f32_scalar(a, b, result)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_f32_scalar(a, b, result)
        }
    }

    /// Safe wrapper for NEON dot product
    pub fn dot_product_f32_safe(a: &[f32], b: &[f32]) -> Result<f32> {
        #[cfg(target_arch = "aarch64")]
        {
            if Self::is_neon_available() {
                unsafe { Self::dot_product_f32_neon(a, b) }
            } else {
                Self::dot_product_f32_scalar(a, b)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::dot_product_f32_scalar(a, b)
        }
    }

    /// Scalar fallback for f32 addition
    fn add_f32_scalar(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Scalar fallback for f32 multiplication
    fn mul_f32_scalar(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Scalar fallback for f32 dot product
    fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(TorshError::dimension_error_with_context(
                "Array lengths must match",
                "simd_operation",
            ));
        }

        let mut result = 0.0;
        for i in 0..a.len() {
            result += a[i] * b[i];
        }

        Ok(result)
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl ArmSimdOps {
    /// Stub implementation for non-ARM platforms
    pub fn is_neon_available() -> bool {
        false
    }
    pub fn is_asimd_available() -> bool {
        false
    }
    pub fn is_fp16_available() -> bool {
        false
    }
    pub fn is_dotprod_available() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_availability() {
        #[cfg(target_arch = "aarch64")]
        {
            // Test availability checks - these should not panic
            let _ = ArmSimdOps::is_neon_available();
            let _ = ArmSimdOps::is_asimd_available();
            let _ = ArmSimdOps::is_fp16_available();
            let _ = ArmSimdOps::is_dotprod_available();
        }
    }

    #[test]
    fn test_safe_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];

        // Test addition
        ArmSimdOps::add_f32_safe(&a, &b, &mut result).unwrap();
        let expected_add = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
        assert_eq!(result, expected_add);

        // Test multiplication
        ArmSimdOps::mul_f32_safe(&a, &b, &mut result).unwrap();
        let expected_mul = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];
        assert_eq!(result, expected_mul);

        // Test dot product
        let dot_result = ArmSimdOps::dot_product_f32_safe(&a, &b).unwrap();
        let expected_dot = 240.0; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9
        assert_eq!(dot_result, expected_dot);
    }

    #[test]
    fn test_error_handling() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let mut result = vec![0.0; 3];

        // Test mismatched lengths
        assert!(ArmSimdOps::add_f32_safe(&a, &b, &mut result).is_err());
        assert!(ArmSimdOps::dot_product_f32_safe(&a, &b).is_err());
    }
}
