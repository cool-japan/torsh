//! RISC-V Vector Extension (RVV) optimized operations
//!
//! This module provides high-performance tensor operations using RISC-V Vector Extensions.
//! It supports dynamic vector length and provides optimized implementations for common
//! tensor operations.

use crate::cpu::feature_detection::{global_detector, CpuFeature};
use std::marker::PhantomData;

/// RISC-V Vector operation dispatcher
pub struct RiscVVectorOps<T> {
    _phantom: PhantomData<T>,
}

impl<T> RiscVVectorOps<T> {
    /// Create a new RISC-V vector operations instance
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Check if RISC-V vector extensions are available
    pub fn is_available() -> bool {
        global_detector().has_feature(CpuFeature::V)
    }

    /// Check if floating-point vector operations are available
    pub fn is_float_available() -> bool {
        global_detector().has_feature(CpuFeature::V)
            && (global_detector().has_feature(CpuFeature::F)
                || global_detector().has_feature(CpuFeature::D))
    }
}

impl RiscVVectorOps<f32> {
    /// Element-wise addition for f32 vectors
    pub fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_add_f32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val + b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val + b_val;
            }
        }
    }

    /// Element-wise multiplication for f32 vectors
    pub fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_mul_f32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val * b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val * b_val;
            }
        }
    }

    /// Fused multiply-add for f32 vectors
    pub fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        assert_eq!(a.len(), result.len());

        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_fma_f32(
                        a.as_ptr(),
                        b.as_ptr(),
                        c.as_ptr(),
                        result.as_mut_ptr(),
                        a.len(),
                    );
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for (((a_val, b_val), c_val), result_val) in
                    a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
                {
                    *result_val = a_val.mul_add(*b_val, *c_val);
                }
            }
        } else {
            // Scalar fallback
            for (((a_val, b_val), c_val), result_val) in
                a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
            {
                *result_val = a_val.mul_add(*b_val, *c_val);
            }
        }
    }

    /// Dot product for f32 vectors
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                a.iter()
                    .zip(b.iter())
                    .map(|(a_val, b_val)| a_val * b_val)
                    .sum()
            }
        } else {
            // Scalar fallback
            a.iter()
                .zip(b.iter())
                .map(|(a_val, b_val)| a_val * b_val)
                .sum()
        }
    }

    /// Sum reduction for f32 vectors
    pub fn sum(input: &[f32]) -> f32 {
        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_sum_f32(input.as_ptr(), input.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                input.iter().sum()
            }
        } else {
            // Scalar fallback
            input.iter().sum()
        }
    }

    /// Matrix multiplication for f32 matrices (row-major)
    pub fn matmul(a: &[f32], b: &[f32], result: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(result.len(), m * n);

        if Self::is_float_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_matmul_f32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), m, n, k);
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback to standard matrix multiplication
                Self::matmul_scalar(a, b, result, m, n, k);
            }
        } else {
            // Scalar fallback
            Self::matmul_scalar(a, b, result, m, n, k);
        }
    }

    /// Scalar matrix multiplication fallback
    fn matmul_scalar(a: &[f32], b: &[f32], result: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }
}

impl RiscVVectorOps<f64> {
    /// Element-wise addition for f64 vectors
    pub fn add(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_available() && global_detector().has_feature(CpuFeature::D) {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_add_f64(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val + b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val + b_val;
            }
        }
    }

    /// Element-wise multiplication for f64 vectors
    pub fn mul(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_available() && global_detector().has_feature(CpuFeature::D) {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_mul_f64(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val * b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val * b_val;
            }
        }
    }

    /// Dot product for f64 vectors
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());

        if Self::is_available() && global_detector().has_feature(CpuFeature::D) {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_dot_f64(a.as_ptr(), b.as_ptr(), a.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                a.iter()
                    .zip(b.iter())
                    .map(|(a_val, b_val)| a_val * b_val)
                    .sum()
            }
        } else {
            // Scalar fallback
            a.iter()
                .zip(b.iter())
                .map(|(a_val, b_val)| a_val * b_val)
                .sum()
        }
    }

    /// Sum reduction for f64 vectors
    pub fn sum(input: &[f64]) -> f64 {
        if Self::is_available() && global_detector().has_feature(CpuFeature::D) {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_sum_f64(input.as_ptr(), input.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                input.iter().sum()
            }
        } else {
            // Scalar fallback
            input.iter().sum()
        }
    }
}

impl RiscVVectorOps<i32> {
    /// Element-wise addition for i32 vectors
    pub fn add(a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_add_i32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val + b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val + b_val;
            }
        }
    }

    /// Element-wise multiplication for i32 vectors
    pub fn mul(a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if Self::is_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe {
                    riscv_vector_mul_i32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
                }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                    *result_val = a_val * b_val;
                }
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val * b_val;
            }
        }
    }

    /// Sum reduction for i32 vectors
    pub fn sum(input: &[i32]) -> i32 {
        if Self::is_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_sum_i32(input.as_ptr(), input.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                input.iter().sum()
            }
        } else {
            // Scalar fallback
            input.iter().sum()
        }
    }

    /// Max reduction for i32 vectors
    pub fn max(input: &[i32]) -> i32 {
        if input.is_empty() {
            return 0;
        }

        if Self::is_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_max_i32(input.as_ptr(), input.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                *input
                    .iter()
                    .max()
                    .expect("input should not be empty after guard")
            }
        } else {
            // Scalar fallback
            *input
                .iter()
                .max()
                .expect("input should not be empty after guard")
        }
    }

    /// Min reduction for i32 vectors
    pub fn min(input: &[i32]) -> i32 {
        if input.is_empty() {
            return 0;
        }

        if Self::is_available() {
            #[cfg(target_arch = "riscv64")]
            {
                unsafe { riscv_vector_min_i32(input.as_ptr(), input.len()) }
            }
            #[cfg(not(target_arch = "riscv64"))]
            {
                // Fallback implementation
                *input
                    .iter()
                    .min()
                    .expect("input should not be empty after guard")
            }
        } else {
            // Scalar fallback
            *input
                .iter()
                .min()
                .expect("input should not be empty after guard")
        }
    }
}

// RISC-V Vector Assembly Functions
// These would be implemented in inline assembly or separate assembly files

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_add_f32(a: *const f32, b: *const f32, result: *mut f32, len: usize) {
    // RISC-V Vector Extension implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform addition
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vfadd.vv v2, v0, v1",    // Vector floating-point add
            "vse32.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_mul_f32(a: *const f32, b: *const f32, result: *mut f32, len: usize) {
    // RISC-V Vector Extension multiplication implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform multiplication
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vfmul.vv v2, v0, v1",    // Vector floating-point multiply
            "vse32.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_fma_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    result: *mut f32,
    len: usize,
) {
    // RISC-V Vector Extension fused multiply-add implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform fused multiply-add
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vle32.v v2, ({c})",      // Load from c
            "vfmadd.vv v2, v0, v1",   // v2 = v0 * v1 + v2 (fused multiply-add)
            "vse32.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            c = in(reg) c.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    // RISC-V Vector Extension dot product implementation
    use std::arch::asm;

    let mut sum = 0.0f32;
    let mut i = 0;

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Perform vectorized multiply and accumulate
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vfmul.vv v2, v0, v1",    // Vector multiply
            "vfredosum.vs v3, v2, v3", // Ordered sum reduction
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final sum from the vector register
    asm!(
        "vfmv.f.s {sum}, v3",
        sum = out(freg) sum,
        options(pure, nomem, nostack),
    );

    sum
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_sum_f32(input: *const f32, len: usize) -> f32 {
    // RISC-V Vector Extension sum reduction implementation
    use std::arch::asm;

    let mut sum = 0.0f32;
    let mut i = 0;

    // Initialize accumulator vector register
    asm!(
        "vsetvli zero, x0, e32, m1, ta, ma",
        "vmv.v.x v3, zero", // Clear accumulator
        options(nostack),
    );

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load and accumulate
        asm!(
            "vle32.v v0, ({input})",    // Load input vector
            "vfredosum.vs v3, v0, v3",  // Ordered sum reduction into v3
            input = in(reg) input.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final sum from the vector register
    asm!(
        "vfmv.f.s {sum}, v3",
        sum = out(freg) sum,
        options(pure, nomem, nostack),
    );

    sum
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_matmul_f32(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    // Placeholder for RVV matrix multiplication
    // Real implementation would use optimized blocking and vector operations
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += *a.add(i * k + l) * *b.add(l * n + j);
            }
            *result.add(i * n + j) = sum;
        }
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_add_f64(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
    // RISC-V Vector Extension f64 addition implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e64, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform addition
        asm!(
            "vle64.v v0, ({a})",      // Load from a
            "vle64.v v1, ({b})",      // Load from b
            "vfadd.vv v2, v0, v1",    // Vector floating-point add
            "vse64.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_mul_f64(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
    // RISC-V Vector Extension f64 multiplication implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e64, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform multiplication
        asm!(
            "vle64.v v0, ({a})",      // Load from a
            "vle64.v v1, ({b})",      // Load from b
            "vfmul.vv v2, v0, v1",    // Vector floating-point multiply
            "vse64.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_dot_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    // RISC-V Vector Extension f64 dot product implementation
    use std::arch::asm;

    let mut sum = 0.0f64;
    let mut i = 0;

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e64, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Perform vectorized multiply and accumulate
        asm!(
            "vle64.v v0, ({a})",      // Load from a
            "vle64.v v1, ({b})",      // Load from b
            "vfmul.vv v2, v0, v1",    // Vector multiply
            "vfredosum.vs v3, v2, v3", // Ordered sum reduction
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final sum from the vector register
    asm!(
        "vfmv.f.s {sum}, v3",
        sum = out(freg) sum,
        options(pure, nomem, nostack),
    );

    sum
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_sum_f64(input: *const f64, len: usize) -> f64 {
    // RISC-V Vector Extension f64 sum reduction implementation
    use std::arch::asm;

    let mut sum = 0.0f64;
    let mut i = 0;

    // Initialize accumulator vector register
    asm!(
        "vsetvli zero, x0, e64, m1, ta, ma",
        "vmv.v.x v3, zero", // Clear accumulator
        options(nostack),
    );

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e64, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load and accumulate
        asm!(
            "vle64.v v0, ({input})",    // Load input vector
            "vfredosum.vs v3, v0, v3",  // Ordered sum reduction into v3
            input = in(reg) input.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final sum from the vector register
    asm!(
        "vfmv.f.s {sum}, v3",
        sum = out(freg) sum,
        options(pure, nomem, nostack),
    );

    sum
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_add_i32(a: *const i32, b: *const i32, result: *mut i32, len: usize) {
    // RISC-V Vector Extension i32 addition implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform addition
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vadd.vv v2, v0, v1",     // Vector integer add
            "vse32.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_mul_i32(a: *const i32, b: *const i32, result: *mut i32, len: usize) {
    // RISC-V Vector Extension i32 multiplication implementation
    use std::arch::asm;

    let mut i = 0;
    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load vectors and perform multiplication
        asm!(
            "vle32.v v0, ({a})",      // Load from a
            "vle32.v v1, ({b})",      // Load from b
            "vmul.vv v2, v0, v1",     // Vector integer multiply
            "vse32.v v2, ({result})", // Store result
            a = in(reg) a.add(i),
            b = in(reg) b.add(i),
            result = in(reg) result.add(i),
            options(nostack),
        );

        i += vl;
    }
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_sum_i32(input: *const i32, len: usize) -> i32 {
    // RISC-V Vector Extension i32 sum reduction implementation
    use std::arch::asm;

    let mut sum = 0i32;
    let mut i = 0;

    // Initialize accumulator vector register
    asm!(
        "vsetvli zero, x0, e32, m1, ta, ma",
        "vmv.v.x v3, zero", // Clear accumulator
        options(nostack),
    );

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load and accumulate
        asm!(
            "vle32.v v0, ({input})",    // Load input vector
            "vredsum.vs v3, v0, v3",    // Sum reduction into v3
            input = in(reg) input.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final sum from the vector register
    asm!(
        "vmv.x.s {sum}, v3",
        sum = out(reg) sum,
        options(pure, nomem, nostack),
    );

    sum
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_max_i32(input: *const i32, len: usize) -> i32 {
    // RISC-V Vector Extension i32 max reduction implementation
    use std::arch::asm;

    let mut max_val = i32::MIN;
    let mut i = 0;

    // Initialize accumulator with first element
    if len > 0 {
        max_val = *input;
        i = 1;
    }

    // Initialize accumulator vector register
    asm!(
        "vsetvli zero, x0, e32, m1, ta, ma",
        "vmv.v.x v3, {max_val}",  // Initialize with max_val
        max_val = in(reg) max_val,
        options(nostack),
    );

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load and find max
        asm!(
            "vle32.v v0, ({input})",    // Load input vector
            "vredmax.vs v3, v0, v3",    // Max reduction into v3
            input = in(reg) input.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final max from the vector register
    asm!(
        "vmv.x.s {max_val}, v3",
        max_val = out(reg) max_val,
        options(pure, nomem, nostack),
    );

    max_val
}

#[cfg(target_arch = "riscv64")]
unsafe fn riscv_vector_min_i32(input: *const i32, len: usize) -> i32 {
    // RISC-V Vector Extension i32 min reduction implementation
    use std::arch::asm;

    let mut min_val = i32::MAX;
    let mut i = 0;

    // Initialize accumulator with first element
    if len > 0 {
        min_val = *input;
        i = 1;
    }

    // Initialize accumulator vector register
    asm!(
        "vsetvli zero, x0, e32, m1, ta, ma",
        "vmv.v.x v3, {min_val}",  // Initialize with min_val
        min_val = in(reg) min_val,
        options(nostack),
    );

    while i < len {
        let vl: usize;

        // Set vector length for this iteration
        asm!(
            "vsetvli {vl}, {len}, e32, m1, ta, ma",
            vl = out(reg) vl,
            len = in(reg) len - i,
            options(pure, nomem, nostack),
        );

        // Load and find min
        asm!(
            "vle32.v v0, ({input})",    // Load input vector
            "vredmin.vs v3, v0, v3",    // Min reduction into v3
            input = in(reg) input.add(i),
            options(nostack),
        );

        i += vl;
    }

    // Extract the final min from the vector register
    asm!(
        "vmv.x.s {min_val}, v3",
        min_val = out(reg) min_val,
        options(pure, nomem, nostack),
    );

    min_val
}

/// Performance information for RISC-V vector operations
#[derive(Debug, Clone)]
pub struct RiscVVectorPerformanceInfo {
    pub vector_length: Option<usize>,
    pub max_vector_length: Option<usize>,
    pub supports_dynamic_vl: bool,
    pub supports_fp32: bool,
    pub supports_fp64: bool,
    pub supports_integer: bool,
}

impl RiscVVectorPerformanceInfo {
    /// Get performance information for the current RISC-V processor
    pub fn detect() -> Self {
        Self {
            vector_length: Self::detect_vector_length(),
            max_vector_length: Self::detect_max_vector_length(),
            supports_dynamic_vl: true, // RVV always supports dynamic VL
            supports_fp32: global_detector().has_feature(CpuFeature::F),
            supports_fp64: global_detector().has_feature(CpuFeature::D),
            supports_integer: global_detector().has_feature(CpuFeature::V),
        }
    }

    fn detect_vector_length() -> Option<usize> {
        // This would query the current vector length (VL)
        // For now, return a reasonable default
        if global_detector().has_feature(CpuFeature::V) {
            Some(128) // bits, typical for many RISC-V implementations
        } else {
            None
        }
    }

    fn detect_max_vector_length() -> Option<usize> {
        // This would query the maximum vector length (VLEN)
        if global_detector().has_feature(CpuFeature::V) {
            Some(512) // bits, common maximum for current implementations
        } else {
            None
        }
    }

    /// Estimate performance relative to scalar operations
    pub fn performance_multiplier(&self) -> f64 {
        if let Some(vl) = self.vector_length {
            // Estimate based on vector length and data type
            if self.supports_fp64 {
                (vl / 64) as f64 // f64 elements
            } else if self.supports_fp32 {
                (vl / 32) as f64 // f32 elements
            } else {
                (vl / 32) as f64 // Assume 32-bit integers
            }
        } else {
            1.0 // No vectorization available
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riscv_vector_add_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        RiscVVectorOps::<f32>::add(&a, &b, &mut result);

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_riscv_vector_mul_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        RiscVVectorOps::<f32>::mul(&a, &b, &mut result);

        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_riscv_vector_dot_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = RiscVVectorOps::<f32>::dot(&a, &b);

        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    }

    #[test]
    fn test_riscv_vector_sum_f32() {
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = RiscVVectorOps::<f32>::sum(&input);

        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_riscv_vector_matmul_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut result = vec![0.0; 4];

        RiscVVectorOps::<f32>::matmul(&a, &b, &mut result, 2, 2, 2);

        // Expected: [1*5 + 2*7, 1*6 + 2*8; 3*5 + 4*7, 3*6 + 4*8]
        //          = [19, 22; 43, 50]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_riscv_performance_info() {
        let info = RiscVVectorPerformanceInfo::detect();

        // Should have reasonable defaults even if not on RISC-V hardware
        assert!(info.performance_multiplier() >= 1.0);
        assert!(info.supports_dynamic_vl);
    }

    #[test]
    fn test_riscv_vector_availability() {
        // Should not crash even on non-RISC-V hardware
        let _available = RiscVVectorOps::<f32>::is_available();
        let _float_available = RiscVVectorOps::<f32>::is_float_available();
    }

    #[test]
    fn test_riscv_vector_f64_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        // Test f64 addition
        RiscVVectorOps::<f64>::add(&a, &b, &mut result);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);

        // Test f64 multiplication
        RiscVVectorOps::<f64>::mul(&a, &b, &mut result);
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);

        // Test f64 dot product
        let dot_result = RiscVVectorOps::<f64>::dot(&a, &b);
        assert_eq!(dot_result, 40.0);

        // Test f64 sum
        let sum_result = RiscVVectorOps::<f64>::sum(&a);
        assert_eq!(sum_result, 10.0);
    }

    #[test]
    fn test_riscv_vector_i32_operations() {
        let a = vec![1, 2, 3, 4];
        let b = vec![2, 3, 4, 5];
        let mut result = vec![0; 4];

        // Test i32 addition
        RiscVVectorOps::<i32>::add(&a, &b, &mut result);
        assert_eq!(result, vec![3, 5, 7, 9]);

        // Test i32 multiplication
        RiscVVectorOps::<i32>::mul(&a, &b, &mut result);
        assert_eq!(result, vec![2, 6, 12, 20]);

        // Test i32 sum
        let sum_result = RiscVVectorOps::<i32>::sum(&a);
        assert_eq!(sum_result, 10);
    }

    #[test]
    fn test_riscv_vector_i32_reductions() {
        let input = vec![5, 2, 8, 1, 9, 3];

        // Test max reduction
        let max_result = RiscVVectorOps::<i32>::max(&input);
        assert_eq!(max_result, 9);

        // Test min reduction
        let min_result = RiscVVectorOps::<i32>::min(&input);
        assert_eq!(min_result, 1);

        // Test empty arrays
        let empty: Vec<i32> = vec![];
        assert_eq!(RiscVVectorOps::<i32>::max(&empty), 0);
        assert_eq!(RiscVVectorOps::<i32>::min(&empty), 0);
    }

    #[test]
    fn test_riscv_vector_fma_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; 4];

        RiscVVectorOps::<f32>::fma(&a, &b, &c, &mut result);

        // Expected: a[i] * b[i] + c[i]
        // [1*2 + 1, 2*3 + 1, 3*4 + 1, 4*5 + 1] = [3, 7, 13, 21]
        assert_eq!(result, vec![3.0, 7.0, 13.0, 21.0]);
    }

    #[test]
    fn test_riscv_vector_large_arrays() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];

        // Test large array addition
        RiscVVectorOps::<f32>::add(&a, &b, &mut result);

        // Verify first few and last few elements
        assert_eq!(result[0], 1.0); // 0 + 1
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[size - 1], (2 * size - 1) as f32); // (size-1) + size

        // Test large array dot product
        let dot_result = RiscVVectorOps::<f32>::dot(&a, &b);
        let expected: f32 = (0..size).map(|i| (i * (i + 1)) as f32).sum();
        assert_eq!(dot_result, expected);
    }
}
