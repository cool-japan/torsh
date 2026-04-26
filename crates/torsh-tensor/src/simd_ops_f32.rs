//! SIMD-backed f32 fast-path helpers for zero-allocation element-wise arithmetic.
//!
//! This module provides slice-based in-place and into-buffer f32 operations that
//! delegate to `scirs2_core::simd_ops::SimdUnifiedOps` for AVX2/NEON acceleration
//! where available, with automatic scalar fallback.
//!
//! # Design rationale
//!
//! All functions operate on raw `&[f32]` / `&mut [f32]` rather than `ndarray`
//! types so that callers (e.g. `math_ops.rs` dispatchers) can work directly
//! against storage buffers without extra view construction overhead.
//!
//! # NaN semantics
//!
//! Binary arithmetic operations (`add`, `sub`, `mul`, `div`) follow IEEE 754:
//! NaN propagates naturally through the SIMD path.
//!
//! Activation operations (`relu_assign_f32`, `clamp_assign_f32`) follow
//! PyTorch semantics: NaN **passes through unchanged**.  Rust's `f32::max` and
//! `_mm256_max_ps` both implement IEEE 754-2008 maxNum which returns the
//! non-NaN operand when one argument is NaN, which would silently eat NaN
//! inputs.  The activation helpers therefore use explicit `is_nan()` guards
//! instead of calling `scirs2_core::simd::simd_relu_f32`.

use scirs2_core::simd_ops::SimdUnifiedOps;

// ─────────────────────────────────────────────────────────────────────────────
// Part 1 – In-place arithmetic assign  (zero-allocation)
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise in-place addition: `out[i] += rhs[i]`.
///
/// Uses AVX2 (x86_64) or NEON (aarch64) when available, scalar fallback
/// otherwise.  `out` and `rhs` must have the same length.
///
/// # Panics
/// Panics if `out.len() != rhs.len()`.
pub fn add_assign_f32(out: &mut [f32], rhs: &[f32]) {
    <f32 as SimdUnifiedOps>::simd_add_inplace(out, rhs);
}

/// Element-wise in-place subtraction: `out[i] -= rhs[i]`.
///
/// # Panics
/// Panics if `out.len() != rhs.len()`.
pub fn sub_assign_f32(out: &mut [f32], rhs: &[f32]) {
    <f32 as SimdUnifiedOps>::simd_sub_inplace(out, rhs);
}

/// Element-wise in-place multiplication: `out[i] *= rhs[i]`.
///
/// # Panics
/// Panics if `out.len() != rhs.len()`.
pub fn mul_assign_f32(out: &mut [f32], rhs: &[f32]) {
    <f32 as SimdUnifiedOps>::simd_mul_inplace(out, rhs);
}

/// Element-wise in-place division: `out[i] /= rhs[i]`.
///
/// Division by zero follows IEEE 754: produces `±Inf` or `NaN` (0/0).
///
/// # Panics
/// Panics if `out.len() != rhs.len()`.
pub fn div_assign_f32(out: &mut [f32], rhs: &[f32]) {
    <f32 as SimdUnifiedOps>::simd_div_inplace(out, rhs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 2 – Into-buffer arithmetic  (caller-supplies output buffer)
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise addition writing into a pre-allocated buffer: `out[i] = a[i] + b[i]`.
///
/// All three slices must have the same length.
///
/// # Panics
/// Panics if lengths do not match.
pub fn add_into_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    <f32 as SimdUnifiedOps>::simd_add_into(a, b, out);
}

/// Element-wise subtraction writing into a pre-allocated buffer: `out[i] = a[i] - b[i]`.
///
/// # Panics
/// Panics if lengths do not match.
pub fn sub_into_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    <f32 as SimdUnifiedOps>::simd_sub_into(a, b, out);
}

/// Element-wise multiplication writing into a pre-allocated buffer: `out[i] = a[i] * b[i]`.
///
/// # Panics
/// Panics if lengths do not match.
pub fn mul_into_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    <f32 as SimdUnifiedOps>::simd_mul_into(a, b, out);
}

/// Element-wise division writing into a pre-allocated buffer: `out[i] = a[i] / b[i]`.
///
/// Division by zero follows IEEE 754: produces `±Inf` or `NaN` (0/0).
///
/// # Panics
/// Panics if lengths do not match.
pub fn div_into_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    <f32 as SimdUnifiedOps>::simd_div_into(a, b, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 3 – In-place activation assign
// ─────────────────────────────────────────────────────────────────────────────

/// ReLU in-place: `out[i] = max(0.0, out[i])`.
///
/// Follows PyTorch semantics: NaN inputs are passed through unchanged.
///
/// Note: `scirs2_core::simd::simd_relu_f32` and Rust's `f32::max` both follow
/// IEEE 754-2008 maxNum, which silently converts NaN to `0.0`.  This function
/// uses an explicit `is_nan()` guard to preserve PyTorch's NaN-passthrough
/// contract instead.
pub fn relu_assign_f32(out: &mut [f32]) {
    for x in out.iter_mut() {
        if !x.is_nan() && *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Leaky ReLU in-place.
///
/// ```text
/// out[i] = if out[i] >= 0.0 { out[i] } else { negative_slope * out[i] }
/// ```
///
/// NaN inputs are passed through unchanged (PyTorch semantics).
pub fn leaky_relu_assign_f32(out: &mut [f32], negative_slope: f32) {
    for x in out.iter_mut() {
        if !x.is_nan() && *x < 0.0 {
            *x *= negative_slope;
        }
    }
}

/// Clamp in-place: `out[i] = out[i].clamp(min_val, max_val)`.
///
/// Follows PyTorch semantics: NaN inputs are passed through unchanged.
///
/// Note: Rust's `f32::clamp` panics when `min_val > max_val` and propagates
/// NaN from the bounds but **not** from the value (it returns the bound when
/// the value is NaN in some versions).  This implementation uses explicit
/// comparisons to guarantee NaN passthrough regardless of compiler version.
pub fn clamp_assign_f32(out: &mut [f32], min_val: f32, max_val: f32) {
    for x in out.iter_mut() {
        if x.is_nan() {
            // NaN passes through – PyTorch semantics.
            continue;
        }
        if *x < min_val {
            *x = min_val;
        } else if *x > max_val {
            *x = max_val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 4 – Op-kind enum for dispatcher use
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which binary f32 SIMD operation to perform.
///
/// Used by `math_ops.rs` dispatchers to select a concrete kernel without
/// branching at every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryF32Op {
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
}

impl BinaryF32Op {
    /// Dispatch the operation, writing into the caller-supplied output buffer.
    ///
    /// `a`, `b`, and `out` must all have the same length.
    ///
    /// # Panics
    /// Panics if lengths do not match.
    pub fn dispatch_into(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        match self {
            BinaryF32Op::Add => add_into_f32(a, b, out),
            BinaryF32Op::Sub => sub_into_f32(a, b, out),
            BinaryF32Op::Mul => mul_into_f32(a, b, out),
            BinaryF32Op::Div => div_into_f32(a, b, out),
        }
    }

    /// Dispatch in-place: `out op= rhs`.
    ///
    /// `out` and `rhs` must have the same length.
    ///
    /// # Panics
    /// Panics if lengths do not match.
    pub fn dispatch_inplace(self, out: &mut [f32], rhs: &[f32]) {
        match self {
            BinaryF32Op::Add => add_assign_f32(out, rhs),
            BinaryF32Op::Sub => sub_assign_f32(out, rhs),
            BinaryF32Op::Mul => mul_assign_f32(out, rhs),
            BinaryF32Op::Div => div_assign_f32(out, rhs),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TEST_SIZES: &[usize] = &[0, 1, 7, 8, 16, 1023, 1024, 4096, 65536];

    fn make_vecs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.1).collect();
        (a, b)
    }

    // ── Part 1: assign variants ──────────────────────────────────────────────

    #[test]
    fn test_add_assign_parity_with_scalar() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out_simd = a.clone();
            add_assign_f32(&mut out_simd, &b);
            let out_scalar: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            for (s, r) in out_simd.iter().zip(out_scalar.iter()) {
                assert_relative_eq!(s, r, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_sub_assign_parity_with_scalar() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out_simd = a.clone();
            sub_assign_f32(&mut out_simd, &b);
            let out_scalar: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
            for (s, r) in out_simd.iter().zip(out_scalar.iter()) {
                assert_relative_eq!(s, r, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_mul_assign_parity_with_scalar() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out_simd = a.clone();
            mul_assign_f32(&mut out_simd, &b);
            let out_scalar: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            for (s, r) in out_simd.iter().zip(out_scalar.iter()) {
                assert_relative_eq!(s, r, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_div_assign_parity_with_scalar() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out_simd = a.clone();
            div_assign_f32(&mut out_simd, &b);
            let out_scalar: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
            for (s, r) in out_simd.iter().zip(out_scalar.iter()) {
                assert_relative_eq!(s, r, epsilon = 1e-5);
            }
        }
    }

    // ── Part 2: into-buffer variants ─────────────────────────────────────────

    #[test]
    fn test_add_into_f32_parity() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out = vec![0.0f32; n];
            add_into_f32(&a, &b, &mut out);
            for ((aa, bb), rr) in a.iter().zip(b.iter()).zip(out.iter()) {
                assert_relative_eq!(*rr, *aa + *bb, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_sub_into_f32_parity() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out = vec![0.0f32; n];
            sub_into_f32(&a, &b, &mut out);
            for ((aa, bb), rr) in a.iter().zip(b.iter()).zip(out.iter()) {
                assert_relative_eq!(*rr, *aa - *bb, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_mul_into_f32_parity() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out = vec![0.0f32; n];
            mul_into_f32(&a, &b, &mut out);
            for ((aa, bb), rr) in a.iter().zip(b.iter()).zip(out.iter()) {
                assert_relative_eq!(*rr, *aa * *bb, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_div_into_f32_parity() {
        for &n in TEST_SIZES {
            let (a, b) = make_vecs(n);
            let mut out = vec![0.0f32; n];
            div_into_f32(&a, &b, &mut out);
            for ((aa, bb), rr) in a.iter().zip(b.iter()).zip(out.iter()) {
                assert_relative_eq!(*rr, *aa / *bb, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_div_by_zero_produces_inf() {
        let a = vec![1.0f32, -1.0, 0.0];
        let b = vec![0.0f32, 0.0, 0.0];
        let mut out = vec![0.0f32; 3];
        div_into_f32(&a, &b, &mut out);
        assert!(out[0].is_infinite() && out[0] > 0.0);
        assert!(out[1].is_infinite() && out[1] < 0.0);
        assert!(out[2].is_nan(), "0.0 / 0.0 should be NaN, got {}", out[2]);
    }

    // ── Part 3: activation variants ──────────────────────────────────────────

    #[test]
    fn test_relu_assign_edge_cases() {
        let mut data = vec![
            -1.0f32,
            -0.0,
            0.0,
            0.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        relu_assign_f32(&mut data);
        // -1.0 → 0.0
        assert_eq!(data[0], 0.0);
        // -0.0 → -0.0 (not negative, so passes through; compare as bits for sign)
        // PyTorch: relu(-0.0) = 0.0 (not < 0), so -0.0 passes through.
        assert!(!data[1].is_sign_negative() || data[1] == 0.0);
        // 0.0 → 0.0
        assert_eq!(data[2], 0.0);
        // 0.5 unchanged
        assert_eq!(data[3], 0.5);
        // NaN must pass through (PyTorch contract)
        assert!(
            data[4].is_nan(),
            "NaN should pass through relu, got {}",
            data[4]
        );
        // +INF unchanged
        assert_eq!(data[5], f32::INFINITY);
        // -INF → 0.0
        assert_eq!(data[6], 0.0);
    }

    #[test]
    fn test_relu_assign_large() {
        for &n in TEST_SIZES {
            let mut data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
            let expected: Vec<f32> = data
                .iter()
                .map(|&x| if x >= 0.0 { x } else { 0.0 })
                .collect();
            relu_assign_f32(&mut data);
            for (got, exp) in data.iter().zip(expected.iter()) {
                assert_relative_eq!(got, exp, epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_leaky_relu_assign() {
        let slope = 0.01_f32;
        let mut data = vec![-2.0f32, -1.0, 0.0, 1.0, f32::NAN, f32::NEG_INFINITY];
        leaky_relu_assign_f32(&mut data, slope);
        assert_relative_eq!(data[0], -0.02_f32, epsilon = 1e-7);
        assert_relative_eq!(data[1], -0.01_f32, epsilon = 1e-7);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 1.0);
        assert!(data[4].is_nan(), "NaN should pass through leaky relu");
        // -INF * 0.01 = -INF
        assert!(data[5].is_infinite() && data[5] < 0.0);
    }

    #[test]
    fn test_clamp_nan_passthrough() {
        let mut data = vec![f32::NAN, -2.0, 0.5, 2.0];
        clamp_assign_f32(&mut data, -1.0, 1.0);
        assert!(
            data[0].is_nan(),
            "NaN should pass through clamp, got {}",
            data[0]
        );
        assert_eq!(data[1], -1.0, "clamped to min");
        assert_eq!(data[2], 0.5, "unchanged");
        assert_eq!(data[3], 1.0, "clamped to max");
    }

    #[test]
    fn test_clamp_edge_values() {
        let mut data = vec![f32::NEG_INFINITY, f32::INFINITY, -1.0, 1.0, 0.0];
        clamp_assign_f32(&mut data, -1.0, 1.0);
        assert_eq!(data[0], -1.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], -1.0);
        assert_eq!(data[3], 1.0);
        assert_eq!(data[4], 0.0);
    }

    // ── Part 4: BinaryF32Op enum ─────────────────────────────────────────────

    #[test]
    fn test_binary_f32op_dispatch_into() {
        let a = vec![2.0f32, 4.0, 6.0, 8.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];

        BinaryF32Op::Add.dispatch_into(&a, &b, &mut out);
        assert_eq!(out, vec![3.0, 6.0, 9.0, 12.0]);

        BinaryF32Op::Sub.dispatch_into(&a, &b, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

        BinaryF32Op::Mul.dispatch_into(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 8.0, 18.0, 32.0]);

        BinaryF32Op::Div.dispatch_into(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_binary_f32op_dispatch_inplace() {
        let a = vec![2.0f32, 4.0, 6.0, 8.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];

        let mut out = a.clone();
        BinaryF32Op::Add.dispatch_inplace(&mut out, &b);
        assert_eq!(out, vec![3.0, 6.0, 9.0, 12.0]);

        let mut out = a.clone();
        BinaryF32Op::Sub.dispatch_inplace(&mut out, &b);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

        let mut out = a.clone();
        BinaryF32Op::Mul.dispatch_inplace(&mut out, &b);
        assert_eq!(out, vec![2.0, 8.0, 18.0, 32.0]);

        let mut out = a.clone();
        BinaryF32Op::Div.dispatch_inplace(&mut out, &b);
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_binary_f32op_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        // Must not panic on empty slices.
        BinaryF32Op::Add.dispatch_into(&a, &b, &mut out);
        BinaryF32Op::Add.dispatch_inplace(&mut out, &b);
    }

    #[test]
    fn test_binary_f32op_debug() {
        // Verify Debug impl doesn't panic (required for downstream logging).
        let _ = format!("{:?}", BinaryF32Op::Add);
        let _ = format!("{:?}", BinaryF32Op::Sub);
        let _ = format!("{:?}", BinaryF32Op::Mul);
        let _ = format!("{:?}", BinaryF32Op::Div);
    }
}
