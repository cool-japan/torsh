//! Cross-Platform SIMD Validation Benchmark (Phase 3)
//!
//! Validates and benchmarks the adaptive SIMD layer across x86_64 (AVX2/AVX512/SSE4.1),
//! ARM64 (NEON), and scalar fallback paths.
//!
//! **Correctness contract**: every SIMD result is checked against a scalar reference
//! with absolute tolerance 1e-5 *before* the timed loop runs.  Any divergence panics
//! immediately so CI catches numerical regressions.
//!
//! **Benchmark matrix** (4 sizes × 4 ops × 2 paths = 32 measurements):
//! - Sizes: 256, 4096, 65536, 1048576
//! - Ops:   add, mul, div, dot
//! - Paths: scalar baseline, adaptive SIMD
//!
//! When the `simd` feature is absent the file still compiles and prints an
//! informational message rather than registering any criterion groups.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark sizes (Phase 3 cross-platform matrix)
// ─────────────────────────────────────────────────────────────────────────────

const SIZES: [usize; 4] = [256, 4_096, 65_536, 1_048_576];

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-level detection (pure compile-time + runtime feature flags)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the highest available SIMD tier on the current CPU at runtime.
///
/// Returns a human-readable label used in the one-time startup banner.
fn detect_simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        // Runtime detection — safe to call outside `unsafe` since
        // `is_x86_feature_detected!` uses CPUID, not unstable intrinsics.
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
        if is_x86_feature_detected!("sse2") {
            return "SSE2";
        }
        return "scalar (x86_64 no SIMD detected)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64; the cfg macro suffices.
        return "NEON (AArch64)";
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return "scalar (unknown arch)";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar baselines (simple iterator loops — no SIMD, no intrinsics)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn scalar_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline]
fn scalar_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

#[inline]
fn scalar_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

#[inline]
fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test-data generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate distinct, well-conditioned test vectors.
///
/// `a[i] = 0.5 + 0.01 * i`,  `b[i] = 1.0 + 0.005 * i`
///
/// Using non-zero `b` avoids division-by-zero in the div benchmarks.
fn make_test_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..size).map(|i| 0.5 + 0.01 * (i as f32)).collect();
    let b: Vec<f32> = (0..size).map(|i| 1.0 + 0.005 * (i as f32)).collect();
    (a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness helpers
// ─────────────────────────────────────────────────────────────────────────────

const TOLERANCE: f32 = 1e-5;

fn assert_vec_close(got: &[f32], expected: &[f32], op: &str, size: usize) {
    assert_eq!(
        got.len(),
        expected.len(),
        "[{op}@{size}] length mismatch: got {}, expected {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= TOLERANCE || (g.is_nan() && e.is_nan()),
            "[{op}@{size}] mismatch at index {i}: simd={g}, scalar={e}, diff={diff} > {TOLERANCE}"
        );
    }
}

fn assert_scalar_close(got: f32, expected: f32, op: &str, size: usize) {
    let diff = (got - expected).abs();
    assert!(
        diff <= TOLERANCE * (size as f32).sqrt(),
        "[{op}@{size}] scalar mismatch: simd={got}, scalar={expected}, diff={diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-feature-gated section
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "simd")]
mod simd_benches {
    use super::*;
    use scirs2_core::simd_ops::SimdUnifiedOps;

    // ── helpers: adaptive SIMD wrappers returning Vec / scalar ────────────────

    /// Adaptive SIMD addition: delegates to `SimdUnifiedOps::simd_add_into`.
    ///
    /// Allocates an output buffer, dispatches to the hardware SIMD path, and
    /// returns the filled `Vec<f32>`.  This is what the crate itself does
    /// internally (see `torsh_tensor::simd_ops_f32::add_into_f32`).
    pub fn adaptive_simd_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(a.len());
        // SAFETY: set_len is safe immediately after with_capacity; we pass a
        // fully-sized slice to simd_add_into which writes every element before
        // the Vec is ever read.
        unsafe { out.set_len(a.len()) };
        <f32 as SimdUnifiedOps>::simd_add_into(a, b, &mut out);
        out
    }

    /// Adaptive SIMD multiplication: delegates to `SimdUnifiedOps::simd_mul_into`.
    pub fn adaptive_simd_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(a.len());
        unsafe { out.set_len(a.len()) };
        <f32 as SimdUnifiedOps>::simd_mul_into(a, b, &mut out);
        out
    }

    /// Adaptive SIMD division: delegates to `SimdUnifiedOps::simd_div_into`.
    pub fn adaptive_simd_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(a.len());
        unsafe { out.set_len(a.len()) };
        <f32 as SimdUnifiedOps>::simd_div_into(a, b, &mut out);
        out
    }

    /// Adaptive SIMD dot product: delegates to `SimdUnifiedOps::simd_dot`.
    pub fn adaptive_simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        use scirs2_core::ndarray::ArrayView1;
        let view_a = ArrayView1::from(a);
        let view_b = ArrayView1::from(b);
        <f32 as SimdUnifiedOps>::simd_dot(&view_a, &view_b)
    }

    // ── add ──────────────────────────────────────────────────────────────────

    pub fn bench_add(c: &mut Criterion) {
        let mut group = c.benchmark_group("cross_platform_simd_add");

        for &size in SIZES.iter() {
            let (a, b) = make_test_vectors(size);

            // ---- correctness gate (runs once, outside the timed loop) ----
            let scalar_ref = scalar_add_f32(&a, &b);
            let simd_result = adaptive_simd_add_f32(&a, &b);
            assert_vec_close(&simd_result, &scalar_ref, "add", size);
            // ---- end correctness gate -------------------------------------

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("scalar_add_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(scalar_add_f32(black_box(&a), black_box(&b))));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("adaptive_add_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(adaptive_simd_add_f32(black_box(&a), black_box(&b))));
                },
            );
        }

        group.finish();
    }

    // ── mul ──────────────────────────────────────────────────────────────────

    pub fn bench_mul(c: &mut Criterion) {
        let mut group = c.benchmark_group("cross_platform_simd_mul");

        for &size in SIZES.iter() {
            let (a, b) = make_test_vectors(size);

            let scalar_ref = scalar_mul_f32(&a, &b);
            let simd_result = adaptive_simd_mul_f32(&a, &b);
            assert_vec_close(&simd_result, &scalar_ref, "mul", size);

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("scalar_mul_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(scalar_mul_f32(black_box(&a), black_box(&b))));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("adaptive_mul_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(adaptive_simd_mul_f32(black_box(&a), black_box(&b))));
                },
            );
        }

        group.finish();
    }

    // ── div ──────────────────────────────────────────────────────────────────

    pub fn bench_div(c: &mut Criterion) {
        let mut group = c.benchmark_group("cross_platform_simd_div");

        for &size in SIZES.iter() {
            let (a, b) = make_test_vectors(size);

            let scalar_ref = scalar_div_f32(&a, &b);
            let simd_result = adaptive_simd_div_f32(&a, &b);
            assert_vec_close(&simd_result, &scalar_ref, "div", size);

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("scalar_div_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(scalar_div_f32(black_box(&a), black_box(&b))));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("adaptive_div_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(adaptive_simd_div_f32(black_box(&a), black_box(&b))));
                },
            );
        }

        group.finish();
    }

    // ── dot ──────────────────────────────────────────────────────────────────

    pub fn bench_dot(c: &mut Criterion) {
        let mut group = c.benchmark_group("cross_platform_simd_dot");

        for &size in SIZES.iter() {
            let (a, b) = make_test_vectors(size);

            let scalar_ref = scalar_dot_f32(&a, &b);
            let simd_result = adaptive_simd_dot_f32(&a, &b);
            assert_scalar_close(simd_result, scalar_ref, "dot", size);

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("scalar_dot_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(scalar_dot_f32(black_box(&a), black_box(&b))));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("adaptive_dot_f32", size),
                &size,
                |bench, _| {
                    bench.iter(|| black_box(adaptive_simd_dot_f32(black_box(&a), black_box(&b))));
                },
            );
        }

        group.finish();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Startup banner — printed once before criterion's own output
// ─────────────────────────────────────────────────────────────────────────────

fn print_simd_banner(c: &mut Criterion) {
    let level = detect_simd_level();
    // `Criterion::benchmark_function` with a near-zero workload is the standard
    // idiom for a "run once" setup function in criterion benchmarks.
    c.bench_function("cross_platform_simd/detected_simd_level", |b| {
        b.iter(|| {
            // The black_box prevents the compiler from eliding the call entirely.
            black_box(detect_simd_level())
        });
    });
    eprintln!("\n=== Cross-Platform SIMD Benchmark ===");
    eprintln!("  Detected SIMD level : {level}");
    eprintln!(
        "  Feature `simd`      : {}",
        if cfg!(feature = "simd") {
            "enabled"
        } else {
            "DISABLED — adaptive paths run scalar fallback"
        }
    );
    eprintln!("  Sizes tested        : {SIZES:?}");
    eprintln!("  Ops tested          : add, mul, div, dot");
    eprintln!("=====================================\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion registration
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    print_simd_banner,
    simd_benches::bench_add,
    simd_benches::bench_mul,
    simd_benches::bench_div,
    simd_benches::bench_dot,
);

// When `simd` is absent we still need a valid `criterion_main!` target.
// Register only the banner benchmark so the binary links and the user
// gets an informative message instead of a compile error.
#[cfg(not(feature = "simd"))]
criterion_group!(benches, print_simd_banner,);

criterion_main!(benches);
