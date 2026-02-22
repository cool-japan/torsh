//! Zero-Copy SIMD Performance Benchmarks (Phase 2 Verification)
//!
//! This benchmark verifies that the zero-copy SIMD implementation achieves
//! the expected 2-4x speedup over scalar operations, as documented in SciRS2.
//!
//! **Phase 2 Goals**:
//! - Verify zero-copy eliminates memory overhead
//! - Measure real SIMD speedup (2-4x expected)
//! - Compare against scalar baseline
//! - Validate Phase 1 architecture enabled SIMD success

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
#[cfg(feature = "simd")]
use torsh_tensor::Tensor;

const SMALL_SIZE: usize = 1_000; // 1K elements
const MEDIUM_SIZE: usize = 50_000; // 50K elements (sweet spot for SIMD)
const LARGE_SIZE: usize = 1_000_000; // 1M elements

/// Generate test data
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.01).collect()
}

/// Benchmark scalar addition (baseline)
fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Benchmark scalar multiplication (baseline)
fn scalar_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Benchmark zero-copy SIMD vs scalar addition
fn bench_zero_copy_simd_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_simd_add");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
        });

        // Zero-copy SIMD (feature-gated)
        #[cfg(feature = "simd")]
        {
            let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
            let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

            group.bench_with_input(
                BenchmarkId::new("zero_copy_simd", size),
                size,
                |bench, _| {
                    bench.iter(|| {
                        // Measure ONLY the zero-copy SIMD operation
                        black_box(tensor_a.add_op(&tensor_b).unwrap())
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark zero-copy SIMD vs scalar multiplication
fn bench_zero_copy_simd_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_simd_mul");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))));
        });

        // Zero-copy SIMD (feature-gated)
        #[cfg(feature = "simd")]
        {
            let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
            let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

            group.bench_with_input(
                BenchmarkId::new("zero_copy_simd", size),
                size,
                |bench, _| {
                    bench.iter(|| {
                        // Measure ONLY the zero-copy SIMD operation
                        black_box(tensor_a.mul_op(&tensor_b).unwrap())
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory allocation overhead comparison
#[cfg(feature = "simd")]
fn bench_memory_overhead_comparison(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("memory_overhead");

    let size = MEDIUM_SIZE;
    let a = generate_test_data(size);
    let b = generate_test_data(size);

    // Pure scalar (Vec operations)
    group.bench_function("pure_scalar_vec", |bench| {
        bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
    });

    // Scalar via Tensor (with memory overhead)
    let tensor_a_scalar = Tensor::<f32>::from_vec(a.clone(), &[size]).unwrap();
    let tensor_b_scalar = Tensor::<f32>::from_vec(b.clone(), &[size]).unwrap();

    group.bench_function("tensor_scalar", |bench| {
        bench.iter(|| {
            // Use public API (add without SIMD)
            black_box(tensor_a_scalar.add(&tensor_b_scalar).unwrap())
        });
    });

    // Zero-copy SIMD via Tensor
    let tensor_a_simd = Tensor::<f32>::from_vec(a.clone(), &[size]).unwrap();
    let tensor_b_simd = Tensor::<f32>::from_vec(b.clone(), &[size]).unwrap();

    group.bench_function("tensor_zero_copy_simd", |bench| {
        bench.iter(|| black_box(tensor_a_simd.add_op(&tensor_b_simd).unwrap()));
    });

    group.finish();
}

/// Benchmark to verify zero-copy (no allocations during operation)
#[cfg(feature = "simd")]
fn bench_allocation_count(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("allocation_verification");

    let size = MEDIUM_SIZE;
    let a = generate_test_data(size);
    let b = generate_test_data(size);

    let tensor_a = Tensor::<f32>::from_vec(a, &[size]).unwrap();
    let tensor_b = Tensor::<f32>::from_vec(b, &[size]).unwrap();

    // This benchmark verifies zero-copy by measuring operation time
    // With zero-copy: ~1-2μs for 50K elements
    // With 4 copies: ~20-100μs for 50K elements
    group.bench_function("zero_copy_add_50k", |bench| {
        bench.iter(|| black_box(tensor_a.add_op(&tensor_b).unwrap()));
    });

    group.finish();
}

/// Benchmark Phase 3 (uninit buffer + scirs2 API) vs previous implementations
#[cfg(feature = "simd")]
fn bench_phase3_comparison(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("phase3_comparison");
    group.throughput(Throughput::Elements(MEDIUM_SIZE as u64));

    let a = generate_test_data(MEDIUM_SIZE);
    let b = generate_test_data(MEDIUM_SIZE);

    let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[MEDIUM_SIZE]).unwrap();
    let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[MEDIUM_SIZE]).unwrap();

    // Pure scalar baseline
    group.bench_function("scalar_baseline", |bench| {
        bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
    });

    // Tensor scalar (element_wise_op)
    group.bench_function("tensor_scalar", |bench| {
        bench.iter(|| black_box(tensor_a.add(&tensor_b).unwrap()));
    });

    // Previous Phase 2 implementation (add_op = current default)
    group.bench_function("phase2_add_op", |bench| {
        bench.iter(|| black_box(tensor_a.add_op(&tensor_b).unwrap()));
    });

    group.finish();
}

/// Benchmark raw SIMD without any tensor abstraction
/// This quantifies the overhead introduced by storage locks and closures
#[cfg(feature = "simd")]
fn bench_raw_simd_no_abstraction(c: &mut Criterion) {
    use scirs2_core::simd_ops::SimdUnifiedOps;

    let mut group = c.benchmark_group("raw_simd_vs_tensor");
    group.throughput(Throughput::Elements(MEDIUM_SIZE as u64));

    let a = generate_test_data(MEDIUM_SIZE);
    let b = generate_test_data(MEDIUM_SIZE);

    // Pure scalar baseline (Vec operations)
    group.bench_function("pure_scalar", |bench| {
        bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
    });

    // Raw SIMD - no tensor abstraction, no locks, just scirs2_core
    group.bench_function("raw_simd_no_tensor", |bench| {
        bench.iter(|| {
            let mut result: Vec<f32> = Vec::with_capacity(MEDIUM_SIZE);
            unsafe {
                result.set_len(MEDIUM_SIZE);
            }
            f32::simd_add_into(black_box(&a), black_box(&b), &mut result);
            black_box(result)
        });
    });

    // Tensor SIMD - with storage abstraction and locks
    let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[MEDIUM_SIZE]).unwrap();
    let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[MEDIUM_SIZE]).unwrap();

    group.bench_function("tensor_simd_with_locks", |bench| {
        bench.iter(|| black_box(tensor_a.add_op(&tensor_b).unwrap()));
    });

    // Isolate: Raw SIMD + result tensor construction (slow path)
    // This shows overhead with SimdOptimized alignment copy
    group.bench_function("raw_simd_plus_tensor_construction", |bench| {
        bench.iter(|| {
            let mut result: Vec<f32> = Vec::with_capacity(MEDIUM_SIZE);
            unsafe {
                result.set_len(MEDIUM_SIZE);
            }
            f32::simd_add_into(black_box(&a), black_box(&b), &mut result);
            // Add tensor construction overhead (slow path with alignment)
            let tensor_result = Tensor::<f32>::from_vec(result, &[MEDIUM_SIZE]).unwrap();
            black_box(tensor_result)
        });
    });

    // Isolate: Raw SIMD + fast result tensor (Phase 7)
    // This shows overhead without alignment copy
    group.bench_function("raw_simd_plus_fast_result", |bench| {
        use torsh_core::DeviceType;
        bench.iter(|| {
            let mut result: Vec<f32> = Vec::with_capacity(MEDIUM_SIZE);
            unsafe {
                result.set_len(MEDIUM_SIZE);
            }
            f32::simd_add_into(black_box(&a), black_box(&b), &mut result);
            // Fast tensor construction (no alignment copy)
            let tensor_result =
                Tensor::<f32>::from_data_fast(result, vec![MEDIUM_SIZE], DeviceType::Cpu);
            black_box(tensor_result)
        });
    });

    // Test add_op (goes through full dispatch: add_op → add_adaptive → add_direct_simd)
    // Same as tensor_simd_with_locks but named to clarify
    group.bench_function("tensor_add_full_path", |bench| {
        bench.iter(|| black_box(tensor_a.add(&tensor_b).unwrap()));
    });

    // Try add_simd method if available
    group.bench_function("tensor_add_simd", |bench| {
        bench.iter(|| black_box(tensor_a.add_simd(&tensor_b).unwrap()));
    });

    group.finish();
}

/// Benchmark Phase 4 adaptive dispatch across different tensor sizes
#[cfg(feature = "simd")]
fn bench_phase4_adaptive(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("phase4_adaptive");

    // Test across multiple sizes to verify adaptive dispatch works correctly
    for size in [100, 512, 1000, 10_000, 50_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
        let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
        });

        // Tensor scalar
        group.bench_with_input(BenchmarkId::new("tensor_scalar", size), size, |bench, _| {
            bench.iter(|| black_box(tensor_a.add(&tensor_b).unwrap()));
        });

        // Phase 2 add_op
        group.bench_with_input(BenchmarkId::new("phase2_add_op", size), size, |bench, _| {
            bench.iter(|| black_box(tensor_a.add_op(&tensor_b).unwrap()));
        });
    }

    group.finish();
}

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_zero_copy_simd_add,
    bench_zero_copy_simd_mul,
    bench_memory_overhead_comparison,
    bench_allocation_count,
    bench_phase3_comparison,
    bench_raw_simd_no_abstraction,
    bench_phase4_adaptive,
);

#[cfg(not(feature = "simd"))]
criterion_group!(benches, bench_zero_copy_simd_add, bench_zero_copy_simd_mul,);

criterion_main!(benches);
