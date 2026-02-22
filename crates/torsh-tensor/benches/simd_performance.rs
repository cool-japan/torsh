//! SIMD Performance Benchmarks for ToRSh Tensor Operations
//!
//! Phase 3: Memory-Aligned SIMD Performance Validation
//!
//! **Performance Targets**:
//! - Adaptive SIMD vs scalar: 2-4x speedup
//! - Medium arrays (10K-100K elements): Up to 14.17x speedup
//! - Cross-platform consistency (x86_64 AVX2, ARM64 NEON)
//!
//! **Test Matrix**:
//! - Small arrays (64-1K elements): Validate overhead vs benefit
//! - Medium arrays (10K-100K elements): Target maximum speedup
//! - Large arrays (1M+ elements): Validate scalability
//! - Activation functions: relu, sigmoid, gelu with SIMD
//! - Element-wise operations: add, mul, div with SIMD
//! - Reduction operations: dot product with SIMD

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use torsh_tensor::Tensor;

const SMALL_SIZE: usize = 1_000; // 1K elements
const MEDIUM_SIZE: usize = 50_000; // 50K elements (target for 14.17x)
const LARGE_SIZE: usize = 1_000_000; // 1M elements

/// Generate test data with specific size
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

/// Benchmark scalar dot product (baseline)
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Benchmark SIMD vs Scalar Addition
fn bench_simd_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_add");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))));
        });

        // SIMD optimized (when available)
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensors ONCE outside the measurement loop
            let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
            let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    black_box(tensor_a.add_op(&tensor_b).unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD vs Scalar Multiplication
fn bench_simd_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_mul");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))));
        });

        // SIMD optimized
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensors ONCE outside the measurement loop
            let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
            let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    black_box(tensor_a.mul_op(&tensor_b).unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD vs Scalar Dot Product
fn bench_simd_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dot");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_test_data(*size);
        let b = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(scalar_dot(black_box(&a), black_box(&b))));
        });

        // SIMD optimized
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensors ONCE outside the measurement loop
            let tensor_a = Tensor::<f32>::from_vec(a.clone(), &[*size]).unwrap();
            let tensor_b = Tensor::<f32>::from_vec(b.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    let mul_result = tensor_a.mul_op(&tensor_b).unwrap();
                    black_box(mul_result.sum().unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD Activation Functions (ReLU)
fn bench_simd_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_relu");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01 - 250.0).collect(); // Mix of positive and negative

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(data.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>()));
        });

        // SIMD optimized
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensor ONCE outside the measurement loop
            let tensor = Tensor::<f32>::from_vec(data.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    black_box(tensor.relu().unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD Activation Functions (Sigmoid)
fn bench_simd_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sigmoid");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data = generate_test_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(
                    data.iter()
                        .map(|&x| 1.0 / (1.0 + (-x).exp()))
                        .collect::<Vec<_>>(),
                )
            });
        });

        // SIMD optimized
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensor ONCE outside the measurement loop
            let tensor = Tensor::<f32>::from_vec(data.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    black_box(tensor.sigmoid().unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD Activation Functions (GELU)
fn bench_simd_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_gelu");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data = generate_test_data(*size);

        // Scalar baseline (approximate GELU)
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(
                    data.iter()
                        .map(|&x| {
                            0.5 * x
                                * (1.0
                                    + ((2.0 / std::f32::consts::PI).sqrt()
                                        * (x + 0.044715 * x.powi(3)))
                                    .tanh())
                        })
                        .collect::<Vec<_>>(),
                )
            });
        });

        // SIMD optimized
        #[cfg(feature = "simd")]
        {
            // Setup: Create tensor ONCE outside the measurement loop
            let tensor = Tensor::<f32>::from_vec(data.clone(), &[*size]).unwrap();

            group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
                bench.iter(|| {
                    // Measure ONLY the operation, not tensor creation
                    black_box(tensor.gelu().unwrap())
                });
            });
        }
    }

    group.finish();
}

/// Cross-platform SIMD feature detection benchmark
fn bench_simd_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_features");

    // Detect and report available SIMD features
    let features = vec![
        ("avx512f", cfg!(target_feature = "avx512f")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("sse2", cfg!(target_feature = "sse2")),
        ("neon", cfg!(target_arch = "aarch64")),
    ];

    println!("\n=== SIMD Feature Detection ===");
    for (name, available) in features {
        println!("{}: {}", name, if available { "✓" } else { "✗" });
    }

    // Benchmark feature detection overhead
    group.bench_function("feature_detection", |bench| {
        bench.iter(|| {
            black_box(
                cfg!(target_feature = "avx512f")
                    || cfg!(target_feature = "avx2")
                    || cfg!(target_feature = "sse2")
                    || cfg!(target_arch = "aarch64"),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_add,
    bench_simd_mul,
    bench_simd_dot,
    bench_simd_relu,
    bench_simd_sigmoid,
    bench_simd_gelu,
    bench_simd_features,
);

criterion_main!(benches);
