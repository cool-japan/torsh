//! Phase 4 Chunking Validation Benchmarks
//!
//! Validates that the chunked kernels from `optimized_kernels.rs` are not slower
//! than naive scalar baselines, and confirms the claimed 15-30% improvement from
//! the Phase 4 intelligent-chunking integration.
//!
//! Benchmark groups:
//! - `matmul`       — `optimized_matmul` vs. naive triple-loop (32², 128², 512²)
//! - `elementwise`  — `chunked_elementwise` vs. naive element-add (1K, 64K, 1M)
//! - `sum`          — `chunked_sum` vs. `Iterator::sum` (1K, 64K, 1M)
//! - `mean`         — `chunked_mean` vs. naive mean (1K, 64K, 1M)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use torsh_backend::cpu::optimized_kernels::{
    chunked_elementwise, chunked_mean, chunked_sum, optimized_matmul,
};

// ---------------------------------------------------------------------------
// Naive baseline implementations
// ---------------------------------------------------------------------------

fn naive_matmul(a: &[f32], b: &[f32], result: &mut [f32], m: usize, n: usize, k: usize) {
    result.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                result[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

fn naive_elementwise_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    for ((r, &ai), &bi) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *r = ai + bi;
    }
}

fn naive_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

fn naive_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}

// ---------------------------------------------------------------------------
// Input generators — deterministic, no external RNG dependency
// ---------------------------------------------------------------------------

fn make_vec_a(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * 0.001).collect()
}

fn make_vec_b(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * 0.001 + 1.0).collect()
}

// ---------------------------------------------------------------------------
// Benchmark: matrix multiplication
// ---------------------------------------------------------------------------

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for &n in &[32_usize, 128, 512] {
        // Throughput expressed as FLOPs proxy: n³ multiply-add pairs
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a = make_vec_a(n * n);
        let b = make_vec_b(n * n);

        group.bench_with_input(BenchmarkId::new("chunked", n), &n, |bench, &size| {
            let mut result = vec![0.0_f32; size * size];
            bench.iter(|| {
                optimized_matmul(
                    black_box(&a),
                    black_box(&b),
                    &mut result,
                    size,
                    size,
                    size,
                    false,
                    false,
                )
                .unwrap();
                black_box(result[0]);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bench, &size| {
            let mut result = vec![0.0_f32; size * size];
            bench.iter(|| {
                naive_matmul(black_box(&a), black_box(&b), &mut result, size, size, size);
                black_box(result[0]);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: elementwise add
// ---------------------------------------------------------------------------

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    for &len in &[1_024_usize, 65_536, 1_048_576] {
        group.throughput(Throughput::Elements(len as u64));

        let a = make_vec_a(len);
        let b = make_vec_b(len);

        group.bench_with_input(BenchmarkId::new("chunked", len), &len, |bench, &size| {
            let mut result = vec![0.0_f32; size];
            bench.iter(|| {
                chunked_elementwise(black_box(&a), black_box(&b), &mut result, |x, y| x + y)
                    .unwrap();
                black_box(result[0]);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", len), &len, |bench, &size| {
            let mut result = vec![0.0_f32; size];
            bench.iter(|| {
                naive_elementwise_add(black_box(&a), black_box(&b), &mut result);
                black_box(result[0]);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: sum reduction
// ---------------------------------------------------------------------------

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for &len in &[1_024_usize, 65_536, 1_048_576] {
        group.throughput(Throughput::Elements(len as u64));

        let data = make_vec_a(len);

        group.bench_with_input(BenchmarkId::new("chunked", len), &len, |bench, _| {
            bench.iter(|| {
                let s = chunked_sum(black_box(&data));
                black_box(s);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", len), &len, |bench, _| {
            bench.iter(|| {
                let s = naive_sum(black_box(&data));
                black_box(s);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: mean reduction
// ---------------------------------------------------------------------------

fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean");

    for &len in &[1_024_usize, 65_536, 1_048_576] {
        group.throughput(Throughput::Elements(len as u64));

        let data = make_vec_a(len);

        group.bench_with_input(BenchmarkId::new("chunked", len), &len, |bench, _| {
            bench.iter(|| {
                let m = chunked_mean(black_box(&data));
                black_box(m);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", len), &len, |bench, _| {
            bench.iter(|| {
                let m = naive_mean(black_box(&data));
                black_box(m);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_matmul,
    bench_elementwise,
    bench_sum,
    bench_mean
);
criterion_main!(benches);
