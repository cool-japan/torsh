//! Performance regression baselines for torsh-tensor SIMD operations.
//!
//! To record a baseline:
//!   cargo bench --bench regression_baselines -- --save-baseline v0.1.2-pre
//!
//! To compare against the baseline:
//!   cargo bench --bench regression_baselines -- --baseline v0.1.2-pre

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use torsh_core::device::DeviceType;
use torsh_tensor::simd_ops_f32;
use torsh_tensor::Tensor;

// ─── Raw SIMD helper benchmarks ─────────────────────────────────────────────

fn bench_add_into_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_into_f32");
    for &n in &[256usize, 4096, 65536] {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                simd_ops_f32::add_into_f32(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_add_assign_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_assign_f32");
    for &n in &[256usize, 4096, 65536] {
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut a = vec![1.0f32; n];
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                simd_ops_f32::add_assign_f32(black_box(&mut a), black_box(&b));
            });
        });
    }
    group.finish();
}

fn bench_relu_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu_inplace");
    for &n in &[256usize, 4096, 65536] {
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                simd_ops_f32::relu_assign_f32(black_box(&mut data));
            });
        });
    }
    group.finish();
}

fn bench_clamp_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp_inplace");
    for &n in &[256usize, 4096, 65536] {
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                simd_ops_f32::clamp_assign_f32(
                    black_box(&mut data),
                    black_box(-100.0),
                    black_box(100.0),
                );
            });
        });
    }
    group.finish();
}

// ─── Tensor-level benchmarks ─────────────────────────────────────────────────

fn bench_tensor_add_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add_f32");
    for &n in &[256usize, 4096, 65536] {
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = black_box(a.add(&b)).expect("add should succeed");
            });
        });
    }
    group.finish();
}

fn bench_tensor_relu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_relu_f32");
    for &n in &[256usize, 4096, 65536] {
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        let a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = black_box(a.relu()).expect("relu should succeed");
            });
        });
    }
    group.finish();
}

fn bench_tensor_add_inplace_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add_inplace_f32");
    for &n in &[256usize, 4096, 65536] {
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let mut a = Tensor::<f32>::from_data(a_data.clone(), vec![n], DeviceType::Cpu)
                    .expect("tensor creation should succeed");
                a.add_(&b).expect("add_ should succeed");
                black_box(a)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add_into_f32,
    bench_add_assign_f32,
    bench_relu_inplace,
    bench_clamp_inplace,
    bench_tensor_add_f32,
    bench_tensor_add_inplace_f32,
    bench_tensor_relu_f32,
);
criterion_main!(benches);
