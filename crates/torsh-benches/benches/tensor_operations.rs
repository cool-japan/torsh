//! Tensor operations benchmarks

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use torsh_tensor::{Tensor, creation::*};
use torsh_core::dtype::DType;

fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));
        
        group.bench_with_input(
            BenchmarkId::new("zeros_f32", size),
            size,
            |b, &size| {
                b.iter(|| {
                    zeros::<f32>(&[size, size])
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ones_f32", size),
            size,
            |b, &size| {
                b.iter(|| {
                    ones::<f32>(&[size, size])
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rand_f32", size),
            size,
            |b, &size| {
                b.iter(|| {
                    rand::<f32>(&[size, size])
                });
            },
        );
    }
    
    group.finish();
}

fn bench_tensor_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_arithmetic");
    
    for size in [64, 128, 256, 512].iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));
        
        // Pre-create tensors for benchmarking
        let a = rand::<f32>(&[*size, *size]);
        let b = rand::<f32>(&[*size, *size]);
        
        group.bench_with_input(
            BenchmarkId::new("add", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    a.add(&b).unwrap()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mul", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    a.mul(&b).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    for size in [32, 64, 128, 256].iter() {
        // FLOPS for matrix multiplication: 2 * n^3
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));
        
        let a = rand::<f32>(&[*size, *size]);
        let b = rand::<f32>(&[*size, *size]);
        
        group.bench_with_input(
            BenchmarkId::new("matmul_f32", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    a.matmul(&b).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_tensor_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_reductions");
    
    for size in [100, 500, 1000, 5000].iter() {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));
        
        let tensor = rand::<f32>(&[*size, *size]);
        
        group.bench_with_input(
            BenchmarkId::new("sum", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    tensor.sum().unwrap()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mean", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    tensor.mean().unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    for size in [64, 128, 256, 512].iter() {
        let bytes = size * size * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));
        
        group.bench_with_input(
            BenchmarkId::new("allocation", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    // Allocate and immediately drop
                    let _tensor = zeros::<f32>(&[size, size]);
                });
            },
        );
        
        let source = rand::<f32>(&[*size, *size]);
        group.bench_with_input(
            BenchmarkId::new("clone", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    source.clone()
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_arithmetic,
    bench_matrix_multiplication,
    bench_tensor_reductions,
    bench_memory_operations
);

criterion_main!(benches);