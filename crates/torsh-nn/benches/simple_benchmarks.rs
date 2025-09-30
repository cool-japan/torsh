//! Simple benchmarks for neural network layers in torsh-nn
//!
//! This benchmark suite measures the performance of basic neural network layers

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_nn::container::Sequential;
use torsh_nn::layers::{Dropout, Linear, ReLU};
use torsh_nn::Module;
use torsh_tensor::creation::*;

/// Benchmark linear layers with different sizes
fn bench_linear_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Layers");

    let configs = vec![
        ("Small", 128, 64),
        ("Medium", 512, 256),
        ("Large", 2048, 1024),
    ];

    for (name, input_size, output_size) in configs {
        let linear = Linear::new(input_size, output_size, true);
        let input = randn(&[32, input_size]);
        let throughput = Throughput::Elements((32 * input_size) as u64);
        group.throughput(throughput);

        group.bench_with_input(
            BenchmarkId::new(name, "forward"),
            &(linear, input),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark activation functions
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Activations");

    let input = randn(&[32, 512]);
    let throughput = Throughput::Elements((32 * 512) as u64);
    group.throughput(throughput);

    let relu = ReLU::new();
    group.bench_with_input(
        BenchmarkId::new("ReLU", "forward"),
        &(relu, input.clone()),
        |b, (layer, input)| {
            b.iter(|| {
                let output = layer.forward(black_box(input));
                black_box(output)
            })
        },
    );

    group.finish();
}

/// Benchmark simple sequential models
fn bench_sequential_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequential_Models");

    // Create a simple MLP
    let mlp = Sequential::new()
        .add_op(Linear::new(784, 256, true))
        .add_op(ReLU::new())
        .add_op(Dropout::new(0.5))
        .add_op(Linear::new(256, 128, true))
        .add_op(ReLU::new())
        .add_op(Linear::new(128, 10, true));

    let input = randn(&[32, 784]);
    let throughput = Throughput::Elements((32 * 784) as u64);
    group.throughput(throughput);

    group.bench_with_input(
        BenchmarkId::new("MLP", "MNIST_like"),
        &(mlp, input),
        |b, (model, input)| {
            b.iter(|| {
                let output = model.forward(black_box(input));
                black_box(output)
            })
        },
    );

    group.finish();
}

/// Benchmark memory efficiency with different batch sizes
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Scaling");

    let batch_sizes = vec![1, 4, 16, 64];
    let input_size = 512;
    let output_size = 256;

    for &batch_size in &batch_sizes {
        let linear = Linear::new(input_size, output_size, true);
        let input = randn(&[batch_size, input_size]);
        let throughput = Throughput::Elements((batch_size * input_size) as u64);
        group.throughput(throughput);

        group.bench_with_input(
            BenchmarkId::new("Linear_Scaling", batch_size),
            &(linear, input),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_layers,
    bench_activations,
    bench_sequential_models,
    bench_memory_scaling
);

criterion_main!(benches);
