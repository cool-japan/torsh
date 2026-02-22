//! Comprehensive benchmarks for neural network layers in torsh-nn
//!
//! This benchmark suite measures the performance of different neural network layers
//! including forward pass time, memory usage, and throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_nn::container::Sequential;
use torsh_nn::layers::*;
use torsh_nn::Module;
use torsh_tensor::creation::*;

/// Benchmark configuration for different layer types
#[derive(Clone)]
struct BenchConfig {
    name: &'static str,
    input_shape: Vec<usize>,
    batch_sizes: Vec<usize>,
}

impl BenchConfig {
    fn new(name: &'static str, input_shape: Vec<usize>, batch_sizes: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            batch_sizes,
        }
    }
}

/// Benchmark linear layers with different sizes
fn bench_linear_layers(c: &mut Criterion) {
    let configs = vec![
        BenchConfig::new("Linear-Small", vec![128], vec![1, 8, 32, 128]),
        BenchConfig::new("Linear-Medium", vec![512], vec![1, 8, 32, 128]),
        BenchConfig::new("Linear-Large", vec![2048], vec![1, 8, 32, 128]),
        BenchConfig::new("Linear-XLarge", vec![4096], vec![1, 8, 32, 128]),
    ];

    let mut group = c.benchmark_group("Linear_Layers");

    for config in configs {
        let input_size = config.input_shape[0];
        let output_size = input_size / 2; // Reduce dimensionality by half

        // Create the linear layer
        let linear = Linear::new(input_size, output_size, true);

        for &batch_size in &config.batch_sizes {
            let input = randn(&[batch_size, input_size]);
            let throughput = Throughput::Elements((batch_size * input_size) as u64);
            group.throughput(throughput);

            group.bench_with_input(
                BenchmarkId::new(config.name, format!("batch_{}", batch_size)),
                &(linear.clone(), input),
                |b, (layer, input)| {
                    b.iter(|| {
                        let output = layer.forward(black_box(input));
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark convolutional layers
fn bench_conv_layers(c: &mut Criterion) {
    let configs = vec![
        BenchConfig::new("Conv2d-Small", vec![3, 32, 32], vec![1, 8, 32]),
        BenchConfig::new("Conv2d-Medium", vec![64, 64, 64], vec![1, 8, 32]),
        BenchConfig::new("Conv2d-Large", vec![128, 128, 128], vec![1, 8, 16]),
    ];

    let mut group = c.benchmark_group("Conv_Layers");

    for config in configs {
        let [channels, height, width] = [
            config.input_shape[0],
            config.input_shape[1],
            config.input_shape[2],
        ];

        // Create different conv configurations
        let conv_configs = vec![
            (
                "3x3_stride1",
                Conv2d::new(channels, channels, 3, 1, 1, 1, 1, true),
            ),
            (
                "3x3_stride2",
                Conv2d::new(channels, channels, 3, 2, 1, 1, 1, true),
            ),
            (
                "1x1_stride1",
                Conv2d::new(channels, channels, 1, 1, 0, 1, 1, true),
            ),
        ];

        for (conv_name, conv_layer) in conv_configs {
            for &batch_size in &config.batch_sizes {
                let input = randn(&[batch_size, channels, height, width]);
                let throughput =
                    Throughput::Elements((batch_size * channels * height * width) as u64);
                group.throughput(throughput);

                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}_{}", config.name, conv_name),
                        format!("batch_{}", batch_size),
                    ),
                    &(conv_layer.clone(), input),
                    |b, (layer, input)| {
                        b.iter(|| {
                            let output = layer.forward(black_box(input));
                            black_box(output)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark activation functions
fn bench_activation_functions(c: &mut Criterion) {
    let input_shapes = vec![
        ("Small", vec![32, 128]),
        ("Medium", vec![32, 512]),
        ("Large", vec![32, 2048]),
    ];

    let mut group = c.benchmark_group("Activation_Functions");

    for (size_name, shape) in input_shapes {
        let input = randn(&shape);
        let throughput = Throughput::Elements(shape.iter().product::<usize>() as u64);
        group.throughput(throughput);

        // Test different activation functions
        let activations: Vec<(&str, Box<dyn Module>)> = vec![
            ("ReLU", Box::new(ReLU::new())),
            ("LeakyReLU", Box::new(LeakyReLU::new(0.01))),
            ("GELU", Box::new(GELU::new())),
            ("Swish", Box::new(Swish::new())),
            ("Mish", Box::new(Mish::new())),
            ("Sigmoid", Box::new(Sigmoid::new())),
            ("Tanh", Box::new(Tanh::new())),
            ("Softmax", Box::new(Softmax::new(Some(1)))),
        ];

        for (act_name, activation) in activations {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", act_name, size_name), "forward"),
                &(activation, input.clone()),
                |b, (layer, input)| {
                    b.iter(|| {
                        let output = layer.forward(black_box(input));
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark normalization layers
fn bench_normalization_layers(c: &mut Criterion) {
    let configs = vec![
        ("BatchNorm2d", vec![8, 64, 32, 32]),
        ("LayerNorm", vec![8, 64, 32, 32]),
        ("GroupNorm", vec![8, 64, 32, 32]),
        ("InstanceNorm2d", vec![8, 64, 32, 32]),
    ];

    let mut group = c.benchmark_group("Normalization_Layers");

    for (norm_name, shape) in configs {
        let input = randn(&shape);
        let throughput = Throughput::Elements(shape.iter().product::<usize>() as u64);
        group.throughput(throughput);

        let layer: Box<dyn Module> = match norm_name {
            "BatchNorm2d" => Box::new(BatchNorm2d::new(shape[1]).unwrap()),
            "LayerNorm" => Box::new(LayerNorm::new(shape[1..].to_vec()).unwrap()),
            "GroupNorm" => Box::new(GroupNorm::new(8, shape[1]).unwrap()), // 8 groups
            "InstanceNorm2d" => Box::new(InstanceNorm2d::new(shape[1]).unwrap()),
            _ => unreachable!(),
        };

        group.bench_with_input(
            BenchmarkId::new(norm_name, "forward"),
            &(layer, input),
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

/// Benchmark pooling layers
fn bench_pooling_layers(c: &mut Criterion) {
    let input_shapes = vec![
        ("Small", vec![8, 32, 32, 32]),
        ("Medium", vec![8, 64, 64, 64]),
        ("Large", vec![8, 128, 128, 128]),
    ];

    let mut group = c.benchmark_group("Pooling_Layers");

    for (size_name, shape) in input_shapes {
        let input = randn(&shape);
        let throughput = Throughput::Elements(shape.iter().product::<usize>() as u64);
        group.throughput(throughput);

        let pooling_layers: Vec<(&str, Box<dyn Module>)> = vec![
            (
                "MaxPool2d",
                Box::new(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false)),
            ),
            (
                "AvgPool2d",
                Box::new(AvgPool2d::new((2, 2), Some((2, 2)), (0, 0), false, true)),
            ),
            (
                "AdaptiveMaxPool2d",
                Box::new(AdaptiveMaxPool2d::new((Some(8), Some(8)))),
            ),
            (
                "AdaptiveAvgPool2d",
                Box::new(AdaptiveAvgPool2d::new((Some(8), Some(8)))),
            ),
        ];

        for (pool_name, pool_layer) in pooling_layers {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", pool_name, size_name), "forward"),
                &(pool_layer, input.clone()),
                |b, (layer, input)| {
                    b.iter(|| {
                        let output = layer.forward(black_box(input));
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark recurrent layers
fn bench_recurrent_layers(c: &mut Criterion) {
    let configs = vec![
        ("Small", 64, 128, 10), // input_size, hidden_size, seq_len
        ("Medium", 128, 256, 20),
        ("Large", 256, 512, 50),
    ];

    let mut group = c.benchmark_group("Recurrent_Layers");

    for (size_name, input_size, hidden_size, seq_len) in configs {
        let batch_sizes = vec![1, 8, 32];

        for &batch_size in &batch_sizes {
            let input = randn(&[batch_size, seq_len, input_size]);
            let throughput = Throughput::Elements((batch_size * seq_len * input_size) as u64);
            group.throughput(throughput);

            let rnn_layers: Vec<(&str, Box<dyn Module>)> = vec![
                ("RNN", Box::new(RNN::new(input_size, hidden_size, 1))),
                ("LSTM", Box::new(LSTM::new(input_size, hidden_size, 1))),
                ("GRU", Box::new(GRU::new(input_size, hidden_size, 1))),
            ];

            for (rnn_name, rnn_layer) in rnn_layers {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}_{}_{}", rnn_name, size_name, batch_size),
                        "forward",
                    ),
                    &(rnn_layer, input.clone()),
                    |b, (layer, input)| {
                        b.iter(|| {
                            let output = layer.forward(black_box(input));
                            black_box(output)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark complete neural network architectures
fn bench_complete_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("Complete_Models");

    // Create a simple MLP model
    let mlp = Sequential::new()
        .add_op(Linear::new(784, 256, true))
        .add_op(ReLU::new())
        .add_op(Dropout::new(0.5))
        .add_op(Linear::new(256, 128, true))
        .add_op(ReLU::new())
        .add_op(Dropout::new(0.5))
        .add_op(Linear::new(128, 10, true));

    // Create a simple conv stack
    let conv_stack = Sequential::new()
        .add_op(Conv2d::new(3, 32, 3, 1, 1, 1, 1, true))
        .add_op(ReLU::new())
        .add_op(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
        .add_op(Conv2d::new(32, 64, 3, 1, 1, 1, 1, true))
        .add_op(ReLU::new())
        .add_op(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false));

    // Benchmark conv stack
    let conv_input = randn(&[8, 3, 32, 32]);
    let conv_throughput = Throughput::Elements((8 * 3 * 32 * 32) as u64);
    group.throughput(conv_throughput);

    group.bench_with_input(
        BenchmarkId::new("ConvStack", "feature_extraction"),
        &(conv_stack, conv_input),
        |b, (model, input)| {
            b.iter(|| {
                let output = model.forward(black_box(input));
                black_box(output)
            })
        },
    );

    // Benchmark MLP
    let mlp_input = randn(&[32, 784]);
    let mlp_throughput = Throughput::Elements((32 * 784) as u64);
    group.throughput(mlp_throughput);

    group.bench_with_input(
        BenchmarkId::new("MLP", "MNIST_like"),
        &(mlp, mlp_input),
        |b, (model, input)| {
            b.iter(|| {
                let output = model.forward(black_box(input));
                black_box(output)
            })
        },
    );

    group.finish();
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Efficiency");

    // Test different batch sizes to understand memory scaling
    let batch_sizes = vec![1, 4, 16, 64, 256];
    let input_size = 1024;
    let hidden_size = 512;

    for &batch_size in &batch_sizes {
        let input = randn(&[batch_size, input_size]);
        let linear = Linear::new(input_size, hidden_size, true);

        let throughput = Throughput::Elements((batch_size * input_size) as u64);
        group.throughput(throughput);

        group.bench_with_input(
            BenchmarkId::new("Memory_Scaling", batch_size),
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
    bench_conv_layers,
    bench_activation_functions,
    bench_normalization_layers,
    bench_pooling_layers,
    bench_recurrent_layers,
    bench_complete_models,
    bench_memory_efficiency
);

criterion_main!(benches);
