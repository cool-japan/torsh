//! Advanced benchmarks for cutting-edge neural network components in torsh-nn
//!
//! This benchmark suite measures the performance of advanced neural network layers
//! including attention mechanisms, transformer components, and research features.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_nn::container::Sequential;
use torsh_nn::layers::*;
use torsh_nn::research::*;
use torsh_nn::Module;
use torsh_tensor::creation::*;

/// Benchmark attention mechanisms
fn bench_attention_mechanisms(c: &mut Criterion) {
    let configs = vec![
        ("Small", 64, 8, 32), // embed_dim, num_heads, seq_len
        ("Medium", 256, 8, 64),
        ("Large", 512, 16, 128),
    ];

    let mut group = c.benchmark_group("Attention_Mechanisms");

    for (size_name, embed_dim, num_heads, seq_len) in configs {
        let batch_sizes = vec![1, 4, 16];

        for &batch_size in &batch_sizes {
            let input = randn(&[batch_size, seq_len, embed_dim]);
            let throughput = Throughput::Elements((batch_size * seq_len * embed_dim) as u64);
            group.throughput(throughput);

            // Multi-head attention
            let mha = MultiheadAttention::new(embed_dim, num_heads, 0.1, true).unwrap();
            group.bench_with_input(
                BenchmarkId::new(
                    format!("MultiheadAttention_{}_{}", size_name, batch_size),
                    "self_attention",
                ),
                &(mha, input.clone()),
                |b, (layer, input)| {
                    b.iter(|| {
                        let output = layer.forward(black_box(input));
                        black_box(output)
                    })
                },
            );

            // Scaled dot-product attention (if available)
            if embed_dim % num_heads == 0 {
                let head_dim = embed_dim / num_heads;
                let q = randn(&[batch_size, seq_len, embed_dim]);
                let k = randn(&[batch_size, seq_len, embed_dim]);
                let v = randn(&[batch_size, seq_len, embed_dim]);

                group.bench_function(
                    BenchmarkId::new(
                        format!("ScaledDotProduct_{}_{}", size_name, batch_size),
                        "attention",
                    ),
                    |b| {
                        b.iter(|| {
                            // Simplified attention computation for benchmarking
                            let q_scaled = q.div_scalar((head_dim as f32).sqrt()).unwrap();
                            let scores = q_scaled.matmul(&k.transpose(-2, -1).unwrap()).unwrap();
                            let attn_weights = scores.softmax(-1).unwrap();
                            let output = attn_weights.matmul(&v).unwrap();
                            black_box(output)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark transformer components
fn bench_transformer_components(c: &mut Criterion) {
    let configs = vec![
        ("Small", 128, 4, 512), // d_model, nhead, dim_feedforward
        ("Medium", 256, 8, 1024),
        ("Large", 512, 16, 2048),
    ];

    let mut group = c.benchmark_group("Transformer_Components");

    for (size_name, d_model, nhead, dim_feedforward) in configs {
        let seq_len = 32;
        let batch_size = 8;
        let input = randn(&[batch_size, seq_len, d_model]);
        let throughput = Throughput::Elements((batch_size * seq_len * d_model) as u64);
        group.throughput(throughput);

        // Transformer encoder layer
        let encoder_layer =
            TransformerEncoderLayer::new(d_model, nhead, dim_feedforward, 0.1, "relu").unwrap();

        group.bench_with_input(
            BenchmarkId::new(format!("TransformerEncoder_{}", size_name), "forward"),
            &(encoder_layer, input.clone()),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );

        // Transformer decoder layer
        let decoder_layer =
            TransformerDecoderLayer::new(d_model, nhead, dim_feedforward, 0.1, "relu").unwrap();

        group.bench_with_input(
            BenchmarkId::new(format!("TransformerDecoder_{}", size_name), "forward"),
            &(decoder_layer, input.clone()),
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

/// Benchmark research components (Neural ODE, DARTS, etc.)
fn bench_research_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("Research_Components");

    // Neural ODE benchmarks
    let batch_size = 4;
    let hidden_dim = 64;
    let input = randn(&[batch_size, hidden_dim]);
    let throughput = Throughput::Elements((batch_size * hidden_dim) as u64);
    group.throughput(throughput);

    // Neural ODE with different solvers
    let solvers = vec!["euler", "rk4", "dopri5"];
    for solver in solvers {
        let func = Sequential::new()
            .add_op(Linear::new(hidden_dim, hidden_dim * 2, true))
            .add_op(ReLU::new())
            .add_op(Linear::new(hidden_dim * 2, hidden_dim, true));

        let ode_layer = NeuralODE::new(func, solver.to_string(), 0.1, 1e-6, 100);

        group.bench_with_input(
            BenchmarkId::new(format!("NeuralODE_{}", solver), "forward"),
            &(ode_layer, input.clone()),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );
    }

    // DARTS cell benchmarks
    let num_nodes = 4;
    let operations = vec!["conv3x3", "conv5x5", "maxpool", "avgpool", "skip"];
    let darts_cell = DARTSCell::new(hidden_dim, hidden_dim, num_nodes, operations);

    group.bench_with_input(
        BenchmarkId::new("DARTS_Cell", "architecture_search"),
        &(darts_cell, input.clone()),
        |b, (layer, input)| {
            b.iter(|| {
                let output = layer.forward(black_box(input));
                black_box(output)
            })
        },
    );

    // Capsule network components
    let in_capsules = 8;
    let out_capsules = 4;
    let in_dim = 8;
    let out_dim = 16;
    let capsule_input = randn(&[batch_size, in_capsules, in_dim]);

    let capsule_layer = CapsuleLayer::new(in_capsules, out_capsules, in_dim, out_dim, 3);

    group.bench_with_input(
        BenchmarkId::new("CapsuleLayer", "routing"),
        &(capsule_layer, capsule_input),
        |b, (layer, input)| {
            b.iter(|| {
                let output = layer.forward(black_box(input));
                black_box(output)
            })
        },
    );

    group.finish();
}

/// Benchmark graph neural networks
fn bench_graph_neural_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Graph_Neural_Networks");

    let configs = vec![
        ("Small", 16, 32, 64), // num_nodes, in_features, out_features
        ("Medium", 64, 64, 128),
        ("Large", 256, 128, 256),
    ];

    for (size_name, num_nodes, in_features, out_features) in configs {
        let batch_size = 1; // Graph operations typically use batch_size=1
        let node_features = randn(&[batch_size, num_nodes, in_features]);
        let adjacency = randn(&[num_nodes, num_nodes]); // Random adjacency matrix
        let throughput = Throughput::Elements((num_nodes * in_features) as u64);
        group.throughput(throughput);

        // Graph Convolution Layer
        let gcn_layer = GraphConvLayer::new(in_features, out_features);

        group.bench_function(
            BenchmarkId::new(format!("GraphConv_{}", size_name), "forward"),
            |b| {
                b.iter(|| {
                    let output = gcn_layer
                        .forward_with_adj(black_box(&node_features), black_box(&adjacency));
                    black_box(output)
                })
            },
        );

        // Graph Attention Layer
        let gat_layer = GraphAttentionLayer::new(in_features, out_features, 8, 0.1); // 8 attention heads

        group.bench_function(
            BenchmarkId::new(format!("GraphAttention_{}", size_name), "forward"),
            |b| {
                b.iter(|| {
                    let output = gat_layer
                        .forward_with_adj(black_box(&node_features), black_box(&adjacency));
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark quantization and mixed precision operations
fn bench_quantization_mixed_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantization_MixedPrecision");

    let input_shapes = vec![
        ("Small", vec![8, 128]),
        ("Medium", vec![32, 512]),
        ("Large", vec![64, 2048]),
    ];

    for (size_name, shape) in input_shapes {
        let input = randn(&shape);
        let throughput = Throughput::Elements(shape.iter().product::<usize>() as u64);
        group.throughput(throughput);

        // Standard linear layer for comparison
        let linear_fp32 = Linear::new(shape[1], shape[1] / 2, true);

        group.bench_with_input(
            BenchmarkId::new(format!("Linear_FP32_{}", size_name), "forward"),
            &(linear_fp32, input.clone()),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );

        // Quantization-aware training linear layer (if available)
        // This would benchmark QAT operations when implemented

        // Mixed precision operations would go here
        // This would benchmark FP16/BF16 operations when available
    }

    group.finish();
}

/// Benchmark memory efficiency with different techniques
fn bench_memory_efficiency_techniques(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Efficiency_Techniques");

    let batch_sizes = vec![1, 8, 32, 128];
    let seq_len = 128;
    let embed_dim = 512;

    for &batch_size in &batch_sizes {
        let input = randn(&[batch_size, seq_len, embed_dim]);
        let throughput = Throughput::Elements((batch_size * seq_len * embed_dim) as u64);
        group.throughput(throughput);

        // Standard attention (if available)
        let std_attention = MultiheadAttention::new(embed_dim, 8, 0.1, true).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Standard_Attention", batch_size),
            &(std_attention, input.clone()),
            |b, (layer, input)| {
                b.iter(|| {
                    let output = layer.forward(black_box(input));
                    black_box(output)
                })
            },
        );

        // Flash attention would be benchmarked here when available
        // Memory-efficient attention patterns would go here
    }

    group.finish();
}

criterion_group!(
    advanced_benches,
    bench_attention_mechanisms,
    bench_transformer_components,
    bench_research_components,
    bench_graph_neural_networks,
    bench_quantization_mixed_precision,
    bench_memory_efficiency_techniques
);

criterion_main!(advanced_benches);
