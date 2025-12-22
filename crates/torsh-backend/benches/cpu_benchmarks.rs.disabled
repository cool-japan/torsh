use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use torsh_backend::cpu::*;

/// Benchmark CPU-specific SIMD operations
fn bench_cpu_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_simd");
    group.measurement_time(Duration::from_secs(8));

    let sizes = [256, 1024, 4096, 16384, 65536];

    for size in sizes.iter() {
        let a = vec![1.0f32; *size];
        let b = vec![2.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_add_f32", size),
            size,
            |bench, &size| {
                let mut result = vec![0.0f32; size];
                bench.iter(|| {
                    simd::simd_add_f32(black_box(&a), black_box(&b), black_box(&mut result));
                    black_box(&result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_mul_f32", size),
            size,
            |bench, &size| {
                let mut result = vec![0.0f32; size];
                bench.iter(|| {
                    simd::simd_mul_f32(black_box(&a), black_box(&b), black_box(&mut result));
                    black_box(&result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_dot_f32", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    let result = simd::simd_dot_f32(black_box(&a), black_box(&b));
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark CPU platform optimizations
fn bench_cpu_platform_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_platform_optimization");
    group.measurement_time(Duration::from_secs(6));

    group.bench_function("cpu_info_detection", |b| {
        b.iter(|| {
            let info = platform_optimization::detect_cpu_info();
            black_box(info);
        })
    });

    group.bench_function("microarchitecture_detection", |b| {
        b.iter(|| {
            let arch = platform_optimization::detect_microarchitecture();
            black_box(arch);
        })
    });

    group.bench_function("optimization_parameters", |b| {
        b.iter(|| {
            let params = platform_optimization::get_optimization_parameters();
            black_box(params);
        })
    });

    group.finish();
}

/// Benchmark CPU memory operations
fn bench_cpu_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_memory");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [1024, 8192, 65536, 262144]; // 1KB to 256KB

    for size in sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    if let Ok(memory_manager) = memory::CpuMemoryManager::new() {
                        if let Ok(ptr) = memory_manager.allocate_raw(black_box(size), 64) {
                            black_box(ptr);
                        }
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cache_aligned_access", size),
            size,
            |b, &size| {
                let data = vec![1u8; size];
                b.iter(|| {
                    let optimizer = memory_patterns::AccessPatternOptimizer::new();
                    optimizer.optimize_sequential_access(black_box(&data));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark CPU feature detection
fn bench_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_feature_detection");
    group.measurement_time(Duration::from_secs(4));

    group.bench_function("feature_detector_creation", |b| {
        b.iter(|| {
            let detector = feature_detection::FeatureDetector::new();
            black_box(detector);
        })
    });

    group.bench_function("feature_detection_check", |b| {
        let detector = feature_detection::FeatureDetector::new();
        b.iter(|| {
            let has_simd = detector.has_feature(black_box(feature_detection::CpuFeature::SIMD));
            black_box(has_simd);
        })
    });

    group.bench_function("arch_info_detection", |b| {
        b.iter(|| {
            let info = feature_detection::get_arch_info();
            black_box(info);
        })
    });

    group.finish();
}

/// Benchmark CPU convolution operations
fn bench_cpu_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_convolution");
    group.measurement_time(Duration::from_secs(12));

    group.bench_function("convolution_config", |b| {
        b.iter(|| {
            let config = convolution::ConvolutionConfig {
                input_shape: vec![1, 3, 32, 32],
                kernel_shape: vec![16, 3, 3, 3],
                stride: vec![1, 1],
                padding: vec![1, 1],
                dilation: vec![1, 1],
                groups: 1,
            };
            black_box(config);
        })
    });

    group.bench_function("algorithm_selection", |b| {
        let config = convolution::ConvolutionConfig {
            input_shape: vec![1, 3, 224, 224],
            kernel_shape: vec![64, 3, 7, 7],
            stride: vec![2, 2],
            padding: vec![3, 3],
            dilation: vec![1, 1],
            groups: 1,
        };

        b.iter(|| {
            let ops = convolution::CpuConvolutionOps::new();
            let algorithm = ops.select_algorithm(black_box(&config));
            black_box(algorithm);
        })
    });

    group.finish();
}

/// Benchmark CPU RNN operations
fn bench_cpu_rnn(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_rnn");
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("rnn_config_creation", |b| {
        b.iter(|| {
            let config = rnn::RnnConfig {
                input_size: 512,
                hidden_size: 1024,
                num_layers: 2,
                bias: true,
                batch_first: true,
                dropout: 0.1,
                bidirectional: false,
                cell_type: rnn::RnnCellType::LSTM,
            };
            black_box(config);
        })
    });

    group.bench_function("weight_buffer_calculation", |b| {
        let config = rnn::RnnConfig {
            input_size: 256,
            hidden_size: 512,
            num_layers: 1,
            bias: true,
            batch_first: true,
            dropout: 0.0,
            bidirectional: false,
            cell_type: rnn::RnnCellType::LSTM,
        };

        b.iter(|| {
            let size = rnn::calculate_weight_buffer_size_lstm(black_box(&config));
            black_box(size);
        })
    });

    group.finish();
}

/// Benchmark CPU auto-tuning
fn bench_cpu_autotuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_autotuning");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("autotuner_creation", |b| {
        let config = autotuning::TuningConfig {
            chunk_sizes: vec![64, 256, 1024],
            thread_counts: vec![1, 2, 4],
            optimization_level: autotuning::OptimizationLevel::Balanced,
            cache_results: true,
        };

        b.iter(|| {
            let tuner = autotuning::AutoTuner::new(black_box(config.clone()));
            black_box(tuner);
        })
    });

    group.bench_function("performance_measurement", |b| {
        b.iter(|| {
            let measurement = autotuning::measure_performance(|| {
                // Simulate computational work
                let mut sum = 0;
                for i in 0..10000 {
                    sum += i * i;
                }
                black_box(sum);
            });
            black_box(measurement);
        })
    });

    group.finish();
}

/// Benchmark CPU optimized kernels
fn bench_cpu_optimized_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_optimized_kernels");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [64, 256, 1024];

    for size in sizes.iter() {
        let a = vec![1.0f32; *size];
        let b = vec![2.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("optimized_dot", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    let result = optimized_kernels::optimized_dot(black_box(&a), black_box(&b));
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel_sum", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    let result = optimized_kernels::parallel_sum(black_box(&a));
                    black_box(result);
                })
            },
        );
    }

    // Matrix benchmarks
    let matrix_size = 128;
    let matrix_a = vec![1.0f32; matrix_size * matrix_size];
    let matrix_b = vec![2.0f32; matrix_size * matrix_size];

    group.bench_function("optimized_matmul", |b| {
        b.iter(|| {
            let result = optimized_kernels::optimized_matmul_basic(
                black_box(&matrix_a),
                black_box(&matrix_b),
                black_box(matrix_size),
                black_box(matrix_size),
                black_box(matrix_size),
            );
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    cpu_benches,
    bench_cpu_simd_operations,
    bench_cpu_platform_optimization,
    bench_cpu_memory_operations,
    bench_cpu_feature_detection,
    bench_cpu_convolution,
    bench_cpu_rnn,
    bench_cpu_autotuning,
    bench_cpu_optimized_kernels
);

criterion_main!(cpu_benches);
