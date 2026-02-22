use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use torsh_backend::*;
use torsh_core::{DType, Device};

/// Benchmark memory allocation operations
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Benchmark different allocation sizes
    let sizes = [1024, 8192, 65536, 1048576]; // 1KB, 8KB, 64KB, 1MB

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("allocation", size), size, |b, &size| {
            b.iter(|| {
                if let Ok(backend) = Backend::cpu() {
                    let descriptor = buffer::BufferDescriptor {
                        size,
                        usage: buffer::BufferUsage::Storage,
                        location: buffer::MemoryLocation::Host,
                    };
                    if let Ok(buffer) = backend.create_buffer(&descriptor) {
                        black_box(buffer);
                    }
                }
            })
        });
    }

    group.finish();
}

/// Benchmark device creation and management
fn bench_device_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_operations");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("cpu_device_creation", |b| {
        b.iter(|| black_box(Device::cpu()))
    });

    group.bench_function("device_features_query", |b| {
        if let Ok(device) = Device::cpu() {
            b.iter(|| black_box(device.supports_feature(&device::DeviceFeature::SIMD)))
        }
    });

    group.finish();
}

/// Benchmark kernel creation and execution
fn bench_kernel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_operations");
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("kernel_creation", |b| {
        if let Ok(backend) = Backend::cpu() {
            let descriptor = kernel::KernelDescriptor {
                name: "test_kernel".to_string(),
                source: kernel::KernelSource::Code("simple test".to_string()),
                language: kernel::KernelLanguage::Rust,
                entry_point: "main".to_string(),
                metadata: None,
            };

            b.iter(|| {
                if let Ok(kernel) = backend.create_kernel(&descriptor) {
                    black_box(kernel);
                }
            })
        }
    });

    group.finish();
}

/// Benchmark SIMD operations performance
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(10));

    // Test different array sizes for SIMD operations
    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("simd_threshold", size),
            size,
            |b, &size| b.iter(|| black_box(cpu::simd::should_use_simd(size))),
        );
    }

    group.finish();
}

/// Benchmark memory patterns and optimization
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("cache_aligned_allocation", |b| {
        b.iter(|| {
            let allocator = cpu::memory_patterns::CacheAlignedAllocator::new(64);
            if let Ok(ptr) = allocator.allocate(1024) {
                black_box(ptr);
            }
        })
    });

    group.bench_function("bandwidth_calculation", |b| {
        b.iter(|| {
            let bandwidth = cpu::memory_patterns::calculate_memory_bandwidth(
                black_box(1000000),                               // bytes
                black_box(std::time::Duration::from_millis(100)), // time
            );
            black_box(bandwidth);
        })
    });

    group.finish();
}

/// Benchmark cross-backend validation operations
fn bench_cross_backend_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_backend_validation");
    group.measurement_time(Duration::from_secs(6));

    group.bench_function("validator_creation", |b| {
        b.iter(|| {
            let validator = cross_backend_validation::CrossBackendValidator::new();
            black_box(validator);
        })
    });

    group.bench_function("floating_point_comparison", |b| {
        b.iter(|| {
            let result = cross_backend_validation::is_close_f64(
                black_box(1.0000001),
                black_box(1.0000002),
                cross_backend_validation::F64_TOLERANCE,
            );
            black_box(result);
        })
    });

    group.finish();
}

/// Benchmark auto-tuning system performance
fn bench_autotuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("autotuning");
    group.measurement_time(Duration::from_secs(12));

    group.bench_function("tuning_config_creation", |b| {
        b.iter(|| {
            let config = cpu::autotuning::TuningConfig {
                chunk_sizes: vec![64, 256, 1024],
                thread_counts: vec![1, 2, 4],
                optimization_level: cpu::autotuning::OptimizationLevel::Balanced,
                cache_results: true,
            };
            black_box(config);
        })
    });

    group.bench_function("performance_measurement", |b| {
        b.iter(|| {
            let measurement = cpu::autotuning::measure_performance(|| {
                // Simulate some work
                for i in 0..1000 {
                    black_box(i * 2);
                }
            });
            black_box(measurement);
        })
    });

    group.finish();
}

/// Benchmark quantization operations
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");
    group.measurement_time(Duration::from_secs(8));

    let data = vec![1.0f32; 1024];

    group.bench_function("int8_quantization", |b| {
        b.iter(|| {
            let params = quantization::QuantizationParams::new(255.0, 128);
            let result = quantization::quantize_to_int8(&data, &params);
            black_box(result);
        })
    });

    group.bench_function("dequantization", |b| {
        let params = quantization::QuantizationParams::new(255.0, 128);
        if let Ok(quantized) = quantization::quantize_to_int8(&data, &params) {
            b.iter(|| {
                let result = quantization::dequantize_from_int8(&quantized, &params);
                black_box(result);
            })
        }
    });

    group.finish();
}

/// Benchmark FFT operations
fn bench_fft_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_operations");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [64, 256, 1024, 4096];

    for size in sizes.iter() {
        group.bench_with_input(BenchmarkId::new("fft_plan", size), size, |b, &size| {
            b.iter(|| {
                let plan = fft::FftPlan::new_1d(black_box(size), fft::FftDirection::Forward);
                black_box(plan);
            })
        });
    }

    group.finish();
}

/// Benchmark sparse operations
fn bench_sparse_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_operations");
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("sparse_matrix_creation", |b| {
        b.iter(|| {
            let matrix = sparse_ops::SparseMatrix::new_coo(black_box(100), black_box(100));
            black_box(matrix);
        })
    });

    group.bench_function("sparse_format_conversion", |b| {
        let mut matrix = sparse_ops::SparseMatrix::new_coo(10, 10);
        for i in 0..5 {
            let _ = matrix.insert_coo(i, i, 1.0);
        }

        b.iter(|| {
            if let Ok(csr) = matrix.to_csr() {
                black_box(csr);
            }
        })
    });

    group.finish();
}

/// Benchmark profiler operations
fn bench_profiler(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("profiler_event_creation", |b| {
        let mut profiler = profiler::SimpleProfiler::new();
        b.iter(|| {
            let event = profiler.start_event(black_box("test_event"));
            black_box(event);
        })
    });

    group.bench_function("profiler_event_completion", |b| {
        let mut profiler = profiler::SimpleProfiler::new();
        b.iter(|| {
            let mut event = profiler.start_event("bench_event");
            event.finish();
            black_box(event);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_allocation,
    bench_device_operations,
    bench_kernel_operations,
    bench_simd_operations,
    bench_memory_patterns,
    bench_cross_backend_validation,
    bench_autotuning,
    bench_quantization,
    bench_fft_operations,
    bench_sparse_operations,
    bench_profiler
);

criterion_main!(benches);
