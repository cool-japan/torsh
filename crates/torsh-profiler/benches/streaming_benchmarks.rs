//! Streaming module benchmarks
//!
//! Comprehensive benchmarks for the real-time streaming capabilities
//! including adaptive bitrate, compression, and buffering performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Instant;
use torsh_profiler::{
    create_high_performance_streaming_engine, create_low_latency_streaming_engine,
    create_streaming_engine, BufferedEvent, CompressionConfig, EventPriority, ProfileEvent,
    StreamingConfig,
};

/// Benchmark streaming engine creation
fn bench_engine_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_creation");

    group.bench_function("basic_engine", |b| {
        b.iter(|| {
            let engine = create_streaming_engine();
            black_box(engine);
        });
    });

    group.bench_function("high_performance_engine", |b| {
        b.iter(|| {
            let engine = create_high_performance_streaming_engine();
            black_box(engine);
        });
    });

    group.bench_function("low_latency_engine", |b| {
        b.iter(|| {
            let engine = create_low_latency_streaming_engine();
            black_box(engine);
        });
    });

    group.finish();
}

/// Benchmark event buffering
fn bench_event_buffering(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_buffering");

    for event_count in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*event_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(event_count),
            event_count,
            |b, &event_count| {
                let engine = create_streaming_engine();
                b.iter(|| {
                    for i in 0..event_count {
                        let event = ProfileEvent {
                            name: format!("stream_op_{}", i),
                            category: "streaming".to_string(),
                            start_us: i * 100,
                            duration_us: 50,
                            thread_id: 1,
                            operation_count: Some(100),
                            flops: Some(500),
                            bytes_transferred: Some(256),
                            stack_trace: None,
                        };
                        engine.add_event(black_box(event));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark event priority calculation
fn bench_priority_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_calculation");

    let categories = vec![
        ("memory", "Memory events"),
        ("performance", "Performance events"),
        ("error", "Error events"),
        ("debug", "Debug events"),
        ("compute", "Compute events"),
    ];

    for (category, label) in categories {
        group.bench_function(label, |b| {
            let engine = create_streaming_engine();
            b.iter(|| {
                for i in 0..100 {
                    let event = ProfileEvent {
                        name: format!("op_{}", i),
                        category: category.to_string(),
                        start_us: i * 100,
                        duration_us: 50,
                        thread_id: 1,
                        operation_count: None,
                        flops: None,
                        bytes_transferred: None,
                        stack_trace: None,
                    };
                    engine.add_event(black_box(event));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark compression simulation
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    let event_sizes = vec![
        (100, "Small events"),
        (1000, "Medium events"),
        (5000, "Large events"),
    ];

    for (size, label) in event_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(label, |b| {
            use torsh_profiler::streaming::CompressionManager;

            let config = CompressionConfig {
                enabled: true,
                algorithm: torsh_profiler::CompressionAlgorithm::Zlib,
                level: 6,
                adaptive: true,
                threshold: 1024,
            };

            let manager = CompressionManager::new(config);

            b.iter(|| {
                let event = BufferedEvent {
                    event: ProfileEvent {
                        name: "compression_test".to_string(),
                        category: "benchmark".to_string(),
                        start_us: 0,
                        duration_us: 100,
                        thread_id: 1,
                        operation_count: None,
                        flops: None,
                        bytes_transferred: None,
                        stack_trace: Some("a".repeat(size)),
                    },
                    priority: EventPriority::Normal,
                    timestamp: Instant::now(),
                    size_bytes: size,
                    compressed: false,
                    category: "benchmark".to_string(),
                };

                black_box(&manager);
                black_box(event);
            });
        });
    }

    group.finish();
}

/// Benchmark adaptive bitrate controller
fn bench_adaptive_bitrate(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_bitrate");

    group.bench_function("bitrate_adjustment", |b| {
        use torsh_profiler::{streaming::AdaptiveRateController, AdaptiveBitrateConfig};

        let config = AdaptiveBitrateConfig {
            enabled: true,
            min_bitrate: 10,
            max_bitrate: 1000,
            initial_bitrate: 100,
            adaptation_threshold: 0.1,
            adjustment_factor: 1.2,
        };

        let controller = AdaptiveRateController::new(config);

        b.iter(|| {
            black_box(&controller);
        });
    });

    group.finish();
}

/// Benchmark event buffer operations
fn bench_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_operations");

    group.bench_function("add_to_buffer", |b| {
        use torsh_profiler::streaming::EventBuffer;

        let mut buffer = EventBuffer::new(10000);

        b.iter(|| {
            let event = BufferedEvent {
                event: ProfileEvent {
                    name: "buffer_test".to_string(),
                    category: "benchmark".to_string(),
                    start_us: 0,
                    duration_us: 100,
                    thread_id: 1,
                    operation_count: None,
                    flops: None,
                    bytes_transferred: None,
                    stack_trace: None,
                },
                priority: EventPriority::Normal,
                timestamp: Instant::now(),
                size_bytes: 100,
                compressed: false,
                category: "benchmark".to_string(),
            };

            buffer.add_event(black_box(event));
        });
    });

    group.finish();
}

/// Benchmark streaming configuration creation
fn bench_config_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_creation");

    group.bench_function("default_config", |b| {
        b.iter(|| {
            let config = StreamingConfig::default();
            black_box(config);
        });
    });

    group.bench_function("custom_config", |b| {
        use torsh_profiler::{AdaptiveBitrateConfig, CompressionAlgorithm, CompressionConfig};

        b.iter(|| {
            let mut config = StreamingConfig::default();
            config.adaptive_bitrate = AdaptiveBitrateConfig {
                enabled: true,
                min_bitrate: 50,
                max_bitrate: 500,
                initial_bitrate: 200,
                adaptation_threshold: 0.15,
                adjustment_factor: 1.5,
            };
            config.compression = CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Zstd,
                level: 8,
                adaptive: true,
                threshold: 2048,
            };
            black_box(config);
        });
    });

    group.finish();
}

/// Benchmark throughput at various bitrates
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for bitrate in [100, 500, 1000, 2000].iter() {
        group.throughput(Throughput::Elements(*bitrate as u64));
        group.bench_with_input(
            BenchmarkId::new("events_per_sec", bitrate),
            bitrate,
            |b, &bitrate| {
                let engine = create_streaming_engine();
                b.iter(|| {
                    for i in 0..bitrate {
                        let event = ProfileEvent {
                            name: format!("throughput_op_{}", i),
                            category: "throughput".to_string(),
                            start_us: i * 100,
                            duration_us: 50,
                            thread_id: 1,
                            operation_count: Some(100),
                            flops: Some(500),
                            bytes_transferred: Some(256),
                            stack_trace: None,
                        };
                        engine.add_event(black_box(event));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    group.bench_function("small_buffer_high_churn", |b| {
        let engine = create_low_latency_streaming_engine(); // Small buffer
        b.iter(|| {
            for i in 0..1000 {
                let event = ProfileEvent {
                    name: format!("churn_op_{}", i),
                    category: "memory".to_string(),
                    start_us: i * 10,
                    duration_us: 5,
                    thread_id: 1,
                    operation_count: Some(10),
                    flops: Some(50),
                    bytes_transferred: Some(64),
                    stack_trace: None,
                };
                engine.add_event(black_box(event));
            }
        });
    });

    group.bench_function("large_buffer_low_churn", |b| {
        let engine = create_high_performance_streaming_engine(); // Large buffer
        b.iter(|| {
            for i in 0..100 {
                let event = ProfileEvent {
                    name: format!("stable_op_{}", i),
                    category: "memory".to_string(),
                    start_us: i * 1000,
                    duration_us: 500,
                    thread_id: 1,
                    operation_count: Some(1000),
                    flops: Some(5000),
                    bytes_transferred: Some(2048),
                    stack_trace: None,
                };
                engine.add_event(black_box(event));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_engine_creation,
    bench_event_buffering,
    bench_priority_calculation,
    bench_compression,
    bench_adaptive_bitrate,
    bench_buffer_operations,
    bench_config_creation,
    bench_throughput,
    bench_memory_patterns,
);

criterion_main!(benches);
